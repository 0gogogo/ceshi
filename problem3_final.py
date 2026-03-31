"""
═══════════════════════════════════════════════════════════════════════════════
问题3 最终版: 物理约束 + 数值稳定 + 贝叶斯不确定性量化
Physical-Constrained GPR Baseline + MCMC Bayesian Inference

数据规格 (来自题目):
  附件 1/2 (SiC): 波数 ~400–4000 cm⁻¹ | 反射率 0–100% | 每档 ~7000 点
  附件 3/4 (Si) : 同上

设计原则:
  ① GPR 长度尺度下界 = 3 × 条纹周期 (物理约束: 基线不得拟合条纹)
  ② 反射率归一化为小数 (÷100) → FP 模型输出量纲统一
  ③ SiC Reststrahlen 声子带 (750–1000 cm⁻¹) 自动排除
  ④ 噪声方差由 FFT 高频残差 (>3f₀) 估计 (物理意义: 仪器噪声)
  ⑤ MCMC 先验由 Sellmeier 折射率 + 文献材料参数严格约束
  ⑥ 收敛诊断: Gelman-Rubin R̂, 有效样本量 ESS, 自相关时间
═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq
from scipy.signal import find_peaks
from scipy.optimize import minimize
from scipy.stats import gaussian_kde
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel, Matern, WhiteKernel
)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 130

# ══════════════════════════════════════════════════════════════════════════════
# §0  材料物理常数与有效波数范围
# ══════════════════════════════════════════════════════════════════════════════

# SIGMA_MIN 已废弃: 改用数据驱动的 auto_detect_valid_range() 函数

# 各材料外延层折射率 (Sellmeier, 在透明区)
def n_SiC_4H(sigma_cm1):
    """4H-SiC 寻常光 Sellmeier: n²=A+B·λ²/(λ²-C²), λ单位μm"""
    sigma_cm1 = np.asarray(sigma_cm1, dtype=float)
    lam   = np.clip(1e4 / np.clip(sigma_cm1, 100, None), 1e-3, 1e3)
    A, B, C2 = 5.5230, 1.6466, 0.02827
    n2    = np.clip(A + B * lam**2 / (lam**2 - C2), 1.0, 30.0)
    return np.sqrt(n2)

def n_Si_IR(sigma_cm1):
    """Si 红外 Sellmeier"""
    sigma_cm1 = np.asarray(sigma_cm1, dtype=float)
    lam   = np.clip(1e4 / np.clip(sigma_cm1, 100, None), 1e-3, 1e3)
    n2    = np.clip(11.6608 + 0.2441 / (lam**2 - 0.04083), 1.0, 30.0)
    return np.sqrt(n2)

# ══════════════════════════════════════════════════════════════════════════════
# §0b  数据驱动有效波数区间检测
# ══════════════════════════════════════════════════════════════════════════════
def MAD(x, scale='normal'):
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if scale == 'normal':
        mad *= 1.4826
    return mad
def auto_detect_valid_range(sigma, R, margin_pts=20):
    """
    数据驱动检测有效干涉波数区间 — 无材料先验, 完全普适

    三条件联合掩码:
      A. R < Q75 + 2.5·IQR  → 排除高反射不透明带 (声子带/截止区)
         使用四分位距 (IQR) 而非 MAD, 对多峰分布更鲁棒
      B. R > max(Q25 - 2.5·IQR, 0.005)  → 排除低信噪比截止端
      C. |ΔR| < median(|ΔR|) + 8·σ_MAD(|ΔR|) 的连续区
         → 排除声子带边界的剧烈跳变及其邻域 (±margin_pts)

    最终取满足三条件的最长连续段 → 保证分析区域内部连续、无间断

    设计原则:
      · 不依赖任何材料常数 (无硬编码 1050/500/Reststrahlen)
      · 对 SiC (有声子带) 和 Si (无声子带) 行为一致
      · IQR 对约 50% 的数据为异常时仍稳健
      · margin_pts 缓冲区防止跳变边缘的过渡段污染分析
    """
    N = len(R)

    # ── 预处理: 鲁棒离群替换 ────────────────────────────────────────────
    # 用插值替换 [Q1, Q99] 之外的极端离群点 (仪器饱和/记录错误)
    # 目的: 防止稀疏离群点在差分中产生大量虚假跳变
    q1, q99  = np.percentile(R, [1, 99])
    outlier  = (R < q1) | (R > q99)
    R_work   = R.copy()
    if outlier.sum() > 0 and outlier.sum() < N * 0.20:
        good     = np.where(~outlier)[0]
        R_work[outlier] = np.interp(np.where(outlier)[0], good, R[good])

    # ── 条件A/B: 基于 IQR 的 R 范围约束 ─────────────────────────────────
    q25, q75 = np.percentile(R_work, [25, 75])
    iqr      = q75 - q25
    hi_A     = q75 + 2.5 * iqr          # 上界: 鲁棒排除高反射异常
    lo_B     = max(q25 - 2.5 * iqr, 0.005)  # 下界: 排除截止/过暗区

    # ── 条件C: 跳变检测 (在 R_work 上进行) ──────────────────────────────
    dR          = np.abs(np.diff(R_work))
    dR_med      = float(np.median(dR))
    dR_mad_sig  = float(MAD(dR, scale='normal'))
    jump_thresh = dR_med + 8.0 * dR_mad_sig

    bad = np.zeros(N, dtype=bool)
    jump_idx = np.where(dR > jump_thresh)[0]
    for j in jump_idx:
        bad[max(0, j - margin_pts) : min(N, j + margin_pts + 1)] = True

    # ── 联合掩码 ─────────────────────────────────────────────────────────
    valid = (R_work < hi_A) & (R_work > lo_B) & ~bad

    # ── 最长连续有效段 ────────────────────────────────────────────────────
    best_s = best_l = cur_s = cur_l = 0
    for i, v in enumerate(valid):
        if v:
            if cur_l == 0: cur_s = i
            cur_l += 1
            if cur_l > best_l:
                best_l, best_s = cur_l, cur_s
        else:
            cur_l = 0

    i0 = best_s
    i1 = best_s + best_l - 1
    return i0, i1, float(sigma[i0]), float(sigma[i1])


# ══════════════════════════════════════════════════════════════════════════════
# §1  数据加载与物理预处理
# ══════════════════════════════════════════════════════════════════════════════

def load_and_preprocess(path, material='SiC'):
    """
    加载附件数据并执行物理约束预处理

    物理约束:
      · 反射率 R ÷ 100 → 归一化为 [0, 1] 小数 (与 FP 模型量纲一致)
      · 排除 R < 0 或 R > 1 的异常点 (仪器饱和/噪声极值)
      · SiC: 排除 Reststrahlen 声子带 (750–1000 cm⁻¹)
      · 排除波数 < 500 cm⁻¹ (噪声底, 探测器截止)
      · 对非均匀采样插值到均匀网格 (FFT 的前提)
    """
    df    = pd.read_excel(path, header=0)
    sigma = df.iloc[:, 0].values.astype(float)
    R_pct = df.iloc[:, 1].values.astype(float)

    # ── 1a. 单位归一化: % → 小数 ─────────────────────────────────────────
    R = R_pct / 100.0

    # 升序排列
    idx   = np.argsort(sigma)
    sigma = sigma[idx];  R = R[idx]

    # ── 1b. 数据驱动有效区间检测 ──────────────────────────────────────────
    # 完全不依赖材料先验 (无硬编码 SIGMA_MIN)
    # 使用 auto_detect_valid_range: IQR阈值 + 跳变检测 → 最长连续有效段
    i0, i1, sigma_lo, sigma_hi = auto_detect_valid_range(sigma, R)
    sigma = sigma[i0:i1+1];  R = R[i0:i1+1]

    # 物理范围钳制 (插值误差)
    R = np.clip(R, 0.0, 1.0)

    # ── 1c. 均匀网格插值 ─────────────────────────────────────────────────
    N_grid  = len(sigma)
    sigma_u = np.linspace(sigma.min(), sigma.max(), N_grid)
    R_u     = np.clip(np.interp(sigma_u, sigma, R), 0.0, 1.0)

    ds = sigma_u[1] - sigma_u[0]
    print(f"   原始点数: {len(df)}  有效段: [{sigma_lo:.0f},{sigma_hi:.0f}] cm⁻¹  N={N_grid}")
    print(f"   反射率范围: {R_u.min():.4f}–{R_u.max():.4f}")
    print(f"   均匀步长: {ds:.4f} cm⁻¹")
    return sigma_u, R_u

# ══════════════════════════════════════════════════════════════════════════════
# §2  FFT 初始估计 (为后续 GPR 约束与 MCMC 先验提供物理基础)
# ══════════════════════════════════════════════════════════════════════════════

def fft_preestimate(sigma_u, R_u, n_func, theta_i_deg, interp=8):
    """
    Hanning 窗 + 零填充 FFT → 初始厚度估计 d₀

    同时返回:
      · 条纹周期 Δσ (cm⁻¹): GPR length_scale 下界的物理依据
      · 噪声 RMS: 从 FFT 高频成分 (>3f₀) 直接估计
        (物理意义: 频率 >3f₀ 处不应有干涉信号, 纯噪声)
    """
    ds      = sigma_u[1] - sigma_u[0]
    N       = len(R_u)

    # ── 移动平均去趋势: 必须在 FFT 前去除基线斜率 ────────────────────────
    # 仅减均值对 SiC 致命: 声子带尾巴带来的单调下降(幅度>条纹10倍)会把
    # FFT 最强峰压在极低频率(对应基线斜率), 而非干涉条纹频率
    # 窗口 = 15% 谱宽, 远大于任何合理条纹周期(确保只去基线)
    _ma_w  = max(int(N * 0.15) | 1, 21)
    _bl_rough = np.convolve(R_u, np.ones(_ma_w) / _ma_w, mode='same')
    R_dt   = R_u - _bl_rough          # 去趋势后仅剩振荡 + 噪声

    win     = np.hanning(N)
    R_w     = R_dt * win
    N_pad   = N * interp

    freqs   = rfftfreq(N_pad, d=ds)
    amps    = np.abs(rfft(R_w, n=N_pad))
    amps[0] = 0.0

    # ── 找基频峰 ──────────────────────────────────────────────────────────
    pks, _ = find_peaks(amps, prominence=amps.max() * 0.04, distance=5)
    if len(pks) == 0:
        pks = np.array([np.argmax(amps)])
    peak_i = pks[np.argmax(amps[pks])]

    # 抛物线插值精化 (sub-bin)
    if 1 <= peak_i < len(amps) - 1:
        y0, y1, y2 = amps[peak_i-1], amps[peak_i], amps[peak_i+1]
        denom  = y0 - 2*y1 + y2
        offset = 0.5*(y0 - y2) / (denom + 1e-30) if abs(denom) > 1e-30 else 0.0
        f_peak = freqs[peak_i] + offset * (freqs[1] - freqs[0])
    else:
        f_peak = freqs[peak_i]

    # ── 厚度估计 ──────────────────────────────────────────────────────────
    sigma_c = sigma_u.mean()
    n_c     = float(np.asarray(n_func(sigma_c)).flat[0])
    sin_r   = np.sin(np.radians(theta_i_deg)) / n_c
    cos_r   = float(np.sqrt(max(0.0, 1.0 - sin_r**2)))
    d_um    = f_peak / (2.0 * n_c * cos_r) * 1e4   # cm→μm

    # ── 条纹周期 (物理约束 GPR 用) ────────────────────────────────────────
    fringe_period_cm1 = 1.0 / (f_peak + 1e-30)   # cm⁻¹

    # ── 高频噪声 RMS: 取 FFT 振幅 > 3f₀ 的均方根 ────────────────────────
    noise_mask  = freqs > 3.0 * f_peak
    noise_level = float(np.sqrt(np.mean(amps[noise_mask]**2)) / (N_pad / 2))
    # 换算为时域幅值 (归一化)
    noise_rms   = max(noise_level, 1e-5)

    print(f"   FFT d₀ = {d_um:.3f} μm  |  f₀ = {f_peak:.5f} cm  |  "
          f"条纹周期 = {fringe_period_cm1:.1f} cm⁻¹")
    print(f"   FFT 高频噪声 RMS = {noise_rms:.6f}  (归一化小数)")

    return d_um, freqs, amps, f_peak, fringe_period_cm1, n_c, cos_r, noise_rms

# ══════════════════════════════════════════════════════════════════════════════
# §3  物理约束 GPR 基线估计
# ══════════════════════════════════════════════════════════════════════════════

def gpr_baseline_physical(sigma_u, R_u, fringe_period_cm1,
                          n_subsample=300, verbose=True):
    """
    物理约束高斯过程基线估计
    ─────────────────────────────────────────────────────────────────
    核心物理约束:
      length_scale_min = 3 × fringe_period_cm1
        → 强制基线只能拟合比条纹慢3倍以上的包络变化
        → 防止 GPR 把干涉条纹当"基线"吸收掉

    归一化坐标:
      x_norm = (σ - σ̄) / (σ_max - σ_min)  ∈ [-0.5, 0.5]
      length_scale 下界对应: l_min_norm = 3 × fringe_period / span

    核结构: C(x,x') = C₀ · Matern(ν=3/2, l) + σ_n²δ
      · Matern ν=3/2: 一阶导连续 (物理基线应光滑但非无穷可微)
      · WhiteKernel: 吸收非基线残差 (条纹 + 噪声)
      · ConstantKernel: 估计基线幅值尺度

    子采样策略 (7000点→300点):
      · 均匀间隔取样保证频谱覆盖
      · 保留波段边界处的点 (防止 GP 外推振荡)
    """
    span   = sigma_u[-1] - sigma_u[0]

    # ── 物理约束: length_scale 下界 ───────────────────────────────────────
    ls_min_phys = 3.0 * fringe_period_cm1          # cm⁻¹
    ls_max_phys = 0.5 * span                        # 最大半谱宽
    ls_init_phys= max(ls_min_phys * 2.0, span * 0.1)

    # 归一化为 [-0.5, 0.5] 坐标
    ls_min_norm = ls_min_phys / span
    ls_max_norm = ls_max_phys / span
    ls_init_norm= ls_init_phys / span

    # 安全约束: 确保 ls_min < ls_max (短谱段时可能违反)
    if ls_min_norm >= ls_max_norm:
        ls_min_norm = ls_max_norm * 0.1   # 降级: 允许更小的 length_scale

    if verbose:
        print(f"   GPR length_scale 约束: "
              f"[{ls_min_phys:.0f}, {ls_max_phys:.0f}] cm⁻¹  "
              f"(物理: 3×条纹周期 ≤ l ≤ 半谱宽)")

    # ── 子采样 ────────────────────────────────────────────────────────────
    N = len(sigma_u)
    step = max(1, N // n_subsample)
    idx_s = np.arange(0, N, step)
    # 确保包含端点
    idx_s = np.unique(np.concatenate([idx_s, [0, N//4, N//2, 3*N//4, N-1]]))
    xs = (sigma_u[idx_s] - sigma_u.mean()) / span   # 归一化
    ys = R_u[idx_s]

    # ── 核函数 ────────────────────────────────────────────────────────────
    kernel = (
        ConstantKernel(
            constant_value    = float(np.var(R_u)),
            constant_value_bounds = (1e-6, 1.0)
        ) *
        Matern(
            length_scale        = ls_init_norm,
            length_scale_bounds = (ls_min_norm, ls_max_norm),
            nu=1.5
        ) +
        WhiteKernel(
            noise_level        = 1e-4,
            noise_level_bounds = (1e-8, 0.1)
        )
    )

    gpr = GaussianProcessRegressor(
        kernel             = kernel,
        alpha              = 1e-9,
        normalize_y        = True,
        n_restarts_optimizer = 8
    )
    gpr.fit(xs.reshape(-1, 1), ys)

    # ── 预测全网格 ────────────────────────────────────────────────────────
    x_all = (sigma_u - sigma_u.mean()) / span
    bl_mu, bl_std = gpr.predict(x_all.reshape(-1, 1), return_std=True)

    # ── 物理合理性检验 ────────────────────────────────────────────────────
    # 基线必须介于数据 min/max 之间
    bl_mu = np.clip(bl_mu, R_u.min() * 0.95, R_u.max() * 1.05)
    # 基线标准差不能大于数据幅值 (否则 GP 在外推)
    bl_std = np.clip(bl_std, 1e-6, float(R_u.std()) * 2.0)

    # ── 提取最优 length_scale (物理单位) ─────────────────────────────────
    fitted_ls_norm = gpr.kernel_.get_params().get(
        'k1__k2__length_scale',
        gpr.kernel_.get_params().get('k1__length_scale', ls_init_norm)
    )
    fitted_ls_phys = fitted_ls_norm * span

    if verbose:
        print(f"   GPR 拟合 length_scale = {fitted_ls_phys:.1f} cm⁻¹  "
              f"({'✓ >' + str(int(ls_min_phys)) + ' cm⁻¹' if fitted_ls_phys >= ls_min_phys else '⚠ 违反下界!'})")
        print(f"   基线噪声 σ_BL(均值) = {bl_std.mean():.6f}")

    R_det = R_u - bl_mu
    return R_det, bl_mu, bl_std, gpr, fitted_ls_phys

# ══════════════════════════════════════════════════════════════════════════════
# §4  Fabry-Pérot 物理模型 (数值稳定版)
# ══════════════════════════════════════════════════════════════════════════════

def fresnel_coeffs(n1, n2, theta_i_deg):
    """
    菲涅耳 s偏振振幅系数 (n₀=1 空气)
    数值稳定性保护:
      · 分母加 ε 防止 n1=n2 时除以零
      · sin_r₂ > 1 时 (全内反射极限) 用临界角处理
    """
    ti   = np.radians(theta_i_deg)
    c0   = np.cos(ti)
    sr1  = np.sin(ti) / max(n1, 1e-6)
    cr1  = float(np.sqrt(max(0.0, 1.0 - sr1**2)))
    sr2  = n1 * sr1 / max(n2, 1e-6)
    cr2  = float(np.sqrt(max(0.0, 1.0 - sr2**2))) if abs(sr2) <= 1.0 else 0.0

    eps = 1e-12
    r01 = (1.0*c0 - n1*cr1) / (1.0*c0 + n1*cr1 + eps)
    r12 = (n1*cr1 - n2*cr2) / (n1*cr1 + n2*cr2 + eps)
    return float(r01), float(r12), cr1

def R_FP(sigma, d_cm, n1, n2, theta_i_deg, scale=1.0, offset=0.0):
    """
    精确 Fabry-Pérot 反射率 (含所有多次反射)

    R_FP = (r₀₁² + r₁₂² + 2r₀₁r₁₂cos2δ) / (1 + r₀₁²r₁₂²+ 2r₀₁r₁₂cos2δ)
    δ(σ) = 2π·n₁·d·cosθᵣ·σ

    数值稳定性:
      · 分母 > ε (r₀₁²r₁₂² < 1 对非全反射材料恒成立)
      · scale, offset 处理仪器系统响应: R_obs = scale·R_FP + offset
      · 输出范围钳制在 [0, 1] (物理约束: 反射率不超过1)
    """
    r01, r12, cr1 = fresnel_coeffs(n1, n2, theta_i_deg)
    delta = 2.0 * np.pi * n1 * np.clip(d_cm, 1e-8, None) * cr1 * sigma
    cos2d = np.cos(2.0 * delta)

    denom = 1.0 + (r01*r12)**2 + 2*r01*r12*cos2d
    denom = np.where(np.abs(denom) < 1e-10, 1e-10, denom)  # 防零除

    R_th  = (r01**2 + r12**2 + 2*r01*r12*cos2d) / denom
    return np.clip(scale * R_th + offset, 0.0, 1.0)

# ══════════════════════════════════════════════════════════════════════════════
# §4b  数据驱动 n₂ 估计 (替换硬编码 MATERIAL_N2)
# ══════════════════════════════════════════════════════════════════════════════

def estimate_n2_from_data(R_u, n1_c, theta_i_deg):
    """
    从测量反射率谱数据驱动估计衬底折射率 n₂ 的先验中心和范围

    原理:
      · 条纹振幅 A ≈ 2|r₀₁||r₁₂| / (1 + r₀₁²r₁₂²)  ≈  2|r₀₁||r₁₂|  (小r₁₂)
      · 由振幅解出 |r₁₂|, 再由 Fresnel 近正入射公式还原 n₂
      · r₁₂ = (n₁·cosθᵣ - n₂·cosθₜ) / (n₁·cosθᵣ + n₂·cosθₜ)
        近正入射简化: r₁₂ ≈ (n₁ - n₂)/(n₁ + n₂)
        → n₂ = n₁·(1 + |r₁₂|)/(1 - |r₁₂|)  (n₂ > n₁, 衬底光密)
           或  n₂ = n₁·(1 - |r₁₂|)/(1 + |r₁₂|)  (n₂ < n₁, 取较大值)

    返回: (n2_mu, n2_range)
      · n2_mu    : 先验中心 (估计值)
      · n2_range : 允许偏移范围 (±n2_range, 作为 MCMC 支撑半宽)
    """
    # 1. 用鲁棒统计估计条纹振幅 (P90-P10 比 max-min 抗噪更好)
    A_fringe = (np.percentile(R_u, 90) - np.percentile(R_u, 10)) / 2.0

    # 2. 空气-外延层界面 Fresnel 系数 (用于解出 r12)
    _cr1 = float(np.sqrt(max(0.0, 1.0 - (np.sin(np.radians(theta_i_deg))/n1_c)**2)))
    r01  = (np.cos(np.radians(theta_i_deg)) - n1_c * _cr1) /            (np.cos(np.radians(theta_i_deg)) + n1_c * _cr1 + 1e-12)

    # 3. 估计 |r₁₂|: A ≈ 2|r01||r12| → |r12| ≈ A / (2|r01|)
    r12_est = float(np.clip(A_fringe / (2.0 * abs(r01) + 1e-8), 0.001, 0.60))

    # 4. 近正入射近似: n₂ = n₁·(1+|r12|)/(1-|r12|) or n₁·(1-|r12|)/(1+|r12|)
    n2_hi = n1_c * (1.0 + r12_est) / max(1.0 - r12_est, 0.01)
    n2_lo = n1_c * max(1.0 - r12_est, 0.01) / (1.0 + r12_est)
    # 衬底通常比外延层光密(折射率更高)
    n2_mu = n2_hi if n2_hi > n1_c else n2_lo
    # 留足余量: ±30% 的估计值作为范围 (折射率色散+掺杂影响)
    n2_range = max(abs(n2_mu - n1_c) * 0.5 + 0.05, 0.10)

    return float(n2_mu), float(n2_range)

# ══════════════════════════════════════════════════════════════════════════════
# §5  MCMC (Metropolis-Hastings) — 物理先验约束版
# ══════════════════════════════════════════════════════════════════════════════

# 参数索引
_D, _N2, _SC, _OFF = 0, 1, 2, 3

class PhysicalMCMC:
    """
    贝叶斯推断: 物理先验 + 自适应 MH 采样

    参数向量 θ = [d (cm),  n₂,  scale,  offset]

    先验设计 (来自材料物理约束):
      d      ~ LogNormal(log d₀, σ_d)    σ_d 由 FFT 峰宽估计
               支撑: [0.3·d₀, 10·d₀]     (实际样品厚度范围)
      n₂     ~ Truncated-Normal(μ_n2, 0.05)
               支撑: [n₁+0.001, n₁+n2_range]  (衬底比外延层光密)
      scale  ~ Truncated-Normal(1.0, 0.1)
               支撑: [0.70, 1.30]          (仪器响应偏差±30%)
      offset ~ Normal(0, 0.03)
               支撑: [-0.10, +0.10]        (DC偏置±10%)

    似然:
      p(R|θ) = ∏ᵢ N(Rᵢ; R_FP(σᵢ;θ), σ_noise²)
      σ_noise 由 FFT 高频残差估计 (物理依据: 仪器热噪声)

    步长自适应:
      目标接受率 Robbins-Monro: α* = 0.234 (最优 MH 接受率)
      每 200 步更新一次, 使用 AM (Adaptive Metropolis) 方案
    """

    def __init__(self, sigma_u, R_u, noise_rms,
                 n1_func, theta_i_deg,
                 d_init_cm, n2_mu, n2_range,
                 d_sigma_frac=0.30):
        self.sigma     = sigma_u
        self.R_obs     = R_u
        self.var_n     = float(np.clip(noise_rms, 1e-6, 0.5))**2
        self.n1_func   = n1_func
        self.theta     = theta_i_deg
        self.d0        = float(d_init_cm)
        self.n2_mu     = float(n2_mu)
        self.n2_range  = float(n2_range)
        # 外延层中心折射率 (固定, 由 Sellmeier 给出)
        self.n1_c      = float(np.asarray(n1_func(sigma_u.mean())).flat[0])

        # ── d 的物理可探测范围 (谱区间驱动, 不依赖 d0) ─────────────────
        # d_min: 谱内出现至少 1 个完整条纹
        #        Δσ_span = f_max - f_min, 1 fringe → d = 1/(2n·cosθ·Δσ)
        # d_max: 奈奎斯特极限, 条纹周期 ≥ 2·ds
        #        d = 1/(2n·cosθ·2·ds)
        _span = float(sigma_u[-1] - sigma_u[0])
        _ds   = float(sigma_u[1]  - sigma_u[0])
        _cr   = float(np.sqrt(max(0.0, 1.0 - (np.sin(np.radians(theta_i_deg))/self.n1_c)**2)))
        _d_nyq = 1.0 / (2.0 * self.n1_c * _cr * 2.0 * _ds + 1e-30)  # cm
        _d_min = 1.0 / (2.0 * self.n1_c * _cr * _span + 1e-30)       # cm
        self.d_lo  = _d_min * 0.5          # 半个最小可检测厚度
        self.d_hi  = _d_nyq * 1.5          # 奈奎斯特极限×1.5 缓冲

        # d_log_sig: LogNormal 先验宽度 — 以 d0 为中心但范围宽(1σ覆盖0.7倍)
        self.d_log_sig = float(np.clip(d_sigma_frac, 0.05, 1.0))

    # ── 先验 ────────────────────────────────────────────────────────────────
    def _log_prior(self, th):
        d, n2, sc, off = th
        # 支撑检查 (先验为零的区域)
        if not (self.d_lo < d < self.d_hi):          return -np.inf
        if not (self.n1_c + 0.001 < n2 <
                self.n1_c + self.n2_range + 0.5):    return -np.inf
        if not (0.50 < sc < 2.00):                   return -np.inf
        if not (-0.10 < off < 0.10):                 return -np.inf

        # LogNormal(d; log d₀, σ_d)
        lp  = -0.5 * ((np.log(d) - np.log(self.d0)) / self.d_log_sig)**2
        # Gaussian(n₂; n₂_μ, 0.05)
        lp += -0.5 * ((n2 - self.n2_mu) / 0.05)**2
        # Gaussian(scale; 1, 0.20) — 宽先验允许仪器响应偏差±40%
        lp += -0.5 * ((sc - 1.0) / 0.20)**2
        # Gaussian(offset; 0, 0.03)
        lp += -0.5 * (off / 0.03)**2
        return lp

    # ── 对数似然 ────────────────────────────────────────────────────────────
    def _log_lik(self, th):
        d, n2, sc, off = th
        try:
            R_pred = R_FP(self.sigma, d, self.n1_c, n2, self.theta, sc, off)
        except Exception:
            return -np.inf
        if np.any(np.isnan(R_pred)):
            return -np.inf
        return -0.5 * np.sum((self.R_obs - R_pred)**2) / self.var_n

    def _log_post(self, th):
        lp = self._log_prior(th)
        return lp + self._log_lik(th) if np.isfinite(lp) else -np.inf

    # ── MAP 初始化 ───────────────────────────────────────────────────────────
    def _map_init(self):
        """Nelder-Mead MAP, 多起点增强鲁棒性"""
        best_res, best_val = None, np.inf
        # 7个起点: d0附近±20%, 以及±50%覆盖误差较大时的情况
        starts = [
            [self.d0,       self.n2_mu,       1.0,  0.0],
            [self.d0*1.20,  self.n2_mu+0.03,  1.0,  0.0],
            [self.d0*0.80,  self.n2_mu-0.03,  1.0,  0.0],
            [self.d0*1.50,  self.n2_mu,       1.05, 0.0],
            [self.d0*0.67,  self.n2_mu,       0.95, 0.0],
            [self.d0*1.20,  self.n2_mu,       1.0,  0.01],
            [self.d0*0.80,  self.n2_mu,       1.0, -0.01],
        ]
        for x0 in starts:
            res = minimize(
                lambda x: -self._log_post(x),
                x0=x0, method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_res = res
        return best_res.x, -best_val

    # ── 主采样循环 ───────────────────────────────────────────────────────────
    def run(self, n_warmup=3000, n_samples=8000, n_chains=4, seed=42):
        """
        多链 MH-MCMC + 自适应步长

        收敛诊断:
          · Gelman-Rubin R̂: 链间/链内方差比  (收敛: R̂ < 1.05)
          · 有效样本量 ESS = n / (1 + 2·Σ τₖ)  (要求 ESS > 200)
          · 自相关时间 τ: 通过 FFT 功率谱估计
        """
        np.random.seed(seed)
        theta_map, lp_map = self._map_init()
        print(f"   MAP: d={theta_map[0]*1e4:.4f}μm  "
              f"n₂={theta_map[1]:.4f}  "
              f"scale={theta_map[2]:.4f}  "
              f"log-post={lp_map:.2f}")

        # MAP 处的 Hessian 数值估计 → 初始步长 (AM 方案)
        eps_h = 1e-5 * np.abs(theta_map)
        eps_h = np.clip(eps_h, [self.d0*1e-4, 1e-4, 1e-4, 1e-4],
                                [self.d0*0.5,  0.1,  0.1,  0.05])

        all_chains = []

        for c in range(n_chains):
            # 初始点: MAP ± 小扰动
            th_cur = theta_map + np.random.randn(4) * eps_h * 0.3
            th_cur[0] = np.clip(th_cur[0], self.d_lo*1.1, self.d_hi*0.9)
            th_cur[1] = np.clip(th_cur[1], self.n1_c+0.005,
                                self.n1_c + self.n2_range + 0.4)
            th_cur[2] = np.clip(th_cur[2], 0.52, 1.98)
            th_cur[3] = np.clip(th_cur[3], -0.09, 0.09)
            lp_cur    = self._log_post(th_cur)

            # 初始步长 (独立各参数)
            step = eps_h.copy()
            n_acc_w = 0

            # ── Warm-up: Robbins-Monro 自适应步长 ─────────────────────────
            for i in range(n_warmup):
                proposal = th_cur + np.random.randn(4) * step
                lp_prop  = self._log_post(proposal)
                if np.log(np.random.rand() + 1e-300) < lp_prop - lp_cur:
                    th_cur, lp_cur = proposal, lp_prop
                    n_acc_w += 1
                # 每 200 步调整 (Robbins-Monro 系数 γ = 1/(i+1)^0.6)
                if (i + 1) % 200 == 0:
                    rate   = n_acc_w / (i + 1)
                    gamma  = 1.0 / ((i / 200 + 1) ** 0.6)
                    step  *= np.exp(gamma * (rate - 0.234))
                    # 物理约束步长上界 (防止步长爆炸)
                    step   = np.clip(step,
                                     eps_h * 0.01,
                                     [self.d0*0.10, 0.05, 0.10, 0.05])

            # ── 采样阶段 ───────────────────────────────────────────────────
            samples = np.empty((n_samples, 4))
            n_acc   = 0
            for i in range(n_samples):
                proposal = th_cur + np.random.randn(4) * step
                lp_prop  = self._log_post(proposal)
                if np.log(np.random.rand() + 1e-300) < lp_prop - lp_cur:
                    th_cur, lp_cur = proposal, lp_prop
                    n_acc += 1
                samples[i] = th_cur

            accept_rate = n_acc / n_samples
            d_med_c     = np.median(samples[:, 0]) * 1e4
            print(f"   链{c+1}: 接受率={accept_rate:.3f}  "
                  f"d_中位数={d_med_c:.4f}μm  "
                  f"{'✓' if 0.15 < accept_rate < 0.60 else '△'}")
            all_chains.append(samples)

        # ── 收敛诊断 ────────────────────────────────────────────────────────
        diag = self._convergence_diagnostics(all_chains, n_samples)
        return all_chains, diag

    # ── 收敛诊断 ─────────────────────────────────────────────────────────────
    @staticmethod
    def _convergence_diagnostics(chains, n_samples):
        """
        三项收敛指标:
          R̂  (Gelman-Rubin): 比较链间与链内方差, <1.05 为收敛
          ESS (Effective Sample Size): n / τ_int, 要求 >200
          τ_int (积分自相关时间): 由功率谱估计
        """
        n_param = 4
        param_names = ['d', 'n₂', 'scale', 'offset']
        M  = len(chains)

        # 丢弃前 50% 作为额外 burn-in
        half = n_samples // 2
        chains_trim = [c[half:] for c in chains]
        n   = chains_trim[0].shape[0]

        Rhat = np.zeros(n_param)
        ESS  = np.zeros(n_param)
        tau  = np.zeros(n_param)

        for p in range(n_param):
            chain_data = np.array([c[:, p] for c in chains_trim])  # (M, n)
            chain_mu   = chain_data.mean(axis=1)
            chain_var  = chain_data.var(axis=1, ddof=1)
            grand_mu   = chain_mu.mean()

            # R̂
            B_n  = np.var(chain_mu, ddof=1)         # 链间方差 (÷n后)
            W    = chain_var.mean()                   # 链内方差
            var_p = (1 - 1/n) * W + B_n
            Rhat[p] = np.sqrt(var_p / (W + 1e-15))

            # 自相关时间 (用第一条链, FFT 法)
            x    = chain_data[0] - chain_data[0].mean()
            fft_x= np.abs(rfft(x))**2
            acf_full = np.real(np.fft.ifft(
                np.concatenate([fft_x, fft_x[1:-1][::-1]])
            ))[:n]
            acf  = acf_full / (acf_full[0] + 1e-15)
            # 截断: 第一个负值处
            cutoff = next((i for i in range(1, len(acf)) if acf[i] < 0), len(acf)//4)
            tau[p]  = 1.0 + 2.0 * np.sum(acf[1:cutoff])
            ESS[p]  = M * n / max(tau[p], 1.0)

        print(f"\n   ── 收敛诊断 ─────────────────────────────────────────")
        for p, nm in enumerate(param_names):
            rhat_ok = '✓' if Rhat[p] < 1.05 else ('△' if Rhat[p] < 1.10 else '✗')
            ess_ok  = '✓' if ESS[p] > 200 else '△'
            print(f"   {nm:8s}:  R̂={Rhat[p]:.4f}{rhat_ok}  "
                  f"ESS={ESS[p]:.0f}{ess_ok}  τ_int={tau[p]:.1f}")

        return dict(Rhat=Rhat, ESS=ESS, tau=tau,
                    param_names=param_names, converged=bool(np.all(Rhat < 1.10)))

# ══════════════════════════════════════════════════════════════════════════════
# §6  Bootstrap 不确定性
# ══════════════════════════════════════════════════════════════════════════════

def bootstrap_d(sigma_u, R_u, n1_func, theta_i_deg,
                d_init_cm, n2_mu, n1_c, n_boot=300, n_sub=250):
    """
    有放回 Bootstrap 重采样 → 数据层面不确定性

    每次重采样后做 MAP 点估计 (Nelder-Mead), 统计 d 的经验分布
    稀疏化到 n_sub 点加速 (7000 → 250)
    """
    step = max(1, len(sigma_u) // n_sub)
    su   = sigma_u[::step]
    Ru   = R_u[::step]
    N    = len(su)
    d_bs = []

    for _ in range(n_boot):
        idx  = np.sort(np.random.choice(N, N, replace=True))
        su_b = su[idx];  Ru_b = Ru[idx]

        def _obj(x):
            d, n2, sc, off = x
            if d <= 0 or n2 <= n1_c + 0.001: return 1e9
            try:
                rp = R_FP(su_b, d, n1_c, n2, theta_i_deg, sc, off)
                return float(np.mean((Ru_b - rp)**2))
            except Exception:
                return 1e9

        res = minimize(_obj, x0=[d_init_cm, n2_mu, 1.0, 0.0],
                       method='Nelder-Mead',
                       options={'maxiter': 4000, 'xatol': 1e-11})
        if res.fun < 1.0:
            d_bs.append(res.x[0] * 1e4)

    d_bs = np.array(d_bs)
    valid = (d_bs > d_init_cm*0.3*1e4) & (d_bs < d_init_cm*10*1e4)
    return d_bs[valid]

# ══════════════════════════════════════════════════════════════════════════════
# §7  完整误差预算 (GUM)
# ══════════════════════════════════════════════════════════════════════════════

def error_budget(chains_flat, d_boot, n1_func, theta_i_deg, d_med_um,
                 delta_theta_deg=0.2, delta_n_frac=0.002):
    """
    GUM 合成不确定度

    u₁ (MCMC后验):  后验样本 d 的标准差
                    → 包含: 仪器噪声 + 模型不确定性
    u₂ (Bootstrap): d 的 Bootstrap 标准差
                    → 包含: 数据采样 + 基线估计误差
    u₃ (角度误差):  灵敏度系数 × Δθ
                    ∂d/∂θᵢ ≈ d·sinθᵢcosθᵢ / (n²−sin²θᵢ)  (解析导数)
    u₄ (折射率不确定性): |∂d/∂n| · Δn = d/n · Δn_frac

    合成: u_c² = u₁² + u₂² + u₃² + u₄²
    扩展: U = k·u_c   k=2 对应 ~95% (正态假设)
    """
    d_post = chains_flat[:, 0] * 1e4
    u1     = float(np.std(d_post))
    u2     = float(np.std(d_boot)) if len(d_boot) > 2 else 0.0

    # 角度灵敏度 (解析)
    n_c   = float(np.asarray(n1_func(np.array([3000.0]))).flat[0])
    ti    = np.radians(theta_i_deg)
    denom = n_c**2 - np.sin(ti)**2
    sens_angle = d_med_um * np.sin(ti) * np.cos(ti) / max(denom, 1e-6)
    u3 = abs(sens_angle) * np.radians(delta_theta_deg)

    # 折射率灵敏度
    u4 = d_med_um * delta_n_frac

    uc = float(np.sqrt(u1**2 + u2**2 + u3**2 + u4**2))
    U  = 2.0 * uc

    return {
        'u₁ MCMC后验 (噪声+模型)': u1,
        'u₂ Bootstrap (采样+基线)': u2,
        'u₃ 角度误差 (±0.2°)':     u3,
        'u₄ 折射率不确定性':        u4,
        'u_c 合成标准不确定度':      uc,
        'U   扩展不确定度 (k=2,95%)': U,
    }

# ══════════════════════════════════════════════════════════════════════════════
# §8  可视化
# ══════════════════════════════════════════════════════════════════════════════

def plot_pipeline(sigma_u, R_u, bl_mu, bl_std, R_fit,
                  freqs, amps, f0,
                  chains_flat, d_boot, budget,
                  diag, d_med, title, savepath=None):
    """
    4×2 综合图:
      行1: 原始谱 + GPR基线 + 拟合  |  GPR置信带放大
      行2: 去基线振荡 + FP拟合      |  拟合残差 + 自相关
      行3: FFT频谱 (谐波)            |  d 后验分布
      行4: d vs n₂ 联合后验          |  误差预算饼图
    """
    fig = plt.figure(figsize=(16, 18))
    gs  = gridspec.GridSpec(4, 2, figure=fig,
                            hspace=0.55, wspace=0.35)

    # ── (0,0): 原始谱 + 基线 + FP拟合 ───────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(sigma_u, R_u,    'b-', lw=0.5, alpha=0.7, label='测量 R(σ)')
    ax.plot(sigma_u, bl_mu,  'r-', lw=1.5, label='GPR 基线')
    ax.fill_between(sigma_u, bl_mu-2*bl_std, bl_mu+2*bl_std,
                    alpha=0.25, color='red', label='GPR ±2σ')
    ax.plot(sigma_u, R_fit, 'g-', lw=0.8, alpha=0.8,
            label=f'FP拟合  d={d_med:.3f}μm')
    ax.set_title(f'全谱拟合  {title}', fontsize=9)
    ax.set_xlabel('波数 (cm⁻¹)', fontsize=8); ax.set_ylabel('反射率', fontsize=8)
    ax.legend(fontsize=6, loc='upper right'); ax.grid(True, alpha=0.25)

    # ── (0,1): GPR基线置信带细节 ─────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    mid = len(sigma_u)//2
    sl  = slice(mid - 500, mid + 500)
    ax.plot(sigma_u[sl], R_u[sl],   'b-', lw=0.5, alpha=0.7)
    ax.plot(sigma_u[sl], bl_mu[sl], 'r-', lw=1.5, label='GPR 基线')
    ax.fill_between(sigma_u[sl], (bl_mu-2*bl_std)[sl], (bl_mu+2*bl_std)[sl],
                    alpha=0.30, color='red')
    ax.set_title('GPR 基线细节 (中段放大)', fontsize=9)
    ax.set_xlabel('波数 (cm⁻¹)', fontsize=8); ax.legend(fontsize=7)
    ax.grid(True, alpha=0.25)

    # ── (1,0): 去基线振荡 + FP 振荡 ─────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    R_det = R_u - bl_mu
    R_fit_det = R_fit - bl_mu
    ax.plot(sigma_u, R_det,     'b-', lw=0.5, alpha=0.6, label='ΔR 测量')
    ax.plot(sigma_u, R_fit_det, 'r-', lw=0.8, alpha=0.7, label='FP 振荡')
    ax.axhline(0, color='k', lw=0.5, ls='--')
    ax.set_title('去基线干涉振荡', fontsize=9)
    ax.set_xlabel('波数 (cm⁻¹)', fontsize=8); ax.legend(fontsize=7)
    ax.grid(True, alpha=0.25)

    # ── (1,1): 残差 + 自相关 ────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    resid = R_u - R_fit
    ax.plot(sigma_u, resid, 'g-', lw=0.4, alpha=0.7)
    ax.axhline(0, color='k', lw=0.5, ls='--')
    ax.fill_between(sigma_u, -2*resid.std(), 2*resid.std(),
                    alpha=0.12, color='gray', label=f'±2σ_res={resid.std():.5f}')
    ss_res = np.sum(resid**2); ss_tot = np.sum((R_u - R_u.mean())**2)
    R2 = 1 - ss_res/ss_tot
    ax.set_title(f'拟合残差  R²={R2:.6f}', fontsize=9)
    ax.set_xlabel('波数 (cm⁻¹)', fontsize=8); ax.legend(fontsize=7)
    ax.grid(True, alpha=0.25)

    # ── (2,0): FFT 谱 ───────────────────────────────────────────────────
    ax = fig.add_subplot(gs[2, 0])
    mask = freqs > 0
    ax.plot(freqs[mask]*1e4, amps[mask], 'b-', lw=0.7)  # 频率×1e4→μm⁻¹
    ax.axvline(f0*1e4, color='r', lw=1.5, ls='--', label=f'f₀→d₀={d_med:.3f}μm')
    for k in range(2, 5):
        fk = k*f0
        if fk < freqs[mask].max():
            ax.axvline(fk*1e4, color='orange', lw=0.8, ls=':',
                       label=f'{k}f₀' if k == 2 else None)
    ax.set_xlabel('频率 (μm⁻¹)', fontsize=8); ax.set_ylabel('FFT 幅值', fontsize=8)
    ax.set_title('FFT 频谱 & 谐波', fontsize=9)
    ax.legend(fontsize=7); ax.grid(True, alpha=0.25); ax.set_xlim(left=0)

    # ── (2,1): d 后验分布 ───────────────────────────────────────────────
    ax = fig.add_subplot(gs[2, 1])
    d_samp = chains_flat[:, 0] * 1e4
    ax.hist(d_samp, bins=80, density=True, color='steelblue',
            alpha=0.55, label='MCMC 后验')
    if len(d_samp) > 50:
        kde = gaussian_kde(d_samp, bw_method='scott')
        xs  = np.linspace(d_samp.min(), d_samp.max(), 400)
        ax.plot(xs, kde(xs), 'r-', lw=2, label='KDE')
    lo95, hi95 = np.percentile(d_samp, [2.5, 97.5])
    ax.axvspan(lo95, hi95, alpha=0.15, color='orange',
               label=f'95%CI [{lo95:.3f},{hi95:.3f}]μm')
    ax.axvline(d_med, color='darkred', lw=2, ls='--',
               label=f'中位数 {d_med:.4f}μm')
    if len(d_boot) > 5:
        ax.axvline(np.median(d_boot), color='green', lw=1.5, ls=':',
                   label=f'Bootstrap {np.median(d_boot):.4f}μm')
    ax.set_xlabel('d (μm)', fontsize=8); ax.set_ylabel('后验密度', fontsize=8)
    ax.set_title('d 后验分布 (MCMC)', fontsize=9)
    ax.legend(fontsize=6); ax.grid(True, alpha=0.25)

    # ── (3,0): d–n₂ 联合后验 ────────────────────────────────────────────
    ax = fig.add_subplot(gs[3, 0])
    n2s = chains_flat[:, 1]
    idx_sc = np.random.choice(len(d_samp), min(4000, len(d_samp)), replace=False)
    sc = ax.scatter(d_samp[idx_sc], n2s[idx_sc], s=1.5, alpha=0.25,
                    c=chains_flat[idx_sc, 2], cmap='viridis')
    plt.colorbar(sc, ax=ax, label='scale', pad=0.02)
    ax.set_xlabel('d (μm)', fontsize=8); ax.set_ylabel('n₂ (衬底)', fontsize=8)
    ax.set_title('d–n₂ 联合后验 (色=scale)', fontsize=9)
    ax.grid(True, alpha=0.25)

    # ── (3,1): 误差预算 ─────────────────────────────────────────────────
    ax = fig.add_subplot(gs[3, 1])
    keys_plot = [k for k in budget if k.startswith('u') and '合成' not in k and '扩展' not in k]
    vals_plot  = [budget[k]**2 for k in keys_plot]
    total_v    = sum(vals_plot)
    fracs      = [v/total_v*100 for v in vals_plot]
    colors_p   = ['#3A86FF','#FF006E','#FFBE0B','#8338EC']
    wedges, texts, autotexts = ax.pie(
        fracs,
        labels    = [f"{k.split(' ')[0]}\n{f:.1f}%" for k,f in zip(keys_plot, fracs)],
        autopct   = '%1.1f%%',
        colors    = colors_p[:len(keys_plot)],
        textprops = {'fontsize': 7},
        startangle= 140
    )
    ax.set_title('方差贡献 (误差预算)', fontsize=9)

    # ── 收敛信息文字框 ───────────────────────────────────────────────────
    rhat_str = "  ".join([f"R̂({nm})={rv:.3f}"
                          for nm,rv in zip(diag['param_names'][:2],
                                          diag['Rhat'][:2])])
    ess_str  = "  ".join([f"ESS({nm})={int(ev)}"
                          for nm,ev in zip(diag['param_names'][:2],
                                          diag['ESS'][:2])])
    fig.text(0.5, 0.01,
             f"收敛诊断:  {rhat_str}  |  {ess_str}  |  "
             f"{'✓ 收敛' if diag['converged'] else '⚠ 需检查'}",
             ha='center', fontsize=8,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle(f'贝叶斯推断完整报告  {title}', fontsize=12, y=1.002)
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=130, bbox_inches='tight')
        print(f"   图像保存: {savepath}")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# §9  单数据集完整流程
# ══════════════════════════════════════════════════════════════════════════════

def analyze_one(path, material, theta_i_deg,
                n2_mu, n2_range,
                n_warmup=3000, n_samples=8000, n_chains=4, n_boot=300,
                label=''):
    """
    单附件完整分析流程:
      §1 加载+物理预处理 → §2 FFT初估 → §3 物理约束GPR
      → §4 噪声估计 → §5 MCMC → §6 Bootstrap → §7 误差预算
    """
    n_func = n_SiC_4H if material == 'SiC' else n_Si_IR
    print(f"\n{'═'*64}")
    print(f"  {label}  ({material}, θᵢ={theta_i_deg}°)")
    print(f"{'═'*64}")

    # §1 加载预处理
    print("\n  [§1] 加载 + 物理预处理")
    sigma_u, R_u = load_and_preprocess(path, material)

    # §2 FFT 初始估计
    print("\n  [§2] FFT 初始估计")
    (d_fft, freqs, amps, f0,
     fringe_period, n_c, cos_r, noise_rms) = fft_preestimate(
        sigma_u, R_u, n_func, theta_i_deg)

    # §3 物理约束 GPR 基线
    print("\n  [§3] 物理约束 GPR 基线估计")
    R_det, bl_mu, bl_std, gpr, ls_fit = gpr_baseline_physical(
        sigma_u, R_u, fringe_period, n_subsample=300)

    # §3b 数据驱动 n₂ 估计 (替换硬编码 n2_mu/n2_range)
    n2_mu_data, n2_range_data = estimate_n2_from_data(R_u, n_c, theta_i_deg)
    print(f"   数据驱动 n₂ 估计: n₂_mu={n2_mu_data:.4f}  range=±{n2_range_data:.4f}")
    # 融合先验与数据估计: 若外部传入有意义的 n2_mu 则取加权平均
    if abs(n2_mu - n_c) > 0.01:   # 外部先验有效
        n2_mu    = 0.5 * n2_mu + 0.5 * n2_mu_data
        n2_range = max(n2_range, n2_range_data)
    else:
        n2_mu    = n2_mu_data
        n2_range = n2_range_data

    # §4 用 GPR 基线残差重新估计噪声 (更准确)
    #    R_u - bl_mu = 条纹 + 仪器噪声
    #    再做FFT取高频部分估计纯噪声
    R_det2 = R_u - bl_mu
    ds     = sigma_u[1] - sigma_u[0]
    Nf     = len(R_det2)
    amps2  = np.abs(rfft(R_det2))
    freqs2 = rfftfreq(Nf, d=ds)
    mask_hf = freqs2 > 3.0 * f0
    if mask_hf.sum() > 10:
        noise_rms = float(np.sqrt(np.mean(amps2[mask_hf]**2)) / (Nf/2))
    # 下界层次: max(FFT高频估计, 信号1%, FTIR仪器典型噪声底0.3%)
    # 0.003 对应归一化反射率 0.3%, 是 FTIR 仪器的物理噪声下限
    # 若低于此值则 var_n 过小 → MCMC 接受率趋近0 (似然惩罚过严)
    noise_rms = max(noise_rms, R_u.std() * 0.01, 0.003)
    print(f"   基线后噪声 RMS = {noise_rms:.6f}")

    # §5 MCMC 贝叶斯推断
    print(f"\n  [§5] MCMC ({n_chains}链 × {n_samples}采样 + {n_warmup}预热)")
    sampler = PhysicalMCMC(
        sigma_u, R_u, noise_rms,
        n_func, theta_i_deg,
        d_init_cm = d_fft * 1e-4,
        n2_mu     = n2_mu,
        n2_range  = n2_range
    )
    all_chains, diag = sampler.run(n_warmup, n_samples, n_chains)

    # 合并后半段 (额外去掉前50%作为burn-in)
    half         = n_samples // 2
    chains_flat  = np.vstack([c[half:] for c in all_chains])
    d_samp       = chains_flat[:, 0] * 1e4
    d_med        = float(np.median(d_samp))
    d_lo95, d_hi95 = np.percentile(d_samp, [2.5, 97.5])

    # MAP 后验中位数模型用于拟合图
    n2_med  = float(np.median(chains_flat[:, 1]))
    sc_med  = float(np.median(chains_flat[:, 2]))
    off_med = float(np.median(chains_flat[:, 3]))
    R_fit   = R_FP(sigma_u, d_med*1e-4, n_c, n2_med, theta_i_deg, sc_med, off_med)
    ss_res  = float(np.sum((R_u - R_fit)**2))
    ss_tot  = float(np.sum((R_u - R_u.mean())**2))
    R2      = 1.0 - ss_res/ss_tot

    print(f"\n   MCMC 结果: d = {d_med:.4f} μm  "
          f"95%CI = [{d_lo95:.4f}, {d_hi95:.4f}] μm  R² = {R2:.6f}")

    # §6 Bootstrap
    print(f"\n  [§6] Bootstrap ({n_boot}次)")
    d_boot = bootstrap_d(sigma_u, R_u, n_func, theta_i_deg,
                         d_fft*1e-4, n2_mu, n_c, n_boot=n_boot)
    if len(d_boot) > 2:
        print(f"   Bootstrap: d = {np.median(d_boot):.4f} ± {np.std(d_boot):.4f} μm  "
              f"(n={len(d_boot)})")

    # §7 误差预算
    print(f"\n  [§7] 误差预算")
    budget = error_budget(chains_flat, d_boot, n_func, theta_i_deg, d_med)
    for k, v in budget.items():
        pct = f"{'━'*max(1,int(v/max(budget.values())*20))}"
        print(f"   {k:<40s}: {v:.5f} μm  {pct}")

    # §8 可视化
    plot_pipeline(
        sigma_u, R_u, bl_mu, bl_std, R_fit,
        freqs, amps, f0,
        chains_flat, d_boot, budget, diag,
        d_med, label,
        savepath=f"{label.replace(' ','_').replace('/','_')}.png"
    )

    # 最终输出
    U = budget['U   扩展不确定度 (k=2,95%)']
    print(f"\n  ╔══════════════════════════════════════════════════════╗")
    print(f"  ║  {label}")
    print(f"  ║  d = {d_med:.4f} ± {U:.4f} μm  (k=2, 95%)")
    print(f"  ║  95%CI: [{d_lo95:.4f}, {d_hi95:.4f}] μm")
    print(f"  ║  R² = {R2:.6f}   模型: FP精确")
    print(f"  ║  R̂(d)={diag['Rhat'][0]:.4f}  ESS(d)={int(diag['ESS'][0])}")
    print(f"  ╚══════════════════════════════════════════════════════╝")

    return dict(
        d_med=d_med, d_lo95=d_lo95, d_hi95=d_hi95, U=U,
        R2=R2, diag=diag, budget=budget,
        d_boot=d_boot, chains_flat=chains_flat,
        sigma_u=sigma_u, R_u=R_u, R_fit=R_fit,
        freqs=freqs, amps=amps, f0=f0,
        d_fft=d_fft, n_c=n_c, n2_med=n2_med,
        theta=theta_i_deg, material=material, label=label
    )


# ══════════════════════════════════════════════════════════════════════════════
# §10  双角度加权合并
# ══════════════════════════════════════════════════════════════════════════════

def combine_two_angles(res1, res2, material):
    """
    加权最小方差合并两独立估计
    w_i = 1/u_i²  →  d_combined = Σwᵢdᵢ/Σwᵢ  u_combined = 1/√(Σwᵢ)
    一致性检验: Z = |d₁-d₂|/√(u₁²+u₂²) < 2 为一致
    """
    d1 = res1['d_med'];  u1 = res1['budget']['u_c 合成标准不确定度']
    d2 = res2['d_med'];  u2 = res2['budget']['u_c 合成标准不确定度']
    w1, w2 = 1/u1**2, 1/u2**2
    d_comb = (w1*d1 + w2*d2) / (w1+w2)
    u_comb = np.sqrt(1/(w1+w2))
    Z      = abs(d1-d2) / np.sqrt(u1**2+u2**2)

    print(f"\n{'═'*64}")
    print(f"  {material} 双角度合并")
    print(f"{'═'*64}")
    print(f"  θ={res1['theta']}°: d={d1:.4f}±{u1:.4f}μm")
    print(f"  θ={res2['theta']}°: d={d2:.4f}±{u2:.4f}μm")
    print(f"  Z一致性检验 = {Z:.3f}  {'✓ (Z<2)' if Z<2 else '△ (Z≥2, 差异偏大)'}")
    print(f"\n  合并结果: d = {d_comb:.4f} ± {2*u_comb:.4f} μm  (k=2, 95%)")
    return d_comb, u_comb


# ══════════════════════════════════════════════════════════════════════════════
# §11  主程序
# ══════════════════════════════════════════════════════════════════════════════

def main():
    FILES = {
        'SiC_10': ('附件1.xlsx', 'SiC', 10),
        'SiC_15': ('附件2.xlsx', 'SiC', 15),
        'Si_10' : ('附件3.xlsx', 'Si',  10),
        'Si_15' : ('附件4.xlsx', 'Si',  15),
    }

    # SiC 衬底: n₂ ≈ 2.75–2.85 (重掺, 自由载流子降低实部)
    # Si  衬底: n₂ ≈ 3.44–3.56 (重掺)
    MATERIAL_N2 = {
        'SiC': (2.80, 0.20),   # (均值, 允许范围)
        'Si' : (3.50, 0.20),
    }

    MCMC_PARAMS = dict(
        n_warmup=3000, n_samples=8000, n_chains=4, n_boot=300
    )

    results = {}
    for key, (fname, mat, ang) in FILES.items():
        try:
            n2_mu, n2_range = MATERIAL_N2[mat]
            res = analyze_one(
                fname, mat, ang, n2_mu, n2_range,
                label=f"{mat} {'附件'+fname[2]} θ={ang}°",
                **MCMC_PARAMS
            )
            results[key] = res
        except FileNotFoundError:
            print(f"\n  ⚠  {fname} 未找到, 跳过")
        except Exception as e:
            print(f"\n  ✗  {key} 分析失败: {e}")
            import traceback; traceback.print_exc()

    # 双角度合并
    for mat in ['SiC', 'Si']:
        k10 = f'{mat}_10'; k15 = f'{mat}_15'
        if k10 in results and k15 in results:
            combine_two_angles(results[k10], results[k15], mat)

    # 最终汇总
    print(f"\n{'╔'+'═'*60+'╗'}")
    print(f"║{'最终结果汇总 (Fabry-Pérot + MCMC + GPR基线)':^52}║")
    print(f"╠{'═'*60}╣")
    for key, res in results.items():
        U = res['budget']['U   扩展不确定度 (k=2,95%)']
        conv = '✓' if res['diag']['converged'] else '△'
        print(f"║  {res['label']:<28}: "
              f"d={res['d_med']:8.4f}±{U:.4f}μm  "
              f"R²={res['R2']:.5f}  {conv}║")
    print(f"╚{'═'*60}╝")


if __name__ == '__main__':
    main()
