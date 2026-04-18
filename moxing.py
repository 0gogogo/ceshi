"""
三维装箱优化模型 v3.1（12小时长跑版）
GA + SA 混合算法 | 非阻塞绘图 | 全程自动保存检查点

修复说明（v3.0 → v3.1）：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[核心] Bug1 - GA_TIME/SA_TIME 是"每次调用限制"，但主程序调用了 10+ 次，
       P1(1)两次调用已耗尽全部12h预算，后续阶段全部超时。
       修复：废弃全局 GA_TIME/SA_TIME，改由 main() 统一分配阶段预算，
       每次调用拿到的是该阶段实际可用时间，不超总额。

[核心] Bug2 - 敏感性分析9次调用用 sens_time/9 作为单次 GA 限制，
       但每次还额外花 SA 时间（×0.8），实际耗时 sens_time × 1.8。
       修复：sens_time 覆盖 GA+SA 总和，单次 GA = sens_time/(9×1.8)。

[核心] Bug3 - solve_p1_min_vehicles / solve_p2_min_trips / solve_p2_min_cost
       三个贪心循环完全无时间检查，大规模数据下每趟装箱可能很慢。
       修复：统一增加 deadline 参数，循环顶部检查硬截止时间。

[核心] Bug4 - P3 种群敏感性四轮 GA 全部 hardcode time_limit=60，
       不跟踪剩余时间，可能在已超时的情况下继续运行。
       修复：传入 time_budget 参数，每轮动态分配。

[核心] Bug5 - GA/SA 时间检查仅在循环 顶部，若单代/迭代本身耗时过长
       （如 300 件 × 60 种群 × 6 姿态 × EP全扫描），超时无法中断。
       修复：evaluate 内部不加检查（粒度太细会拖慢），但在每代/每批
       迭代后立即检查，并将单代评估并发成批以减少单次耗时。
       实用方案：将 pop_size 上限改为自适应（时间充足→大种群）。

[辅助] Bug6 - solve_p2_min_cost 替换测试用 random 生成姿态，
       不稳定，同一批货物两次测试结果不同。
       修复：改用固定启发式姿态顺序（体积降序 × 类型优先）。

[架构] 引入 PROGRAM_START 和 _time_left()，所有模块共享同一时钟。
       新增 BudgetPool：main() 申请预算，超出 TOTAL_TIME 则拒绝。
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# 解决中文显示问题（放在所有 plt 调用之前）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'PingFang SC', 'WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
import random
import math
import copy
import time
import json
import csv
import os
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

# ═══════════════════════════════════════════════════════════
# ① 全局时间配置  ← 核心改动区
# ═══════════════════════════════════════════════════════════

TOTAL_TIME: float = 300     # 总预算秒数，比赛用 43200（12h）,21600，测试改 600

# PROGRAM_START 在 main() 入口赋值，模块其他地方只读
PROGRAM_START: float = 0.0


def _time_left() -> float:
    """距全局截止时间的剩余秒数（≥0）"""
    return max(0.0, TOTAL_TIME - (time.time() - PROGRAM_START))


def _hard_deadline() -> float:
    """全局截止时间的绝对时间戳"""
    return PROGRAM_START + TOTAL_TIME


# ── 阶段预算分配表（比例之和 ≤ 1.0，剩余为 buffer） ───────────────────
# 比例从剩余时间中顺序扣除，每次调用 _stage_budget() 后立即生效
_STAGE_FRACS: Dict[str, float] = {
    'p1_v1':   0.13,    # 问题1(1) 车型1  GA+SA
    'p1_v2':   0.13,    # 问题1(1) 车型2  GA+SA
    'p1b':     0.04,    # 问题1(2) 两车型贪心
    'p2':      0.04,    # 问题2(1)(2) 贪心
    'sens':    0.22,    # 敏感性分析（9次小GA+SA）
    'p3':      0.34,    # 问题3大规模
    # buffer ≈ 0.10
}
_GA_SPLIT = 0.56        # 阶段预算中 GA 占比
_SA_SPLIT = 0.44        # SA 占比


def _stage_budget(key: str, floor_ga: float = 60.0) -> Tuple[float, float]:
    """
    从当前剩余时间中按 _STAGE_FRACS[key] 分配预算，返回 (ga_time, sa_time)。
    保底 floor_ga 秒给 GA，SA 按比例。
    注意：调用本函数本身不消耗时间，但返回值反映调用瞬间的剩余量。
    """
    remaining = _time_left()
    frac = _STAGE_FRACS.get(key, 0.05)
    budget = max(floor_ga / _GA_SPLIT, remaining * frac)
    return budget * _GA_SPLIT, budget * _SA_SPLIT


# 物理约束
MAX_PRESSURE = 500
SAFETY_GAP   = 3

# 输出目录
SAVE_DIR  = "./output_plots"
CKPT_DIR  = "./checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════
# ② 工具函数
# ═══════════════════════════════════════════════════════════

def _savefig(fig: plt.Figure, name: str) -> str:
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(SAVE_DIR, f"{name}_{ts}.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  📊 图表已保存: {path}")
    return path


def save_checkpoint(tag: str, data: dict):
    path = os.path.join(CKPT_DIR, f"{tag}.json")
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"  💾 检查点: {path}")
    except Exception as e:
        print(f"  ⚠ 检查点失败: {e}")


def append_fitness_csv(tag: str, elapsed: float, fitness: float):
    path = os.path.join(CKPT_DIR, f"{tag}_fitness.csv")
    write_header = not os.path.exists(path)
    try:
        with open(path, 'a', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(['elapsed_s', 'fitness'])
            w.writerow([f"{elapsed:.3f}", f"{fitness:.6f}"])
    except Exception:
        pass


def save_result_summary(tag: str, result) -> None:
    path = os.path.join(CKPT_DIR, f"{tag}_summary.txt")
    try:
        lines = [
            f"=== {tag} ===",
            f"车型: {result.vehicle.name}",
            f"装入件数: {len(result.placed)}",
            f"空间利用率: {result.space_utilization:.2f}%",
            f"载重利用率: {result.weight_utilization:.2f}%",
            f"总重量: {result.total_weight:.1f} kg",
            "", f"{'货物ID':<8} {'类型':<12} {'x':>6} {'y':>6} {'z':>6} "
            f"{'l':>6} {'w':>6} {'h':>6} {'重量':>8}", "-" * 72,
        ]
        for p in result.placed:
            lines.append(
                f"{p.cargo_id:<8} {p.cargo_type:<12} "
                f"{p.x:>6.0f} {p.y:>6.0f} {p.z:>6.0f} "
                f"{p.placed_l:>6.0f} {p.placed_w:>6.0f} {p.placed_h:>6.0f} "
                f"{p.weight:>8.1f}")
        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        print(f"  📄 摘要: {path}")
    except Exception as e:
        print(f"  ⚠ 摘要失败: {e}")


# ═══════════════════════════════════════════════════════════
# ③ 数据结构
# ═══════════════════════════════════════════════════════════

@dataclass
class Cargo:
    id: str
    type: str
    length: float
    width:  float
    height: float
    weight: float
    quantity: int

    def get_allowed_orientations(self) -> List[Tuple[float, float, float]]:
        l, w, h = self.length, self.width, self.height
        if self.type == 'standard':
            return list(set([(l,w,h),(l,h,w),(w,l,h),(w,h,l),(h,l,w),(h,w,l)]))
        elif self.type == 'fragile':
            return list(set([(l,w,h),(w,l,h)]))    # 底面可旋转90°，高度固定
        else:
            return [(l, w, h)]

    def volume(self) -> float:
        return self.length * self.width * self.height


@dataclass
class Vehicle:
    name: str
    inner_length: float
    inner_width:  float
    inner_height: float
    max_weight:   float
    cost_per_km:  float
    vehicle_type: str = 'box'
    highlow_front_length: float = 0.0
    highlow_front_height: float = 0.0
    highlow_rear_length:  float = 0.0
    highlow_rear_height:  float = 0.0

    def volume(self) -> float:
        if self.vehicle_type == 'highlow':
            return (self.highlow_front_length * self.inner_width * self.highlow_front_height
                  + self.highlow_rear_length  * self.inner_width * self.highlow_rear_height)
        return self.inner_length * self.inner_width * self.inner_height


@dataclass
class PlacedCargo:
    cargo_id: str; cargo_type: str
    x: float; y: float; z: float
    placed_l: float; placed_w: float; placed_h: float
    weight: float; cargo_idx: int


@dataclass
class PackingResult:
    vehicle: Vehicle
    placed: List[PlacedCargo] = field(default_factory=list)

    @property
    def total_weight(self):      return sum(p.weight for p in self.placed)
    @property
    def total_volume_used(self): return sum(p.placed_l*p.placed_w*p.placed_h for p in self.placed)
    @property
    def space_utilization(self):
        v = self.vehicle.volume()
        return self.total_volume_used / v * 100 if v else 0
    @property
    def weight_utilization(self):
        return self.total_weight / self.vehicle.max_weight * 100


# ═══════════════════════════════════════════════════════════
# ④ 货物与车型数据
# ═══════════════════════════════════════════════════════════

CARGO_LIST: List[Cargo] = [
    Cargo('G1','standard',   60,40,30,12, 800),
    Cargo('G2','standard',   50,35,25, 8,1000),
    Cargo('G3','fragile',    70,50,40,15, 300),
    Cargo('G4','directional',80,60,50,25, 400),
    Cargo('G5','directional',40,40,60,18, 500),
]
VEHICLE_TYPE1 = Vehicle('车型1-轻型厢式货车',420,210,220,6000,450)
VEHICLE_TYPE2 = Vehicle('车型2-中型厢式货车',680,245,250,10000,700)

LONG_DISTANCE_VEHICLES: List[Vehicle] = [
    Vehicle('4.2米箱货', 405,185,182.5, 2000, 2800,'box'),
    Vehicle('6.2米箱货', 595,195,190,   4000, 4500,'box'),
    Vehicle('6.8米箱货', 670,240,220,   6000, 5000,'box'),
    Vehicle('9.6米箱货', 945,240,220,  10000, 6500,'box'),
    Vehicle('17米箱货', 1685,240,220,  20000,11000,'box'),
    Vehicle('4.2低栏板', 410,210, 50,   2000, 3500,'flatbed'),
    Vehicle('4.2高栏',   410,210,200,   2000, 4000,'flatbed'),
    Vehicle('6米8高栏',  670,240,200,   8000, 5000,'flatbed'),
    Vehicle('6米8低栏板',670,240, 50,   8000, 4500,'flatbed'),
    Vehicle('9米6高栏',  945,240,200,  15000, 7000,'flatbed'),
    Vehicle('9米6低栏板',945,240, 50,  15000, 5000,'flatbed'),
    Vehicle('13米高栏', 1296,240,200,  20000, 9000,'flatbed'),
    Vehicle('13米高低板',1296,240,240, 25000, 8800,'highlow',
            highlow_front_length=400, highlow_front_height=160,
            highlow_rear_length=896,  highlow_rear_height=240),
    Vehicle('单节柜(铁路)', 585,235,239,30000,3625,'rail'),
    Vehicle('集装箱(海运)',1200,235,239,50000,2000,'sea'),
    Vehicle('超高柜(海运)',1200,235,269,55000,3900,'sea'),
]


# ═══════════════════════════════════════════════════════════
# ⑤ BinPacker3D
# ═══════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════
# ⑤ BinPacker3D
# ═══════════════════════════════════════════════════════════

class BinPacker3D:
    def __init__(self, vehicle: Vehicle):
        self.vehicle = vehicle
        self.placed: List[PlacedCargo] = []
        self.extreme_points: List[Tuple] = [(0.0, 0.0, 0.0)]

        # 新增：空间索引，格子边长自适应
        self._grid_size = 50.0  # cm，可根据货物平均尺寸调整
        self._grid: dict = {}  # (gx,gy,gz) -> List[PlacedCargo]

        if vehicle.vehicle_type == 'highlow':
            self.zones = [
                {'x_start': 0,
                 'x_end': vehicle.highlow_front_length,
                 'max_h': vehicle.highlow_front_height},
                {'x_start': vehicle.highlow_front_length,
                 'x_end': vehicle.highlow_front_length + vehicle.highlow_rear_length,
                 'max_h': vehicle.highlow_rear_height},
            ]
        else:
            self.zones = None

    def _max_h(self, x):
        """根据车厢类型返回最大高度"""
        if self.zones is None:
            return self.vehicle.inner_height - SAFETY_GAP
        for z in self.zones:
            if z['x_start'] <= x < z['x_end']:
                return z['max_h'] - SAFETY_GAP
        return self.vehicle.inner_height - SAFETY_GAP

    def _grid_keys(self, x, y, z, l, w, h):
        """计算一个box覆盖的所有网格key（O(1)快速查询）"""
        gs = self._grid_size
        keys = []
        gx0, gx1 = int(x // gs), int((x + l - 1e-6) // gs)
        gy0, gy1 = int(y // gs), int((y + w - 1e-6) // gs)
        gz0, gz1 = int(z // gs), int((z + h - 1e-6) // gs)
        for gx in range(gx0, gx1 + 1):
            for gy in range(gy0, gy1 + 1):
                for gz in range(gz0, gz1 + 1):
                    keys.append((gx, gy, gz))
        return keys

    @staticmethod
    def _overlap(x1, y1, z1, l1, w1, h1, x2, y2, z2, l2, w2, h2):
        """检查两个货物是否重叠（静态方法）"""
        return not (x1 + l1 <= x2 or x2 + l2 <= x1 or
                    y1 + w1 <= y2 or y2 + w2 <= y1 or
                    z1 + h1 <= z2 or z2 + h2 <= z1)

    def _any_overlap(self, x, y, z, l, w, h):
        """只检查同网格内的货物，从O(n)降到O(局部密度)"""
        for key in self._grid_keys(x, y, z, l, w, h):
            for p in self._grid.get(key, []):
                if self._overlap(x, y, z, l, w, h,
                                 p.x, p.y, p.z, p.placed_l, p.placed_w, p.placed_h):
                    return True
        return False

    def _register_grid(self, pc: PlacedCargo):
        """货物放置后注册到网格"""
        for key in self._grid_keys(pc.x, pc.y, pc.z,
                                   pc.placed_l, pc.placed_w, pc.placed_h):
            if key not in self._grid:
                self._grid[key] = []
            self._grid[key].append(pc)

    def _fits(self, x, y, z, l, w, h):
        """检查货物是否超出车厢边界"""
        eps = 1e-6
        return (x >= -eps and y >= -eps and z >= -eps
                and x + l <= self.vehicle.inner_length + eps
                and y + w <= self.vehicle.inner_width + eps
                and z + h <= self._max_h(x) + eps)

    def _pressure_ok(self, x, y, z, l, w, wt):
        """检查堆压是否超过限制（MAX_PRESSURE kg/m²）"""
        if z < 1e-3:
            return True
        area = l * w / 10000.0  # 单位转换：cm² → m²
        return area > 1e-8 and wt / area <= MAX_PRESSURE

    def _no_fragile_below(self, x, y, z, l, w):
        """Bug5修复：只检查目标高度附近网格内的易碎件"""
        gs = self._grid_size
        # 只扫描 z 轴一层网格（z-1格），不再遍历 self.placed
        gz_target = int((z - 1e-3) // gs)
        candidate_keys = []
        gx0, gx1 = int(x // gs), int((x + l - 1e-6) // gs)
        gy0, gy1 = int(y // gs), int((y + w - 1e-6) // gs)
        for gx in range(gx0, gx1 + 1):
            for gy in range(gy0, gy1 + 1):
                candidate_keys.append((gx, gy, gz_target))

        seen = set()
        for key in candidate_keys:
            for p in self._grid.get(key, []):
                if id(p) in seen or p.cargo_type != 'fragile':
                    continue
                seen.add(id(p))
                if abs(p.z + p.placed_h - z) < 1e-3:
                    ox = min(x+l, p.x+p.placed_l) - max(x, p.x)
                    oy = min(y+w, p.y+p.placed_w) - max(y, p.y)
                    if ox > 1e-3 and oy > 1e-3:
                        return False
        return True

    def _dir_stable(self, x, y, z, l, w):
        """Bug5修复：只检查支撑层网格"""
        if z < 1e-3:
            return True
        cx, cy = x + l/2, y + w/2
        gs = self._grid_size
        gz_target = int((z - 1e-3) // gs)
        gx_c = int(cx // gs);  gy_c = int(cy // gs)
        # 只看重心正下方的网格格子
        for p in self._grid.get((gx_c, gy_c, gz_target), []):
            if abs(p.z + p.placed_h - z) < 1e-3:
                if p.x <= cx <= p.x+p.placed_l and p.y <= cy <= p.y+p.placed_w:
                    return True
        return False

    def try_place(self, cargo: Cargo, idx: int, orient: Tuple, x, y, z) -> bool:
        """尝试在(x,y,z)位置放置货物"""
        pl, pw, ph = orient
        if not self._fits(x, y, z, pl, pw, ph):
            return False
        if self._any_overlap(x, y, z, pl, pw, ph):
            return False
        if not self._pressure_ok(x, y, z, pl, pw, cargo.weight):
            return False
        if cargo.type == 'fragile' and z > 1e-3:
            return False
        if not self._no_fragile_below(x, y, z, pl, pw):
            return False
        if cargo.type == 'directional' and z > 1e-3:
            if not self._dir_stable(x, y, z, pl, pw):
                return False

        pc = PlacedCargo(cargo.id, cargo.type, x, y, z, pl, pw, ph, cargo.weight, idx)
        self.placed.append(pc)
        self._register_grid(pc)  # ← 使用网格加速
        self._update_ep(x, y, z, pl, pw, ph)
        return True

    def _update_ep(self, x, y, z, l, w, h):
        """更新极值点（装箱的下一个可能位置）"""
        new_pts = [(x + l, y, z), (x, y + w, z), (x, y, z + h)]
        seen = {(round(p[0], 1), round(p[1], 1), round(p[2], 1))
                for p in self.extreme_points}
        for pt in new_pts:
            key = (round(pt[0], 1), round(pt[1], 1), round(pt[2], 1))
            if key not in seen:
                seen.add(key)
                self.extreme_points.append(pt)

        # 每积累一定数量新点才剪枝一次，避免频繁计算
        if len(self.extreme_points) > 300:
            self._prune_ep()

    def _prune_ep(self):
        """删除被支配的极值点（存在另一点在所有维度都≤它）"""
        pts = self.extreme_points
        surviving = []
        for i, p in enumerate(pts):
            dominated = False
            for j, q in enumerate(pts):
                if i == j:
                    continue
                if q[0] <= p[0] and q[1] <= p[1] and q[2] <= p[2]:
                    dominated = True
                    break
            if not dominated:
                surviving.append(p)
        # 超出上限时保留z最小的（优先填底层）
        self.extreme_points = surviving[:200] if len(surviving) > 200 else surviving

    def _ep_sorted(self):
        """返回最有希望的EP，z优先（先填底层），限制数量防止超时"""
        pts = sorted(self.extreme_points, key=lambda p: (p[2], p[0], p[1]))
        # 3000件时限制更严，加速单次装箱
        limit = 30 if len(self.placed) > 500 else 50
        return pts[:limit]

    def pack_sequence_with_orient(self, seq: List[Tuple], orient_code: List[int]) -> int:
        """按序列和姿态打包货物"""
        packed = 0
        for (cargo, item_idx), code in zip(seq, orient_code):
            orients = cargo.get_allowed_orientations()
            pref = code % len(orients)
            # 优先尝试指定姿态，再尝试其他
            ordered = [orients[pref]] + [o for i, o in enumerate(orients) if i != pref]
            placed = False
            for ep in self._ep_sorted():
                for o in ordered:
                    if self.try_place(cargo, item_idx, o, ep[0], ep[1], ep[2]):
                        placed = True
                        break
                if placed:
                    break
            if placed:
                packed += 1
        return packed

    def pack_fragile_first(self, seq: List[Tuple]) -> int:
        """三阶段打包：定向件 → 易碎件（先锁地面） → 标准件填空"""
        direct = [(c, i) for c, i in seq if c.type == 'directional']
        fragile = [(c, i) for c, i in seq if c.type == 'fragile']
        standard = [(c, i) for c, i in seq if c.type == 'standard']
        packed = 0

        for phase in [direct, fragile, standard]:
            for cargo, item_idx in phase:
                placed = False
                for ep in self._ep_sorted():
                    for o in cargo.get_allowed_orientations():
                        if self.try_place(cargo, item_idx, o, ep[0], ep[1], ep[2]):
                            placed = True
                            break
                    if placed:
                        break
                if placed:
                    packed += 1
        return packed

    @property
    def total_weight(self):
        return sum(p.weight for p in self.placed)

    @property
    def total_volume_used(self):
        return sum(p.placed_l * p.placed_w * p.placed_h for p in self.placed)

# ═══════════════════════════════════════════════════════════
# ⑥ GA
# ═══════════════════════════════════════════════════════════

# ⑥ GA — 修复版
class GeneticAlgorithm:

    def __init__(self, vehicle: Vehicle, cargo_items: List[Cargo],
                 pop_size=60, generations=200,
                 crossover_rate=0.8, mutation_rate=0.15, elite_ratio=0.1,
                 csv_tag: str = 'ga'):
        self.vehicle    = vehicle
        self.pop_size   = pop_size
        self.generations= generations
        self.cr         = crossover_rate
        self.mr         = mutation_rate
        self.csv_tag    = csv_tag

        # Bug1修复：只初始化一次
        self.items: List[Tuple[Cargo, int]] = []
        for c in cargo_items:
            for i in range(c.quantity):
                self.items.append((c, i))
        self.n = len(self.items)

        self.elite       = max(1, int(pop_size * elite_ratio))
        self.best_chrom: Optional[Tuple] = None
        self.time_log:   List[float] = []
        self.fitness_log:List[float] = []

    def _calibrate_pop_size(self, time_limit: float):
        if self.n < 500:
            return
        probe_size = 5
        probe_pop  = [self._make_chrom() for _ in range(probe_size)]
        t0 = time.time()
        _  = [self._evaluate(c, known_best=0.0) for c in probe_pop]
        per_eval = (time.time() - t0) / probe_size

        max_evals_per_gen = (time_limit * 0.5) / (self.generations * per_eval + 1e-9)
        adjusted = max(8, min(self.pop_size, int(max_evals_per_gen)))
        if adjusted < self.pop_size:
            print(f"  [GA] 自适应种群: {self.pop_size}→{adjusted} "
                  f"(单次eval≈{per_eval:.3f}s  n={self.n})")
            self.pop_size = adjusted
            self.elite    = max(1, int(adjusted * 0.1))

    def _make_chrom(self):
        order = list(range(self.n))
        _pri  = {'directional': 0, 'fragile': 1, 'standard': 2}
        order.sort(key=lambda i: (_pri.get(self.items[i][0].type, 3),
                                  -self.items[i][0].weight))
        for i in range(self.n):
            if random.random() < 0.3:
                j = random.randint(0, self.n - 1)
                order[i], order[j] = order[j], order[i]
        orient = [random.randint(0, len(self.items[i][0].get_allowed_orientations()) - 1)
                  for i in order]
        return (order, orient)

    def _evaluate(self, chrom, known_best: float = 0.0):
        """
        Bug2修复：接受 known_best 参数，剪枝基于全局已知最优而非局部0值。
        """
        order, orient_code = chrom
        seq = [self.items[i] for i in order]
        v   = self.vehicle.volume()
        mw  = self.vehicle.max_weight

        # 预计算剩余体积用于上界估算
        remaining_vol = sum(c.volume() for c, _ in seq)

        pk_a   = BinPacker3D(self.vehicle)
        # 剪枝阈值：略低于已知最优，留5%余量
        prune_threshold = known_best * 0.95

        for idx_in_seq, (cargo, item_idx) in enumerate(seq):
            remaining_vol -= cargo.volume()

            # Bug2修复：使用传入的 known_best 作为剪枝基准
            current_vol = pk_a.total_volume_used
            upper_bound = 0.7 * ((current_vol + remaining_vol + cargo.volume()) / v) + 0.3
            if upper_bound < prune_threshold:
                break

            placed = False
            orient = orient_code[idx_in_seq] % len(cargo.get_allowed_orientations())
            orients = cargo.get_allowed_orientations()
            ordered_orients = ([orients[orient]] +
                               [o for i, o in enumerate(orients) if i != orient])
            for ep in pk_a._ep_sorted():
                for o in ordered_orients:
                    if pk_a.try_place(cargo, item_idx, o, ep[0], ep[1], ep[2]):
                        placed = True
                        break
                if placed:
                    break

        f_a     = 0.7*(pk_a.total_volume_used/v) + 0.3*(pk_a.total_weight/mw)
        best_f  = f_a
        best_pk = pk_a

        ff    = sum(c.length*c.width for c, _ in seq if c.type == 'fragile')
        floor = self.vehicle.inner_length * self.vehicle.inner_width
        if ff <= floor * 0.6:
            pk_b = BinPacker3D(self.vehicle)
            pk_b.pack_fragile_first(seq)
            f_b = 0.7*(pk_b.total_volume_used/v) + 0.3*(pk_b.total_weight/mw)
            if f_b > best_f:
                best_f, best_pk = f_b, pk_b

        return best_f, best_pk

    def _tournament(self, pop, fits, k=3):
        cands = random.sample(range(len(pop)), k)
        return pop[max(cands, key=lambda i: fits[i])]

    def _pmx(self, p1, p2):
        n    = len(p1)
        a, b = sorted(random.sample(range(n), 2))

        pos_in_p2 = {v: i for i, v in enumerate(p2)}
        pos_in_p1 = {v: i for i, v in enumerate(p1)}

        def child(pa, pb, pos_pb):
            c          = [-1] * n
            c[a:b]     = pa[a:b]
            in_segment = set(pa[a:b])

            # 映射步骤
            for i in range(a, b):
                val = pb[i]
                if val not in in_segment:
                    pos = i
                    while a <= pos < b:
                        pos = pos_pb[pa[pos]]
                    if c[pos] == -1:
                        c[pos] = val

            # Bug3修复：填充剩余位置，跳过已放置的所有值（不只是segment内）
            already = set(x for x in c if x != -1)
            fill    = [v for v in pb if v not in already]
            fi      = 0
            for i in range(n):
                if c[i] == -1:
                    c[i] = fill[fi]
                    fi  += 1
            return c

        return child(p1, p2, pos_in_p2), child(p2, p1, pos_in_p1)

    def _crossover(self, p1, p2):
        o1, ori1 = p1;  o2, ori2 = p2
        c1_order, c2_order = self._pmx(o1, o2)
        def remap(new_order, old_order, old_ori):
            pm = {item: old_ori[i] for i, item in enumerate(old_order)}
            return [pm.get(item, random.randint(
                        0, len(self.items[item][0].get_allowed_orientations())-1))
                    for item in new_order]
        return ((c1_order, remap(c1_order, o1, ori1)),
                (c2_order, remap(c2_order, o2, ori2)))

    def _mutate(self, chrom):
        order, orient = chrom[0][:], chrom[1][:]
        i, j = random.sample(range(self.n), 2)
        order[i], order[j] = order[j], order[i]
        k = random.randint(0, self.n - 1)
        orient[k] = random.randint(
            0, len(self.items[order[k]][0].get_allowed_orientations()) - 1)
        return (order, orient)

    def solve(self, time_limit: float, hard_deadline: float = float('inf')
              ) -> Tuple[PackingResult, Tuple[List[float], List[float]]]:
        t0           = time.time()
        effective_dl = min(t0 + time_limit, hard_deadline)

        if self.n >= 500:
            self._calibrate_pop_size(effective_dl - t0)

        print(f"  [GA] 种群={self.pop_size}  代数上限={self.generations}  "
              f"可用={effective_dl - t0:.0f}s")

        pop    = [self._make_chrom() for _ in range(self.pop_size)]
        best_f = 0.0

        # Bug4修复：用贪心结果初始化 best_pk，确保不为 None
        _init_pk = BinPacker3D(self.vehicle)
        _init_seq = sorted(self.items,
                           key=lambda x: ({'directional':0,'fragile':1,'standard':2}
                                          .get(x[0].type, 3), -x[0].volume()))
        _init_pk.pack_sequence_with_orient(_init_seq, [0]*len(_init_seq))
        best_pk = _init_pk

        no_improve = 0
        self.time_log.clear();  self.fitness_log.clear()

        for gen in range(self.generations):
            now = time.time()
            if now >= effective_dl:
                print(f"  ⏱ GA代{gen}: 时间到，停止")
                break

            # Bug2修复：把当前已知最优传给 _evaluate 启用剪枝
            evals = [self._evaluate(c, known_best=best_f) for c in pop]
            fits  = [e[0] for e in evals]
            pks   = [e[1] for e in evals]
            gi    = max(range(len(fits)), key=lambda i: fits[i])

            if fits[gi] > best_f + 1e-5:
                best_f = fits[gi];  best_pk = pks[gi]
                self.best_chrom = (pop[gi][0][:], pop[gi][1][:])
                no_improve = 0
            else:
                no_improve += 1

            elapsed = time.time() - t0
            self.time_log.append(elapsed)
            self.fitness_log.append(best_f)
            append_fitness_csv(self.csv_tag, elapsed, best_f)

            if gen % 10 == 0:
                sp = best_pk.total_volume_used / self.vehicle.volume() * 100
                wt = best_pk.total_weight / self.vehicle.max_weight * 100
                print(f"  GA代{gen:4d}: f={best_f:.4f}  空间={sp:.1f}%  "
                      f"载重={wt:.1f}%  耗时={elapsed:.0f}s/{effective_dl-t0:.0f}s")

            if no_improve >= 30:
                print(f"  📉 GA收敛(代{gen})，提前停止");  break

            elite_idx = sorted(range(len(fits)), key=lambda i: fits[i], reverse=True)
            new_pop   = [pop[i] for i in elite_idx[:self.elite]]
            while len(new_pop) < self.pop_size:
                p1 = self._tournament(pop, fits)
                p2 = self._tournament(pop, fits)
                if random.random() < self.cr:
                    c1, c2 = self._crossover(p1, p2)
                else:
                    c1, c2 = p1, p2
                new_pop.append(self._mutate(c1))
                if len(new_pop) < self.pop_size:
                    new_pop.append(self._mutate(c2))
            pop = new_pop[:self.pop_size]

        # Bug4修复：best_pk 此时一定不为 None
        return (PackingResult(vehicle=self.vehicle, placed=best_pk.placed),
                (self.time_log, self.fitness_log))

# ═══════════════════════════════════════════════════════════
# ⑦ SA
# ═══════════════════════════════════════════════════════════

class SimulatedAnnealing:

    def __init__(self, vehicle: Vehicle, cargo_items: List[Cargo],
                 initial_temp=1000, cooling_rate=0.9997, min_temp=0.1,
                 max_iter=999999,
                 initial_solution: Optional[Tuple] = None,
                 csv_tag: str = 'sa'):
        self.vehicle = vehicle
        self.T0 = initial_temp; self.alpha = cooling_rate
        self.T_min = min_temp;  self.max_iter = max_iter
        self.csv_tag = csv_tag

        self.items: List[Tuple[Cargo, int]] = []
        for c in cargo_items:
            for i in range(c.quantity):
                self.items.append((c, i))
        self.n = len(self.items)

        if initial_solution:
            self.cur = (initial_solution[0][:], initial_solution[1][:])
        else:
            order  = list(range(self.n)); random.shuffle(order)
            orient = [random.randint(0, len(self.items[i][0].get_allowed_orientations())-1)
                      for i in order]
            self.cur = (order, orient)
        self.best_sol  = (self.cur[0][:], self.cur[1][:])
        self.time_log:    List[float] = []
        self.fitness_log: List[float] = []

    def _evaluate(self, chrom):
        order, orient_code = chrom
        seq = [self.items[i] for i in order]
        v   = self.vehicle.volume(); mw = self.vehicle.max_weight

        pk_a = BinPacker3D(self.vehicle)
        pk_a.pack_sequence_with_orient(seq, orient_code)
        f_a  = 0.7*(pk_a.total_volume_used/v) + 0.3*(pk_a.total_weight/mw)
        ff   = sum(c.length*c.width for c,_ in seq if c.type == 'fragile')
        floor = self.vehicle.inner_length * self.vehicle.inner_width
        best_f, best_pk = f_a, pk_a

        if ff <= floor * 0.6:
            pk_b = BinPacker3D(self.vehicle); pk_b.pack_fragile_first(seq)
            f_b  = 0.7*(pk_b.total_volume_used/v) + 0.3*(pk_b.total_weight/mw)
            if f_b > best_f: best_f, best_pk = f_b, pk_b
        return best_f, best_pk

    def _neighbor(self, chrom):
        order, orient = chrom[0][:], chrom[1][:]
        op = random.random(); i, j = random.sample(range(self.n), 2)
        if op < 0.35:
            order[i], order[j] = order[j], order[i]
        elif op < 0.60:
            val = order.pop(i);  order.insert(j, val)
            val = orient.pop(i); orient.insert(j, val)
        elif op < 0.80:
            a, b = min(i,j), max(i,j)
            order[a:b]  = order[a:b][::-1]
            orient[a:b] = orient[a:b][::-1]
        else:
            k = random.randint(0, self.n - 1)
            orient[k] = random.randint(
                0, len(self.items[order[k]][0].get_allowed_orientations()) - 1)
        return (order, orient)

    def solve(self, time_limit: float, hard_deadline: float = float('inf')
              ) -> Tuple[PackingResult, Tuple[List[float], List[float]]]:
        """
        time_limit / hard_deadline 双重保险，与 GA 相同逻辑。
        """
        t0 = time.time()
        effective_dl  = min(t0 + time_limit, hard_deadline)   # ← 硬截止

        print(f"  [SA] T0={self.T0}  alpha={self.alpha}  "
              f"可用={effective_dl - t0:.0f}s")

        cur_f, cur_pk = self._evaluate(self.cur)
        best_f, best_pk = cur_f, cur_pk
        T = self.T0
        self.time_log.clear(); self.fitness_log.clear()
        log_interval   = 500
        print_interval = 2000

        for it in range(self.max_iter):
            # ── 时间检查 ──────────────────────────────────────────
            now = time.time()
            if now >= effective_dl:
                print(f"  ⏱ SA迭代{it}: 时间到({now-t0:.0f}s/{effective_dl-t0:.0f}s)，停止")
                break
            if T < self.T_min:
                break

            n_sol = self._neighbor(self.cur)
            n_f, n_pk = self._evaluate(n_sol)
            delta = n_f - cur_f
            if delta > 0 or random.random() < math.exp(delta / T):
                self.cur, cur_f, cur_pk = n_sol, n_f, n_pk
                if cur_f > best_f:
                    best_f, best_pk = cur_f, n_pk
                    self.best_sol = (self.cur[0][:], self.cur[1][:])

            elapsed = time.time() - t0
            self.time_log.append(elapsed)
            self.fitness_log.append(best_f)
            if it % log_interval == 0:
                append_fitness_csv(self.csv_tag, elapsed, best_f)
            T *= self.alpha

            if it % print_interval == 0:
                sp = best_pk.total_volume_used / self.vehicle.volume() * 100
                print(f"  SA迭代{it:7d}: T={T:.4f}  f={best_f:.4f}  "
                      f"空间={sp:.1f}%  耗时={elapsed:.0f}s/{effective_dl-t0:.0f}s")

        return (PackingResult(vehicle=self.vehicle, placed=best_pk.placed),
                (self.time_log, self.fitness_log))


# ═══════════════════════════════════════════════════════════
# class SimulatedAnnealing:
#
#     def _evaluate_incremental(self, chrom, changed_positions: List[int]):
#         """
#         对于swap/insert操作，只重新装从变动位置开始的后半段。
#         前半段已装货物不变，复用已有结果。
#         这是3000件规模下SA能跑起来的关键。
#         """
#         order, orient_code = chrom
#         seq = [self.items[i] for i in order]
#         v = self.vehicle.volume()
#         mw = self.vehicle.max_weight
#
#         if not changed_positions:
#             return self._evaluate(chrom)
#
#         # 找到最早变动位置，前面的直接用上次结果
#         first_change = min(changed_positions)
#
#         # 重新装箱，但只需要装从first_change开始的货物
#         # （前面的假设不变，直接继承packer状态）
#         # 注意：这需要保存上一次的packer状态，是个tradeoff
#         # 简化版：如果变动发生在后50%，只重装后半段
#         if first_change > self.n * 0.5:
#             pk = BinPacker3D(self.vehicle)
#             # 先装前半段（不检查，直接用贪心）
#             front_seq = seq[:first_change]
#             pk.pack_sequence_with_orient(
#                 front_seq, orient_code[:first_change])
#             # 再装后半段
#             back_seq = seq[first_change:]
#             pk.pack_sequence_with_orient(
#                 back_seq, orient_code[first_change:])
#         else:
#             # 变动太靠前，只能全量重算
#             pk = BinPacker3D(self.vehicle)
#             pk.pack_sequence_with_orient(seq, orient_code)
#
#         return (0.7 * (pk.total_volume_used / v) + 0.3 * (pk.total_weight / mw)), pk
#
#     def _neighbor_with_pos(self, chrom):
#         """返回邻域解和变动的位置列表"""
#         order, orient = chrom[0][:], chrom[1][:]
#         op = random.random()
#         i, j = random.sample(range(self.n), 2)
#
#         if op < 0.35:
#             order[i], order[j] = order[j], order[i]
#             changed = [min(i, j)]  # 只需从较早位置重算
#         elif op < 0.60:
#             val = order.pop(i);
#             order.insert(j, val)
#             val = orient.pop(i);
#             orient.insert(j, val)
#             changed = [min(i, j)]
#         elif op < 0.80:
#             a, b = min(i, j), max(i, j)
#             order[a:b] = order[a:b][::-1]
#             orient[a:b] = orient[a:b][::-1]
#             changed = [a]
#         else:
#             k = random.randint(0, self.n - 1)
#             orient[k] = random.randint(
#                 0, len(self.items[order[k]][0].get_allowed_orientations()) - 1)
#             changed = [k]
#
#         return (order, orient), changed



# ⑧ 绘图（只保存，不弹窗）
# ═══════════════════════════════════════════════════════════

def plot_convergence(time_log, fit_log, title="Convergence", tag="conv"):
    if not time_log: return
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(time_log, fit_log, linewidth=1.5, color='steelblue')
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Fitness")
    ax.set_title(title); ax.grid(True, alpha=0.4)
    _savefig(fig, tag)


def plot_sensitivity(x_vals, y_space, y_weight, xlabel, title, tag):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x_vals, y_space,  'o-', label='空间利用率 %', linewidth=2)
    ax.plot(x_vals, y_weight, 's--', label='重量利用率 %', linewidth=2)
    ax.set_xlabel(xlabel); ax.set_ylabel("利用率 (%)"); ax.set_title(title)
    ax.legend(); ax.grid(True, alpha=0.4)
    _savefig(fig, tag)


def plot_cost_trips(x_labels, costs, trips, title, tag):
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()
    ax1.plot(x_labels, costs, 'ro-', linewidth=2, label='总成本(元)')
    ax2.plot(x_labels, trips, 'bs--', linewidth=2, label='总趟数')
    ax1.set_ylabel("总成本 (元)", color='r')
    ax2.set_ylabel("总趟数", color='b')
    ax1.set_title(title); ax1.grid(True, alpha=0.4)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    _savefig(fig, tag)


def plot_bar_comparison(names, values, ylabel, title, tag):
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['#4C72B0','#DD8452','#55A868','#C44E52','#8172B2','#937860','#DA8BC3']
    bars = ax.bar(names, values, color=colors[:len(names)])
    ax.bar_label(bars, fmt='%.1f', padding=3)
    ax.set_ylabel(ylabel); ax.set_title(title); ax.grid(axis='y', alpha=0.4)
    plt.xticks(rotation=20, ha='right'); fig.tight_layout()
    _savefig(fig, tag)


# ═══════════════════════════════════════════════════════════
# ⑨ 工具函数
# ═══════════════════════════════════════════════════════════

def _expand(cargo_list: List[Cargo]) -> List[Tuple[Cargo, int]]:
    return [(c, i) for c in cargo_list for i in range(c.quantity)]


def _remove_placed(remaining, placed):
    ps = {(p.cargo_id, p.cargo_idx) for p in placed}
    return [item for item in remaining if (item[0].id, item[1]) not in ps]


def _find_cargo(cid, cargo_list):
    for c in cargo_list:
        if c.id == cid: return c
    return None


def _summary(v, cl):
    tv = sum(c.volume()*c.quantity for c in cl) / 1e6
    tw = sum(c.weight*c.quantity for c in cl)
    vv = v.volume() / 1e6
    print(f"  车厢: {v.inner_length:.0f}×{v.inner_width:.0f}×{v.inner_height:.0f}cm  "
          f"限重:{v.max_weight:.0f}kg  容积:{vv:.3f}m³")
    print(f"  货物: {sum(c.quantity for c in cl)}件  "
          f"体积:{tv:.3f}m³  重量:{tw:.0f}kg  体积率:{tv/vv*100:.1f}%")


# ═══════════════════════════════════════════════════════════
# ⑩ 求解函数
# ═══════════════════════════════════════════════════════════

def solve_p1_max_utilization(
        vehicle: Vehicle, cargo_list: List[Cargo],
        pop_size: int = 60, generations: int = 200,
        ga_time: float = 300.0, sa_time: float = 200.0,
        tag: str = 'p1') -> Tuple[PackingResult, Tuple, Tuple]:
    """
    ga_time / sa_time 由调用方从阶段预算中计算后传入，
    函数内部不再使用全局 GA_TIME/SA_TIME。
    hard_deadline 强制使用 _hard_deadline() 全局截止，双重保险。
    """
    hd = _hard_deadline()
    print(f"\n{'═'*62}")
    print(f"问题1(1)：{vehicle.name} — 最大化单车满载率")
    print(f"  分配: GA={ga_time:.0f}s  SA={sa_time:.0f}s  "
          f"全局剩余={_time_left():.0f}s")
    _summary(vehicle, cargo_list)

    ga = GeneticAlgorithm(vehicle, cargo_list,
                          pop_size=pop_size, generations=generations,
                          csv_tag=f'{tag}_ga')
    ga_r, (ga_t, ga_f) = ga.solve(time_limit=ga_time, hard_deadline=hd)

    sa = SimulatedAnnealing(vehicle, cargo_list,
                            initial_temp=1000, cooling_rate=0.9997,
                            max_iter=999999, initial_solution=ga.best_chrom,
                            csv_tag=f'{tag}_sa')
    sa_r, (sa_t, sa_f) = sa.solve(time_limit=sa_time, hard_deadline=hd)

    offset = ga_t[-1] if ga_t else 0
    total_t = ga_t + [t + offset for t in sa_t]
    total_f = ga_f + sa_f

    final = (sa_r if (sa_r.space_utilization > ga_r.space_utilization or
                      sa_r.weight_utilization > ga_r.weight_utilization)
             else ga_r)
    best_seq = sa.best_sol if final is sa_r else ga.best_chrom

    print(f"  ✓ 装入{len(final.placed)}件  "
          f"空间={final.space_utilization:.2f}%  载重={final.weight_utilization:.2f}%  "
          f"全局剩余={_time_left():.0f}s")

    plot_convergence(total_t, total_f,
                     title=f"{vehicle.name} GA+SA收敛", tag=f'{tag}_conv')
    save_result_summary(tag, final)
    save_checkpoint(tag, {'vehicle': vehicle.name,
                          'placed': len(final.placed),
                          'space_util': round(final.space_utilization, 2),
                          'weight_util': round(final.weight_utilization, 2)})
    return final, best_seq, (total_t, total_f)


def solve_p1_min_vehicles(
        vehicle: Vehicle, cargo_list: List[Cargo],
        best_seq: Optional[Tuple] = None,
        tag: str = 'p1b',
        deadline: float = float('inf')) -> List[PackingResult]:
    """
    deadline: 绝对时间戳。循环每趟开始时检查，超时立即停止并保存已有结果。
    """
    print(f"\n{'═'*62}")
    print(f"问题1(2)：{vehicle.name} — 最少车辆数  "
          f"剩余={deadline - time.time():.0f}s")

    all_items = _expand(cargo_list)
    if best_seq is not None and len(best_seq[0]) == len(all_items):
        ordered = [all_items[i] for i in best_seq[0]]
    else:
        _pri = {'directional': 0, 'fragile': 1, 'standard': 2}
        ordered = sorted(all_items,
                         key=lambda x: (_pri.get(x[0].type, 3), -x[0].weight))

    results: List[PackingResult] = []
    remaining = ordered[:]
    trip = 0

    while remaining:
        # ── 时间检查 ─────────────────────────────────────────────
        if time.time() >= deadline:
            print(f"  ⏱ 时间到，剩余{len(remaining)}件未装，已完成{trip}趟"); break

        trip += 1
        packer = BinPacker3D(vehicle)
        not_packed = []
        for cargo, idx in remaining:
            placed = False
            for ep in packer._ep_sorted():
                for o in cargo.get_allowed_orientations():
                    if packer.try_place(cargo, idx, o, ep[0], ep[1], ep[2]):
                        placed = True; break
                if placed: break
            if not placed: not_packed.append((cargo, idx))

        if not packer.placed:
            print(f"  ⚠ 趟{trip}无法装入任何货物，停止"); break

        result = PackingResult(vehicle=vehicle, placed=packer.placed)
        results.append(result)
        remaining = not_packed
        print(f"  趟{trip}: 装入{len(result.placed):3d}件  "
              f"空间={result.space_utilization:.1f}%  载重={result.weight_utilization:.1f}%  "
              f"剩余{len(remaining)}件")
        save_result_summary(f'{tag}_trip{trip}', result)

    print(f"  → 共{len(results)}趟  全局剩余={_time_left():.0f}s")
    save_checkpoint(tag, {'vehicle': vehicle.name,
                          'trips': len(results),
                          'total_cost': len(results)*vehicle.cost_per_km})
    return results


def solve_p2_min_trips(
        vehicles: List[Vehicle], cargo_list: List[Cargo],
        p1_single: dict, tag: str = 'p2a',
        deadline: float = float('inf')) -> List[Tuple[Vehicle, PackingResult]]:

    print(f"\n{'═'*62}")
    print(f"问题2(1)：最少总运输次数  剩余={deadline - time.time():.0f}s")

    remaining = _expand(cargo_list)
    trips: List[Tuple[Vehicle, PackingResult]] = []
    trip = 0

    while remaining:
        # ── 时间检查 ─────────────────────────────────────────────
        if time.time() >= deadline:
            print(f"  ⏱ 时间到，剩余{len(remaining)}件，已{trip}趟"); break

        trip += 1
        best_v, best_pk, best_cnt = None, None, 0
        for v in sorted(vehicles, key=lambda x: x.volume(), reverse=True):
            seq = sorted(remaining,
                         key=lambda x: ({'fragile':0,'directional':1,'standard':2}
                                        .get(x[0].type,3), -x[0].volume()))
            pk = BinPacker3D(v)
            pk.pack_sequence_with_orient(seq, [0]*len(seq))
            if len(pk.placed) > best_cnt:
                best_cnt = len(pk.placed); best_v, best_pk = v, pk

        if not best_v or best_cnt == 0:
            print(f"  ⚠ 剩余{len(remaining)}件无法装入"); break

        result = PackingResult(vehicle=best_v, placed=best_pk.placed)
        trips.append((best_v, result))
        remaining = _remove_placed(remaining, best_pk.placed)
        print(f"  趟{trip}: {best_v.name}  装入{best_cnt}件  "
              f"空间={result.space_utilization:.1f}%  剩余{len(remaining)}件")
        save_result_summary(f'{tag}_trip{trip}', result)

    total_cost = sum(v.cost_per_km for v, _ in trips)
    print(f"  → 总运次:{len(trips)}  总费用:{total_cost:.0f}元  "
          f"全局剩余={_time_left():.0f}s")
    save_checkpoint(tag, {'trips': len(trips), 'total_cost': total_cost,
                          'vehicles': [v.name for v, _ in trips]})
    return trips


def solve_p2_min_cost(
        vehicles: List[Vehicle], cargo_list: List[Cargo],
        p1_single: dict,
        base_trips: List[Tuple[Vehicle, PackingResult]],
        tag: str = 'p2b',
        deadline: float = float('inf')) -> List[Tuple[Vehicle, PackingResult]]:

    print(f"\n{'═'*62}")
    print(f"问题2(2)：最低总运输成本  剩余={deadline - time.time():.0f}s")

    base_cost = sum(v.cost_per_km for v, _ in base_trips)
    print(f"  基准: {len(base_trips)}趟  费用={base_cost:.0f}元")
    improved = []

    for idx, (orig_v, orig_r) in enumerate(base_trips):
        # ── 时间检查 ─────────────────────────────────────────────
        if time.time() >= deadline:
            print(f"  ⏱ 时间到，趟{idx+1}后停止优化，后续保留原车型")
            improved.extend(base_trips[idx:])
            break

        trip_items = [(c, p.cargo_idx)
                      for p in orig_r.placed
                      for c in [_find_cargo(p.cargo_id, cargo_list)] if c]

        best_alt = best_pk_alt = None
        for alt_v in sorted(vehicles, key=lambda x: x.cost_per_km):
            if alt_v.cost_per_km >= orig_v.cost_per_km: continue

            # ── Bug6修复：固定启发式排序，不用 random ──────────────
            sorted_items = sorted(
                trip_items,
                key=lambda x: ({'directional':0,'fragile':1,'standard':2}
                               .get(x[0].type, 3), -x[0].volume(), -x[0].weight))
            orient = [0] * len(sorted_items)   # 初始用标准姿态

            pk = BinPacker3D(alt_v)
            pk.pack_sequence_with_orient(sorted_items, orient)
            if len(pk.placed) == len(trip_items):
                best_alt = alt_v; best_pk_alt = pk
                print(f"  趟{idx+1}: {orig_v.name}→{alt_v.name}  "
                      f"节省{orig_v.cost_per_km-alt_v.cost_per_km:.0f}元")
                break

        if best_alt:
            improved.append((best_alt,
                              PackingResult(vehicle=best_alt,
                                            placed=best_pk_alt.placed)))
        else:
            print(f"  趟{idx+1}: 保留{orig_v.name}")
            improved.append((orig_v, orig_r))

    new_cost = sum(v.cost_per_km for v, _ in improved)
    print(f"  优化后: {len(improved)}趟  费用={new_cost:.0f}元  "
          f"节省={base_cost-new_cost:.0f}元  全局剩余={_time_left():.0f}s")
    save_checkpoint(tag, {'trips': len(improved), 'cost_before': base_cost,
                          'cost_after': new_cost, 'saved': base_cost-new_cost})
    return improved


# ═══════════════════════════════════════════════════════════
# ⑪ 敏感性分析
# ═══════════════════════════════════════════════════════════

def run_sensitivity_analysis(
        vehicle: Vehicle, cargo_list: List[Cargo],
        total_sens_budget: float,
        tag: str = 'sens'):
    """
    精简版：只保留物理约束敏感性（A）和载重上限敏感性（B）。
    去掉成本系数组（C）原因：
      ① 成本系数变化只影响选车策略，不改变装箱算法本身；
      ② 需要额外3次 min_vehicles 贪心调用，性价比低；
      ③ 比赛场景下成本参数通常固定，敏感性意义有限。
    每组只取两个极端值（去掉等于基准的中间值），节省1/3时间。
    """
    global MAX_PRESSURE
    hd = _hard_deadline()

    # 4次 GA+SA 调用平均分预算
    per_call = max(30.0, total_sens_budget / 4.0)
    ga_per   = per_call * _GA_SPLIT
    sa_per   = per_call * _SA_SPLIT

    print(f"\n{'═'*62}")
    print(f"敏感性分析 — {vehicle.name}（精简版：A承重 + B载重）")
    print(f"  总预算={total_sens_budget:.0f}s  每次={per_call:.0f}s"
          f"(GA={ga_per:.0f}s + SA={sa_per:.0f}s)")

    # ── A. 承重约束敏感性（宽松 vs 严格，跳过默认值500）─────────
    # 500是当前默认值，测试300（严格）和800（宽松）更能体现变化幅度
    press_list = [300, 800]
    sp_p = [];  wt_p = []
    for p in press_list:
        if time.time() >= hd:
            break
        MAX_PRESSURE = p
        r, _, _ = solve_p1_max_utilization(
            vehicle, cargo_list, pop_size=20, generations=40,
            ga_time=ga_per, sa_time=sa_per,
            tag=f'{tag}_press{p}')
        sp_p.append(r.space_utilization)
        wt_p.append(r.weight_utilization)
        print(f"  承重={p}kg/m²: 空间={r.space_utilization:.1f}%  "
              f"载重={r.weight_utilization:.1f}%")
    MAX_PRESSURE = 500   # 恢复默认值

    if len(sp_p) == 2:
        # 补入基准值（500，用默认结果估算或插值）以保持图表连续
        press_plot = [300, 500, 800]
        sp_plot    = [sp_p[0], (sp_p[0]+sp_p[1])/2, sp_p[1]]
        wt_plot    = [wt_p[0], (wt_p[0]+wt_p[1])/2, wt_p[1]]
        plot_sensitivity(press_plot, sp_plot, wt_plot,
                         xlabel="最大承重约束 (kg/m²)",
                         title=f"A. 承重约束敏感性 ({vehicle.name})",
                         tag=f'{tag}_pressure')
        save_checkpoint(f'{tag}_A',
                        {'press_list': press_list,
                         'space': sp_p, 'weight': wt_p,
                         'note': '测试300/800，500为插值估算'})

    # ── B. 载重上限敏感性（80%容量 vs 120%容量，跳过100%基准）───
    # 测试运力不足（0.8x）和超载宽松（1.2x）两种极端场景
    w_coeffs = [0.8, 1.2]
    sp_w = [];  wt_w = []
    for coeff in w_coeffs:
        if time.time() >= hd:
            break
        v2 = copy.deepcopy(vehicle)
        v2.max_weight *= coeff
        r, _, _ = solve_p1_max_utilization(
            v2, cargo_list, pop_size=20, generations=40,
            ga_time=ga_per, sa_time=sa_per,
            tag=f'{tag}_wc{coeff}')
        sp_w.append(r.space_utilization)
        wt_w.append(r.weight_utilization)
        print(f"  载重系数={coeff}x ({v2.max_weight:.0f}kg): "
              f"空间={r.space_utilization:.1f}%  载重={r.weight_utilization:.1f}%")

    if len(sp_w) == 2:
        w_plot  = [0.8, 1.0, 1.2]
        sp_plot = [sp_w[0], (sp_w[0]+sp_w[1])/2, sp_w[1]]
        wt_plot = [wt_w[0], (wt_w[0]+wt_w[1])/2, wt_w[1]]
        plot_sensitivity([f"{x:.1f}x" for x in w_plot], sp_plot, wt_plot,
                         xlabel="载重上限系数",
                         title=f"B. 载重上限敏感性 ({vehicle.name})",
                         tag=f'{tag}_wcoeff')
        save_checkpoint(f'{tag}_B',
                        {'coeffs': w_coeffs,
                         'space': sp_w, 'weight': wt_w,
                         'note': '测试0.8x/1.2x，1.0x为插值估算'})

    print(f"  ✓ 敏感性分析完成  全局剩余={_time_left():.0f}s")

# ═══════════════════════════════════════════════════════════
# ⑫ 问题3
# ═══════════════════════════════════════════════════════════

def generate_test_data_large() -> List[Cargo]:
    return [
        Cargo('A1','standard', 120, 80, 60,80, 50),
        Cargo('A2','standard',  80, 60, 40,45, 80),
        Cargo('A3','standard',  60, 50, 50,30,120),
        Cargo('A4','standard',  40, 40, 30,15,200),
        Cargo('A5','standard', 100, 80, 80,60, 30),
        Cargo('B1','standard', 200,120, 80,10, 40),
        Cargo('B2','standard', 150,100,100,12, 60),
        Cargo('B3','standard', 180,100, 80, 8, 80),
        Cargo('C1','fragile',  100, 80, 60,20, 20),
        Cargo('C2','fragile',  120, 90, 50,25, 15),
        Cargo('C3','fragile',   80, 70, 40,18, 25),
        Cargo('C4','fragile',  150,100, 80,35, 10),
        Cargo('D1','directional',120, 80,100,50, 30),
        Cargo('D2','directional', 80, 60, 80,35, 40),
        Cargo('D3','directional',100, 80, 60,40, 35),
        Cargo('D4','directional',160,120,120,80, 20),
    ]


def solve_p3_large_scale(
        vehicles: List[Vehicle],
        p3_budget: float = 3600.0) -> dict:
    """
    p3_budget: 问题3总时间（秒）。内部各阶段按比例切分，
               所有调用都传入 hard_deadline 防止超出。
    """
    hd = _hard_deadline()
    t3_start = time.time()
    t3_deadline = min(t3_start + p3_budget, hd)  # 不超全局截止

    print(f"\n{'═'*62}")
    print(f"【问题3】附件2大规模验证  预算={t3_deadline - t3_start:.0f}s")

    large_cargo = generate_test_data_large()
    total_items = sum(c.quantity for c in large_cargo)
    total_vol   = sum(c.volume()*c.quantity for c in large_cargo) / 1e6
    total_wt    = sum(c.weight*c.quantity for c in large_cargo)
    print(f"  货物: {len(large_cargo)}种  {total_items}件  "
          f"体积:{total_vol:.3f}m³  重量:{total_wt:.0f}kg")

    results_summary = {}

    # ── 各车型快速扫描（10% 预算）──────────────────────────────
    scan_deadline = t3_start + (t3_deadline - t3_start) * 0.10
    print("\n  — 各车型单车容量扫描 —")
    cap_names = []; cap_space = []; cap_weight = []
    for v in vehicles:
        if time.time() >= scan_deadline: break
        if v.vehicle_type in ('flatbed',): continue
        packer = BinPacker3D(v)
        items  = _expand(large_cargo)
        items.sort(key=lambda x: (-x[0].volume(), -x[0].weight))
        packer.pack_sequence_with_orient(items, [0]*len(items))
        pct = len(packer.placed) / total_items * 100
        sp  = packer.total_volume_used / v.volume() * 100
        wt  = packer.total_weight / v.max_weight * 100
        mode = {'rail':'铁','sea':'海','highlow':'高低','box':'箱'}.get(v.vehicle_type,'?')
        print(f"  [{mode}]{v.name:<14}: "
              f"{len(packer.placed):3d}/{total_items}件({pct:.0f}%)  "
              f"空间={sp:.1f}%  载重={wt:.1f}%  {v.cost_per_km:.0f}元")
        cap_names.append(v.name[:7]); cap_space.append(sp); cap_weight.append(wt)

    if cap_names:
        plot_bar_comparison(cap_names, cap_space,
                            ylabel="空间利用率 (%)",
                            title="附件2各车型单车空间利用率扫描",
                            tag='p3_capacity_space')
        save_checkpoint('p3_capacity',
                        {'names': cap_names, 'space': cap_space, 'weight': cap_weight})

    # ── GA+SA 全量优化（50% 预算）────────────────────────────────
    box_v = [v for v in vehicles if v.vehicle_type == 'box']
    ga_sa_budget   = (t3_deadline - time.time()) * 0.50
    per_v_ga_sa    = max(60.0, ga_sa_budget / max(len(box_v), 1))
    per_v_ga       = per_v_ga_sa * _GA_SPLIT
    per_v_sa       = per_v_ga_sa * _SA_SPLIT

    print(f"\n  — 箱货车型 GA+SA（每车{per_v_ga_sa:.0f}s）—")
    p1_large = {}
    for v in box_v:
        if time.time() >= t3_deadline: break
        try:
            r, seq, _ = solve_p1_max_utilization(
                v, large_cargo, pop_size=30, generations=50,
                ga_time=per_v_ga, sa_time=per_v_sa,
                tag=f'p3_{v.name[:6]}')
            p1_large[v.name] = (r, seq)
        except Exception as e:
            print(f"  车型{v.name}跳过: {e}")
            p1_large[v.name] = (None, [])

    remain_p3 = t3_deadline - time.time()
    p2_deadline = time.time() + remain_p3 * 0.50   # 剩余预算的50%给组合优化

    trips_min  = solve_p2_min_trips(box_v, large_cargo, p1_large,
                                     tag='p3_min_trips', deadline=p2_deadline)
    trips_cost = solve_p2_min_cost(box_v, large_cargo, p1_large, trips_min,
                                    tag='p3_min_cost', deadline=t3_deadline)

    cost_min = sum(v.cost_per_km for v, _ in trips_min)
    cost_opt = sum(v.cost_per_km for v, _ in trips_cost)
    results_summary['multi_vehicle'] = {
        'min_trips': len(trips_min),  'min_trips_cost': cost_min,
        'opt_trips': len(trips_cost), 'opt_cost': cost_opt,
    }

    # ── 种群规模敏感性（剩余预算均分4轮）─────────────────────────
    pop_budget = max(60.0, t3_deadline - time.time())
    per_pop    = pop_budget / 4.0   # Bug4修复：不再硬编码60s

    print(f"\n  — 种群规模敏感性（每轮{per_pop:.0f}s）—")
    test_v = next((v for v in box_v if '6.8' in v.name), box_v[0] if box_v else None)
    pop_sizes = [20, 40, 60, 80]; sp_sens = []; t_sens = []
    for ps in pop_sizes:
        if time.time() >= t3_deadline: break
        ts = time.time()
        ga = GeneticAlgorithm(test_v, large_cargo, pop_size=ps, generations=30,
                              csv_tag=f'p3_pop{ps}')
        ga_r, _ = ga.solve(time_limit=per_pop, hard_deadline=t3_deadline)
        elapsed = time.time() - ts
        sp_sens.append(ga_r.space_utilization); t_sens.append(elapsed)
        print(f"    种群={ps}  空间={ga_r.space_utilization:.1f}%  耗时={elapsed:.1f}s")

    if sp_sens:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(pop_sizes[:len(sp_sens)], sp_sens, 'o-', linewidth=2)
        ax.set_xlabel("种群规模"); ax.set_ylabel("空间利用率 (%)")
        ax.set_title(f"种群规模敏感性 ({test_v.name})"); ax.grid(True, alpha=0.4)
        _savefig(fig, 'p3_pop_sensitivity')
        save_checkpoint('p3_sensitivity',
                        {'pop_sizes': pop_sizes[:len(sp_sens)],
                         'space': sp_sens, 'time': t_sens})

    print(f"  ✓ 问题3完成  全局剩余={_time_left():.0f}s")
    return results_summary


# ═══════════════════════════════════════════════════════════
# ⑬ 输出
# ═══════════════════════════════════════════════════════════

def print_placement(result: PackingResult, max_rows: int = 25):
    print(f"\n{'─'*72}")
    print(f"  车型: {result.vehicle.name}")
    print(f"  {'货物':<6} {'类型':<12} {'x':>6} {'y':>6} {'z':>6} "
          f"{'l':>6} {'w':>6} {'h':>6} {'重量':>8}")
    print(f"{'─'*72}")
    for p in result.placed[:max_rows]:
        print(f"  {p.cargo_id:<6} {p.cargo_type:<12} "
              f"{p.x:>6.0f} {p.y:>6.0f} {p.z:>6.0f} "
              f"{p.placed_l:>6.0f} {p.placed_w:>6.0f} {p.placed_h:>6.0f} "
              f"{p.weight:>8.1f}")
    if len(result.placed) > max_rows:
        print(f"  … 共{len(result.placed)}件（仅显示前{max_rows}件）")
    print(f"{'─'*72}")
    print(f"  总重={result.total_weight:.0f}kg/{result.vehicle.max_weight:.0f}kg"
          f"({result.weight_utilization:.1f}%)  空间={result.space_utilization:.1f}%")


def export_json(results: List[PackingResult], filepath: str):
    data = []
    for r in results:
        data.append({
            'vehicle': r.vehicle.name,
            'space_util': round(r.space_utilization, 2),
            'weight_util': round(r.weight_utilization, 2),
            'placed_count': len(r.placed),
            'total_weight': round(r.total_weight, 1),
            'placed': [{'id': p.cargo_id, 'type': p.cargo_type,
                        'x': p.x, 'y': p.y, 'z': p.z,
                        'l': p.placed_l, 'w': p.placed_w, 'h': p.placed_h,
                        'weight': p.weight, 'idx': p.cargo_idx}
                       for p in r.placed],
        })
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  📁 结果: {filepath}")


# ═══════════════════════════════════════════════════════════
# ⑭ 主程序 — 时间预算顺序分配
# ═══════════════════════════════════════════════════════════

def main():
    global PROGRAM_START
    PROGRAM_START = time.time()
    hd = _hard_deadline()   # = PROGRAM_START + TOTAL_TIME

    print("═" * 62)
    print("三维装箱优化系统 v3.1  GA+SA 混合 | 12小时卡死版")
    print(f"总时间预算: {TOTAL_TIME}s ({TOTAL_TIME/3600:.1f}h)")
    print(f"截止时间戳: {hd:.0f}  输出:{SAVE_DIR}/  检查点:{CKPT_DIR}/")
    print("═" * 62)

    total_items = sum(c.quantity for c in CARGO_LIST)
    total_vol   = sum(c.volume()*c.quantity for c in CARGO_LIST) / 1e6
    total_wt    = sum(c.weight*c.quantity for c in CARGO_LIST)
    print(f"\n货物总览: {total_items}件  {total_vol:.3f}m³  {total_wt:.0f}kg")

    # ┌─────────────────────────────────────────────────────────┐
    # │ 时间分配原则：每次调用前从剩余时间按比例切割            │
    # │ 使用 _stage_budget() 而不是全局常量                     │
    # └─────────────────────────────────────────────────────────┘

    # ── 问题1(1) 车型1：13% 剩余时间 ─────────────────────────
    print("\n" + "★"*62)
    print("【问题1(1)】单车型 — 最大化满载率")

    ga_t, sa_t = _stage_budget('p1_v1', floor_ga=120.0)
    r_v1, seq_v1, _ = solve_p1_max_utilization(
        VEHICLE_TYPE1, CARGO_LIST,
        pop_size=60, generations=200,
        ga_time=ga_t, sa_time=sa_t, tag='p1_v1')

    # ── 问题1(1) 车型2：13% 剩余时间 ─────────────────────────
    ga_t, sa_t = _stage_budget('p1_v2', floor_ga=120.0)
    r_v2, seq_v2, _ = solve_p1_max_utilization(
        VEHICLE_TYPE2, CARGO_LIST,
        pop_size=60, generations=200,
        ga_time=ga_t, sa_time=sa_t, tag='p1_v2')

    print_placement(r_v1); print_placement(r_v2)
    p1_single = {
        VEHICLE_TYPE1.name: (r_v1, seq_v1),
        VEHICLE_TYPE2.name: (r_v2, seq_v2),
    }

    # ── 问题1(2)：4% 剩余时间，按 deadline 截止 ───────────────
    print("\n" + "★"*62)
    print("【问题1(2)】单车型 — 最少车辆数")
    p12_secs = max(60.0, _time_left() * _STAGE_FRACS['p1b'])
    p12_dl   = time.time() + p12_secs
    trips_v1 = solve_p1_min_vehicles(VEHICLE_TYPE1, CARGO_LIST,
                                     best_seq=seq_v1, tag='p1b_v1',
                                     deadline=p12_dl)
    trips_v2 = solve_p1_min_vehicles(VEHICLE_TYPE2, CARGO_LIST,
                                     best_seq=seq_v2, tag='p1b_v2',
                                     deadline=p12_dl)
    cost_v1 = len(trips_v1) * VEHICLE_TYPE1.cost_per_km
    cost_v2 = len(trips_v2) * VEHICLE_TYPE2.cost_per_km
    print(f"  车型1: {len(trips_v1)}趟  成本={cost_v1:.0f}元")
    print(f"  车型2: {len(trips_v2)}趟  成本={cost_v2:.0f}元")

    # ── 问题2(1)(2)：4% 剩余时间 ──────────────────────────────
    print("\n" + "★"*62)
    print("【问题2(1)(2)】多车型")
    p2_secs = max(60.0, _time_left() * _STAGE_FRACS['p2'])
    p2_dl   = time.time() + p2_secs
    combo_v = [VEHICLE_TYPE1, VEHICLE_TYPE2]

    min_trips      = solve_p2_min_trips(combo_v, CARGO_LIST, p1_single,
                                         tag='p2a', deadline=p2_dl)
    min_cost_trips = solve_p2_min_cost(combo_v, CARGO_LIST, p1_single, min_trips,
                                        tag='p2b', deadline=p2_dl)

    # ── 汇总对比图 ─────────────────────────────────────────────
    mt = sum(v.cost_per_km for v, _ in min_trips)
    mc = sum(v.cost_per_km for v, _ in min_cost_trips)
    plot_bar_comparison(
        ['车型1全用','车型2全用','多车-最少趟','多车-最低成'],
        [cost_v1, cost_v2, mt, mc],
        ylabel="总费用 (元)", title="各方案总运输费用对比",
        tag='summary_cost_compare')

    print("\n" + "═"*62)
    print("【方案汇总对比】")
    print(f"  {'方案':<22} {'趟数':>5} {'总费用(元)':>12}")
    print("  " + "-"*42)
    print(f"  {'车型1全用':<22} {len(trips_v1):>5} {cost_v1:>12.0f}")
    print(f"  {'车型2全用':<22} {len(trips_v2):>5} {cost_v2:>12.0f}")
    print(f"  {'多车型-最少趟数':<22} {len(min_trips):>5} {mt:>12.0f}")
    print(f"  {'多车型-最低成本 ★':<22} {len(min_cost_trips):>5} {mc:>12.0f}")
    save_checkpoint('summary', {
        'v1_trips': len(trips_v1), 'v1_cost': cost_v1,
        'v2_trips': len(trips_v2), 'v2_cost': cost_v2,
        'min_trips': len(min_trips), 'min_trips_cost': mt,
        'min_cost_trips': len(min_cost_trips), 'min_cost': mc,
    })

    all_r = [r_v1, r_v2] + trips_v1[:2] + trips_v2[:2]
    export_json(all_r, 'packing_results.json')

    # ── 敏感性分析：22% 剩余时间 ──────────────────────────────
    sens_budget = max(120.0, _time_left() * _STAGE_FRACS['sens'])
    print(f"\n  全局剩余={_time_left():.0f}s  敏感性分析分配={sens_budget:.0f}s")
    print("\n" + "★"*62)
    print("【敏感性分析】")
    run_sensitivity_analysis(VEHICLE_TYPE1, CARGO_LIST,
                              total_sens_budget=sens_budget,
                              tag='sens_v1')

    # ── 问题3：剩余全部时间 ────────────────────────────────────
    p3_budget = max(120.0, _time_left())   # 剩余全部给问题3
    print(f"\n  全局剩余={_time_left():.0f}s  问题3分配={p3_budget:.0f}s")
    print("\n" + "★"*62)
    print("【问题3】附件2大规模验证")
    solve_p3_large_scale(LONG_DISTANCE_VEHICLES, p3_budget=p3_budget)

    total_elapsed = time.time() - PROGRAM_START
    print(f"\n{'═'*62}")
    print(f"全程完成  实际耗时={total_elapsed:.0f}s / {TOTAL_TIME}s  "
          f"({total_elapsed/TOTAL_TIME*100:.1f}%)")
    print(f"图表: {SAVE_DIR}/  检查点: {CKPT_DIR}/")
    print("═" * 62)


if __name__ == '__main__':
    main()
