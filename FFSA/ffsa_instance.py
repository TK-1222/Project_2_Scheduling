"""
FFSA 인스턴스 생성기
=====================
주문 모델:
  - 정규주문: t=0 도착, 제품 종류·수량·납기 지정
  - 긴급주문: t>0 랜덤 도착, 짧은 납기
  - 주문 내 unit들은 납기 공유
  - Job = 조립 전 구성품 하나 (에이전트가 개별 의사결정)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


# ──────────────────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────────────────

@dataclass
class InstanceConfig:
    """FFSA 인스턴스 생성 설정"""
    num_products: int = 8
    components_range: Tuple[int, int] = (2, 4)   # 제품별 컴포넌트 수 유니폼 분포 [min, max]
    num_stages: int = 8
    assembly_stage_idx: int = 4
    machines_per_stage: Optional[List[int]] = field(default_factory=lambda: [3, 3, 3, 3, 3, 3, 3, 3])
    processing_time_range: Tuple[float, float] = (10.0, 60.0)
    setup_time_range: Tuple[float, float] = (10.0, 30.0)
    assembly_setup_time_range: Tuple[float, float] = (30.0, 80.0)
    buffer_capacity: int = 10
    weight_range: Tuple[float, float] = (1.0, 5.0)
    machine_stage_prob: float = 0.7    # 기계가 스테이지 처리 가능한 확률
    product_stage_prob: float = 0.7    # 제품이 스테이지를 방문할 확률
    use_assembly: bool = True
    use_setup: bool = True
    use_finite_buffer: bool = True
    seed: Optional[int] = 42
    # 정규주문 (t=0 도착)
    num_regular_orders: int = 15
    regular_quantity_range: Tuple[int, int] = (1, 5)
    regular_due_date_range: Tuple[float, float] = (300.0, 800.0)
    # 긴급주문 (포아송 프로세스, 실시간 생성)
    num_urgent_orders: int = 4                              # 에피소드당 최대 긴급주문 수
    urgent_inter_arrival_mean: float = 150.0               # 평균 도착 간격 (1/λ)
    urgent_quantity_range: Tuple[int, int] = (1, 2)
    urgent_due_date_offset_range: Tuple[float, float] = (40.0, 100.0)  # arrival + offset


# ──────────────────────────────────────────────────────────
# 데이터 구조
# ──────────────────────────────────────────────────────────

@dataclass
class OrderData:
    """주문 정보: 어떤 제품을 몇 개, 언제까지, 언제 도착"""
    order_id: int
    product_id: int
    quantity: int
    due_date: float
    arrival_time: float        # 0 = 정규주문, >0 = 긴급주문
    is_urgent: bool = False
    component_job_ids: List[int] = field(default_factory=list)
    final_job_ids: List[int] = field(default_factory=list)


@dataclass
class ProductData:
    """제품 정보"""
    product_id: int
    weight: float              # wp (tardiness 가중치)
    num_components: int = 2    # 조립에 필요한 컴포넌트 수 (제품마다 다름)
    route: List[int] = field(default_factory=list)  # 방문할 스테이지 목록, 순서대로


@dataclass
class JobData:
    """Job 정보: 조립 전 구성품 하나 또는 final job"""
    job_id: int
    product_id: int
    order_id: int
    arrival_time: float        # 주문 도착 시점 상속
    route: List[int] = field(default_factory=list)
    is_component: bool = False
    component_type_idx: int = 0
    is_final_job: bool = False
    assembly_stage: Optional[int] = None
    order_unit_idx: int = 0    # 주문 내 몇 번째 unit인지
    due_date: float = 0.0      # 주문 납기 상속


@dataclass
class MachineData:
    """기계 정보"""
    machine_id: int
    stage_id: int
    compatible_stages: List[int] = field(default_factory=list)


@dataclass
class FFSAInstance:
    """생성된 FFSA 인스턴스"""
    config: InstanceConfig
    products: Dict[int, ProductData]
    orders: Dict[int, OrderData]
    jobs: Dict[int, JobData]
    machines: Dict[int, MachineData]
    num_stages: int
    num_products: int
    num_jobs: int
    num_machines: int
    machines_by_stage: Dict[int, List[int]]
    processing_times: Dict[Tuple[int, int, int], float]
    setup_times: Dict[Tuple[int, int, int, int], float]
    buffer_capacities: Dict[int, int]
    machine_stage_matrix: Dict[int, List[int]]   # machine_id → 처리 가능한 stage_id 리스트
    product_stage_matrix: Dict[int, List[int]]   # product_id → 방문할 stage_id 리스트


# ──────────────────────────────────────────────────────────
# 인스턴스 생성
# ──────────────────────────────────────────────────────────

def generate_instance(config: InstanceConfig) -> FFSAInstance:
    rng = np.random.RandomState(config.seed)

    stages = list(range(config.num_stages))
    mps = config.machines_per_stage or [2] * config.num_stages

    # ── 기계 생성 ──
    machines: Dict[int, MachineData] = {}
    machines_by_stage: Dict[int, List[int]] = {}
    mid = 0
    for sid in stages:
        machines_by_stage[sid] = []
        for _ in range(mps[sid]):
            machines[mid] = MachineData(machine_id=mid, stage_id=sid)
            machines_by_stage[sid].append(mid)
            mid += 1
    num_machines = mid

    # ── 제품 생성 ──
    products: Dict[int, ProductData] = {}
    for p in range(config.num_products):
        num_comp = (
            int(rng.randint(config.components_range[0], config.components_range[1] + 1))
            if config.use_assembly else 1
        )
        products[p] = ProductData(
            product_id=p,
            weight=float(rng.uniform(*config.weight_range)),
            num_components=num_comp,
        )

    # ── 주문 생성 ──
    orders: Dict[int, OrderData] = {}
    oid = 0

    for _ in range(config.num_regular_orders):
        p = int(rng.randint(config.num_products))
        qty = int(rng.randint(config.regular_quantity_range[0],
                               config.regular_quantity_range[1] + 1))
        due = float(rng.uniform(*config.regular_due_date_range))
        orders[oid] = OrderData(
            order_id=oid, product_id=p, quantity=qty,
            due_date=due, arrival_time=0.0, is_urgent=False,
        )
        oid += 1

    # 긴급주문은 에피소드 진행 중 환경에서 실시간 생성 (포아송 프로세스)

    # ── Job 생성 (주문 → unit → component/final) ──
    jobs: Dict[int, JobData] = {}
    jid = 0

    if config.use_assembly:
        for order in orders.values():
            prod_route = products[order.product_id].route
            for unit_idx in range(order.quantity):
                num_comp = products[order.product_id].num_components
                for comp_type in range(num_comp):
                    jobs[jid] = JobData(
                        job_id=jid,
                        product_id=order.product_id,
                        order_id=order.order_id,
                        arrival_time=order.arrival_time,
                        route=[s for s in prod_route if s < config.assembly_stage_idx],
                        is_component=True,
                        component_type_idx=comp_type,
                        order_unit_idx=unit_idx,
                        due_date=order.due_date,
                    )
                    order.component_job_ids.append(jid)
                    jid += 1

                jobs[jid] = JobData(
                    job_id=jid,
                    product_id=order.product_id,
                    order_id=order.order_id,
                    arrival_time=order.arrival_time,
                    route=[s for s in prod_route if s >= config.assembly_stage_idx],
                    is_final_job=True,
                    assembly_stage=config.assembly_stage_idx,
                    order_unit_idx=unit_idx,
                    due_date=order.due_date,
                )
                order.final_job_ids.append(jid)
                jid += 1
    else:
        for order in orders.values():
            prod_route = products[order.product_id].route
            for unit_idx in range(order.quantity):
                jobs[jid] = JobData(
                    job_id=jid,
                    product_id=order.product_id,
                    order_id=order.order_id,
                    arrival_time=order.arrival_time,
                    route=list(prod_route),
                    is_final_job=False,   # assembly 없을 때는 final job 개념 없음
                    order_unit_idx=unit_idx,
                    due_date=order.due_date,
                )
                order.final_job_ids.append(jid)  # 완료 추적용 리스트는 유지
                jid += 1

    num_jobs = jid

    # ── 기계-스테이지 호환성 행렬 ──
    for m in machines.values():
        for sid in stages:
            if rng.random() < config.machine_stage_prob:
                m.compatible_stages.append(sid)
        if not m.compatible_stages:
            m.compatible_stages.append(int(rng.choice(stages)))

    # 각 스테이지에 처리 가능한 기계가 최소 1개 보장
    for sid in stages:
        compat = [m for m in machines.values() if sid in m.compatible_stages]
        if not compat:
            forced = machines[int(rng.choice(list(machines.keys())))]
            forced.compatible_stages.append(sid)

    machine_stage_matrix = {m.machine_id: m.compatible_stages for m in machines.values()}

    # ── 제품-스테이지 방문 행렬 ──
    pre_asm = stages[:config.assembly_stage_idx]
    post_asm_non_asm = stages[config.assembly_stage_idx + 1:]
    asm_stage = config.assembly_stage_idx

    for p, prod in products.items():
        # 비조립 pre 스테이지: 확률 기반
        visited_pre = [s for s in pre_asm if rng.random() < config.product_stage_prob]
        if not visited_pre:
            visited_pre = [int(rng.choice(pre_asm))]
        # 조립 스테이지: use_assembly면 항상 포함
        visited_asm = [asm_stage] if config.use_assembly else []
        # 비조립 post 스테이지: 확률 기반
        visited_post = [s for s in post_asm_non_asm if rng.random() < config.product_stage_prob]
        if not visited_post:
            visited_post = [int(rng.choice(post_asm_non_asm))] if post_asm_non_asm else []
        prod.route = sorted(visited_pre + visited_asm + visited_post)

    product_stage_matrix = {p: prod.route for p, prod in products.items()}

    # ── 처리시간 ──
    processing_times: Dict[Tuple[int, int, int], float] = {}
    for j in jobs.values():
        for sid in j.route:
            for m in machines_by_stage[sid]:
                if sid in machines[m].compatible_stages:
                    processing_times[(j.job_id, sid, m)] = float(
                        rng.uniform(*config.processing_time_range)
                    )

    # ── Setup time ──
    setup_times: Dict[Tuple[int, int, int, int], float] = {}
    if config.use_setup:
        for pf in range(config.num_products):
            for pt in range(config.num_products):
                if pf == pt:
                    continue
                for sid in stages:
                    st_range = config.assembly_setup_time_range if sid == config.assembly_stage_idx else config.setup_time_range
                    for m in machines_by_stage[sid]:
                        setup_times[(pf, pt, sid, m)] = float(
                            rng.uniform(*st_range)
                        )

    # ── 버퍼 용량 ──
    buffer_capacities: Dict[int, int] = {}
    for sid in stages:
        if sid == stages[0]:
            buffer_capacities[sid] = -1
        else:
            buffer_capacities[sid] = config.buffer_capacity if config.use_finite_buffer else -1

    return FFSAInstance(
        config=config,
        products=products,
        orders=orders,
        jobs=jobs,
        machines=machines,
        num_stages=config.num_stages,
        num_products=config.num_products,
        num_jobs=num_jobs,
        num_machines=num_machines,
        machines_by_stage=machines_by_stage,
        processing_times=processing_times,
        setup_times=setup_times,
        buffer_capacities=buffer_capacities,
        machine_stage_matrix=machine_stage_matrix,
        product_stage_matrix=product_stage_matrix,
    )


# ──────────────────────────────────────────────────────────
# Preset 설정
# ──────────────────────────────────────────────────────────

def simple_config(**kwargs) -> InstanceConfig:
    """Step 1: 단순 FFSA (assembly 없음, setup 없음, 무한 버퍼)"""
    defaults = dict(
        num_stages=8, machines_per_stage=[3, 3, 3, 3, 3, 3, 3, 3],
        use_assembly=False, use_setup=False, use_finite_buffer=False,
        num_urgent_orders=0,
    )
    defaults.update(kwargs)
    return InstanceConfig(**defaults)


def assembly_config(**kwargs) -> InstanceConfig:
    """Step 2: Assembly 포함 (스케일업 기본값 사용)"""
    defaults = dict(use_setup=False, use_finite_buffer=False)
    defaults.update(kwargs)
    return InstanceConfig(**defaults)


def full_config(**kwargs) -> InstanceConfig:
    """Step 3: Setup + Buffer 포함 (스케일업 기본값 사용)"""
    return InstanceConfig(**kwargs)
