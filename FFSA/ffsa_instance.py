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
    num_products: int = 2
    components_range: Tuple[int, int] = (2, 3)   # 제품별 컴포넌트 수 유니폼 분포 [min, max]
    num_stages: int = 6
    assembly_stage_idx: int = 3
    machines_per_stage: Optional[List[int]] = field(default_factory=lambda: [3, 3, 3, 3, 3, 3])
    processing_time_range: Tuple[float, float] = (10.0, 60.0)
    setup_time_range: Tuple[float, float] = (10.0, 30.0)
    assembly_setup_time_range: Tuple[float, float] = (30.0, 80.0)
    buffer_capacity: int = 10
    weight_range: Tuple[float, float] = (1.0, 5.0)
    machine_stage_prob: float = 0.7    # 기계가 스테이지 처리 가능한 확률
    component_stage_prob: float = 0.7  # 컴포넌트별 스테이지 방문 확률
    final_stage_prob: float = 0.7      # final job 스테이지 방문 확률
    use_assembly: bool = True
    use_setup: bool = True
    use_finite_buffer: bool = True
    seed: Optional[int] = 42
    # 정규주문 (t=0 도착)
    num_regular_orders: int = 6
    regular_quantity_range: Tuple[int, int] = (1, 3)
    regular_due_date_range: Tuple[float, float] = (600.0, 1400.0)
    # 긴급주문 (포아송 프로세스, 실시간 생성)
    num_urgent_orders: int = 2                              # 에피소드당 최대 긴급주문 수
    urgent_inter_arrival_mean: float = 150.0               # 평균 도착 간격 (1/λ)
    urgent_quantity_range: Tuple[int, int] = (1, 2)
    urgent_due_date_offset_range: Tuple[float, float] = (80.0, 160.0)  # arrival + offset


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
    compatible_component_ops: List[Tuple[int, int]] = field(default_factory=list)  # (product_id, component_type_idx)
    compatible_final_ops: List[int] = field(default_factory=list)                  # product_id


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
    processing_times: Dict[Tuple[int, int, int, int], float]       # (product_id, comp_type, stage_id, machine_id)
    processing_times_final: Dict[Tuple[int, int, int], float]      # (product_id, stage_id, machine_id)
    setup_times: Dict[Tuple[int, int, int, int], float]
    buffer_capacities: Dict[int, int]
    component_stage_matrix: Dict[Tuple[int, int], List[int]]       # (product_id, comp_type) → [stage_id,...]
    final_stage_matrix: Dict[int, List[int]]                       # product_id → [stage_id,...]


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

    # ── 기계 호환성 생성 ──
    # 컴포넌트 job 호환성: (product_id, component_type_idx) 조합
    # final job 호환성: product_id
    # 스테이지에 속한 기계만 해당 스테이지 job 처리 가능 (물리적 위치 고정)
    for m in machines.values():
        sid = m.stage_id
        is_asm = (sid == config.assembly_stage_idx)

        if not is_asm:
            # 비조립 스테이지: 컴포넌트 job 호환성
            for p_id, prod in products.items():
                for comp_type in range(prod.num_components):
                    if rng.random() < config.machine_stage_prob:
                        m.compatible_component_ops.append((p_id, comp_type))
            if not m.compatible_component_ops:
                # 최소 1개 보장
                p_id = int(rng.randint(config.num_products))
                comp_type = int(rng.randint(products[p_id].num_components))
                m.compatible_component_ops.append((p_id, comp_type))
        else:
            # 조립 스테이지: final job 호환성 (product_id 단위)
            for p_id in range(config.num_products):
                if rng.random() < config.machine_stage_prob:
                    m.compatible_final_ops.append(p_id)
            if not m.compatible_final_ops:
                m.compatible_final_ops.append(int(rng.randint(config.num_products)))

        # post-assembly 스테이지: final job 호환성
        if sid > config.assembly_stage_idx:
            for p_id in range(config.num_products):
                if rng.random() < config.machine_stage_prob:
                    if p_id not in m.compatible_final_ops:
                        m.compatible_final_ops.append(p_id)
            if not m.compatible_final_ops:
                m.compatible_final_ops.append(int(rng.randint(config.num_products)))

    # ── component_stage_matrix 및 final_stage_matrix 생성 ──
    pre_asm_stages = stages[:config.assembly_stage_idx]
    post_asm_stages = stages[config.assembly_stage_idx + 1:]
    asm_stage = config.assembly_stage_idx

    component_stage_matrix: Dict[Tuple[int, int], List[int]] = {}
    final_stage_matrix: Dict[int, List[int]] = {}

    for p_id, prod in products.items():
        for comp_type in range(prod.num_components):
            visited = [s for s in pre_asm_stages if rng.random() < config.component_stage_prob]
            if not visited:
                visited = [int(rng.choice(pre_asm_stages))]
            component_stage_matrix[(p_id, comp_type)] = sorted(visited)

        # final job: 조립 스테이지 + post-assembly 중 일부
        visited_post = [s for s in post_asm_stages if rng.random() < config.final_stage_prob]
        if not visited_post and post_asm_stages:
            visited_post = [int(rng.choice(post_asm_stages))]
        final_stage_matrix[p_id] = sorted([asm_stage] + visited_post)

    # ── Job 생성 (주문 → unit → component/final) ──
    jobs: Dict[int, JobData] = {}
    jid = 0

    if config.use_assembly:
        for order in orders.values():
            for unit_idx in range(order.quantity):
                num_comp = products[order.product_id].num_components
                for comp_type in range(num_comp):
                    route = component_stage_matrix[(order.product_id, comp_type)]
                    jobs[jid] = JobData(
                        job_id=jid,
                        product_id=order.product_id,
                        order_id=order.order_id,
                        arrival_time=order.arrival_time,
                        route=list(route),
                        is_component=True,
                        component_type_idx=comp_type,
                        order_unit_idx=unit_idx,
                        due_date=order.due_date,
                    )
                    order.component_job_ids.append(jid)
                    jid += 1

                route = final_stage_matrix[order.product_id]
                jobs[jid] = JobData(
                    job_id=jid,
                    product_id=order.product_id,
                    order_id=order.order_id,
                    arrival_time=order.arrival_time,
                    route=list(route),
                    is_final_job=True,
                    assembly_stage=config.assembly_stage_idx,
                    order_unit_idx=unit_idx,
                    due_date=order.due_date,
                )
                order.final_job_ids.append(jid)
                jid += 1
    else:
        for order in orders.values():
            for unit_idx in range(order.quantity):
                # no-assembly: use comp_type=0 route as fallback
                route = component_stage_matrix.get((order.product_id, 0), list(pre_asm_stages))
                jobs[jid] = JobData(
                    job_id=jid,
                    product_id=order.product_id,
                    order_id=order.order_id,
                    arrival_time=order.arrival_time,
                    route=list(route),
                    is_final_job=False,
                    order_unit_idx=unit_idx,
                    due_date=order.due_date,
                )
                order.final_job_ids.append(jid)
                jid += 1

    num_jobs = jid

    # ── 처리시간 ──
    # component job 처리시간: (product_id, comp_type_idx, stage_id, machine_id)
    processing_times: Dict[Tuple[int, int, int, int], float] = {}
    for (p_id, comp_type), stage_list in component_stage_matrix.items():
        for sid in stage_list:
            for m in machines_by_stage[sid]:
                if (p_id, comp_type) in machines[m].compatible_component_ops:
                    processing_times[(p_id, comp_type, sid, m)] = float(
                        rng.uniform(*config.processing_time_range)
                    )

    # final job 처리시간: (product_id, stage_id, machine_id)
    processing_times_final: Dict[Tuple[int, int, int], float] = {}
    for p_id, stage_list in final_stage_matrix.items():
        for sid in stage_list:
            for m in machines_by_stage[sid]:
                if sid == config.assembly_stage_idx:
                    if p_id in machines[m].compatible_final_ops:
                        processing_times_final[(p_id, sid, m)] = float(
                            rng.uniform(*config.processing_time_range)
                        )
                else:
                    if p_id in machines[m].compatible_final_ops:
                        processing_times_final[(p_id, sid, m)] = float(
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
        processing_times_final=processing_times_final,
        setup_times=setup_times,
        buffer_capacities=buffer_capacities,
        component_stage_matrix=component_stage_matrix,
        final_stage_matrix=final_stage_matrix,
    )


def full_config(**kwargs) -> InstanceConfig:
    return InstanceConfig(**kwargs)
