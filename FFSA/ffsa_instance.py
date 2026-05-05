"""
FFSA 인스턴스 생성기
=====================
PPT 02장: 집합/인덱스 (P, I, J, K, Ip, Ji, Kj)
PPT 03장: 파라미터 (pijk, siijk, dp, wp, Bj, epk)

단계적 실험 전략 (PPT Slide 13):
  Step 1: simple  — assembly 없음, setup 없음, 무한 버퍼
  Step 2: assembly — assembly 추가
  Step 3: full    — setup + 유한 버퍼 추가
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
import numpy as np


# ──────────────────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────────────────

@dataclass
class InstanceConfig:
    """FFSA 인스턴스 생성 설정"""
    num_products: int = 4
    components_per_product: int = 2      # 조립 시 component 타입 수
    num_stages: int = 6
    assembly_stage_idx: int = 3          # 조립 stage 인덱스 (0-based)
    machines_per_stage: Optional[List[int]] = field(default_factory=lambda: [2, 2, 2, 2, 2, 2])
    processing_time_range: Tuple[float, float] = (5.0, 40.0)
    setup_time_range: Tuple[float, float] = (2.0, 10.0)
    buffer_capacity: int = 10
    due_date_tightness_range: Tuple[float, float] = (1.2, 2.0)  # 분포로 납기 설정
    weight_range: Tuple[float, float] = (1.0, 5.0)
    machine_product_compatibility: float = 1.0
    use_assembly: bool = True
    use_setup: bool = True
    use_finite_buffer: bool = True
    orders_per_product: List[int] = field(default_factory=lambda: [2, 3, 3, 2])
    seed: Optional[int] = 42


# ──────────────────────────────────────────────────────────
# 데이터 구조
# ──────────────────────────────────────────────────────────

@dataclass
class ProductData:
    """제품 정보 (PPT: p ∈ P)"""
    product_id: int
    weight: float                        # wp
    component_job_ids: List[int] = field(default_factory=list)  # 전체 component job id 목록
    final_job_ids: List[int] = field(default_factory=list)      # 주문별 final job id 목록


@dataclass
class JobData:
    """Job 정보 (PPT: i ∈ I)"""
    job_id: int
    product_id: int
    route: List[int] = field(default_factory=list)
    is_component: bool = False
    component_type_idx: int = 0          # 같은 제품 내 component 타입 (0=A, 1=B, ...)
    is_final_job: bool = False
    assembly_stage: Optional[int] = None  # final job의 조립 시작 stage
    order_idx: int = 0                   # 제품 내 주문 번호
    due_date: float = 0.0                # dp (주문별 납기)


@dataclass
class MachineData:
    """기계 정보 (PPT: k ∈ K)"""
    machine_id: int
    stage_id: int
    compatible_products: List[int] = field(default_factory=list)


@dataclass
class FFSAInstance:
    """생성된 FFSA 인스턴스"""
    config: InstanceConfig
    products: Dict[int, ProductData]
    jobs: Dict[int, JobData]
    machines: Dict[int, MachineData]
    num_stages: int
    num_products: int
    num_jobs: int
    num_machines: int
    machines_by_stage: Dict[int, List[int]]
    processing_times: Dict[Tuple[int, int, int], float]         # (job, stage, machine) → pijk
    setup_times: Dict[Tuple[int, int, int, int], float]         # (prod_from, prod_to, stage, machine) → siijk
    buffer_capacities: Dict[int, int]                           # stage_id → Bj (-1=무한)


# ──────────────────────────────────────────────────────────
# 인스턴스 생성
# ──────────────────────────────────────────────────────────

def generate_instance(config: InstanceConfig) -> FFSAInstance:
    """PPT의 집합/인덱스/파라미터를 랜덤으로 생성"""
    rng = np.random.RandomState(config.seed)

    stages = list(range(config.num_stages))
    mps = config.machines_per_stage or [2] * config.num_stages

    # orders_per_product 길이 보정
    orders = list(config.orders_per_product)
    while len(orders) < config.num_products:
        orders.append(1)
    orders = orders[:config.num_products]

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

    # ── 제품/Job 생성 ──
    products: Dict[int, ProductData] = {}
    jobs: Dict[int, JobData] = {}
    jid = 0

    for p in range(config.num_products):
        prod = ProductData(
            product_id=p,
            weight=float(rng.uniform(*config.weight_range)),
        )
        products[p] = prod

        if config.use_assembly:
            pre_asm = stages[:config.assembly_stage_idx]
            post_asm = stages[config.assembly_stage_idx:]

            for order_idx in range(orders[p]):
                # component jobs: 타입별 1개씩
                for comp_type in range(config.components_per_product):
                    jobs[jid] = JobData(
                        job_id=jid,
                        product_id=p,
                        route=list(pre_asm),
                        is_component=True,
                        component_type_idx=comp_type,
                        order_idx=order_idx,
                    )
                    prod.component_job_ids.append(jid)
                    jid += 1

                # final job: 조립 포함 이후 공정
                jobs[jid] = JobData(
                    job_id=jid,
                    product_id=p,
                    route=list(post_asm),
                    is_final_job=True,
                    assembly_stage=config.assembly_stage_idx,
                    order_idx=order_idx,
                )
                prod.final_job_ids.append(jid)
                jid += 1
        else:
            # 조립 없음: 주문별 1개 job, 모든 stage 통과
            for order_idx in range(orders[p]):
                jobs[jid] = JobData(
                    job_id=jid,
                    product_id=p,
                    route=list(stages),
                    is_final_job=True,
                    order_idx=order_idx,
                )
                prod.final_job_ids.append(jid)
                jid += 1

    num_jobs = jid

    # ── 기계 적합성 ──
    for m in machines.values():
        for p in range(config.num_products):
            if rng.random() < config.machine_product_compatibility:
                m.compatible_products.append(p)
        if not m.compatible_products:
            m.compatible_products.append(int(rng.randint(config.num_products)))

    # 모든 (job, stage) 조합에 최소 1개 호환 기계 보장
    for j in jobs.values():
        for sid in j.route:
            compat = [m for m in machines_by_stage[sid]
                      if j.product_id in machines[m].compatible_products]
            if not compat:
                forced = int(rng.choice(machines_by_stage[sid]))
                if j.product_id not in machines[forced].compatible_products:
                    machines[forced].compatible_products.append(j.product_id)

    # ── 처리시간 (pijk) ──
    processing_times: Dict[Tuple[int, int, int], float] = {}
    for j in jobs.values():
        for sid in j.route:
            for m in machines_by_stage[sid]:
                if j.product_id in machines[m].compatible_products:
                    processing_times[(j.job_id, sid, m)] = float(
                        rng.uniform(*config.processing_time_range)
                    )

    # ── Setup time (siijk) ──
    setup_times: Dict[Tuple[int, int, int, int], float] = {}
    if config.use_setup:
        for pf in range(config.num_products):
            for pt in range(config.num_products):
                if pf == pt:
                    continue
                for sid in stages:
                    for m in machines_by_stage[sid]:
                        setup_times[(pf, pt, sid, m)] = float(
                            rng.uniform(*config.setup_time_range)
                        )

    # ── 버퍼 용량 ──
    buffer_capacities: Dict[int, int] = {}
    for sid in stages:
        if sid == stages[0]:
            buffer_capacities[sid] = -1  # 첫 stage 전 버퍼: 무한
        else:
            buffer_capacities[sid] = config.buffer_capacity if config.use_finite_buffer else -1

    # ── 납기 (dp): tightness 분포로 주문별 독립 설정 ──
    for p, prod in products.items():
        base_time = _estimate_min_completion_time(
            prod, jobs, machines, machines_by_stage, processing_times, config
        )
        for fid in prod.final_job_ids:
            tightness = float(rng.uniform(*config.due_date_tightness_range))
            jobs[fid].due_date = base_time * tightness

    return FFSAInstance(
        config=config,
        products=products,
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
    )


def _estimate_min_completion_time(
    product: ProductData,
    jobs: Dict[int, JobData],
    machines: Dict[int, MachineData],
    machines_by_stage: Dict[int, List[int]],
    processing_times: Dict[Tuple[int, int, int], float],
    config: InstanceConfig,
) -> float:
    """납기 기준값 추정: 가장 오래 걸리는 component 타입 + final job 최소 처리시간"""
    comp_time = 0.0
    if config.use_assembly and product.component_job_ids:
        type_times: Dict[int, float] = {}
        for jid in product.component_job_ids:
            j = jobs[jid]
            t = j.component_type_idx
            if t not in type_times:
                type_times[t] = _job_min_proc_time(jid, j, machines_by_stage, machines, processing_times)
        comp_time = max(type_times.values()) if type_times else 0.0

    final_time = 0.0
    if product.final_job_ids:
        fid = product.final_job_ids[0]
        fj = jobs[fid]
        final_time = _job_min_proc_time(fid, fj, machines_by_stage, machines, processing_times)

    return comp_time + final_time


def _job_min_proc_time(
    job_id: int,
    job: JobData,
    machines_by_stage: Dict[int, List[int]],
    machines: Dict[int, MachineData],
    processing_times: Dict[Tuple[int, int, int], float],
) -> float:
    total = 0.0
    for sid in job.route:
        compat = [m for m in machines_by_stage[sid]
                  if job.product_id in machines[m].compatible_products]
        if compat:
            min_proc = min(
                processing_times.get((job_id, sid, m), float('inf'))
                for m in compat
            )
            if min_proc < float('inf'):
                total += min_proc
    return total


# ──────────────────────────────────────────────────────────
# Preset 설정
# ──────────────────────────────────────────────────────────

def simple_config(**kwargs) -> InstanceConfig:
    """Step 1: 단순 FFSA (assembly 없음, setup 없음, 무한 버퍼)"""
    defaults = dict(
        num_stages=4, machines_per_stage=[2, 2, 2, 2],
        use_assembly=False, use_setup=False, use_finite_buffer=False,
    )
    defaults.update(kwargs)
    return InstanceConfig(**defaults)


def assembly_config(**kwargs) -> InstanceConfig:
    """Step 2: Assembly 포함"""
    defaults = dict(use_setup=False, use_finite_buffer=False)
    defaults.update(kwargs)
    return InstanceConfig(**defaults)


def full_config(**kwargs) -> InstanceConfig:
    """Step 3: Setup + Buffer 포함"""
    return InstanceConfig(**kwargs)
