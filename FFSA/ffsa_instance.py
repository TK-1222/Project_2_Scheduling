"""
FFSA 인스턴스 생성기
=====================
PPT 02장: 집합/인덱스 (P, I, J, K, Ip, Ji, Kj, A(i))
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
    components_per_product: int = 2      # 조립 시 component job 수
    num_stages: int = 6
    assembly_stage_idx: int = 3          # 조립 stage 인덱스 (0-based)
    machines_per_stage: Optional[List[int]] = field(default_factory=lambda: [2, 2, 2, 2, 2, 2])
    processing_time_range: Tuple[float, float] = (5.0, 40.0)
    setup_time_range: Tuple[float, float] = (2.0, 10.0)
    buffer_capacity: int = 10
    due_date_tightness: float = 1.5      # CLB × tightness = due_date
    weight_range: Tuple[float, float] = (1.0, 5.0)
    machine_product_compatibility: float = 1.0  # 1.0 = 모든 기계가 모든 제품 처리 가능
    use_assembly: bool = True
    use_setup: bool = True
    use_finite_buffer: bool = True
    orders_per_product: List[int] = field(default_factory=lambda: [2, 3, 3, 2])  # 제품별 생산 수량
    seed: Optional[int] = 42


# ──────────────────────────────────────────────────────────
# 데이터 구조
# ──────────────────────────────────────────────────────────

@dataclass
class ProductData:
    """제품 정보 (PPT: p ∈ P) — 주문 단위"""
    product_id: int              # order unique ID (products dict key)
    product_type: int            # 제품 타입 (setup time 기준, 0~num_products-1)
    due_date: float              # dp
    weight: float                # wp
    job_ids: List[int] = field(default_factory=list)     # Ip
    final_job_id: Optional[int] = None                   # ip_final


@dataclass
class JobData:
    """Job 정보 (PPT: i ∈ I)"""
    job_id: int
    product_id: int              # 제품 타입 (setup time·기계 호환성 기준)
    order_id: int                # 주문 ID (ProductData key)
    route: List[int] = field(default_factory=list)       # Ji (stage_id 리스트)
    component_jobs: List[int] = field(default_factory=list)  # A(i)
    is_final_job: bool = False
    assembly_stage: Optional[int] = None                 # jasm,i


@dataclass
class MachineData:
    """기계 정보 (PPT: k ∈ K)"""
    machine_id: int
    stage_id: int                # 속한 stage
    compatible_products: List[int] = field(default_factory=list)  # epk=1인 제품들


@dataclass
class FFSAInstance:
    """생성된 FFSA 인스턴스"""
    config: InstanceConfig
    products: Dict[int, ProductData]           # order_id → ProductData
    jobs: Dict[int, JobData]
    machines: Dict[int, MachineData]
    num_stages: int
    num_products: int                          # 제품 타입 수 (setup time 기준)
    num_orders: int                            # 총 주문 수
    num_jobs: int
    num_machines: int
    machines_by_stage: Dict[int, List[int]]   # stage_id → [machine_id]
    processing_times: Dict[Tuple[int, int, int], float]        # (job, stage, machine) → pijk
    setup_times: Dict[Tuple[int, int, int, int], float]        # (prod_type_from, prod_type_to, stage, machine) → siijk
    buffer_capacities: Dict[int, int]          # stage_id → Bj (-1=무한)


# ──────────────────────────────────────────────────────────
# 인스턴스 생성
# ──────────────────────────────────────────────────────────

def generate_instance(config: InstanceConfig) -> FFSAInstance:
    """PPT의 집합/인덱스/파라미터를 랜덤으로 생성"""
    rng = np.random.RandomState(config.seed)

    stages = list(range(config.num_stages))
    mps = config.machines_per_stage or [2] * config.num_stages

    # ── 기계 생성 (K, Kj) ──
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

    # ── 제품/Job 생성 (P, I, Ip, Ji, A(i)) ──
    products: Dict[int, ProductData] = {}
    jobs: Dict[int, JobData] = {}
    jid = 0
    oid = 0  # order_id 카운터

    orders = config.orders_per_product

    for p in range(config.num_products):
        for _ in range(orders[p]):
            job_ids = []

            if config.use_assembly:
                pre_asm = stages[:config.assembly_stage_idx]
                post_asm = stages[config.assembly_stage_idx:]
                comp_ids = []

                # Component jobs
                for _ in range(config.components_per_product):
                    jobs[jid] = JobData(
                        job_id=jid, product_id=p, order_id=oid,
                        route=list(pre_asm),
                        is_final_job=False,
                    )
                    job_ids.append(jid)
                    comp_ids.append(jid)
                    jid += 1

                # Final (assembly) job
                jobs[jid] = JobData(
                    job_id=jid, product_id=p, order_id=oid,
                    route=list(post_asm),
                    component_jobs=comp_ids,
                    is_final_job=True,
                    assembly_stage=config.assembly_stage_idx,
                )
                final_jid = jid
                job_ids.append(jid)
                jid += 1
            else:
                # 조립 없음: 주문당 1개 job, 모든 stage 통과
                jobs[jid] = JobData(
                    job_id=jid, product_id=p, order_id=oid,
                    route=list(stages),
                    is_final_job=True,
                )
                final_jid = jid
                job_ids.append(jid)
                jid += 1

            products[oid] = ProductData(
                product_id=oid,
                product_type=p,
                due_date=0.0,  # 나중에 계산
                weight=float(rng.uniform(*config.weight_range)),
                job_ids=job_ids,
                final_job_id=final_jid,
            )
            oid += 1

    num_jobs = jid
    num_orders = oid

    # ── 기계 적합성 (epk) ──
    for m in machines.values():
        for p in range(config.num_products):
            if rng.random() < config.machine_product_compatibility:
                m.compatible_products.append(p)
        if not m.compatible_products:
            m.compatible_products.append(int(rng.randint(config.num_products)))

    # 모든 (job, stage) 조합에 최소 1개 호환 기계 보장
    for j in jobs.values():
        for sid in j.route:
            compat = [mid for mid in machines_by_stage[sid]
                      if j.product_id in machines[mid].compatible_products]
            if not compat:
                forced = int(rng.choice(machines_by_stage[sid]))
                if j.product_id not in machines[forced].compatible_products:
                    machines[forced].compatible_products.append(j.product_id)

    # ── 처리시간 (pijk) ──
    processing_times: Dict[Tuple[int, int, int], float] = {}
    for j in jobs.values():
        for sid in j.route:
            for mid in machines_by_stage[sid]:
                if j.product_id in machines[mid].compatible_products:
                    processing_times[(j.job_id, sid, mid)] = float(
                        rng.uniform(*config.processing_time_range)
                    )

    # ── Setup time (siijk) ──
    setup_times: Dict[Tuple[int, int, int, int], float] = {}
    if config.use_setup:
        for pf in range(config.num_products):
            for pt in range(config.num_products):
                if pf == pt:
                    continue  # 동일 제품 = 0
                for sid in stages:
                    for mid in machines_by_stage[sid]:
                        setup_times[(pf, pt, sid, mid)] = float(
                            rng.uniform(*config.setup_time_range)
                        )

    # ── 버퍼 용량 (Bj) ──
    # PPT 문제 정의: 첫 번째 스테이지 전 버퍼는 용량이 무한이라 가정
    buffer_capacities: Dict[int, int] = {}
    for sid in stages:
        if sid == stages[0]:
            buffer_capacities[sid] = -1
        else:
            buffer_capacities[sid] = config.buffer_capacity if config.use_finite_buffer else -1

    # ── Due date (dp) = CLB × tightness ──
    for p, prod in products.items():
        clb = _estimate_product_clb(prod, jobs, machines, machines_by_stage, processing_times, setup_times)
        prod.due_date = clb * config.due_date_tightness

    return FFSAInstance(
        config=config,
        products=products,
        jobs=jobs,
        machines=machines,
        num_stages=config.num_stages,
        num_products=config.num_products,
        num_orders=num_orders,
        num_jobs=num_jobs,
        num_machines=num_machines,
        machines_by_stage=machines_by_stage,
        processing_times=processing_times,
        setup_times=setup_times,
        buffer_capacities=buffer_capacities,
    )


def _estimate_product_clb(
    product: ProductData,
    jobs: Dict[int, JobData],
    machines: Dict[int, MachineData],
    machines_by_stage: Dict[int, List[int]],
    processing_times: Dict[Tuple[int, int, int], float],
    setup_times: Dict[Tuple[int, int, int, int], float] = None,
) -> float:
    """인스턴스 생성 시 rough CLB 추정 (due date 계산용)"""
    final = jobs[product.final_job_id]
    all_product_ids = list(set(j.product_id for j in jobs.values()))

    def _job_lb(job_id: int) -> float:
        job = jobs[job_id]
        lb = 0.0
        for sid in job.route:
            compat = [mid for mid in machines_by_stage[sid]
                      if job.product_id in machines[mid].compatible_products]
            if compat:
                min_proc = min(processing_times[(job_id, sid, mid)] for mid in compat)
                # 최소 setup time 추정 (B안: compute_job_clb와 일관성 유지)
                min_setup = min(
                    (setup_times.get((pf, job.product_id, sid, mid), 0.0)
                     for pf in all_product_ids if pf != job.product_id
                     for mid in compat),
                    default=0.0
                ) if setup_times else 0.0
                lb += min_proc + min_setup
        return lb

    if final.component_jobs:
        comp_max = max(_job_lb(cid) for cid in final.component_jobs)
        post_lb = _job_lb(final.job_id)
        return comp_max + post_lb
    else:
        return _job_lb(final.job_id)


# ──────────────────────────────────────────────────────────
# Preset 설정 (PPT Slide 13 단계별)
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
    defaults = dict(
        use_setup=False, use_finite_buffer=False,
    )
    defaults.update(kwargs)
    return InstanceConfig(**defaults)


def full_config(**kwargs) -> InstanceConfig:
    """Step 3: Setup + Buffer 포함"""
    return InstanceConfig(**kwargs)
