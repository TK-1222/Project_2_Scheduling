"""
FFSA 스케줄링 강화학습 환경 (디스패칭 룰 버전)
================================================
액션 후보 생성 방식:
  - 일반 공정: FIFO/EDD/MWKR/SPT/WINQ 5개 룰이 각 1개 후보 생성 → 중복 제거.
    룰이 후보를 못 찾으면 전체 유효 액션 fallback.
  - 조립 공정: 5개 룰이 컴포넌트 페어 후보 선정 → 각 페어 × 모든 호환 유휴 기계 조합.
    Q-네트워크가 (페어, 기계) 조합 전체를 평가해 최종 선택.

Reward: r = -Δ(추정 가중 지연) — dense reward shaping
  텔레스코핑 합: Σr_t = WT_est(0) - WT_est(T), 종료 시 WT_est(T) = WT_actual(T)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Union
import numpy as np
import gymnasium as gym
from gymnasium import spaces

import torch
from torch_geometric.data import HeteroData

from ffsa_instance import (
    InstanceConfig, FFSAInstance, generate_instance,
    OrderData, ProductData, JobData, MachineData,
    OpFeat,
)

# Action 타입 정의
RegularAction = Tuple[int, int]                    # (op_id, machine_id)
AssemblyAction = Tuple[Tuple[int, ...], int]       # ((comp_job_id, ...), machine_id)
Action = Union[RegularAction, AssemblyAction]


# ──────────────────────────────────────────────────────────
# 런타임 상태 구조체
# ──────────────────────────────────────────────────────────

@dataclass
class OperationState:
    """Operation 런타임 상태"""
    op_id: int
    job_id: int
    product_id: int
    stage_id: int
    is_done: bool = False
    is_ready: bool = False
    is_processing: bool = False
    is_assembly: bool = False            # final job의 첫 번째 op (조립 op)
    buffer_waiting: bool = False
    predecessors: List[int] = field(default_factory=list)
    machine_id: Optional[int] = None
    start_time: Optional[float] = None
    completion_time: Optional[float] = None
    active: bool = True                  # final job은 활성화 전까지 False


@dataclass
class MachineState:
    """Machine 런타임 상태"""
    machine_id: int
    stage_id: int
    compatible_component_ops: Set[Tuple[int, int]] = field(default_factory=set)
    compatible_final_ops: Set[int] = field(default_factory=set)
    is_idle: bool = True
    is_blocked: bool = False
    current_op: Optional[int] = None
    remaining_time: float = 0.0
    last_product: Optional[int] = None
    blocked_job: Optional[int] = None
    total_busy_time: float = 0.0


@dataclass
class BufferState:
    """Buffer 런타임 상태"""
    stage_id: int
    capacity: int                        # -1 = 무한
    queue: List[int] = field(default_factory=list)  # job_id 리스트

    def has_space(self) -> bool:
        return self.capacity < 0 or len(self.queue) < self.capacity

    def push(self, job_id: int):
        self.queue.append(job_id)

    def remove(self, job_id: int):
        if job_id in self.queue:
            self.queue.remove(job_id)

    @property
    def occupancy(self) -> float:
        if self.capacity <= 0:
            return 0.0
        return len(self.queue) / self.capacity


# ──────────────────────────────────────────────────────────
# GraphBuilder
# ──────────────────────────────────────────────────────────

class GraphBuilder:
    """이종 그래프 생성"""

    def __init__(self, instance: FFSAInstance):
        self.inst = instance
        all_proc = list(instance.processing_times.values()) + list(instance.processing_times_final.values())
        self.max_proc = max(all_proc) if all_proc else 1.0
        self.max_setup = max(instance.setup_times.values()) if instance.setup_times else 1.0
        self.max_due = max(
            (j.due_date for j in instance.jobs.values() if j.due_date > 0), default=1.0
        )
        self.max_weight = max(o.weight for o in instance.orders.values()) if instance.orders else 1.0

    def build(self, env: "FFSASchedulingEnv") -> HeteroData:
        data = HeteroData()
        data['op'].x = self._build_op_features(env)
        data['machine'].x = self._build_machine_features(env)

        prec_src, prec_dst = self._build_precedence_edges(env)
        data['op', 'precedence', 'op'].edge_index = (
            torch.tensor([prec_src, prec_dst], dtype=torch.long)
            if prec_src else torch.zeros((2, 0), dtype=torch.long)
        )

        cand_src, cand_dst, cand_attr = self._build_candidate_edges(env)
        if cand_src:
            edge_index = torch.tensor([cand_src, cand_dst], dtype=torch.long)
            edge_attr = torch.tensor(cand_attr, dtype=torch.float32)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 2), dtype=torch.float32)

        data['op', 'candidate', 'machine'].edge_index = edge_index
        data['op', 'candidate', 'machine'].edge_attr = edge_attr
        data['machine', 'candidate_rev', 'op'].edge_index = edge_index.flip(0)
        data['machine', 'candidate_rev', 'op'].edge_attr = edge_attr

        return data

    def _build_op_features(self, env: "FFSASchedulingEnv") -> torch.Tensor:
        """Operation Node Feature: OpFeat.DIM 차원 (OpFeat enum 참조)"""
        active_ops = [op for op in env.operations.values() if op.active]
        num_ops = len(active_ops)
        if num_ops == 0:
            return torch.zeros((0, OpFeat.DIM), dtype=torch.float32)

        feats = np.zeros((num_ops, OpFeat.DIM), dtype=np.float32)

        for idx, op in enumerate(active_ops):
            feats[idx, OpFeat.IS_DONE]        = float(op.is_done)
            feats[idx, OpFeat.IS_READY]       = float(op.is_ready)
            feats[idx, OpFeat.IS_PROCESSING]  = float(op.is_processing)
            feats[idx, OpFeat.IS_ASSEMBLY]    = float(op.is_assembly)
            feats[idx, OpFeat.BUFFER_WAITING] = float(op.buffer_waiting)
            feats[idx, OpFeat.STAGE_NORM]     = op.stage_id / max(self.inst.num_stages - 1, 1)
            feats[idx, OpFeat.PRODUCT_NORM]   = op.product_id / max(self.inst.num_products - 1, 1)
            job = self.inst.jobs[op.job_id]
            slack = max(0.0, job.due_date - env.current_time)
            feats[idx, OpFeat.DUE_DATE_NORM]  = min(1.0, slack / self.max_due) if self.max_due > 0 else 0.0
            order = self.inst.orders[self.inst.jobs[op.job_id].order_id]
            feats[idx, OpFeat.WEIGHT_NORM]    = order.weight / self.max_weight if self.max_weight > 0 else 0.0

        return torch.tensor(feats)

    def _build_machine_features(self, env: "FFSASchedulingEnv") -> torch.Tensor:
        """Machine Node Feature: 6차원"""
        num_m = env.num_machines
        feats = np.zeros((num_m, 6), dtype=np.float32)

        for ms in env.machine_states.values():
            i = ms.machine_id
            feats[i, 0] = ms.stage_id / max(self.inst.num_stages - 1, 1)
            feats[i, 1] = float(ms.is_idle)
            feats[i, 2] = ms.remaining_time / self.max_proc if self.max_proc > 0 else 0.0
            avail = env.current_time + ms.remaining_time
            feats[i, 3] = avail / self.max_due if self.max_due > 0 else 0.0
            feats[i, 4] = (ms.last_product / max(self.inst.num_products - 1, 1)
                           if ms.last_product is not None else -1.0)
            feats[i, 5] = (ms.total_busy_time / env.current_time
                           if env.current_time > 0 else 0.0)

        return torch.tensor(feats)

    def _build_precedence_edges(self, env) -> Tuple[List[int], List[int]]:
        src, dst = [], []
        op_id_to_idx = {op.op_id: i for i, op in enumerate(
            op for op in env.operations.values() if op.active
        )}
        for op in env.operations.values():
            if not op.active:
                continue
            for pred_id in op.predecessors:
                pred_op = env.operations.get(pred_id)
                if pred_op and pred_op.active:
                    src.append(op_id_to_idx[pred_id])
                    dst.append(op_id_to_idx[op.op_id])
        return src, dst

    def _build_candidate_edges(self, env) -> Tuple[List[int], List[int], List[List[float]]]:
        src, dst, attrs = [], [], []
        op_id_to_idx = {op.op_id: i for i, op in enumerate(
            op for op in env.operations.values() if op.active
        )}
        for op in env.operations.values():
            if not op.active or op.is_done or op.is_processing:
                continue
            if op.op_id not in op_id_to_idx:
                continue
            for mid in self.inst.machines_by_stage.get(op.stage_id, []):
                m_data = self.inst.machines[mid]
                job = self.inst.jobs[op.job_id]
                if job.is_component:
                    if (job.product_id, job.component_type_idx) not in m_data.compatible_component_ops:
                        continue
                else:  # final job
                    if job.product_id not in m_data.compatible_final_ops:
                        continue
                if job.is_component:
                    pt = self.inst.processing_times.get((job.product_id, job.component_type_idx, op.stage_id, mid), 0.0)
                else:
                    pt = self.inst.processing_times_final.get((job.product_id, op.stage_id, mid), 0.0)
                pt_norm = pt / self.max_proc if self.max_proc > 0 else 0.0
                ms = env.machine_states[mid]
                if ms.last_product is not None and ms.last_product != op.product_id:
                    st = self.inst.setup_times.get(
                        (ms.last_product, op.product_id, op.stage_id, mid), 0.0
                    )
                else:
                    st = 0.0
                st_norm = st / self.max_setup if self.max_setup > 0 else 0.0
                src.append(op_id_to_idx[op.op_id])
                dst.append(mid)
                attrs.append([pt_norm, st_norm])
        return src, dst, attrs


# ──────────────────────────────────────────────────────────
# 환경
# ──────────────────────────────────────────────────────────

class FFSASchedulingEnv(gym.Env):
    """
    Flexible Flow Shop with Assembly 스케줄링 환경

    조립 제약: 조립 전 버퍼에 동일 제품의 A타입 ≥1 AND B타입 ≥1 존재 시 조립 가능
    Reward: r = -Δ(추정 가중 지연) — dense reward shaping (telescoping sum → 실제 WT와 동일 목표)
    """
    metadata = {"render_modes": []}

    def __init__(self, config: InstanceConfig):
        super().__init__()
        self.config = config
        self.instance = generate_instance(config)
        self.graph_builder = GraphBuilder(self.instance)

        self.operations: Dict[int, OperationState] = {}
        self.machine_states: Dict[int, MachineState] = {}
        self.buffers: Dict[int, BufferState] = {}
        self.num_operations = 0
        self.num_machines = self.instance.num_machines
        self.current_time = 0.0

        # 조립 버퍼: {prod_id: {(order_id, unit_idx): {comp_type: job_id}}}  — Constraint 5
        self.assembly_pool: Dict[int, Dict[Tuple[int, int], Dict[int, int]]] = {}
        # 미활성 final job: {(prod_id, order_id, unit_idx): final_job_id}
        self.inactive_final_jobs: Dict[Tuple[int, int, int], int] = {}
        # 조립 버퍼 만원으로 blocked된 컴포넌트  — Constraint 10
        self._pool_blocked: Dict[int, int] = {}  # job_id → machine_id

        self.job_ops: Dict[int, List[int]] = {}
        self.op_to_job_stage: Dict[int, Tuple[int, int]] = {}
        self.job_stage_to_op: Dict[Tuple[int, int], int] = {}

        self._max_actions = 2000
        self.action_space = spaces.Discrete(self._max_actions)
        self.observation_space = spaces.Dict({
            "dummy": spaces.Box(0, 1, (1,), dtype=np.float32)
        })

        self._current_actions: List[Action] = []
        self._deadlock_detected: bool = False
        self._prev_estimated_wt: float = 0.0

    # ──────────────────────────────────────────────────────
    # Reset
    # ──────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_time = 0.0
        self._deadlock_detected = False
        # 인스턴스를 매 에피소드 새로 생성 (정규주문 랜덤 변경)
        self.instance = generate_instance(self.config)
        self.graph_builder = GraphBuilder(self.instance)
        self._init_operations()
        self._init_machines()
        self._init_buffers()
        self._init_assembly_pool()
        self._load_initial_jobs()
        self.update_ready_operations()
        self._prev_estimated_wt = self._compute_estimated_weighted_tardiness()
        return self._get_obs(), {}

    def _init_operations(self):
        self.operations = {}
        self.job_ops = {}
        self.op_to_job_stage = {}
        self.job_stage_to_op = {}
        op_id = 0

        for job in self.instance.jobs.values():
            self.job_ops[job.job_id] = []
            prev_op_id = None
            # 활성 조건: final job이 아닌 경우 (component 또는 no-assembly job)
            is_active = not job.is_final_job

            for stage_id in job.route:
                is_asm = (job.is_final_job and stage_id == job.assembly_stage)
                predecessors = [prev_op_id] if prev_op_id is not None else []

                self.operations[op_id] = OperationState(
                    op_id=op_id,
                    job_id=job.job_id,
                    product_id=job.product_id,
                    stage_id=stage_id,
                    is_assembly=is_asm,
                    predecessors=predecessors,
                    active=is_active,
                )
                self.job_ops[job.job_id].append(op_id)
                self.op_to_job_stage[op_id] = (job.job_id, stage_id)
                self.job_stage_to_op[(job.job_id, stage_id)] = op_id
                prev_op_id = op_id
                op_id += 1

        self.num_operations = op_id

    def _init_machines(self):
        self.machine_states = {}
        for m in self.instance.machines.values():
            self.machine_states[m.machine_id] = MachineState(
                machine_id=m.machine_id,
                stage_id=m.stage_id,
                compatible_component_ops=set(map(tuple, m.compatible_component_ops)),
                compatible_final_ops=set(m.compatible_final_ops),
            )

    def _init_buffers(self):
        self.buffers = {}
        for sid in range(self.instance.num_stages):
            cap = self.instance.buffer_capacities.get(sid, -1)
            self.buffers[sid] = BufferState(stage_id=sid, capacity=cap)

    def _init_assembly_pool(self):
        """조립 버퍼 pool 및 정규주문 final job 초기화"""
        self.assembly_pool = {p: {} for p in self.instance.products}
        self.inactive_final_jobs = {}
        self._pool_blocked = {}
        if not self.config.use_assembly:
            return
        for order in self.instance.orders.values():
            for fid in order.final_job_ids:
                job = self.instance.jobs[fid]
                # (prod_id, order_id, unit_idx) 키로 final job 등록 — Constraint 5
                key = (order.product_id, order.order_id, job.order_unit_idx)
                self.inactive_final_jobs[key] = fid

    def _load_initial_jobs(self):
        """t=0 도착 job을 stage 0 버퍼에 투입 (final job 제외: 조립 dispatch 전까지 비활성)"""
        for job in self.instance.jobs.values():
            if not job.route or job.is_final_job:
                continue
            first_stage = job.route[0]
            self.buffers[first_stage].push(job.job_id)
            first_op_id = self.job_ops[job.job_id][0]
            self.operations[first_op_id].buffer_waiting = True

    # ──────────────────────────────────────────────────────
    # Step
    # ──────────────────────────────────────────────────────

    def step(self, action_idx: int):
        actions = self._current_actions
        if action_idx >= len(actions):
            # 액션 없음 → 진행 불가, truncated 처리
            return self._get_obs(), -1.0, False, True, {"deadlock": True}

        action = actions[action_idx]
        self._deadlock_detected = False

        if isinstance(action[0], tuple):
            self._dispatch_assembly(*action)
        else:
            self._dispatch(*action)

        self.update_ready_operations()
        if not self._has_valid_action():
            self._advance_until_next_decision_point()

        reward = self._reward_fn()
        done = self._check_done()
        truncated = self._deadlock_detected

        obs = self._get_obs()
        active_ops = [op for op in self.operations.values() if op.active]
        info = {
            "time": self.current_time,
            "completed_ops": sum(1 for op in active_ops if op.is_done),
            "total_ops": len(active_ops),
            "deadlock": self._deadlock_detected,
        }
        return obs, reward, done, truncated, info

    # ──────────────────────────────────────────────────────
    # Dispatch
    # ──────────────────────────────────────────────────────

    def _dispatch(self, op_id: int, machine_id: int):
        """일반 operation dispatch"""
        op = self.operations[op_id]
        ms = self.machine_states[machine_id]

        self.buffers[op.stage_id].remove(op.job_id)
        op.buffer_waiting = False
        # Constraint 1: 단일 할당 — op에 machine_id 하나만 기록
        op.machine_id = machine_id

        # Constraint 3+6: 완료시간 = 시작시간 + 준비시간(s) + 처리시간(p)
        setup = self._get_setup_time(ms.last_product, op.product_id, op.stage_id, machine_id)
        job = self.instance.jobs[op.job_id]
        if job.is_component:
            proc = self.instance.processing_times.get((job.product_id, job.component_type_idx, op.stage_id, machine_id), 0.0)
        else:
            proc = self.instance.processing_times_final.get((job.product_id, op.stage_id, machine_id), 0.0)
        total = setup + proc

        op.is_processing = True
        op.is_ready = False
        op.start_time = self.current_time
        op.completion_time = self.current_time + total

        ms.is_idle = False
        ms.current_op = op_id
        ms.remaining_time = total

    def _dispatch_assembly(self, comp_job_ids: Tuple[int, ...], machine_id: int):
        """조립 dispatch: 동일 unit 컴포넌트 소비 → final job 활성화 → 조립 op 시작"""
        first_job = self.instance.jobs[comp_job_ids[0]]
        product_id = first_job.product_id
        unit_key  = (first_job.order_id, first_job.order_unit_idx)
        final_key = (product_id, first_job.order_id, first_job.order_unit_idx)
        asm_stage = self.instance.config.assembly_stage_idx

        # Constraint 5: 동일 unit 컴포넌트만 소비
        for job_id in comp_job_ids:
            comp_type = self.instance.jobs[job_id].component_type_idx
            del self.assembly_pool[product_id][unit_key][comp_type]
        if not self.assembly_pool[product_id][unit_key]:
            del self.assembly_pool[product_id][unit_key]

        # Constraint 10: 조립 후 pool 여유 생기면 blocked 기계 해제
        self._try_unblock_pool()

        # Constraint 5: 동일 unit의 final job 활성화
        final_job_id = self.inactive_final_jobs.pop(final_key)
        for op_id in self.job_ops[final_job_id]:
            self.operations[op_id].active = True

        # 조립 op 직접 dispatch
        asm_op_id = self.job_ops[final_job_id][0]
        asm_op = self.operations[asm_op_id]
        ms = self.machine_states[machine_id]

        asm_op.machine_id = machine_id
        setup = self._get_setup_time(ms.last_product, product_id, asm_stage, machine_id)
        proc = self.instance.processing_times_final.get((product_id, asm_stage, machine_id), 0.0)
        total = setup + proc

        asm_op.is_processing = True
        asm_op.is_ready = False
        asm_op.buffer_waiting = False
        asm_op.start_time = self.current_time
        asm_op.completion_time = self.current_time + total

        ms.is_idle = False
        ms.current_op = asm_op_id
        ms.remaining_time = total

    def _get_setup_time(self, last_prod, curr_prod, stage_id, machine_id) -> float:
        if last_prod is None or last_prod == curr_prod:
            return 0.0
        return self.instance.setup_times.get((last_prod, curr_prod, stage_id, machine_id), 0.0)

    # ──────────────────────────────────────────────────────
    # DES 전진
    # ──────────────────────────────────────────────────────

    def _advance_until_next_decision_point(self):
        while True:
            self.update_ready_operations()
            if self._has_valid_action():
                break

            processing = [op for op in self.operations.values()
                          if op.active and op.is_processing]

            if not processing:
                # 처리 중인 op도 없고 유효 액션도 없음
                # done 상태이면 정상 종료, 아니면 deadlock
                if not self._check_done():
                    self._deadlock_detected = True
                break

            next_time = min(op.completion_time for op in processing)
            dt = next_time - self.current_time

            for ms in self.machine_states.values():
                if not ms.is_idle and not ms.is_blocked:
                    ms.total_busy_time += dt

            self.current_time = next_time
            self._complete_operations_at(next_time)
            self._move_completed_to_next_buffer()
            self._update_machine_remaining()

    def _complete_operations_at(self, t: float):
        for op in self.operations.values():
            if (op.active and op.is_processing
                    and op.completion_time is not None
                    and op.completion_time <= t + 1e-9):
                op.is_processing = False
                op.is_done = True

                if op.machine_id is not None:
                    ms = self.machine_states[op.machine_id]
                    ms.is_idle = True
                    ms.current_op = None
                    ms.remaining_time = 0.0
                    ms.last_product = op.product_id

    def _move_completed_to_next_buffer(self):
        for op in self.operations.values():
            if not op.active or not op.is_done:
                continue

            job_id = op.job_id
            job = self.instance.jobs[job_id]
            op_list = self.job_ops[job_id]
            op_idx = op_list.index(op.op_id)

            # 마지막 op이면
            if op_idx == len(op_list) - 1:
                # component job 완료 → assembly pool 진입
                if job.is_component:
                    comp_type = job.component_type_idx
                    prod_id   = job.product_id
                    unit_key  = (job.order_id, job.order_unit_idx)
                    pool      = self.assembly_pool[prod_id]
                    # 이미 pool에 있으면 재처리 불필요
                    if pool.get(unit_key, {}).get(comp_type) == job_id:
                        continue
                    # 해당 unit이 이미 조립 완료됐으면 재진입 불필요
                    if (prod_id, job.order_id, job.order_unit_idx) not in self.inactive_final_jobs:
                        continue
                    # Constraint 10: 조립 버퍼 용량 제한
                    asm_cap      = self.instance.config.assembly_buffer_capacity
                    total_in_pool = sum(len(ud) for ud in pool.values())
                    if asm_cap > 0 and total_in_pool >= asm_cap:
                        # 버퍼 가득 → 기계 blocked
                        if op.machine_id is not None:
                            ms = self.machine_states[op.machine_id]
                            if not ms.is_blocked:
                                ms.is_blocked = True
                                ms.is_idle    = False
                                ms.blocked_job = job_id
                                self._pool_blocked[job_id] = op.machine_id
                    else:
                        # Constraint 5: unit_key 단위로 pool에 저장
                        if unit_key not in pool:
                            pool[unit_key] = {}
                        pool[unit_key][comp_type] = job_id
                continue

            # 다음 op이 있으면 다음 stage 버퍼로 이동
            next_op_id = op_list[op_idx + 1]
            next_op = self.operations[next_op_id]

            if next_op.buffer_waiting or next_op.is_processing or next_op.is_done:
                continue

            next_stage = next_op.stage_id
            buffer = self.buffers[next_stage]

            # Constraint 9: 일반 버퍼 용량 제한 WIP_j(t) ≤ B_j
            if buffer.has_space():
                if job_id not in buffer.queue:
                    buffer.push(job_id)
                next_op.buffer_waiting = True
            else:
                if op.machine_id is not None:
                    ms = self.machine_states[op.machine_id]
                    if ms.current_op is None and not ms.is_blocked:
                        ms.is_blocked = True
                        ms.is_idle = False
                        ms.blocked_job = job_id

    def _update_machine_remaining(self):
        for ms in self.machine_states.values():
            if ms.current_op is not None and not ms.is_blocked:
                op = self.operations[ms.current_op]
                if op.completion_time is not None:
                    ms.remaining_time = max(0.0, op.completion_time - self.current_time)
            elif ms.is_blocked:
                ms.remaining_time = 0.0

    def _try_unblock_machines(self):
        for ms in self.machine_states.values():
            if not ms.is_blocked or ms.blocked_job is None:
                continue
            job_id = ms.blocked_job
            if job_id in self._pool_blocked:
                continue  # Constraint 10: pool-blocked는 _try_unblock_pool에서 처리
            op_list = self.job_ops[job_id]
            for idx, oid in enumerate(op_list):
                op = self.operations[oid]
                if op.is_done and idx < len(op_list) - 1:
                    next_op = self.operations[op_list[idx + 1]]
                    if not next_op.buffer_waiting and not next_op.is_processing and not next_op.is_done:
                        buf = self.buffers[next_op.stage_id]
                        if buf.has_space():
                            buf.push(job_id)
                            next_op.buffer_waiting = True
                            ms.is_blocked = False
                            ms.is_idle = True
                            ms.blocked_job = None
                            ms.current_op = None
                            break

    def _try_unblock_pool(self):
        """Constraint 10: 조립 버퍼 여유 생기면 대기 중 컴포넌트 투입 & 기계 해제"""
        asm_cap = self.instance.config.assembly_buffer_capacity
        for job_id in list(self._pool_blocked.keys()):
            job      = self.instance.jobs[job_id]
            prod_id  = job.product_id
            comp_type = job.component_type_idx
            unit_key = (job.order_id, job.order_unit_idx)
            pool     = self.assembly_pool[prod_id]
            total_in_pool = sum(len(ud) for ud in pool.values())
            if asm_cap <= 0 or total_in_pool < asm_cap:
                if unit_key not in pool:
                    pool[unit_key] = {}
                pool[unit_key][comp_type] = job_id
                mid = self._pool_blocked.pop(job_id)
                ms  = self.machine_states[mid]
                ms.is_blocked  = False
                ms.is_idle     = True
                ms.blocked_job = None
                ms.current_op  = None

    # ──────────────────────────────────────────────────────
    # Ready 판정
    # ──────────────────────────────────────────────────────

    def update_ready_operations(self):
        self._try_unblock_machines()
        for op in self.operations.values():
            if not op.active or op.is_done or op.is_processing:
                op.is_ready = False
                continue
            # Constraint 4: 선행 operation이 모두 완료되어야 시작 가능
            pred_done = all(
                self.operations[pid].is_done
                for pid in op.predecessors
                if self.operations[pid].active
            )
            op.is_ready = pred_done and op.buffer_waiting

    # ──────────────────────────────────────────────────────
    # Valid Action
    # ──────────────────────────────────────────────────────

    def _is_valid_regular_action(self, op_id: int, machine_id: int) -> bool:
        op = self.operations[op_id]
        ms = self.machine_states[machine_id]
        if not op.active or op.is_done:          return False
        if not op.is_ready:                       return False
        # Constraint 1: 단일 할당 — 이미 처리 중이면 재할당 불가
        if op.is_processing:                      return False
        # Constraint 6: 기계는 동시에 하나의 작업만 처리
        if not ms.is_idle or ms.is_blocked:       return False
        if ms.stage_id != op.stage_id:            return False
        job = self.instance.jobs[op.job_id]
        # Constraint 2: 기계적합성 — 기계가 해당 제품/stage를 처리 가능해야 함
        if job.is_component:
            if (job.product_id, job.component_type_idx) not in ms.compatible_component_ops:
                return False
        else:
            if job.product_id not in ms.compatible_final_ops:
                return False
        return True

    def _get_valid_assembly_actions(self) -> List[AssemblyAction]:
        """Constraint 5: 조립 조건 — 동일 unit의 모든 컴포넌트 타입이 pool에 존재해야 조립 가능"""
        actions = []
        asm_stage = self.instance.config.assembly_stage_idx

        for prod_id, unit_groups in self.assembly_pool.items():
            num_types = self.instance.products[prod_id].num_components
            if num_types < 2:
                continue

            for unit_key, comp_dict in unit_groups.items():
                order_id, unit_idx = unit_key
                # Constraint 5: 동일 unit의 모든 컴포넌트 타입이 pool에 있어야 함
                if not all(t in comp_dict for t in range(num_types)):
                    continue
                # 해당 unit의 미활성 final job 존재 확인
                final_key = (prod_id, order_id, unit_idx)
                if final_key not in self.inactive_final_jobs:
                    continue

                # Constraint 2: 호환 조립 기계 (idle + 제품 호환)
                asm_machines = [
                    mid for mid in self.instance.machines_by_stage.get(asm_stage, [])
                    if (prod_id in self.machine_states[mid].compatible_final_ops
                        and self.machine_states[mid].is_idle
                        and not self.machine_states[mid].is_blocked)
                ]
                if not asm_machines:
                    continue

                # 동일 unit 컴포넌트 tuple (타입 순서 고정)
                comp_ids = tuple(comp_dict[t] for t in range(num_types))
                for mid in asm_machines:
                    actions.append((comp_ids, mid))

        return actions

    # ──────────────────────────────────────────────────────
    # Dispatching Rules
    # ──────────────────────────────────────────────────────

    # ── 조립 공정 헬퍼 ──────────────────────────────────────

    def _assemblable_products(self) -> List[int]:
        """조립 가능 제품 목록: pool 완비 + inactive final_job 존재 + idle 조립 기계 존재"""
        asm_stage = self.instance.config.assembly_stage_idx
        result = []
        for prod_id, unit_groups in self.assembly_pool.items():
            num_types = self.instance.products[prod_id].num_components
            if num_types < 2:
                continue
            has_ready_unit = any(
                all(t in comp_dict for t in range(num_types))
                and (prod_id, uk[0], uk[1]) in self.inactive_final_jobs
                for uk, comp_dict in unit_groups.items()
            )
            if not has_ready_unit:
                continue
            has_machine = any(
                prod_id in self.machine_states[mid].compatible_final_ops
                and self.machine_states[mid].is_idle
                and not self.machine_states[mid].is_blocked
                for mid in self.instance.machines_by_stage.get(asm_stage, [])
            )
            if has_machine:
                result.append(prod_id)
        return result

    def _best_asm_machine(self, prod_id: int) -> Optional[int]:
        """해당 제품 조립 stage에서 처리시간 최소 idle 기계 반환"""
        asm_stage = self.instance.config.assembly_stage_idx
        best_mid, best_t = None, float('inf')
        for mid in self.instance.machines_by_stage.get(asm_stage, []):
            ms = self.machine_states[mid]
            if not ms.is_idle or ms.is_blocked:
                continue
            if prod_id not in ms.compatible_final_ops:
                continue
            t = self.instance.processing_times_final.get((prod_id, asm_stage, mid), float('inf'))
            if t < best_t:
                best_t = t
                best_mid = mid
        return best_mid

    def _select_comp_jobs_fifo(self, prod_id: int) -> Optional[Tuple[int, ...]]:
        """FIFO: 각 타입에서 job_id 가장 작은(가장 먼저 생성된) unit의 comp_job 선택"""
        num_types = self.instance.products[prod_id].num_components
        best_comp_ids, best_key = None, None
        for unit_key, comp_dict in self.assembly_pool[prod_id].items():
            if not all(t in comp_dict for t in range(num_types)):
                continue
            if (prod_id, unit_key[0], unit_key[1]) not in self.inactive_final_jobs:
                continue
            comp_ids = tuple(comp_dict[t] for t in range(num_types))
            key = min(comp_ids)
            if best_key is None or key < best_key:
                best_key = key
                best_comp_ids = comp_ids
        return best_comp_ids

    def _select_comp_jobs_edd(self, prod_id: int) -> Optional[Tuple[int, ...]]:
        """EDD: 납기 가장 빠른 final_job에 대응하는 unit의 comp_job 선택"""
        num_types = self.instance.products[prod_id].num_components
        best_comp_ids, best_due = None, float('inf')
        for unit_key, comp_dict in self.assembly_pool[prod_id].items():
            if not all(t in comp_dict for t in range(num_types)):
                continue
            final_key = (prod_id, unit_key[0], unit_key[1])
            if final_key not in self.inactive_final_jobs:
                continue
            fid = self.inactive_final_jobs[final_key]
            due = self.instance.jobs[fid].due_date
            if due < best_due:
                best_due = due
                best_comp_ids = tuple(comp_dict[t] for t in range(num_types))
        return best_comp_ids

    def _final_job_remaining_work(self, fid: int) -> float:
        """inactive final_job의 잔여 최소 처리시간 합산 (조립 stage + post-asm 전체)"""
        job = self.instance.jobs[fid]
        total = 0.0
        for stage_id in job.route:
            compat = [
                mid for mid in self.instance.machines_by_stage.get(stage_id, [])
                if job.product_id in self.instance.machines[mid].compatible_final_ops
            ]
            times = [
                self.instance.processing_times_final.get((job.product_id, stage_id, mid), 0.0)
                for mid in compat
            ]
            total += min(times) if times else 0.0
        return total

    def _best_ready_unit(self, prod_id: int, key_fn) -> Optional[Tuple[int, ...]]:
        """key_fn 기준으로 최선 unit의 comp_ids 반환 (공통 unit 선택 로직)"""
        num_types = self.instance.products[prod_id].num_components
        best_comp_ids, best_key = None, None
        for unit_key, comp_dict in self.assembly_pool[prod_id].items():
            if not all(t in comp_dict for t in range(num_types)):
                continue
            final_key = (prod_id, unit_key[0], unit_key[1])
            if final_key not in self.inactive_final_jobs:
                continue
            k = key_fn(self.inactive_final_jobs[final_key], unit_key, comp_dict)
            if best_key is None or k < best_key:
                best_key = k
                best_comp_ids = tuple(comp_dict[t] for t in range(num_types))
        return best_comp_ids

    # ── 조립 공정 디스패칭룰 ────────────────────────────────

    def _apply_fifo_asm(self) -> Optional[Tuple[int, ...]]:
        """FIFO_asm: pool에서 job_id 가장 작은(가장 먼저 도착한) comp_job 기준 pair 선택"""
        best_comp_ids, best_key = None, None
        for prod_id in self._assemblable_products():
            comp_ids = self._select_comp_jobs_fifo(prod_id)
            if comp_ids is None:
                continue
            key = min(comp_ids)
            if best_key is None or key < best_key:
                best_key = key
                best_comp_ids = comp_ids
        return best_comp_ids

    def _apply_edd_asm(self) -> Optional[Tuple[int, ...]]:
        """EDD_asm: 납기 가장 빠른 final_job 기준 pair 선택"""
        best_comp_ids, best_due = None, float('inf')
        for prod_id in self._assemblable_products():
            num_types = self.instance.products[prod_id].num_components
            for unit_key, comp_dict in self.assembly_pool[prod_id].items():
                if not all(t in comp_dict for t in range(num_types)):
                    continue
                final_key = (prod_id, unit_key[0], unit_key[1])
                if final_key not in self.inactive_final_jobs:
                    continue
                fid = self.inactive_final_jobs[final_key]
                due = self.instance.jobs[fid].due_date
                if due < best_due:
                    best_due = due
                    best_comp_ids = tuple(comp_dict[t] for t in range(num_types))
        return best_comp_ids

    def _apply_mwkr_asm(self) -> Optional[Tuple[int, ...]]:
        """MWKR_asm: 잔여 작업량 가장 많은 final_job 기준 pair 선택"""
        best_comp_ids, best_work = None, -1.0
        for prod_id in self._assemblable_products():
            num_types = self.instance.products[prod_id].num_components
            for unit_key, comp_dict in self.assembly_pool[prod_id].items():
                if not all(t in comp_dict for t in range(num_types)):
                    continue
                final_key = (prod_id, unit_key[0], unit_key[1])
                if final_key not in self.inactive_final_jobs:
                    continue
                fid = self.inactive_final_jobs[final_key]
                work = self._final_job_remaining_work(fid)
                if work > best_work:
                    best_work = work
                    best_comp_ids = tuple(comp_dict[t] for t in range(num_types))
        return best_comp_ids

    def _apply_spt_asm(self) -> Optional[Tuple[int, ...]]:
        """SPT_asm: 조립 처리시간 가장 짧은 제품 기준 pair 선택 (unit은 EDD)"""
        asm_stage = self.instance.config.assembly_stage_idx
        best_comp_ids, best_t = None, float('inf')
        for prod_id in self._assemblable_products():
            min_t = min(
                (self.instance.processing_times_final.get((prod_id, asm_stage, mid), float('inf'))
                 for mid in self.instance.machines_by_stage.get(asm_stage, [])
                 if self.machine_states[mid].is_idle
                 and not self.machine_states[mid].is_blocked
                 and prod_id in self.machine_states[mid].compatible_final_ops),
                default=float('inf')
            )
            if min_t < best_t:
                comp_ids = self._select_comp_jobs_edd(prod_id)
                if comp_ids is not None:
                    best_t = min_t
                    best_comp_ids = comp_ids
        return best_comp_ids

    def _apply_winq_asm(self) -> Optional[Tuple[int, ...]]:
        """WINQ_asm: 조립 후 다음 stage 대기 작업량 가장 적은 제품 기준 pair 선택"""
        asm_stage = self.instance.config.assembly_stage_idx
        best_comp_ids, best_load = None, float('inf')
        for prod_id in self._assemblable_products():
            final_stages = self.instance.final_stage_matrix.get(prod_id, [])
            try:
                asm_idx = final_stages.index(asm_stage)
            except ValueError:
                asm_idx = -1
            if asm_idx >= 0 and asm_idx < len(final_stages) - 1:
                next_stage = final_stages[asm_idx + 1]
                load = 0.0
                for jid in self.buffers[next_stage].queue:
                    qjob = self.instance.jobs.get(jid)
                    if qjob is None:
                        continue
                    compat = [
                        m for m in self.instance.machines_by_stage.get(next_stage, [])
                        if qjob.product_id in self.instance.machines[m].compatible_final_ops
                    ]
                    times = [self.instance.processing_times_final.get(
                        (qjob.product_id, next_stage, m), 0.0) for m in compat]
                    load += min(times) if times else 0.0
            else:
                load = 0.0
            if load < best_load:
                comp_ids = self._select_comp_jobs_fifo(prod_id)
                if comp_ids is not None:
                    best_load = load
                    best_comp_ids = comp_ids
        return best_comp_ids

    def _remaining_work(self, op: OperationState) -> float:
        """MWKR용: job의 미완료 op 최소 처리시간 합산"""
        total = 0.0
        job = self.instance.jobs[op.job_id]
        for oid in self.job_ops[op.job_id]:
            o = self.operations[oid]
            if o.is_done or o.is_processing:
                continue
            if job.is_component:
                compat = [
                    mid for mid in self.instance.machines_by_stage.get(o.stage_id, [])
                    if (job.product_id, job.component_type_idx) in self.instance.machines[mid].compatible_component_ops
                ]
                times = [self.instance.processing_times.get((job.product_id, job.component_type_idx, o.stage_id, mid), 0.0) for mid in compat]
            else:
                compat = [
                    mid for mid in self.instance.machines_by_stage.get(o.stage_id, [])
                    if job.product_id in self.instance.machines[mid].compatible_final_ops
                ]
                times = [self.instance.processing_times_final.get((job.product_id, o.stage_id, mid), 0.0) for mid in compat]
            total += min(times) if times else 0.0
        return total

    def _best_machine_for_op(self, op: OperationState) -> Optional[int]:
        """유휴 호환 기계 중 처리시간 최소 기계 반환"""
        best_mid: Optional[int] = None
        best_t = float('inf')
        job = self.instance.jobs[op.job_id]
        for mid in self.instance.machines_by_stage.get(op.stage_id, []):
            if not self._is_valid_regular_action(op.op_id, mid):
                continue
            if job.is_component:
                t = self.instance.processing_times.get((job.product_id, job.component_type_idx, op.stage_id, mid), float('inf'))
            else:
                t = self.instance.processing_times_final.get((job.product_id, op.stage_id, mid), float('inf'))
            if t < best_t:
                best_t = t
                best_mid = mid
        return best_mid

    def _apply_fifo(self) -> Optional[int]:
        """FIFO: 가장 일찍 도착한 job의 ready op 선택 (op_id만 반환)"""
        ready = [
            op for op in self.operations.values()
            if op.active and op.is_ready and not op.is_processing and not op.is_assembly
        ]
        ready.sort(key=lambda op: self.instance.jobs[op.job_id].job_id)
        for op in ready:
            if self._best_machine_for_op(op) is not None:
                return op.op_id
        return None

    def _apply_edd(self) -> Optional[int]:
        """EDD: 납기가 가장 빠른 job의 ready op 선택 (op_id만 반환)"""
        ready = [
            op for op in self.operations.values()
            if op.active and op.is_ready and not op.is_processing and not op.is_assembly
        ]
        ready.sort(key=lambda op: self.instance.jobs[op.job_id].due_date)
        for op in ready:
            if self._best_machine_for_op(op) is not None:
                return op.op_id
        return None

    def _apply_mwkr(self) -> Optional[int]:
        """MWKR: 잔여 작업량이 가장 큰 job의 ready op 선택 (op_id만 반환)"""
        ready = [
            op for op in self.operations.values()
            if op.active and op.is_ready and not op.is_processing and not op.is_assembly
        ]
        ready.sort(key=lambda op: self._remaining_work(op), reverse=True)
        for op in ready:
            if self._best_machine_for_op(op) is not None:
                return op.op_id
        return None

    def _apply_spt(self) -> Optional[int]:
        """SPT: 현재 op 처리시간이 가장 짧은 것 우선 (op_id만 반환)"""
        ready = [
            op for op in self.operations.values()
            if op.active and op.is_ready and not op.is_processing and not op.is_assembly
        ]

        def min_proc_time(op: OperationState) -> float:
            job = self.instance.jobs[op.job_id]
            if job.is_component:
                times = [
                    self.instance.processing_times.get((job.product_id, job.component_type_idx, op.stage_id, mid), float('inf'))
                    for mid in self.instance.machines_by_stage.get(op.stage_id, [])
                    if self._is_valid_regular_action(op.op_id, mid)
                ]
            else:
                times = [
                    self.instance.processing_times_final.get((job.product_id, op.stage_id, mid), float('inf'))
                    for mid in self.instance.machines_by_stage.get(op.stage_id, [])
                    if self._is_valid_regular_action(op.op_id, mid)
                ]
            return min(times) if times else float('inf')

        ready.sort(key=min_proc_time)
        for op in ready:
            if self._best_machine_for_op(op) is not None:
                return op.op_id
        return None

    def _apply_winq(self) -> Optional[int]:
        """WINQ: 다음 스테이지 대기 작업량이 가장 적은 op 우선 (op_id만 반환)"""
        ready = [
            op for op in self.operations.values()
            if op.active and op.is_ready and not op.is_processing and not op.is_assembly
        ]

        def next_queue_work(op: OperationState) -> float:
            job_ops = self.job_ops[op.job_id]
            idx = job_ops.index(op.op_id)
            if idx >= len(job_ops) - 1:
                return 0.0
            next_stage = self.operations[job_ops[idx + 1]].stage_id
            total = 0.0
            for jid in self.buffers[next_stage].queue:
                queued_job = self.instance.jobs.get(jid)
                if queued_job is None:
                    continue
                if queued_job.is_component:
                    compat = [
                        mid for mid in self.instance.machines_by_stage.get(next_stage, [])
                        if (queued_job.product_id, queued_job.component_type_idx) in self.instance.machines[mid].compatible_component_ops
                    ]
                    times = [self.instance.processing_times.get((queued_job.product_id, queued_job.component_type_idx, next_stage, mid), 0.0) for mid in compat]
                else:
                    compat = [
                        mid for mid in self.instance.machines_by_stage.get(next_stage, [])
                        if queued_job.product_id in self.instance.machines[mid].compatible_final_ops
                    ]
                    times = [self.instance.processing_times_final.get((queued_job.product_id, next_stage, mid), 0.0) for mid in compat]
                total += min(times) if times else 0.0
            return total

        ready.sort(key=next_queue_work)
        for op in ready:
            if self._best_machine_for_op(op) is not None:
                return op.op_id
        return None

    def _apply_balance(self) -> Optional[int]:
        """BALANCE: comp type 불균형 감지 → 부족한 type의 ready op 중 FIFO 선택."""
        type_counts: Dict[int, int] = {}
        for unit_pool in self.assembly_pool.values():
            for type_map in unit_pool.values():
                for comp_type in type_map:
                    type_counts[comp_type] = type_counts.get(comp_type, 0) + 1
        for op in self.operations.values():
            if op.active and op.is_processing:
                job = self.instance.jobs[op.job_id]
                if job.is_component:
                    ct = job.component_type_idx
                    type_counts[ct] = type_counts.get(ct, 0) + 1

        if len(type_counts) < 2:
            return None
        min_count = min(type_counts.values())
        if min_count == max(type_counts.values()):
            return None  # 균형 상태 — 다른 룰에 위임

        lagging = {ct for ct, cnt in type_counts.items() if cnt == min_count}
        candidates = [
            op for op in self.operations.values()
            if (op.active and op.is_ready and not op.is_processing and not op.is_assembly
                and self.instance.jobs[op.job_id].is_component
                and self.instance.jobs[op.job_id].component_type_idx in lagging
                and self._best_machine_for_op(op) is not None)
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda o: o.job_id).op_id

    def _get_valid_action_pairs(self) -> List[Action]:
        # 일반 공정: 룰 6개 → op 후보 수집 → op × 모든 호환 기계 조합
        op_candidates: List[int] = []
        op_seen: set = set()
        for op_id in [self._apply_fifo(), self._apply_edd(), self._apply_mwkr(),
                      self._apply_spt(), self._apply_winq(), self._apply_balance()]:
            if op_id is not None and op_id not in op_seen:
                op_candidates.append(op_id)
                op_seen.add(op_id)

        regular_candidates: List[Action] = []
        seen: set = set()
        for op_id in op_candidates:
            op = self.operations[op_id]
            for mid in self.instance.machines_by_stage.get(op.stage_id, []):
                if self._is_valid_regular_action(op_id, mid):
                    a: Action = (op_id, mid)
                    if a not in seen:
                        regular_candidates.append(a)
                        seen.add(a)

        # 룰이 후보를 못 찾으면 전체 유효 액션 fallback
        if not regular_candidates:
            for op in self.operations.values():
                if not op.active or not op.is_ready:
                    continue
                for mid in self.instance.machines_by_stage.get(op.stage_id, []):
                    if self._is_valid_regular_action(op.op_id, mid):
                        a: Action = (op.op_id, mid)
                        if a not in seen:
                            regular_candidates.append(a)
                            seen.add(a)

        # 조립 공정: 룰이 pair 후보 선정 → pair × 가능한 기계 전부 조합
        asm_stage = self.instance.config.assembly_stage_idx
        pair_candidates: List[Tuple[int, ...]] = []
        pair_seen: set = set()
        for comp_ids in [self._apply_fifo_asm(), self._apply_edd_asm(), self._apply_mwkr_asm(),
                         self._apply_spt_asm(), self._apply_winq_asm()]:
            if comp_ids is not None and comp_ids not in pair_seen:
                pair_candidates.append(comp_ids)
                pair_seen.add(comp_ids)

        asm_candidates: List[Action] = []
        asm_seen: set = set()
        if pair_candidates:
            for comp_ids in pair_candidates:
                prod_id = self.instance.jobs[comp_ids[0]].product_id
                for mid in self.instance.machines_by_stage.get(asm_stage, []):
                    ms = self.machine_states[mid]
                    if not ms.is_idle or ms.is_blocked:
                        continue
                    if prod_id not in ms.compatible_final_ops:
                        continue
                    a: Action = (comp_ids, mid)
                    if a not in asm_seen:
                        asm_candidates.append(a)
                        asm_seen.add(a)
        else:
            # 룰이 후보를 못 찾으면 전체 유효 조립 액션 fallback
            asm_candidates = self._get_valid_assembly_actions()

        return regular_candidates + asm_candidates

    def _has_valid_action(self) -> bool:
        for op in self.operations.values():
            if not op.active or not op.is_ready:
                continue
            for mid in self.instance.machines_by_stage.get(op.stage_id, []):
                if self._is_valid_regular_action(op.op_id, mid):
                    return True
        return len(self._assemblable_products()) > 0

    # ──────────────────────────────────────────────────────
    # Observation
    # ──────────────────────────────────────────────────────

    def _get_obs(self) -> dict:
        graph = self.graph_builder.build(self)
        actions = self._get_valid_action_pairs()
        self._current_actions = actions

        mask = np.ones(len(actions), dtype=np.float32) if actions else np.array([], dtype=np.float32)

        active_ops = [op for op in self.operations.values() if op.active]
        prev_map: Dict[int, Optional[int]] = {}
        next_map: Dict[int, Optional[int]] = {}
        candidate_machines: Dict[int, List[int]] = {}

        for op in active_ops:
            oid = op.op_id
            prev_map[oid] = op.predecessors[0] if op.predecessors else None
            job_ops = self.job_ops[op.job_id]
            idx = job_ops.index(oid)
            next_map[oid] = job_ops[idx + 1] if idx < len(job_ops) - 1 else None
            job = self.instance.jobs[op.job_id]
            if job.is_component:
                candidate_machines[oid] = [
                    mid for mid in self.instance.machines_by_stage.get(op.stage_id, [])
                    if (job.product_id, job.component_type_idx) in self.instance.machines[mid].compatible_component_ops
                ]
            else:
                candidate_machines[oid] = [
                    mid for mid in self.instance.machines_by_stage.get(op.stage_id, [])
                    if job.product_id in self.instance.machines[mid].compatible_final_ops
                ]

        # 조립 pool 정보 — 모델 호환을 위해 {prod_id: {comp_type: [job_id,...]}} 로 직렬화
        assembly_pool_info: Dict[int, Dict[int, List[int]]] = {}
        for prod_id, unit_groups in self.assembly_pool.items():
            flat: Dict[int, List[int]] = {}
            for comp_dict in unit_groups.values():
                for comp_type, job_id in comp_dict.items():
                    flat.setdefault(comp_type, []).append(job_id)
            assembly_pool_info[prod_id] = flat

        # inactive_final_jobs — {prod_id: [job_id,...]} 로 직렬화
        inactive_info: Dict[int, List[int]] = {}
        for (prod_id, _oid, _uid), job_id in self.inactive_final_jobs.items():
            inactive_info.setdefault(prod_id, []).append(job_id)

        return {
            "graph": graph,
            "actions": actions,
            "action_mask": mask,
            "precedence_info": {
                "prev_map": prev_map,
                "next_map": next_map,
                "candidate_machines": candidate_machines,
            },
            "assembly_pool": assembly_pool_info,
            "inactive_final_jobs": inactive_info,
            "job_op_map": {jid: list(ops) for jid, ops in self.job_ops.items()},
        }

    # ──────────────────────────────────────────────────────
    # Reward
    # ──────────────────────────────────────────────────────

    def _reward_fn(self) -> float:
        """r = -Δ(estimated weighted tardiness) — dense reward shaping"""
        current_est_wt = self._compute_estimated_weighted_tardiness()
        delta = current_est_wt - self._prev_estimated_wt
        self._prev_estimated_wt = current_est_wt
        return -delta

    def _avg_proc_time_for_op(self, op_id: int) -> float:
        """operation 하나에 대해 호환 가능한 기계들의 평균 처리시간. 호환 기계 없으면 35.0 반환."""
        op = self.operations[op_id]
        job = self.instance.jobs[op.job_id]
        times = []
        if job.is_component:
            for mid in self.instance.machines_by_stage.get(op.stage_id, []):
                key = (job.product_id, job.component_type_idx, op.stage_id, mid)
                if key in self.instance.processing_times:
                    times.append(self.instance.processing_times[key])
        else:
            for mid in self.instance.machines_by_stage.get(op.stage_id, []):
                key = (job.product_id, op.stage_id, mid)
                if key in self.instance.processing_times_final:
                    times.append(self.instance.processing_times_final[key])
        return float(np.mean(times)) if times else 35.0

    def _estimate_comp_completion(self, comp_job_id: int) -> float:
        """comp_job 하나가 조립 버퍼에 들어올 예상 시간."""
        job = self.instance.jobs[comp_job_id]
        prod_id = job.product_id

        # Case 1: 이미 assembly_pool에 있음 → 현재 시간 반환
        for comp_dict in self.assembly_pool.get(prod_id, {}).values():
            if comp_job_id in comp_dict.values():
                return self.current_time

        # Case 2: pool_blocked → 처리는 완료, pool 진입 대기
        if comp_job_id in self._pool_blocked:
            last_op = self.operations[self.job_ops[comp_job_id][-1]]
            return last_op.completion_time if last_op.completion_time is not None else self.current_time

        op_list = self.job_ops[comp_job_id]

        # Case 3: 현재 처리 중인 op 존재
        for i, oid in enumerate(op_list):
            op = self.operations[oid]
            if op.is_processing:
                ms = self.machine_states[op.machine_id]
                finish = self.current_time + ms.remaining_time
                remaining = sum(self._avg_proc_time_for_op(op_list[j]) for j in range(i + 1, len(op_list)))
                return finish + remaining

        # Case 4: 버퍼 대기 또는 미시작 — 첫 미완료 op부터 평균 처리시간 합산
        for i, oid in enumerate(op_list):
            if not self.operations[oid].is_done:
                remaining = sum(self._avg_proc_time_for_op(op_list[j]) for j in range(i, len(op_list)))
                return self.current_time + remaining

        # 모든 op 완료 (pool 체크에서 못 잡힌 경우)
        last_op = self.operations[op_list[-1]]
        return last_op.completion_time if last_op.completion_time is not None else self.current_time

    def _estimate_completion_time(self, final_job_id: int) -> float:
        """final_job 하나의 예상 완료시간. 3가지 케이스로 분기."""
        job = self.instance.jobs[final_job_id]
        op_list = self.job_ops[final_job_id]
        if not op_list:
            return self.current_time

        last_op = self.operations[op_list[-1]]

        # Case A: 이미 완료
        if last_op.active and last_op.is_done and last_op.completion_time is not None:
            return last_op.completion_time

        # Case B: active 상태 (조립 이후 진행 중)
        if last_op.active:
            for i, oid in enumerate(op_list):
                op = self.operations[oid]
                if op.is_processing:
                    ms = self.machine_states[op.machine_id]
                    finish = self.current_time + ms.remaining_time
                    remaining = sum(self._avg_proc_time_for_op(op_list[j]) for j in range(i + 1, len(op_list)))
                    return finish + remaining
            # 처리 중인 op 없음 (버퍼 대기)
            for i, oid in enumerate(op_list):
                if not self.operations[oid].is_done:
                    remaining = sum(self._avg_proc_time_for_op(op_list[j]) for j in range(i, len(op_list)))
                    return self.current_time + remaining
            return last_op.completion_time if last_op.completion_time is not None else self.current_time

        # Case C: inactive (조립 전) — comp_type 가용성 기반 추정
        prod_id = job.product_id
        num_comp_types = self.instance.products[prod_id].num_components
        pool = self.assembly_pool.get(prod_id, {})

        comp_ready_times = []
        for comp_type in range(num_comp_types):
            earliest: Optional[float] = None

            # assembly_pool에 이미 있는 경우
            for comp_dict in pool.values():
                if comp_type in comp_dict:
                    earliest = self.current_time
                    break

            # pool_blocked 중에 해당 comp_type
            if earliest is None:
                for blocked_jid in self._pool_blocked:
                    bj = self.instance.jobs[blocked_jid]
                    if bj.product_id == prod_id and bj.component_type_idx == comp_type:
                        bop = self.operations[self.job_ops[blocked_jid][-1]]
                        t = bop.completion_time if bop.completion_time is not None else self.current_time
                        if earliest is None or t < earliest:
                            earliest = t

            # 처리 중 또는 버퍼 대기인 comp_job 탐색
            if earliest is None:
                for order in self.instance.orders.values():
                    for cid in order.component_job_ids:
                        cjob = self.instance.jobs[cid]
                        if cjob.product_id == prod_id and cjob.component_type_idx == comp_type:
                            t = self._estimate_comp_completion(cid)
                            if earliest is None or t < earliest:
                                earliest = t

            comp_ready_times.append(earliest if earliest is not None else self.current_time)

        asm_start = max(comp_ready_times) if comp_ready_times else self.current_time
        final_proc = sum(self._avg_proc_time_for_op(oid) for oid in op_list)
        return asm_start + final_proc

    def _compute_estimated_weighted_tardiness(self) -> float:
        """모든 final_job의 추정 완료시간 기반 가중 지연 합산."""
        total = 0.0
        for order in self.instance.orders.values():
            weight = order.weight
            for fid in order.final_job_ids:
                est_t = self._estimate_completion_time(fid)
                tp = max(0.0, est_t - order.due_date)
                total += weight * tp
        return total

    def _compute_actual_weighted_tardiness(self) -> float:
        """Constraint 8: T_p = max(0, E_last - d_p) — 완료된 final job 기준 가중 지연 합산"""
        total = 0.0
        for order in self.instance.orders.values():
            weight = order.weight
            for fid in order.final_job_ids:
                op_list = self.job_ops[fid]
                if not op_list:
                    continue
                last_op = self.operations[op_list[-1]]
                if last_op.active and last_op.is_done and last_op.completion_time is not None:
                    tp = max(0.0, last_op.completion_time - order.due_date)
                    total += weight * tp
        return total

    # ──────────────────────────────────────────────────────
    # Done 체크
    # ──────────────────────────────────────────────────────

    def _check_done(self) -> bool:
        """모든 final job이 완료되면 종료"""
        for order in self.instance.orders.values():
            for fid in order.final_job_ids:
                op_list = self.job_ops[fid]
                if not op_list:
                    return False
                last_op = self.operations[op_list[-1]]
                if not last_op.active or not last_op.is_done:
                    return False
        return True

    # ──────────────────────────────────────────────────────
    # 유틸리티
    # ──────────────────────────────────────────────────────

    def get_actual_weighted_tardiness(self) -> float:
        return self._compute_actual_weighted_tardiness()

    def get_makespan(self) -> float:
        max_t = 0.0
        for op in self.operations.values():
            if op.active and op.is_done and op.completion_time is not None:
                max_t = max(max_t, op.completion_time)
        return max_t
