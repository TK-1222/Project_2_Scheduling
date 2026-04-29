"""
FFSA 스케줄링 강화학습 환경
============================
PPT 04장: 제약식 C1~C10 → action mask + DES 로직
PPT 05장: MDP 정의, State (이종 그래프), Action, Reward

리뷰 수정사항 반영:
  [1] Operation에 predecessors, machine_id, buffer_waiting 추가
  [2] _dispatch()에서 buffer pop + 상태 갱신
  [3] _advance 루프에서 빈 시퀀스 방어
  [4] _check_done() 연결
  [5] 단순 blocking (Step 1~2)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
import numpy as np
import gymnasium as gym
from gymnasium import spaces

import torch
from torch_geometric.data import HeteroData

from ffsa_instance import (
    InstanceConfig, FFSAInstance, generate_instance,
    ProductData, JobData, MachineData,
)


# ──────────────────────────────────────────────────────────
# 런타임 상태 구조체
# ──────────────────────────────────────────────────────────

@dataclass
class OperationState:
    """Operation 런타임 상태 (PPT: Oij)"""
    op_id: int
    job_id: int
    product_id: int
    stage_id: int
    # 상태 플래그
    is_done: bool = False
    is_ready: bool = False
    is_processing: bool = False
    is_assembly: bool = False
    buffer_waiting: bool = False
    # 구조 정보
    predecessors: List[int] = field(default_factory=list)      # wij: 선행 op_id (job 내부)
    component_last_ops: List[int] = field(default_factory=list) # 조립: component job의 마지막 op
    # 배정 정보
    machine_id: Optional[int] = None
    start_time: Optional[float] = None
    completion_time: Optional[float] = None


@dataclass
class MachineState:
    """Machine 런타임 상태 (PPT: k ∈ K)"""
    machine_id: int
    stage_id: int
    compatible_products: Set[int] = field(default_factory=set)
    is_idle: bool = True
    is_blocked: bool = False
    current_op: Optional[int] = None
    remaining_time: float = 0.0
    last_product: Optional[int] = None
    blocked_job: Optional[int] = None
    total_busy_time: float = 0.0     # utilization 계산용


@dataclass
class BufferState:
    """Buffer 런타임 상태 (PPT: Bj)"""
    stage_id: int
    capacity: int                    # -1 = 무한
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
    """이종 그래프 생성 (PPT Slide 10: State)"""

    def __init__(self, instance: FFSAInstance):
        self.inst = instance
        # 정규화 상수
        self.max_proc = max(instance.processing_times.values()) if instance.processing_times else 1.0
        self.max_setup = max(instance.setup_times.values()) if instance.setup_times else 1.0
        self.max_due = max(p.due_date for p in instance.products.values()) if instance.products else 1.0
        self.max_weight = max(p.weight for p in instance.products.values()) if instance.products else 1.0

    def build(self, env: "FFSASchedulingEnv") -> HeteroData:
        """현재 환경 상태에서 HeteroData 구성"""
        data = HeteroData()

        # ── Operation Node Features ──
        data['op'].x = self._build_op_features(env)

        # ── Machine Node Features ──
        data['machine'].x = self._build_machine_features(env)

        # ── Precedence Edges (op → op) ──
        prec_src, prec_dst = self._build_precedence_edges(env)
        data['op', 'precedence', 'op'].edge_index = torch.tensor(
            [prec_src, prec_dst], dtype=torch.long
        ) if prec_src else torch.zeros((2, 0), dtype=torch.long)

        # ── Assembly Dependency Edges (op → op) ──
        asm_src, asm_dst = self._build_assembly_edges(env)
        data['op', 'assembly_dep', 'op'].edge_index = torch.tensor(
            [asm_src, asm_dst], dtype=torch.long
        ) if asm_src else torch.zeros((2, 0), dtype=torch.long)

        # ── Candidate Edges (op ↔ machine) with edge_attr ──
        cand_src, cand_dst, cand_attr = self._build_candidate_edges(env)
        if cand_src:
            edge_index = torch.tensor([cand_src, cand_dst], dtype=torch.long)
            edge_attr = torch.tensor(cand_attr, dtype=torch.float32)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 2), dtype=torch.float32)

        data['op', 'candidate', 'machine'].edge_index = edge_index
        data['op', 'candidate', 'machine'].edge_attr = edge_attr

        # 역방향
        data['machine', 'candidate_rev', 'op'].edge_index = edge_index.flip(0)
        data['machine', 'candidate_rev', 'op'].edge_attr = edge_attr

        return data

    def _build_op_features(self, env: "FFSASchedulingEnv") -> torch.Tensor:
        """Operation Node Feature (PPT Slide 10): 14차원"""
        num_ops = env.num_operations
        feats = np.zeros((num_ops, 14), dtype=np.float32)
        max_pred = max((len(op.predecessors) + len(op.component_last_ops))
                       for op in env.operations.values()) or 1
        max_comp = max((len(op.component_last_ops))
                       for op in env.operations.values()) or 1

        for op in env.operations.values():
            i = op.op_id
            feats[i, 0] = float(op.is_done)
            feats[i, 1] = float(op.is_ready)
            feats[i, 2] = float(op.is_processing)
            feats[i, 3] = float(op.is_assembly)
            feats[i, 4] = float(op.buffer_waiting)
            # remaining_pred_count
            rem_pred = sum(1 for pid in op.predecessors if not env.operations[pid].is_done)
            feats[i, 5] = rem_pred / max_pred
            # remaining_comp_count
            rem_comp = sum(1 for cid in op.component_last_ops if not env.operations[cid].is_done)
            feats[i, 6] = rem_comp / max_comp
            # stage / product
            feats[i, 7] = op.stage_id / max(self.inst.num_stages - 1, 1)
            feats[i, 8] = op.product_id / max(self.inst.num_products - 1, 1)
            # CLB
            clb = env.compute_product_clb(op.product_id)
            feats[i, 9] = clb / self.max_due if self.max_due > 0 else 0.0
            # due_date / weight
            prod = self.inst.products[op.product_id]
            feats[i, 10] = prod.due_date / self.max_due if self.max_due > 0 else 0.0
            feats[i, 11] = prod.weight / self.max_weight if self.max_weight > 0 else 0.0
            # slack & tardiness_est
            slack = prod.due_date - clb
            feats[i, 12] = np.tanh(slack / self.max_due) if self.max_due > 0 else 0.0
            feats[i, 13] = max(0.0, clb - prod.due_date) / self.max_due if self.max_due > 0 else 0.0

        return torch.tensor(feats)

    def _build_machine_features(self, env: "FFSASchedulingEnv") -> torch.Tensor:
        """Machine Node Feature (PPT Slide 10): 7차원"""
        num_m = env.num_machines
        feats = np.zeros((num_m, 7), dtype=np.float32)

        for ms in env.machine_states.values():
            i = ms.machine_id
            feats[i, 0] = ms.stage_id / max(self.inst.num_stages - 1, 1)
            feats[i, 1] = float(ms.is_idle)
            feats[i, 2] = ms.remaining_time / self.max_proc if self.max_proc > 0 else 0.0
            # available_time
            avail = env.current_time + ms.remaining_time
            feats[i, 3] = avail / self.max_due if self.max_due > 0 else 0.0
            # last_product_id
            feats[i, 4] = (ms.last_product / max(self.inst.num_products - 1, 1)
                           if ms.last_product is not None else -1.0)
            # utilization
            feats[i, 5] = (ms.total_busy_time / env.current_time
                           if env.current_time > 0 else 0.0)
            # buffer_occupancy
            buf = env.buffers.get(ms.stage_id)
            feats[i, 6] = buf.occupancy if buf else 0.0

        return torch.tensor(feats)

    def _build_precedence_edges(self, env) -> Tuple[List[int], List[int]]:
        """(op, precedence, op): job 내 선후행"""
        src, dst = [], []
        for op in env.operations.values():
            for pred_id in op.predecessors:
                src.append(pred_id)
                dst.append(op.op_id)
        return src, dst

    def _build_assembly_edges(self, env) -> Tuple[List[int], List[int]]:
        """(op, assembly_dep, op): component → 조립"""
        src, dst = [], []
        for op in env.operations.values():
            for comp_op_id in op.component_last_ops:
                src.append(comp_op_id)
                dst.append(op.op_id)
        return src, dst

    def _build_candidate_edges(self, env) -> Tuple[List[int], List[int], List[List[float]]]:
        """(op, candidate, machine) with edge_attr=[pijk_norm, siijk_norm]"""
        src, dst, attrs = [], [], []
        for op in env.operations.values():
            if op.is_done or op.is_processing:
                continue
            for mid in self.inst.machines_by_stage.get(op.stage_id, []):
                m_data = self.inst.machines[mid]
                if op.product_id not in m_data.compatible_products:
                    continue
                # Processing time
                pt = self.inst.processing_times.get((op.job_id, op.stage_id, mid), 0.0)
                pt_norm = pt / self.max_proc if self.max_proc > 0 else 0.0
                # Setup time
                ms = env.machine_states[mid]
                if ms.last_product is not None and ms.last_product != op.product_id:
                    st = self.inst.setup_times.get(
                        (ms.last_product, op.product_id, op.stage_id, mid), 0.0
                    )
                else:
                    st = 0.0
                st_norm = st / self.max_setup if self.max_setup > 0 else 0.0

                src.append(op.op_id)
                dst.append(mid)
                attrs.append([pt_norm, st_norm])
        return src, dst, attrs


# ──────────────────────────────────────────────────────────
# 환경
# ──────────────────────────────────────────────────────────

class FFSASchedulingEnv(gym.Env):
    """
    Flexible Flow Shop with Assembly 스케줄링 환경

    PPT Slide 9: DES 루프 구조
      유효 쌍 존재 → policy가 하나 선택 → 배정 → 시간 진행 없이 반복
      유효 쌍 없음 → DES 다음 이벤트까지 진행 → 재판단

    Reward: rt = −α · Δ Σ wp · Tp_est  (PPT Slide 11)
    """
    metadata = {"render_modes": []}

    def __init__(self, config: InstanceConfig):
        super().__init__()
        self.config = config
        self.instance = generate_instance(config)
        self.graph_builder = GraphBuilder(self.instance)

        # 보상 하이퍼파라미터 (PPT Slide 11)
        self.alpha = 1.0
        self.beta = 0.05    # op 완료 보너스
        self.gamma_pen = 0.01   # idle machine 패널티
        self.eta = 0.01     # buffer congestion 패널티
        self.use_completion_bonus = False
        self.use_idle_penalty = False
        self.use_buffer_penalty = False

        # Operation/Machine/Buffer 상태 (reset에서 초기화)
        self.operations: Dict[int, OperationState] = {}
        self.machine_states: Dict[int, MachineState] = {}
        self.buffers: Dict[int, BufferState] = {}
        self.num_operations = 0
        self.num_machines = self.instance.num_machines
        self.current_time = 0.0
        self.prev_est_tardiness = 0.0
        self.completed_ops_last_event = 0

        # 매핑 테이블
        self.job_ops: Dict[int, List[int]] = {}            # job_id → [op_id] (순서)
        self.op_to_job_stage: Dict[int, Tuple[int, int]] = {}  # op_id → (job_id, stage_id)
        self.job_stage_to_op: Dict[Tuple[int, int], int] = {}  # (job_id, stage_id) → op_id

        # Action / Observation spaces (유연하게 정의, 실제는 그래프 기반)
        self._max_actions = 500  # 넉넉한 상한
        self.action_space = spaces.Discrete(self._max_actions)
        self.observation_space = spaces.Dict({
            "dummy": spaces.Box(0, 1, (1,), dtype=np.float32)
        })

        # [Fix #3] reset() 전 step() 호출 방어: 명시적 초기화
        self._current_action_pairs: List[Tuple[int, int]] = []
        # [Fix Critical] 데드락 감지 플래그
        self._deadlock_detected: bool = False

    # ──────────────────────────────────────────────────────
    # Reset
    # ──────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_time = 0.0
        self.completed_ops_last_event = 0
        self._deadlock_detected = False
        self._init_operations()
        self._init_machines()
        self._init_buffers()
        self._load_initial_jobs()
        self.update_ready_operations()
        self.prev_est_tardiness = self.compute_weighted_estimated_tardiness()
        return self._get_obs(), {}

    def _init_operations(self):
        """모든 (job, stage) 조합에서 Operation 생성"""
        self.operations = {}
        self.job_ops = {}
        self.op_to_job_stage = {}
        self.job_stage_to_op = {}
        op_id = 0

        for job in self.instance.jobs.values():
            self.job_ops[job.job_id] = []
            prev_op_id = None

            for stage_id in job.route:
                is_asm = (stage_id == job.assembly_stage) if job.assembly_stage is not None else False
                predecessors = [prev_op_id] if prev_op_id is not None else []

                self.operations[op_id] = OperationState(
                    op_id=op_id,
                    job_id=job.job_id,
                    product_id=job.product_id,
                    stage_id=stage_id,
                    is_assembly=is_asm,
                    predecessors=predecessors,
                )
                self.job_ops[job.job_id].append(op_id)
                self.op_to_job_stage[op_id] = (job.job_id, stage_id)
                self.job_stage_to_op[(job.job_id, stage_id)] = op_id
                prev_op_id = op_id
                op_id += 1

        self.num_operations = op_id

        # 조립 의존성: assembly op의 component_last_ops 설정
        for job in self.instance.jobs.values():
            if job.component_jobs and job.route:
                asm_stage = job.assembly_stage  # j_i^asm: job i의 조립 공정 (PPT 집합/인덱스)
                asm_op_id = self.job_stage_to_op.get((job.job_id, asm_stage))
                if asm_op_id is not None:
                    asm_op = self.operations[asm_op_id]
                    for comp_jid in job.component_jobs:
                        comp_ops = self.job_ops.get(comp_jid, [])
                        if comp_ops:
                            asm_op.component_last_ops.append(comp_ops[-1])

    def _init_machines(self):
        self.machine_states = {}
        for m in self.instance.machines.values():
            self.machine_states[m.machine_id] = MachineState(
                machine_id=m.machine_id,
                stage_id=m.stage_id,
                compatible_products=set(m.compatible_products),
            )

    def _init_buffers(self):
        self.buffers = {}
        for sid in range(self.instance.num_stages):
            cap = self.instance.buffer_capacities.get(sid, -1)
            self.buffers[sid] = BufferState(stage_id=sid, capacity=cap)

    def _load_initial_jobs(self):
        """컴포넌트 job의 첫 op을 해당 stage buffer에 투입 (t=0).

        [Fix #1] Assembly final job은 component job들이 모두 완료된 시점에
        _check_and_enqueue_assembly_jobs()에서 buffer에 진입시킨다.
        t=0에 미리 진입시키면 유한 버퍼 capacity를 불필요하게 소모하고
        buffer_occupancy feature를 왜곡한다.
        """
        for job in self.instance.jobs.values():
            if not job.route:
                continue
            # Assembly final job: component 완료 후 buffer 진입
            if job.component_jobs:
                continue
            first_stage = job.route[0]
            self.buffers[first_stage].push(job.job_id)
            first_op_id = self.job_ops[job.job_id][0]
            self.operations[first_op_id].buffer_waiting = True

    # ──────────────────────────────────────────────────────
    # Step
    # ──────────────────────────────────────────────────────

    def step(self, action: int):
        action_pairs = self._current_action_pairs
        if action >= len(action_pairs):
            # 안전장치 (정상 작동 시 발생하지 않음)
            return self._get_obs(), -1.0, False, False, {}

        op_id, machine_id = action_pairs[action]

        self._deadlock_detected = False  # 매 step 초기화

        # Dispatch
        self._dispatch(op_id, machine_id)

        # DES 루프 (PPT Slide 9)
        self.update_ready_operations()
        if not self._has_valid_action():
            self._advance_until_next_decision_point()

        reward = self._reward_fn()
        done = self._check_done()
        # [Fix Critical] 데드락 발생 시 truncated=True로 에피소드 강제 종료
        truncated = self._deadlock_detected

        obs = self._get_obs()
        info = {
            "time": self.current_time,
            "completed_ops": sum(1 for op in self.operations.values() if op.is_done),
            "total_ops": self.num_operations,
            "weighted_tardiness_est": self.prev_est_tardiness,
            "deadlock": self._deadlock_detected,
        }
        return obs, reward, done, truncated, info

    # ──────────────────────────────────────────────────────
    # Dispatch (수정사항 #2 반영)
    # ──────────────────────────────────────────────────────

    def _dispatch(self, op_id: int, machine_id: int):
        """Operation을 machine에 배정"""
        op = self.operations[op_id]
        ms = self.machine_states[machine_id]

        # ★ Buffer에서 제거
        buffer = self.buffers[op.stage_id]
        buffer.remove(op.job_id)
        op.buffer_waiting = False

        # ★ Machine 할당 기록
        op.machine_id = machine_id

        # Setup + Processing time
        setup = self._get_setup_time(ms.last_product, op.product_id, op.stage_id, machine_id)
        proc = self.instance.processing_times.get((op.job_id, op.stage_id, machine_id), 0.0)
        total = setup + proc

        # Operation 상태
        op.is_processing = True
        op.is_ready = False
        op.start_time = self.current_time
        op.completion_time = self.current_time + total

        # Machine 상태
        ms.is_idle = False
        ms.current_op = op_id
        ms.remaining_time = total

    def _get_setup_time(self, last_prod, curr_prod, stage_id, machine_id) -> float:
        """Setup time 조회 (동일 제품이면 0)"""
        if last_prod is None or last_prod == curr_prod:
            return 0.0
        return self.instance.setup_times.get(
            (last_prod, curr_prod, stage_id, machine_id), 0.0
        )

    # ──────────────────────────────────────────────────────
    # DES 전진 (수정사항 #3 반영)
    # ──────────────────────────────────────────────────────

    def _advance_until_next_decision_point(self):
        """유효한 action이 나올 때까지 시뮬레이션 전진.

        [Fix Critical] 처리 중인 op이 없고 기계가 blocked 상태이면
        더 이상 진행이 불가능한 데드락이다. _deadlock_detected=True로 표시 후 종료.

        [Fix #2] is_blocked인 기계는 실제 작업을 하지 않으므로
        total_busy_time 누적에서 제외한다.
        """
        while True:
            self.update_ready_operations()
            if self._has_valid_action():
                break

            processing = [op for op in self.operations.values() if op.is_processing]

            if not processing:
                # 처리 중인 op이 없음 → 종료 또는 데드락 판정
                blocked = [ms for ms in self.machine_states.values() if ms.is_blocked]
                if blocked:
                    self._deadlock_detected = True
                break

            # 다음 이벤트 시점
            next_time = min(op.completion_time for op in processing)
            dt = next_time - self.current_time

            # [Fix #2] blocked 기계는 생산적 작업을 하지 않으므로 제외
            for ms in self.machine_states.values():
                if not ms.is_idle and not ms.is_blocked:
                    ms.total_busy_time += dt

            self.current_time = next_time
            self._complete_operations_at(next_time)
            self._move_completed_to_next_buffer()
            # [Fix #1 연동] component 완료 시점에 assembly final job buffer 진입 체크
            self._check_and_enqueue_assembly_jobs()
            self._update_machine_remaining()

    def _complete_operations_at(self, t: float):
        """시점 t에 완료되는 operation 처리"""
        self.completed_ops_last_event = 0
        for op in self.operations.values():
            if op.is_processing and op.completion_time is not None and op.completion_time <= t + 1e-9:
                op.is_processing = False
                op.is_done = True
                self.completed_ops_last_event += 1

                # Machine 해제
                if op.machine_id is not None:
                    ms = self.machine_states[op.machine_id]
                    ms.is_idle = True
                    ms.current_op = None
                    ms.remaining_time = 0.0
                    ms.last_product = op.product_id

    def _move_completed_to_next_buffer(self):
        """완료된 op의 job을 다음 stage buffer로 이동"""
        for op in self.operations.values():
            if not op.is_done:
                continue
            # 이미 처리된 op은 건너뜀 (다음 buffer에 이미 들어감)
            job_id = op.job_id
            job = self.instance.jobs[job_id]
            op_list = self.job_ops[job_id]
            op_idx = op_list.index(op.op_id)

            # 마지막 operation이면 job 완료
            if op_idx == len(op_list) - 1:
                continue

            next_op_id = op_list[op_idx + 1]
            next_op = self.operations[next_op_id]

            # 이미 buffer에 있거나 처리 중이면 건너뜀
            if next_op.buffer_waiting or next_op.is_processing or next_op.is_done:
                continue

            next_stage = next_op.stage_id
            buffer = self.buffers[next_stage]

            if buffer.has_space():
                if job_id not in buffer.queue:
                    buffer.push(job_id)
                next_op.buffer_waiting = True
            else:
                # ★ 단순 blocking: machine을 차단 상태로 유지
                # current_op이 None인 경우에만 blocking (다른 job이 이미 점유 중이면 skip)
                if op.machine_id is not None:
                    ms = self.machine_states[op.machine_id]
                    if ms.current_op is None and not ms.is_blocked:
                        ms.is_blocked = True
                        ms.is_idle = False
                        ms.blocked_job = job_id

    def _check_and_enqueue_assembly_jobs(self):
        """[Fix #1] Assembly final job의 buffer 진입 조건을 매 이벤트 시점마다 확인.

        component job들의 마지막 op이 모두 완료된 순간 assembly buffer에 진입시킨다.
        유한 버퍼인 경우 공간이 생길 때까지 대기하며, 다음 호출 시점에 재시도한다.
        """
        for job in self.instance.jobs.values():
            if not job.component_jobs or not job.route:
                continue

            first_op_id = self.job_ops[job.job_id][0]
            first_op = self.operations[first_op_id]

            # 이미 buffer에 있거나 처리 중이거나 완료된 경우 skip
            if first_op.buffer_waiting or first_op.is_processing or first_op.is_done:
                continue

            # 모든 component의 마지막 op 완료 여부 확인
            all_comp_done = all(
                self.operations[cid].is_done
                for cid in first_op.component_last_ops
            )
            if not all_comp_done:
                continue

            # 조립 stage buffer에 공간이 있으면 진입
            # PPT 제약식 10: WIP_asm 은 component 수 기준으로 카운트
            asm_stage = job.route[0]
            buf = self.buffers[asm_stage]
            n_comp = len(first_op.component_last_ops)
            can_enter = (buf.capacity < 0) or (len(buf.queue) + n_comp <= buf.capacity)
            if can_enter:
                if job.job_id not in buf.queue:
                    buf.push(job.job_id)
                first_op.buffer_waiting = True
            # 버퍼가 꽉 찬 경우: 공간 생길 때까지 대기 (다음 호출 시 재시도)

    def _update_machine_remaining(self):
        """Machine remaining_time 갱신"""
        for ms in self.machine_states.values():
            if ms.current_op is not None and not ms.is_blocked:
                op = self.operations[ms.current_op]
                if op.completion_time is not None:
                    ms.remaining_time = max(0.0, op.completion_time - self.current_time)
            elif ms.is_blocked:
                ms.remaining_time = 0.0

    def _try_unblock_machines(self):
        """Blocked machine 해제 시도"""
        for ms in self.machine_states.values():
            if not ms.is_blocked or ms.blocked_job is None:
                continue
            # blocked job의 다음 op의 buffer에 공간이 생겼는지 확인
            job_id = ms.blocked_job
            job = self.instance.jobs[job_id]
            op_list = self.job_ops[job_id]

            # 현재 완료된 op 찾기
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

    # ──────────────────────────────────────────────────────
    # Ready 판정 (PPT C1~C5)
    # ──────────────────────────────────────────────────────

    def update_ready_operations(self):
        self._try_unblock_machines()  # blocking 해제 시도

        for op in self.operations.values():
            if op.is_done or op.is_processing:
                op.is_ready = False
                continue

            # C4: 모든 선행 op 완료
            pred_done = all(self.operations[pid].is_done for pid in op.predecessors)

            # C5: assembly — 모든 component의 마지막 op 완료
            comp_done = all(self.operations[cid].is_done for cid in op.component_last_ops)

            # Buffer 대기 여부
            in_buffer = op.buffer_waiting

            op.is_ready = pred_done and comp_done and in_buffer

    # ──────────────────────────────────────────────────────
    # Action 유효성 (PPT C1~C10 대응)
    # ──────────────────────────────────────────────────────

    def _is_valid_action(self, op_id: int, machine_id: int) -> bool:
        op = self.operations[op_id]
        ms = self.machine_states[machine_id]

        if op.is_done:                     return False  # C1
        if not op.is_ready:                return False  # C4+C5+buffer
        if op.is_processing:               return False
        if not ms.is_idle:                 return False  # C6,C7
        if ms.is_blocked:                  return False
        if ms.stage_id != op.stage_id:     return False  # C2
        if op.product_id not in ms.compatible_products:
            return False                                 # C2

        # C9/C10: downstream buffer 확인 (현재 dispatch 시점에서는 체크 안 함)
        # → _move_completed_to_next_buffer에서 blocking으로 처리

        return True

    def _has_valid_action(self) -> bool:
        for op in self.operations.values():
            if not op.is_ready:
                continue
            for mid in self.instance.machines_by_stage.get(op.stage_id, []):
                if self._is_valid_action(op.op_id, mid):
                    return True
        return False

    def _get_valid_action_pairs(self) -> List[Tuple[int, int]]:
        pairs = []
        for op in self.operations.values():
            if not op.is_ready:
                continue
            for mid in self.instance.machines_by_stage.get(op.stage_id, []):
                if self._is_valid_action(op.op_id, mid):
                    pairs.append((op.op_id, mid))
        return pairs

    # ──────────────────────────────────────────────────────
    # Observation
    # ──────────────────────────────────────────────────────

    def _get_obs(self) -> dict:
        graph = self.graph_builder.build(self)
        action_pairs = self._get_valid_action_pairs()
        self._current_action_pairs = action_pairs  # step()에서 사용

        mask = np.ones(len(action_pairs), dtype=np.float32) if action_pairs else np.array([], dtype=np.float32)

        # Assembly map: asm_op_idx → [comp_last_op_indices]
        assembly_map = {}
        for op in self.operations.values():
            if op.component_last_ops:
                assembly_map[op.op_id] = list(op.component_last_ops)

        # Precedence info: prev/next maps + candidate machines
        prev_map = {}
        next_map = {}
        candidate_machines = {}

        for op in self.operations.values():
            oid = op.op_id
            # prev: job 내 직전 op
            prev_map[oid] = op.predecessors[0] if op.predecessors else None
            # next: job 내 다음 op
            job_ops = self.job_ops[op.job_id]
            idx = job_ops.index(oid)
            next_map[oid] = job_ops[idx + 1] if idx < len(job_ops) - 1 else None
            # candidate machines
            candidate_machines[oid] = [
                mid for mid in self.instance.machines_by_stage.get(op.stage_id, [])
                if op.product_id in self.instance.machines[mid].compatible_products
            ]

        return {
            "graph": graph,
            "action_pairs": action_pairs,
            "action_mask": mask,
            "assembly_map": assembly_map,
            "precedence_info": {
                "prev_map": prev_map,
                "next_map": next_map,
                "candidate_machines": candidate_machines,
            },
        }

    # ──────────────────────────────────────────────────────
    # CLB 계산 (PPT Slide 10: CLB_norm, slack_norm)
    # ──────────────────────────────────────────────────────

    def compute_job_clb(self, job_id: int) -> float:
        """Job의 Completion Lower Bound"""
        job = self.instance.jobs[job_id]

        # 조립 job이면 component CLB 반영
        t = self.current_time
        if job.component_jobs:
            for comp_id in job.component_jobs:
                t = max(t, self.compute_job_clb(comp_id))

        for stage_id in job.route:
            op_id = self.job_stage_to_op.get((job_id, stage_id))
            if op_id is None:
                continue
            op = self.operations[op_id]

            if op.is_done:
                continue
            if op.is_processing and op.completion_time is not None:
                t = max(t, op.completion_time)
                continue
            # 미시작 → 최소 처리시간 + 최소 setup time 추가 (B안: optimistic estimate)
            compatible = [
                mid for mid in self.instance.machines_by_stage.get(stage_id, [])
                if job.product_id in self.instance.machines[mid].compatible_products
            ]
            if compatible:
                min_proc = min(
                    self.instance.processing_times.get((job_id, stage_id, mid), float('inf'))
                    for mid in compatible
                )
                if min_proc < float('inf'):
                    min_setup = min(
                        (self.instance.setup_times.get((prev_p, job.product_id, stage_id, mid), 0.0)
                         for prev_p in range(self.instance.num_products)
                         if prev_p != job.product_id
                         for mid in compatible),
                        default=0.0
                    ) if self.instance.setup_times else 0.0
                    t += min_proc + min_setup
        return t

    def compute_product_clb(self, product_id: int) -> float:
        """제품의 CLB = 최종 job의 CLB"""
        prod = self.instance.products[product_id]
        return self.compute_job_clb(prod.final_job_id)

    def compute_weighted_estimated_tardiness(self) -> float:
        """Σ wp · Tp_est (PPT Slide 11)"""
        total = 0.0
        for prod in self.instance.products.values():
            clb = self.compute_product_clb(prod.product_id)
            tp_est = max(0.0, clb - prod.due_date)
            total += prod.weight * tp_est
        return total

    # ──────────────────────────────────────────────────────
    # Reward (PPT Slide 11)
    # ──────────────────────────────────────────────────────

    def _reward_fn(self) -> float:
        """rt = −α · Δ Σ wp · Tp_est"""
        current_est = self.compute_weighted_estimated_tardiness()
        delta = current_est - self.prev_est_tardiness
        reward = -self.alpha * delta

        if self.use_completion_bonus:
            reward += self.beta * self.completed_ops_last_event

        if self.use_idle_penalty:
            idle_count = sum(1 for ms in self.machine_states.values() if ms.is_idle)
            reward -= self.gamma_pen * idle_count

        if self.use_buffer_penalty:
            congestion = sum(buf.occupancy for buf in self.buffers.values())
            reward -= self.eta * congestion

        self.prev_est_tardiness = current_est
        return reward

    # ──────────────────────────────────────────────────────
    # Done 체크 (수정사항 #4)
    # ──────────────────────────────────────────────────────

    def _check_done(self) -> bool:
        """모든 operation이 완료되면 종료"""
        return all(op.is_done for op in self.operations.values())

    # ──────────────────────────────────────────────────────
    # 유틸리티
    # ──────────────────────────────────────────────────────

    def get_actual_weighted_tardiness(self) -> float:
        """실제 weighted tardiness (에피소드 종료 후 평가용)"""
        total = 0.0
        for prod in self.instance.products.values():
            final_job = self.instance.jobs[prod.final_job_id]
            last_ops = self.job_ops[final_job.job_id]
            if last_ops:
                last_op = self.operations[last_ops[-1]]
                if last_op.is_done and last_op.completion_time is not None:
                    tp = max(0.0, last_op.completion_time - prod.due_date)
                    total += prod.weight * tp
        return total

    def get_makespan(self) -> float:
        """Makespan (마지막 operation 완료 시점)"""
        max_t = 0.0
        for op in self.operations.values():
            if op.is_done and op.completion_time is not None:
                max_t = max(max_t, op.completion_time)
        return max_t
