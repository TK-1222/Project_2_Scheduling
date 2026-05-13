"""
FFSA 스케줄링 강화학습 환경 (디스패칭 룰 버전)
================================================
ffsa_env.py 와 동일하나 _get_valid_action_pairs()를
FIFO / EDD / MWKR 디스패칭 룰 기반 후보 생성으로 대체.

각 룰이 ready op 중 1개를 선택 → 최소 처리시간 기계와 페어링
→ 중복 제거 후 최대 3개 후보를 GNN에 전달.
룰이 후보를 못 찾으면 전체 유효 액션 fallback.
조립 actions는 항상 포함.

Reward: r = -Δ(실제 완료된 주문의 가중 지연)
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
    compatible_products: Set[int] = field(default_factory=set)
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
    """이종 그래프 생성 (PPT Slide 11: State)"""

    def __init__(self, instance: FFSAInstance):
        self.inst = instance
        self.max_proc = max(instance.processing_times.values()) if instance.processing_times else 1.0
        self.max_setup = max(instance.setup_times.values()) if instance.setup_times else 1.0
        self.max_due = max(
            (j.due_date for j in instance.jobs.values() if j.due_date > 0), default=1.0
        )
        self.max_weight = max(p.weight for p in instance.products.values()) if instance.products else 1.0

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
        """Operation Node Feature: 10차원
        0: is_done, 1: is_ready, 2: is_processing, 3: is_assembly, 4: buffer_waiting
        5: rem_pred_count, 6: stage_norm, 7: product_norm
        8: due_date_norm (final job op만), 9: weight_norm
        """
        active_ops = [op for op in env.operations.values() if op.active]
        num_ops = len(active_ops)
        if num_ops == 0:
            return torch.zeros((0, 10), dtype=torch.float32)

        feats = np.zeros((num_ops, 10), dtype=np.float32)
        max_pred = max((len(op.predecessors) for op in active_ops), default=1) or 1

        for idx, op in enumerate(active_ops):
            feats[idx, 0] = float(op.is_done)
            feats[idx, 1] = float(op.is_ready)
            feats[idx, 2] = float(op.is_processing)
            feats[idx, 3] = float(op.is_assembly)
            feats[idx, 4] = float(op.buffer_waiting)
            rem_pred = sum(1 for pid in op.predecessors
                          if env.operations[pid].active and not env.operations[pid].is_done)
            feats[idx, 5] = rem_pred / max_pred
            feats[idx, 6] = op.stage_id / max(self.inst.num_stages - 1, 1)
            feats[idx, 7] = op.product_id / max(self.inst.num_products - 1, 1)
            job = self.inst.jobs[op.job_id]
            feats[idx, 8] = job.due_date / self.max_due if self.max_due > 0 else 0.0
            feats[idx, 9] = self.inst.products[op.product_id].weight / self.max_weight if self.max_weight > 0 else 0.0

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
                if op.product_id not in m_data.compatible_products:
                    continue
                pt = self.inst.processing_times.get((op.job_id, op.stage_id, mid), 0.0)
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
    Reward: r = -Δ(실제 완료 주문의 가중 지연합)
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

        # 조립 버퍼: product_id → {comp_type_idx: [job_id, ...]}
        self.assembly_pool: Dict[int, Dict[int, List[int]]] = {}
        # 미활성 final job: product_id → [job_id, ...]
        self.inactive_final_jobs: Dict[int, List[int]] = {}

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
        self._prev_actual_wt: float = 0.0

        # 긴급주문 실시간 생성용
        self._urgent_rng: np.random.RandomState = np.random.RandomState()
        self._urgent_orders_remaining: int = 0
        self._next_urgent_arrival: float = float('inf')

    # ──────────────────────────────────────────────────────
    # Reset
    # ──────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_time = 0.0
        self._deadlock_detected = False
        self._prev_actual_wt = 0.0
        # 인스턴스를 매 에피소드 새로 생성 (정규주문 랜덤 변경)
        self.instance = generate_instance(self.config)
        self.graph_builder = GraphBuilder(self.instance)
        self._init_operations()
        self._init_machines()
        self._init_buffers()
        self._init_assembly_pool()
        self._load_initial_jobs()
        # 긴급주문 포아송 프로세스 초기화
        rng_seed = None if self.config.seed is None else self.config.seed + id(self) % 10000
        self._urgent_rng = np.random.RandomState(rng_seed)
        self._urgent_orders_remaining = self.config.num_urgent_orders
        self._next_urgent_arrival = self._sample_next_urgent_arrival()
        self.update_ready_operations()
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
            # 활성 조건: t=0 도착이고, final job이 아닌 경우 (component 또는 no-assembly job)
            is_active = job.arrival_time == 0.0 and not job.is_final_job

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
                compatible_products=set(m.compatible_products),
            )

    def _init_buffers(self):
        self.buffers = {}
        for sid in range(self.instance.num_stages):
            cap = self.instance.buffer_capacities.get(sid, -1)
            self.buffers[sid] = BufferState(stage_id=sid, capacity=cap)

    def _init_assembly_pool(self):
        """조립 버퍼 pool 및 정규주문 final job 초기화 (긴급주문은 실시간 추가)"""
        self.assembly_pool = {p: {} for p in self.instance.products}
        self.inactive_final_jobs = {p: [] for p in self.instance.products}
        if not self.config.use_assembly:
            return
        for order in self.instance.orders.values():
            for fid in order.final_job_ids:
                self.inactive_final_jobs[order.product_id].append(fid)

    def _load_initial_jobs(self):
        """t=0 도착 job을 stage 0 버퍼에 투입 (final job 제외: 조립 dispatch 전까지 비활성)"""
        for job in self.instance.jobs.values():
            if job.arrival_time != 0.0 or not job.route or job.is_final_job:
                continue
            first_stage = job.route[0]
            self.buffers[first_stage].push(job.job_id)
            first_op_id = self.job_ops[job.job_id][0]
            self.operations[first_op_id].buffer_waiting = True

    def _sample_next_urgent_arrival(self) -> float:
        """다음 긴급주문 도착 시점 샘플링 (포아송 프로세스)"""
        if self._urgent_orders_remaining <= 0:
            return float('inf')
        inter = self._urgent_rng.exponential(self.config.urgent_inter_arrival_mean)
        return self.current_time + inter

    def _arrive_urgent_order(self):
        """긴급주문 실시간 생성: 구성 결정 → job/op 생성 → 버퍼 투입"""
        cfg = self.config
        rng = self._urgent_rng
        t = self.current_time

        p = int(rng.randint(cfg.num_products))
        qty = int(rng.randint(cfg.urgent_quantity_range[0], cfg.urgent_quantity_range[1] + 1))
        due = t + float(rng.uniform(*cfg.urgent_due_date_offset_range))

        new_oid = max(self.instance.orders.keys(), default=-1) + 1
        order = OrderData(
            order_id=new_oid, product_id=p, quantity=qty,
            due_date=due, arrival_time=t, is_urgent=True,
        )
        self.instance.orders[new_oid] = order

        stages = list(range(cfg.num_stages))
        pre_asm = stages[:cfg.assembly_stage_idx]
        post_asm = stages[cfg.assembly_stage_idx:]

        new_jid = max(self.instance.jobs.keys(), default=-1) + 1
        new_op_id = max(self.operations.keys(), default=-1) + 1

        for unit_idx in range(qty):
            if cfg.use_assembly:
                # component jobs
                num_comp = self.instance.products[p].num_components
                for comp_type in range(num_comp):
                    job = JobData(
                        job_id=new_jid, product_id=p, order_id=new_oid,
                        arrival_time=t, route=list(pre_asm),
                        is_component=True, component_type_idx=comp_type,
                        order_unit_idx=unit_idx, due_date=due,
                    )
                    self.instance.jobs[new_jid] = job
                    order.component_job_ids.append(new_jid)
                    self.job_ops[new_jid] = []
                    # 처리시간 생성
                    for sid in pre_asm:
                        for mid in self.instance.machines_by_stage[sid]:
                            if p in self.instance.machines[mid].compatible_products:
                                self.instance.processing_times[(new_jid, sid, mid)] = float(
                                    rng.uniform(*cfg.processing_time_range)
                                )
                    # operations 생성 및 버퍼 투입
                    prev_op = None
                    for sid in pre_asm:
                        op = OperationState(
                            op_id=new_op_id, job_id=new_jid, product_id=p,
                            stage_id=sid, predecessors=[prev_op] if prev_op is not None else [],
                            active=True,
                        )
                        self.operations[new_op_id] = op
                        self.job_ops[new_jid].append(new_op_id)
                        self.op_to_job_stage[new_op_id] = (new_jid, sid)
                        self.job_stage_to_op[(new_jid, sid)] = new_op_id
                        prev_op = new_op_id
                        new_op_id += 1
                    # 첫 번째 op 버퍼에 투입
                    first_op = self.job_ops[new_jid][0]
                    self.buffers[pre_asm[0]].push(new_jid)
                    self.operations[first_op].buffer_waiting = True
                    new_jid += 1

                # final job
                fjob = JobData(
                    job_id=new_jid, product_id=p, order_id=new_oid,
                    arrival_time=t, route=list(post_asm),
                    is_final_job=True, assembly_stage=cfg.assembly_stage_idx,
                    order_unit_idx=unit_idx, due_date=due,
                )
                self.instance.jobs[new_jid] = fjob
                order.final_job_ids.append(new_jid)
                self.job_ops[new_jid] = []
                for sid in post_asm:
                    for mid in self.instance.machines_by_stage[sid]:
                        if p in self.instance.machines[mid].compatible_products:
                            self.instance.processing_times[(new_jid, sid, mid)] = float(
                                rng.uniform(*cfg.processing_time_range)
                            )
                prev_op = None
                for sid in post_asm:
                    is_asm = (sid == cfg.assembly_stage_idx)
                    op = OperationState(
                        op_id=new_op_id, job_id=new_jid, product_id=p,
                        stage_id=sid, is_assembly=is_asm,
                        predecessors=[prev_op] if prev_op is not None else [],
                        active=False,  # 조립 dispatch까지 비활성
                    )
                    self.operations[new_op_id] = op
                    self.job_ops[new_jid].append(new_op_id)
                    self.op_to_job_stage[new_op_id] = (new_jid, sid)
                    self.job_stage_to_op[(new_jid, sid)] = new_op_id
                    prev_op = new_op_id
                    new_op_id += 1
                self.inactive_final_jobs[p].append(new_jid)
                new_jid += 1
            else:
                job = JobData(
                    job_id=new_jid, product_id=p, order_id=new_oid,
                    arrival_time=t, route=list(stages),
                    is_final_job=False, order_unit_idx=unit_idx, due_date=due,
                )
                self.instance.jobs[new_jid] = job
                order.final_job_ids.append(new_jid)
                self.job_ops[new_jid] = []
                for sid in stages:
                    for mid in self.instance.machines_by_stage[sid]:
                        if p in self.instance.machines[mid].compatible_products:
                            self.instance.processing_times[(new_jid, sid, mid)] = float(
                                rng.uniform(*cfg.processing_time_range)
                            )
                prev_op = None
                for sid in stages:
                    op = OperationState(
                        op_id=new_op_id, job_id=new_jid, product_id=p,
                        stage_id=sid, predecessors=[prev_op] if prev_op is not None else [],
                        active=True,
                    )
                    self.operations[new_op_id] = op
                    self.job_ops[new_jid].append(new_op_id)
                    self.op_to_job_stage[new_op_id] = (new_jid, sid)
                    self.job_stage_to_op[(new_jid, sid)] = new_op_id
                    prev_op = new_op_id
                    new_op_id += 1
                first_op = self.job_ops[new_jid][0]
                self.buffers[stages[0]].push(new_jid)
                self.operations[first_op].buffer_waiting = True
                new_jid += 1

        # GraphBuilder max_due 갱신
        if due > self.graph_builder.max_due:
            self.graph_builder.max_due = due

        self.num_operations = new_op_id
        self._urgent_orders_remaining -= 1
        self._next_urgent_arrival = self._sample_next_urgent_arrival()

    # ──────────────────────────────────────────────────────
    # Step
    # ──────────────────────────────────────────────────────

    def step(self, action_idx: int):
        actions = self._current_actions
        if action_idx >= len(actions):
            return self._get_obs(), -1.0, False, False, {}

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
        op.machine_id = machine_id

        setup = self._get_setup_time(ms.last_product, op.product_id, op.stage_id, machine_id)
        proc = self.instance.processing_times.get((op.job_id, op.stage_id, machine_id), 0.0)
        total = setup + proc

        op.is_processing = True
        op.is_ready = False
        op.start_time = self.current_time
        op.completion_time = self.current_time + total

        ms.is_idle = False
        ms.current_op = op_id
        ms.remaining_time = total

    def _dispatch_assembly(self, comp_job_ids: Tuple[int, ...], machine_id: int):
        """조립 dispatch: 컴포넌트 소비 → final job 활성화 → 조립 op 시작"""
        product_id = self.instance.jobs[comp_job_ids[0]].product_id
        asm_stage = self.instance.config.assembly_stage_idx

        # pool에서 컴포넌트 소비 (가변 개수)
        for job_id in comp_job_ids:
            comp_type = self.instance.jobs[job_id].component_type_idx
            self.assembly_pool[product_id][comp_type].remove(job_id)

        # 미활성 final job 활성화
        final_job_id = self.inactive_final_jobs[product_id].pop(0)
        for op_id in self.job_ops[final_job_id]:
            self.operations[op_id].active = True

        # 조립 op 직접 dispatch
        asm_op_id = self.job_ops[final_job_id][0]
        asm_op = self.operations[asm_op_id]
        ms = self.machine_states[machine_id]

        asm_op.machine_id = machine_id
        setup = self._get_setup_time(ms.last_product, product_id, asm_stage, machine_id)
        proc = self.instance.processing_times.get((final_job_id, asm_stage, machine_id), 0.0)
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
            next_arrival = self._next_urgent_arrival

            if not processing and next_arrival == float('inf'):
                blocked = [ms for ms in self.machine_states.values() if ms.is_blocked]
                if blocked:
                    self._deadlock_detected = True
                break

            next_completion = min(
                (op.completion_time for op in processing), default=float('inf')
            )
            next_time = min(next_completion, next_arrival)
            dt = next_time - self.current_time

            for ms in self.machine_states.values():
                if not ms.is_idle and not ms.is_blocked:
                    ms.total_busy_time += dt

            self.current_time = next_time
            self._complete_operations_at(next_time)
            self._move_completed_to_next_buffer()
            self._update_machine_remaining()

            if self.current_time >= self._next_urgent_arrival - 1e-9:
                self._arrive_urgent_order()

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
                    prod_id = job.product_id
                    pool = self.assembly_pool[prod_id]
                    if comp_type not in pool:
                        pool[comp_type] = []
                    if job_id not in pool[comp_type]:
                        pool[comp_type].append(job_id)
                continue

            # 다음 op이 있으면 다음 stage 버퍼로 이동
            next_op_id = op_list[op_idx + 1]
            next_op = self.operations[next_op_id]

            if next_op.buffer_waiting or next_op.is_processing or next_op.is_done:
                continue

            next_stage = next_op.stage_id
            buffer = self.buffers[next_stage]

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

    # ──────────────────────────────────────────────────────
    # Ready 판정
    # ──────────────────────────────────────────────────────

    def update_ready_operations(self):
        self._try_unblock_machines()
        for op in self.operations.values():
            if not op.active or op.is_done or op.is_processing:
                op.is_ready = False
                continue
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
        if op.is_processing:                      return False
        if not ms.is_idle or ms.is_blocked:       return False
        if ms.stage_id != op.stage_id:            return False
        if op.product_id not in ms.compatible_products:
            return False
        return True

    def _get_valid_assembly_actions(self) -> List[AssemblyAction]:
        """조립 가능 (comp_A, comp_B, machine) 조합 반환"""
        actions = []
        asm_stage = self.instance.config.assembly_stage_idx

        for prod_id, type_pool in self.assembly_pool.items():
            # 제품별 컴포넌트 수 조회
            num_types = self.instance.products[prod_id].num_components
            # 컴포넌트가 1개 이하면 조립 불필요
            if num_types < 2:
                continue
            # 모든 타입이 pool에 ≥1개 존재해야 조립 가능
            if len(type_pool) < num_types:
                continue
            if not all(len(type_pool.get(t, [])) >= 1 for t in range(num_types)):
                continue
            # 미활성 final job이 없으면 조립 불가
            if not self.inactive_final_jobs.get(prod_id):
                continue

            # 호환 조립 기계
            asm_machines = [
                mid for mid in self.instance.machines_by_stage.get(asm_stage, [])
                if (prod_id in self.machine_states[mid].compatible_products
                    and self.machine_states[mid].is_idle
                    and not self.machine_states[mid].is_blocked)
            ]
            if not asm_machines:
                continue

            # 타입 0 × 타입 1 × ... × 타입 (num_types-1) 모든 조합
            from itertools import product as iterproduct
            type_lists = [type_pool[t] for t in range(num_types)]
            for combo in iterproduct(*type_lists):
                for mid in asm_machines:
                    actions.append((combo, mid))   # ((comp1, comp2, ...), machine_id)

        return actions

    # ──────────────────────────────────────────────────────
    # Dispatching Rules
    # ──────────────────────────────────────────────────────

    def _remaining_work(self, op: OperationState) -> float:
        """MWKR용: job의 미완료 op 최소 처리시간 합산"""
        total = 0.0
        for oid in self.job_ops[op.job_id]:
            o = self.operations[oid]
            if o.is_done or o.is_processing:
                continue
            times = [
                self.instance.processing_times.get((op.job_id, o.stage_id, mid), 0.0)
                for mid in self.instance.machines_by_stage.get(o.stage_id, [])
                if o.product_id in self.instance.machines[mid].compatible_products
            ]
            total += min(times) if times else 0.0
        return total

    def _best_machine_for_op(self, op: OperationState) -> Optional[int]:
        """유휴 호환 기계 중 처리시간 최소 기계 반환"""
        best_mid: Optional[int] = None
        best_t = float('inf')
        for mid in self.instance.machines_by_stage.get(op.stage_id, []):
            if not self._is_valid_regular_action(op.op_id, mid):
                continue
            t = self.instance.processing_times.get((op.job_id, op.stage_id, mid), float('inf'))
            if t < best_t:
                best_t = t
                best_mid = mid
        return best_mid

    def _apply_fifo(self) -> Optional[Action]:
        """FIFO: 가장 일찍 도착한 job의 ready op 선택"""
        ready = [
            op for op in self.operations.values()
            if op.active and op.is_ready and not op.is_processing and not op.is_assembly
        ]
        ready.sort(key=lambda op: self.instance.jobs[op.job_id].arrival_time)
        for op in ready:
            mid = self._best_machine_for_op(op)
            if mid is not None:
                return (op.op_id, mid)
        return None

    def _apply_edd(self) -> Optional[Action]:
        """EDD: 납기가 가장 빠른 job의 ready op 선택"""
        ready = [
            op for op in self.operations.values()
            if op.active and op.is_ready and not op.is_processing and not op.is_assembly
        ]
        ready.sort(key=lambda op: self.instance.jobs[op.job_id].due_date)
        for op in ready:
            mid = self._best_machine_for_op(op)
            if mid is not None:
                return (op.op_id, mid)
        return None

    def _apply_mwkr(self) -> Optional[Action]:
        """MWKR: 잔여 작업량이 가장 큰 job의 ready op 선택"""
        ready = [
            op for op in self.operations.values()
            if op.active and op.is_ready and not op.is_processing and not op.is_assembly
        ]
        ready.sort(key=lambda op: self._remaining_work(op), reverse=True)
        for op in ready:
            mid = self._best_machine_for_op(op)
            if mid is not None:
                return (op.op_id, mid)
        return None

    def _get_valid_action_pairs(self) -> List[Action]:
        # 각 디스패칭 룰이 후보 1개씩 생성, 중복 제거
        regular_candidates: List[Action] = []
        seen: set = set()

        for action in [self._apply_fifo(), self._apply_edd(), self._apply_mwkr()]:
            if action is not None and action not in seen:
                regular_candidates.append(action)
                seen.add(action)

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

        # 조립 actions 항상 포함
        return regular_candidates + self._get_valid_assembly_actions()

    def _has_valid_action(self) -> bool:
        for op in self.operations.values():
            if not op.active or not op.is_ready:
                continue
            for mid in self.instance.machines_by_stage.get(op.stage_id, []):
                if self._is_valid_regular_action(op.op_id, mid):
                    return True
        return len(self._get_valid_assembly_actions()) > 0

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
            candidate_machines[oid] = [
                mid for mid in self.instance.machines_by_stage.get(op.stage_id, [])
                if op.product_id in self.instance.machines[mid].compatible_products
            ]

        # 조립 pool 정보 (state feature용)
        assembly_pool_info = {
            prod_id: {t: list(jobs) for t, jobs in type_pool.items()}
            for prod_id, type_pool in self.assembly_pool.items()
        }

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
            "inactive_final_jobs": {p: list(v) for p, v in self.inactive_final_jobs.items()},
            "job_op_map": {jid: list(ops) for jid, ops in self.job_ops.items()},
        }

    # ──────────────────────────────────────────────────────
    # Reward
    # ──────────────────────────────────────────────────────

    def _reward_fn(self) -> float:
        """r = -Δ(weighted tardiness)"""
        current_wt = self._compute_actual_weighted_tardiness()
        delta = current_wt - self._prev_actual_wt
        self._prev_actual_wt = current_wt
        return -delta

    def _compute_actual_weighted_tardiness(self) -> float:
        """완료된 final job에 대해 주문 납기 기준 tardiness 합산"""
        total = 0.0
        for order in self.instance.orders.values():
            weight = self.instance.products[order.product_id].weight
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
        """미도착 긴급주문이 없고, 모든 final job이 완료되면 종료"""
        if self._urgent_orders_remaining > 0:
            return False
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
