# inspect_model.py
import torch
import random
from ffsa_instance import full_config
from ffsa_env_dispatch import FFSASchedulingEnv
from ffsa_model import RegularQNetwork, AssemblyQNetwork, DualDQNAgent

# ── 1. 모델 로드 ──────────────────────────────────────────
CHECKPOINT = "runs/1run/checkpoints/best_model.pt"
HIDDEN_DIM = 16   # 학습 시 사용한 hidden_dim과 동일하게

ckpt    = torch.load(CHECKPOINT, map_location="cpu")
reg_net = RegularQNetwork(hidden_dim=HIDDEN_DIM)
asm_net = AssemblyQNetwork(hidden_dim=HIDDEN_DIM)
reg_net.load_state_dict(ckpt["reg_online"])
asm_net.load_state_dict(ckpt["asm_online"])
print(f"체크포인트 로드: ep={ckpt['ep']}, best_wt={ckpt['best_wt']:.2f}")

agent         = DualDQNAgent(reg_net, asm_net, device="cpu")
agent.epsilon = 0.0   # greedy 평가 (탐색 없음)

# ── 2. 여러 시드로 에피소드 실행 ─────────────────────────
TEST_SEEDS = [42, 123, 777, 1000, 2025]

for seed in TEST_SEEDS:
    config  = full_config(seed=seed)
    env     = FFSASchedulingEnv(config)
    obs, _  = env.reset()
    done = truncated = False
    step = 0

    while not (done or truncated):
        action_idx = agent.select_action(obs)
        obs, reward, done, truncated, info = env.step(action_idx)
        step += 1

        if info.get("deadlock"):
            # ── deadlock 발생 시 상태 덤프 ──
            print(f"\n[seed={seed}] ★ DEADLOCK  step={step}  t={env.current_time:.1f}")

            # pool_blocked 상태
            print(f"  pool_blocked  : {env._pool_blocked}")

            # 미완료 final job
            incomplete = []
            for order in env.instance.orders.values():
                for fid in order.final_job_ids:
                    last_op = env.operations[env.job_ops[fid][-1]]
                    if not last_op.is_done:
                        incomplete.append(fid)
            print(f"  미완료 final  : {incomplete}")

            # ready지만 아직 미처리인 op
            ready_ops = [
                op.op_id for op in env.operations.values()
                if op.active and op.is_ready and not op.is_done and not op.is_processing
            ]
            print(f"  ready op      : {ready_ops}")

            # 처리 중인 op
            processing = [
                (op.op_id, op.completion_time)
                for op in env.operations.values()
                if op.active and op.is_processing
            ]
            print(f"  처리 중 op    : {processing}")

            # 기계 상태
            blocked_machines = [
                (mid, ms.blocked_job)
                for mid, ms in env.machine_states.items()
                if ms.is_blocked
            ]
            print(f"  blocked 기계  : {blocked_machines}")
            break

    wt = env.get_actual_weighted_tardiness()
    status = "DEADLOCK" if truncated else "DONE"
    print(f"[seed={seed}]  {status}  steps={step}  WT={wt:.2f}")