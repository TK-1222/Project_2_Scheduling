"""
디스패칭룰 검증 스크립트 (FIFO / EDD / MWKR)
=============================================
동일 인스턴스에서 3가지 룰 각각 50회 반복.
결과: 룰별 각 에피소드 WT + 평균, 표준편차 출력 (그래프용 값)

사용법:
  python ffsa_validate_dispatch_rules.py
  python ffsa_validate_dispatch_rules.py --num-orders 6
"""

import argparse
import numpy as np

from ffsa_instance import full_config
from ffsa_env_dispatch import FFSASchedulingEnv

# ── RL 검증 스크립트와 동일한 시드 ──
TEST_SEEDS = list(range(10000, 10050))   # 50개


# ──────────────────────────────────────────────────────────
# 룰 선택 함수
# ──────────────────────────────────────────────────────────

def _is_assembly(action) -> bool:
    return isinstance(action[0], tuple)


def _job_id_of(action, env) -> int:
    if _is_assembly(action):
        return env.instance.jobs[action[0][0]].order_id
    op_id = action[0]
    return env.operations[op_id].job_id


def _remaining_ops(job_id, env) -> int:
    """job의 미완료 operation 수"""
    return sum(
        1 for oid in env.job_ops[job_id]
        if not env.operations[oid].is_done
    )


def select_fifo(obs, env) -> int:
    """FIFO: 가장 먼저 등록된 주문(order_id 작은 것) 우선"""
    actions = obs["actions"]
    best_idx, best_key = 0, None
    for i, action in enumerate(actions):
        if _is_assembly(action):
            job   = env.instance.jobs[action[0][0]]
            key   = (job.order_id, job.job_id)
        else:
            op_id, _ = action
            op    = env.operations[op_id]
            job   = env.instance.jobs[op.job_id]
            key   = (job.order_id, job.job_id)
        if best_key is None or key < best_key:
            best_key = key
            best_idx = i
    return best_idx


def select_edd(obs, env) -> int:
    """EDD: 납기 가장 빠른 job 우선"""
    actions = obs["actions"]
    best_idx, best_key = 0, None
    for i, action in enumerate(actions):
        if _is_assembly(action):
            job = env.instance.jobs[action[0][0]]
        else:
            op_id, _ = action
            op  = env.operations[op_id]
            job = env.instance.jobs[op.job_id]
        key = (job.due_date, job.order_id)
        if best_key is None or key < best_key:
            best_key = key
            best_idx = i
    return best_idx


def select_mwkr(obs, env) -> int:
    """MWKR: 잔여 operation 수 가장 많은 job 우선"""
    actions = obs["actions"]
    best_idx, best_key = 0, None
    for i, action in enumerate(actions):
        if _is_assembly(action):
            job_id = action[0][0]
        else:
            op_id, _ = action
            job_id   = env.operations[op_id].job_id
        remaining = _remaining_ops(job_id, env)
        key = (-remaining, job_id)   # 잔여 많을수록 우선 → 음수 사용
        if best_key is None or key < best_key:
            best_key = key
            best_idx = i
    return best_idx


RULES = {
    "FIFO": select_fifo,
    "EDD":  select_edd,
    "MWKR": select_mwkr,
}


# ──────────────────────────────────────────────────────────
# 검증 루프
# ──────────────────────────────────────────────────────────

def run_rule(rule_name, selector, num_orders, seeds=None):
    if seeds is None:
        seeds = TEST_SEEDS
    print(f"\n── {rule_name} ──")
    wt_list = []
    for i, seed in enumerate(seeds):
        config = full_config(seed=seed, num_regular_orders=num_orders)
        env    = FFSASchedulingEnv(config)
        obs, _ = env.reset()
        done = truncated = False
        max_steps = 50000
        step_count = 0

        while not (done or truncated) and step_count < max_steps:
            action_idx = selector(obs, env)
            obs, _, done, truncated, _ = env.step(action_idx)
            step_count += 1

        if step_count >= max_steps:
            print(f"  [{i+1:2d}/50]  seed={seed}  WT=TIMEOUT (max_steps 초과)")
            wt_list.append(float('inf'))
            continue

        wt = env.get_actual_weighted_tardiness()
        wt_list.append(wt)
        print(f"  [{i+1:2d}/50]  seed={seed}  WT={wt:.4f}")

    arr = np.array(wt_list)
    print(f"\n{rule_name} 결과  (num_orders={num_orders}, 50회)")
    print(f"  평균      : {arr.mean():.4f}")
    print(f"  표준편차  : {arr.std():.4f}")
    print(f"  최솟값    : {arr.min():.4f}")
    print(f"  최댓값    : {arr.max():.4f}")
    print(f"\n전체 WT 리스트:")
    print([round(v, 4) for v in wt_list])
    return wt_list


def main():
    parser = argparse.ArgumentParser(description="디스패칭룰 검증 (FIFO/EDD/MWKR)")
    parser.add_argument("--num-orders", type=int, default=6,
                        help="정규주문 수 (기본 6 ≈ 50 jobs)")
    parser.add_argument("--seed", type=int, default=None,
                        help="단일 seed 지정 시 해당 seed 1회만 실행 (기본: 10000~10049)")
    args = parser.parse_args()

    seeds = [args.seed] if args.seed is not None else TEST_SEEDS

    results = {}
    for name, selector in RULES.items():
        results[name] = run_rule(name, selector, args.num_orders, seeds)

    # ── 최종 요약 ──
    print(f"\n{'='*50}")
    print(f"{'룰':<8}  {'평균 WT':>10}  {'표준편차':>10}")
    print(f"{'-'*50}")
    for name, wt_list in results.items():
        arr = np.array(wt_list)
        print(f"{name:<8}  {arr.mean():>10.4f}  {arr.std():>10.4f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
