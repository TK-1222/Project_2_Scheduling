"""
RL 에이전트 검증 스크립트
============================
best_model.pt를 로드해 50회 반복 실험.
결과: 각 에피소드 WT + 평균, 표준편차 출력 (그래프용 값)

--gantt 옵션 사용 시 첫 번째 에피소드 스케줄을 간트 차트로 저장.

사용법:
  python ffsa_validate_rl.py --checkpoint runs/exp_name/checkpoints/best_model.pt
  python ffsa_validate_rl.py --checkpoint runs/exp_name/checkpoints/best_model.pt --gantt
"""

import argparse
import numpy as np
import torch

from ffsa_instance import full_config
from ffsa_env_dispatch import FFSASchedulingEnv
from ffsa_model import RegularQNetwork, AssemblyQNetwork, DualDQNAgent
from ffsa_env import OpFeat

# ── 두 스크립트 공통 시드 (dispatching rule 비교와 동일 인스턴스 사용) ──
TEST_SEEDS = list(range(10000, 10050))   # 50개


def main():
    parser = argparse.ArgumentParser(description="RL 에이전트 검증")
    parser.add_argument("--checkpoint",  type=str, required=True,
                        help="best_model.pt 경로")
    parser.add_argument("--num-orders",  type=int, default=6,
                        help="정규주문 수 (기본 6 ≈ 50 jobs)")
    parser.add_argument("--hidden-dim",  type=int, default=16)
    parser.add_argument("--device",      type=str, default="cpu")
    parser.add_argument("--gantt",       action="store_true",
                        help="첫 번째 에피소드 간트 차트를 PNG로 저장")
    args = parser.parse_args()

    # ── 모델 로드 ──
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    print(f"체크포인트: ep={ckpt['ep']},  best_train_wt={ckpt['best_wt']:.2f}\n")

    reg_net = RegularQNetwork(
        op_feat_dim=OpFeat.DIM, machine_feat_dim=6, edge_feat_dim=2,
        hidden_dim=args.hidden_dim, num_layers=2, mlp_hidden=128,
    ).to(args.device)
    asm_net = AssemblyQNetwork(
        op_feat_dim=OpFeat.DIM, machine_feat_dim=6, edge_feat_dim=2,
        hidden_dim=args.hidden_dim, num_layers=2, mlp_hidden=128,
    ).to(args.device)
    reg_net.load_state_dict(ckpt["reg_online"])
    asm_net.load_state_dict(ckpt["asm_online"])

    agent = DualDQNAgent(
        reg_net=reg_net, asm_net=asm_net,
        lr=1e-4, gamma=0.99, device=args.device,
    )
    agent.epsilon = 0.0   # 완전 greedy

    # ── 검증 ──
    wt_list = []
    for i, seed in enumerate(TEST_SEEDS):
        config = full_config(seed=seed, num_regular_orders=args.num_orders)
        env    = FFSASchedulingEnv(config)
        obs, _ = env.reset()
        done = truncated = False

        while not (done or truncated):
            action_idx = agent.select_action(obs)
            obs, _, done, truncated, _ = env.step(action_idx)

        wt = env.get_actual_weighted_tardiness()
        wt_list.append(wt)
        print(f"  [{i+1:2d}/50]  seed={seed}  WT={wt:.4f}")

        # 첫 번째 에피소드 간트 차트 저장
        if i == 0 and args.gantt:
            from ffsa_viz import draw_gantt
            draw_gantt(env,
                       title=f"RL Agent  (seed={seed}, WT={wt:.1f})",
                       save_path=f"gantt_rl_seed{seed}.png")

    arr = np.array(wt_list)
    print(f"\n{'='*40}")
    print(f"RL 에이전트 결과  (num_orders={args.num_orders}, 50회)")
    print(f"  평균      : {arr.mean():.4f}")
    print(f"  표준편차  : {arr.std():.4f}")
    print(f"  최솟값    : {arr.min():.4f}")
    print(f"  최댓값    : {arr.max():.4f}")
    print(f"\n전체 WT 리스트:")
    print([round(v, 4) for v in wt_list])


if __name__ == "__main__":
    main()
