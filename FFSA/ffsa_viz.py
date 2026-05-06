"""
FFSA 스케줄링 결과 시각화
===========================
[태림]라인3개 각각 표현.py 스타일 적용:
  - networkx DiGraph 기반
  - 기계 노드 (삼각형) / 버퍼 노드 (원)
  - 제품별 색상 엣지로 실제 job 이동 경로 표현

사용 흐름:
  schedule = extract_schedule(env)          # 에피소드 완료 직후 호출
  log_schedule_to_tensorboard(writer, schedule, ep)   # TensorBoard 기록
  visualize_schedule(schedule, ep, save_path="out.png")  # PNG 저장
"""

import os
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams
from matplotlib.font_manager import FontProperties, fontManager
from typing import Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────
# 색상 / 스타일 (태림 코드 스타일 계승)
# ─────────────────────────────────────────────────────────

PRODUCT_COLORS = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
    "#ff7f00", "#a65628", "#f781bf", "#636363",
]

NODE_SIZE = {"machine": 1600, "buffer": 900, "assembly": 2000, "done": 1000}
NODE_SHAPE = {"machine": "^", "buffer": "o", "assembly": "s", "done": "D"}
NODE_COLOR = {
    "machine":  ("#8ecae6", "#5b8bd0"),   # fill, edge (태림 operation 색상)
    "buffer":   ("#86df7f", "#2e8b57"),   # fill, edge (태림 buffer 색상)
    "assembly": ("#ffd166", "#9e5c00"),
    "done":     ("#aaaaaa", "#444444"),
}

EDGE_STRUCTURE = "#dddddd"   # 가능 경로 (연한 회색)
EDGE_JOB_WIDTH = 2.0         # job 경로 엣지 기본 굵기
LABEL_ABOVE = 0.38           # 태림과 동일한 라벨 오프셋
LABEL_BELOW = -0.38


# ─────────────────────────────────────────────────────────
# 한글 폰트 (태림 코드와 동일한 로직)
# ─────────────────────────────────────────────────────────

def _init_font() -> Optional[FontProperties]:
    candidates = [
        r"C:\Windows\Fonts\malgun.ttf",
        "/System/Library/Fonts/AppleGothic.ttf",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    ]
    path = next((p for p in candidates if os.path.exists(p)), None)
    if path is None:
        return None
    fontManager.addfont(path)
    fp = FontProperties(fname=path)
    rcParams["font.family"] = fp.get_name()
    rcParams["axes.unicode_minus"] = False
    return fp


# ─────────────────────────────────────────────────────────
# 스케줄 데이터 추출
# ─────────────────────────────────────────────────────────

def extract_schedule(env) -> dict:
    """
    env에서 시각화에 필요한 스케줄 스냅샷 추출.
    env.reset() 이전에 호출해야 함.
    """
    ops = {}
    for op_id, op in env.operations.items():
        ops[op_id] = {
            "job_id":          op.job_id,
            "stage_id":        op.stage_id,
            "product_id":      op.product_id,
            "is_done":         op.is_done,
            "machine_id":      op.machine_id,
            "start_time":      op.start_time,
            "completion_time": op.completion_time,
        }

    jobs = {}
    for jid, j in env.instance.jobs.items():
        jobs[jid] = {
            "product_id":   j.product_id,
            "is_component": j.is_component,
            "is_final_job": j.is_final_job,
            "due_date":     j.due_date,
            "order_id":     j.order_id,
        }

    machines = {}
    for mid, ms in env.machine_states.items():
        machines[mid] = {
            "stage_id":        ms.stage_id,
            "last_product":    ms.last_product,
            "total_busy_time": ms.total_busy_time,
        }

    return {
        "ops":               ops,
        "jobs":              jobs,
        "job_ops":           {jid: list(ops_list) for jid, ops_list in env.job_ops.items()},
        "machines":          machines,
        "machines_by_stage": {sid: list(mids) for sid, mids in env.instance.machines_by_stage.items()},
        "num_stages":        env.instance.num_stages,
        "num_products":      env.config.num_products,
        "use_assembly":      env.config.use_assembly,
        "assembly_stage":    env.config.assembly_stage_idx,
        "makespan":          env.get_makespan(),
        "wt":                env.get_actual_weighted_tardiness(),
    }


# ─────────────────────────────────────────────────────────
# 그래프 빌드
# ─────────────────────────────────────────────────────────

def build_flow_graph(schedule: dict) -> nx.DiGraph:
    """
    태림 스타일 DiGraph 생성.
    노드: 기계(삼각형), 버퍼(원), 조립(사각형), 완료(다이아몬드)
    엣지: 구조 엣지(연회색) + job 실제 이동 경로(제품색)
    """
    G = nx.DiGraph()
    num_stages   = schedule["num_stages"]
    machines     = schedule["machines"]
    machines_by_stage = schedule["machines_by_stage"]
    use_assembly = schedule["use_assembly"]
    asm_stage    = schedule["assembly_stage"]

    # ── 기계 노드 ──
    for mid, mdata in machines.items():
        sid = mdata["stage_id"]
        is_asm = (use_assembly and sid == asm_stage)
        ntype = "assembly" if is_asm else "machine"
        G.add_node(f"M{mid}",
                   ntype=ntype, stage_id=sid, machine_id=mid,
                   last_product=mdata["last_product"],
                   label=f"M{mid}")

    # ── 버퍼 노드 (stage 앞단) ──
    for sid in range(num_stages):
        G.add_node(f"BUF{sid}",
                   ntype="buffer", stage_id=sid,
                   label=f"Buf{sid}")

    # ── 완료 노드 ──
    G.add_node("DONE", ntype="done", stage_id=num_stages, label="완료")

    # ── 구조 엣지 (배경, 가능한 경로) ──
    for sid in range(num_stages):
        for mid in machines_by_stage.get(sid, []):
            # 버퍼 → 기계
            G.add_edge(f"BUF{sid}", f"M{mid}",
                       etype="structure", color=EDGE_STRUCTURE, width=0.8)
            # 기계 → 다음 버퍼 (마지막 stage 제외)
            if sid < num_stages - 1:
                G.add_edge(f"M{mid}", f"BUF{sid+1}",
                           etype="structure", color=EDGE_STRUCTURE, width=0.8)
            else:
                G.add_edge(f"M{mid}", "DONE",
                           etype="structure", color=EDGE_STRUCTURE, width=0.8)

    # ── Job 실제 이동 경로 엣지 ──
    job_ops = schedule["job_ops"]
    ops     = schedule["ops"]
    jobs    = schedule["jobs"]

    for jid, op_ids in job_ops.items():
        job = jobs.get(jid)
        if job is None:
            continue
        color = PRODUCT_COLORS[job["product_id"] % len(PRODUCT_COLORS)]
        pid   = job["product_id"]

        done_ops = [
            ops[oid] for oid in op_ids
            if oid in ops and ops[oid]["is_done"] and ops[oid]["machine_id"] is not None
        ]
        if not done_ops:
            continue

        for i, op in enumerate(done_ops):
            curr_m = f"M{op['machine_id']}"
            buf    = f"BUF{op['stage_id']}"

            # 버퍼 → 기계 경로 기록
            _add_job_edge(G, buf, curr_m, color, pid)

            if i > 0:
                # 이전 기계 → 이 버퍼
                prev_m = f"M{done_ops[i-1]['machine_id']}"
                _add_job_edge(G, prev_m, buf, color, pid)

        # 마지막 기계 → 완료
        last_m = f"M{done_ops[-1]['machine_id']}"
        _add_job_edge(G, last_m, "DONE", color, pid)

    return G


def _add_job_edge(G: nx.DiGraph, src: str, dst: str, color: str, pid: int):
    """job 경로 엣지 추가 (중복 시 굵기 증가)"""
    if G.has_edge(src, dst) and G[src][dst].get("etype") == "job":
        G[src][dst]["width"] += 0.5
        G[src][dst]["count"] += 1
    else:
        G.add_edge(src, dst, etype="job", color=color,
                   width=EDGE_JOB_WIDTH, product_id=pid, count=1)


# ─────────────────────────────────────────────────────────
# 좌표 배치 (태림 코드와 동일한 고정 좌표 방식)
# ─────────────────────────────────────────────────────────

def get_positions(schedule: dict) -> dict:
    """
    x축: stage, y축: stage 내 기계 번호
    버퍼 노드는 stage 직전 위치 (x - 0.5 * x_spacing)
    """
    pos: dict = {}
    num_stages        = schedule["num_stages"]
    machines_by_stage = schedule["machines_by_stage"]
    machines          = schedule["machines"]

    X_SPACING = 3.5
    Y_SPACING = 2.2
    BUF_OFFSET = 1.5   # 버퍼 x 위치 = stage_x - BUF_OFFSET

    for sid in range(num_stages):
        mids = machines_by_stage.get(sid, [])
        n    = len(mids)
        x    = sid * X_SPACING
        for i, mid in enumerate(mids):
            y = (i - (n - 1) / 2.0) * Y_SPACING
            pos[f"M{mid}"] = (x, y)

        # 버퍼 노드 위치
        buf_x = x - BUF_OFFSET
        pos[f"BUF{sid}"] = (buf_x, 0.0)

    # 완료 노드
    pos["DONE"] = (num_stages * X_SPACING, 0.0)
    return pos


# ─────────────────────────────────────────────────────────
# 그리기 (태림 draw_nodes / draw_edges / draw_labels 패턴)
# ─────────────────────────────────────────────────────────

def _draw_edges(G: nx.DiGraph, pos: dict, ax):
    # 구조 엣지 (연한 배경)
    struct = [(u, v) for u, v, d in G.edges(data=True) if d.get("etype") == "structure"]
    if struct:
        nx.draw_networkx_edges(G, pos, edgelist=struct,
                               edge_color=EDGE_STRUCTURE, width=0.8,
                               arrows=False, ax=ax)

    # job 경로 엣지
    job_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("etype") == "job"]
    if job_edges:
        colors = [G[u][v]["color"] for u, v in job_edges]
        widths = [G[u][v]["width"] for u, v in job_edges]
        nx.draw_networkx_edges(G, pos, edgelist=job_edges,
                               edge_color=colors, width=widths,
                               arrows=True, arrowstyle="-|>", arrowsize=14,
                               connectionstyle="arc3,rad=0.08", ax=ax)


def _draw_nodes(G: nx.DiGraph, pos: dict, ax):
    for ntype in ("machine", "buffer", "assembly", "done"):
        nodelist = [n for n, d in G.nodes(data=True) if d.get("ntype") == ntype]
        if not nodelist:
            continue
        fill_base, edge_c = NODE_COLOR[ntype]
        fills = []
        for n in nodelist:
            d = G.nodes[n]
            lp = d.get("last_product")
            # 기계/조립 노드: 마지막으로 처리한 제품 색상으로 채우기
            if ntype in ("machine", "assembly") and lp is not None:
                fills.append(PRODUCT_COLORS[lp % len(PRODUCT_COLORS)])
            else:
                fills.append(fill_base)

        nx.draw_networkx_nodes(G, pos, nodelist=nodelist,
                               node_shape=NODE_SHAPE[ntype],
                               node_color=fills,
                               edgecolors=edge_c, linewidths=1.5,
                               node_size=NODE_SIZE[ntype], ax=ax)


def _draw_labels(G: nx.DiGraph, pos: dict, schedule: dict,
                 kfont: Optional[FontProperties], ax):
    num_stages = schedule["num_stages"]
    X_SPACING  = 3.5

    for node, data in G.nodes(data=True):
        if node not in pos:
            continue
        x, y = pos[node]
        label = data.get("label", "")
        if not label:
            continue
        ax.text(x, y + LABEL_ABOVE, label,
                fontproperties=kfont, ha="center", va="bottom",
                fontsize=7, color="black")

    # stage 제목 (태림의 라인 타이틀에 해당)
    for sid in range(num_stages):
        x = sid * X_SPACING
        ax.text(x, -3.8, f"Stage {sid}",
                fontproperties=kfont, ha="center", va="top",
                fontsize=9, fontweight="bold")


def _draw_legend(G: nx.DiGraph, schedule: dict,
                 kfont: Optional[FontProperties], ax):
    num_prods = schedule["num_products"]
    patches = [
        mpatches.Patch(color=PRODUCT_COLORS[pid % len(PRODUCT_COLORS)],
                       label=f"제품 {pid}")
        for pid in range(num_prods)
    ]
    patches += [
        mpatches.Patch(facecolor=NODE_COLOR["machine"][0],
                       edgecolor=NODE_COLOR["machine"][1], label="기계 (▲)"),
        mpatches.Patch(facecolor=NODE_COLOR["buffer"][0],
                       edgecolor=NODE_COLOR["buffer"][1],  label="버퍼 (●)"),
    ]
    if schedule["use_assembly"]:
        patches.append(
            mpatches.Patch(facecolor=NODE_COLOR["assembly"][0],
                           edgecolor=NODE_COLOR["assembly"][1], label="조립 (■)")
        )
    ax.legend(handles=patches, loc="upper right", fontsize=8,
              prop=kfont if kfont else {})


# ─────────────────────────────────────────────────────────
# 통합 시각화 함수
# ─────────────────────────────────────────────────────────

def draw_flow_graph(G: nx.DiGraph, pos: dict, schedule: dict,
                    title: str = "", kfont=None) -> plt.Figure:
    """태림 스타일로 FFSA 스케줄 그래프 시각화"""
    num_stages = schedule["num_stages"]
    machines_by_stage = schedule["machines_by_stage"]
    max_machines = max((len(v) for v in machines_by_stage.values()), default=2)

    fig_w = max(18, num_stages * 3.5 + 4)
    fig_h = max(8,  max_machines * 2.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    _draw_edges(G, pos, ax)
    _draw_nodes(G, pos, ax)
    _draw_labels(G, pos, schedule, kfont, ax)
    _draw_legend(G, schedule, kfont, ax)

    ax.set_title(title, fontsize=13)
    ax.axis("off")

    # 좌표 범위 (태림과 같은 방식)
    all_x = [p[0] for p in pos.values()]
    all_y = [p[1] for p in pos.values()]
    margin_x, margin_y = 2.0, 2.0
    ax.set_xlim(min(all_x) - margin_x, max(all_x) + margin_x)
    ax.set_ylim(min(all_y) - margin_y - 2.0, max(all_y) + margin_y)

    plt.tight_layout()
    return fig


def visualize_schedule(schedule: dict, ep: int = 0,
                        save_path: Optional[str] = None,
                        show: bool = False) -> plt.Figure:
    """에피소드 스케줄 시각화 편의 함수"""
    kfont = _init_font()
    G   = build_flow_graph(schedule)
    pos = get_positions(schedule)
    wt  = schedule.get("wt", 0.0)
    ms  = schedule.get("makespan", 0.0)
    fig = draw_flow_graph(G, pos, schedule,
                          title=f"[Ep {ep}]  WT={wt:.1f}  |  Makespan={ms:.1f}",
                          kfont=kfont)
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=100, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def log_schedule_to_tensorboard(writer, schedule: dict, ep: int):
    """TensorBoard writer.add_figure()로 스케줄 그래프 로깅"""
    fig = visualize_schedule(schedule, ep=ep)
    writer.add_figure("schedule/flow_graph", fig, global_step=ep)
    plt.close(fig)
