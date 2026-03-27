import os
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams
from matplotlib.font_manager import FontProperties
from torch_geometric.utils import from_networkx


# =========================================================
# 0. 상수 정의
# =========================================================
NODE_SIZE = {
    "operation": 1900,
    "buffer":    1300,
    "inspection":1100,
}
NODE_COLOR = {
    "operation":  ("#8ecae6", "#5b8bd0"),
    "buffer":     ("#86df7f", "#2e8b57"),
    "inspection": ("white",   "#9e9e9e"),
}
EDGE_COLOR_MAIN = "#d9991e"

# 글자 겹침 해결: 노드 위/아래 분리
# 상단 라인 공정은 라벨을 위쪽, 하단 라인 공정은 라벨을 아래쪽에 배치
LABEL_ABOVE = 0.38   # 노드 위 (y + LABEL_ABOVE)
LABEL_BELOW = -0.38  # 노드 아래 (y + LABEL_BELOW)


# =========================================================
# 1. 한글 폰트 설정
# =========================================================
def set_korean_font() -> FontProperties:
    candidates = [
        r"C:\Windows\Fonts\malgun.ttf",
        "/System/Library/Fonts/AppleGothic.ttf",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    ]
    font_path = next((p for p in candidates if os.path.exists(p)), None)
    if font_path is None:
        raise FileNotFoundError("한글 폰트를 찾지 못했습니다.")

    font_manager.fontManager.addfont(font_path)
    fp = FontProperties(fname=font_path)
    rcParams["font.family"] = fp.get_name()
    rcParams["axes.unicode_minus"] = False
    print(f"[INFO] 적용된 폰트: {fp.get_name()}")
    return fp


# =========================================================
# 2. 그래프 생성
#    ※ 시각용 요소는 그래프 노드에서 제거 → GNN 데이터 오염 방지
#    ※ 후속 연구: CT, 작업방법 등 확장 시 x 벡터에 추가
# =========================================================
def build_taerim_graph() -> nx.DiGraph:
    G = nx.DiGraph()
    lines = ["LINE1", "LINE2", "LINE3"]

    for line in lines:
        node_defs = {
            # 상단 브랜치 (label_side="top" → 라벨 위쪽)
            f"BUF_TOP_0_{line}":  ("",                     "buffer",    "top"),
            f"F_IN_TOP_{line}":   ("자료투입",             "operation", "top"),
            f"BUF_TOP_1_{line}":  ("",                     "buffer",    "top"),
            f"OP20_TOP_{line}":   ("cpointyoke",            "operation", "top"),
            f"BUF_TOP_2_{line}":  ("",                     "buffer",    "top"),
            f"OP30_TOP_{line}":   ("tubeyoke",              "operation", "top"),
            f"INSP_TOP_{line}":   ("",                     "inspection","top"),
            f"OP40_TOP_{line}":   ("inspection\ntubeyoke",  "operation", "top"),
            # 하단 브랜치 (label_side="bot" → 라벨 아래쪽)
            f"BUF_BOT_0_{line}":  ("",                     "buffer",    "bot"),
            f"F_IN_BOT_{line}":   ("자료투입",             "operation", "bot"),
            f"BUF_BOT_1_{line}":  ("",                     "buffer",    "bot"),
            f"OP20_BOT_{line}":   ("bpointyoke",            "operation", "bot"),
            f"BUF_BOT_2_{line}":  ("",                     "buffer",    "bot"),
            f"OP30_BOT_{line}":   ("shaftyoke",             "operation", "bot"),
            f"INSP_BOT_{line}":   ("",                     "inspection","bot"),
            f"OP40_BOT_{line}":   ("inspection\nshaftyoke", "operation", "bot"),
            # 합류 이후 (label_side="bot" → 라벨 아래쪽)
            f"ASSEMBLY_{line}":   ("조립",                 "buffer",    "bot"),
            f"OP50_{line}":       ("Tube+shaft",            "operation", "top"),
            f"BUF_50_60_{line}":  ("",                     "buffer",    "bot"),
            f"OP60_{line}":       ("preassy",               "operation", "top"),
            f"INSP_MAIN_{line}":  ("검사",                  "inspection","bot"),
            f"OP70_{line}":       ("assy",                  "operation", "top"),
            f"POST70_BUF_{line}": ("",                     "buffer",    "bot"),
            f"FGS_{line}":        ("FGS",                  "operation", "top"),
        }

        for nid, (label, ntype, side) in node_defs.items():
            x = [
                1.0 if ntype == "operation"  else 0.0,
                1.0 if ntype == "buffer"     else 0.0,
                1.0 if ntype == "inspection" else 0.0,
            ]
            G.add_node(nid, label=label, type=ntype, side=side, x=x, line=line)

        top_flow    = [f"BUF_TOP_0_{line}", f"F_IN_TOP_{line}", f"BUF_TOP_1_{line}", f"OP20_TOP_{line}",
                       f"BUF_TOP_2_{line}", f"OP30_TOP_{line}", f"INSP_TOP_{line}",
                       f"OP40_TOP_{line}", f"ASSEMBLY_{line}"]
        bot_flow    = [f"BUF_BOT_0_{line}", f"F_IN_BOT_{line}", f"BUF_BOT_1_{line}", f"OP20_BOT_{line}",
                       f"BUF_BOT_2_{line}", f"OP30_BOT_{line}", f"INSP_BOT_{line}",
                       f"OP40_BOT_{line}", f"ASSEMBLY_{line}"]
        merged_flow = [f"ASSEMBLY_{line}", f"OP50_{line}", f"BUF_50_60_{line}",
                       f"OP60_{line}", f"INSP_MAIN_{line}", f"OP70_{line}",
                       f"POST70_BUF_{line}", f"FGS_{line}"]

        for flow in (top_flow, bot_flow, merged_flow):
            for u, v in zip(flow, flow[1:]):
                G.add_edge(u, v)

    return G


# =========================================================
# 3. 좌표 배치
#    라인 간격을 5.0으로 늘려 라벨 겹침 방지
# =========================================================
def get_positions() -> dict:
    pos = {}
    line_offsets = {"LINE1": 5.0, "LINE2": 0.0, "LINE3": -5.0}

    for line, dy in line_offsets.items():
        y_top = 1.5 + dy
        y_bot = 0.0 + dy
        y_mid = 0.75 + dy

        pos.update({
            f"BUF_TOP_0_{line}":  (-1.2, y_top),
            f"F_IN_TOP_{line}":   (0.0,  y_top),
            f"BUF_TOP_1_{line}":  (1.1,  y_top),
            f"OP20_TOP_{line}":   (2.4,  y_top),
            f"BUF_TOP_2_{line}":  (3.6,  y_top),
            f"OP30_TOP_{line}":   (4.9,  y_top),
            f"INSP_TOP_{line}":   (6.0,  y_top),
            f"OP40_TOP_{line}":   (7.1,  y_top),

            f"BUF_BOT_0_{line}":  (-1.2, y_bot),
            f"F_IN_BOT_{line}":   (0.0,  y_bot),
            f"BUF_BOT_1_{line}":  (1.1,  y_bot),
            f"OP20_BOT_{line}":   (2.4,  y_bot),
            f"BUF_BOT_2_{line}":  (3.6,  y_bot),
            f"OP30_BOT_{line}":   (4.9,  y_bot),
            f"INSP_BOT_{line}":   (6.0,  y_bot),
            f"OP40_BOT_{line}":   (7.1,  y_bot),

            f"ASSEMBLY_{line}":   (8.3,  y_mid),
            f"OP50_{line}":       (9.5,  y_mid),
            f"BUF_50_60_{line}":  (10.8, y_mid),
            f"OP60_{line}":       (12.1, y_mid),
            f"INSP_MAIN_{line}":  (13.3, y_mid),
            f"OP70_{line}":       (14.5, y_mid),
            f"POST70_BUF_{line}": (15.8, y_mid),
            f"FGS_{line}":        (17.1, y_mid),
        })
    return pos


# =========================================================
# 4. 노드 그리기
# =========================================================
def draw_nodes(G: nx.DiGraph, pos: dict):
    for ntype in ("operation", "buffer", "inspection"):
        nodelist = [n for n, d in G.nodes(data=True) if d["type"] == ntype]
        fill, edge = NODE_COLOR[ntype]
        shape = "^" if ntype == "operation" else "o"
        nx.draw_networkx_nodes(
            G, pos, nodelist=nodelist,
            node_shape=shape, node_color=fill,
            edgecolors=edge, linewidths=1.5,
            node_size=NODE_SIZE[ntype],
        )


# =========================================================
# 5. 엣지 그리기
# =========================================================
def draw_edges(G: nx.DiGraph, pos: dict):
    nx.draw_networkx_edges(
        G, pos,
        edgelist=list(G.edges()),
        edge_color=EDGE_COLOR_MAIN,
        arrows=False, width=1.8,
    )


# =========================================================
# 6. 라벨 그리기
#    side="top"  → 노드 위쪽 (va="bottom")
#    side="bot"  → 노드 아래쪽 (va="top")
#    겹치는 노드가 없도록 상단/하단 브랜치를 반대 방향으로 분리
# =========================================================
def draw_labels(G: nx.DiGraph, pos: dict, kfont: FontProperties):
    ax = plt.gca()

    for node, data in G.nodes(data=True):
        label = data.get("label", "")
        if not label:
            continue
        x, y = pos[node]
        side = data.get("side", "top")
        if side == "top":
            ax.text(x, y + LABEL_ABOVE, label,
                    fontproperties=kfont, ha="center", va="bottom",
                    fontsize=8, color="black")
        else:
            ax.text(x, y + LABEL_BELOW, label,
                    fontproperties=kfont, ha="center", va="top",
                    fontsize=8, color="black")

    # 라인 타이틀
    line_names   = {"LINE1": "생산라인 1", "LINE2": "생산라인 2", "LINE3": "생산라인 3"}
    line_offsets = {"LINE1": 5.0, "LINE2": 0.0, "LINE3": -5.0}
    for line, dy in line_offsets.items():
        ax.text(-1.5, 0.75 + dy, line_names[line],
                fontproperties=kfont, ha="right", va="center",
                fontsize=12, fontweight="bold")


# =========================================================
# 7. 시각화 통합
# =========================================================
def visualize_graph(G: nx.DiGraph, kfont: FontProperties):
    _, ax = plt.subplots(figsize=(20, 14))
    pos = get_positions()

    draw_edges(G, pos)
    draw_nodes(G, pos)
    draw_labels(G, pos, kfont)

    ax.axis("off")
    ax.set_xlim(-3.5, 18.0)
    ax.set_ylim(-6.5, 8.0)
    plt.tight_layout()
    plt.show()


# =========================================================
# 8. GNN 데이터 변환 및 요약 출력
# =========================================================
def convert_to_pyg(G: nx.DiGraph):
    pyg_data = from_networkx(G)
    print("\n[PyTorch Geometric Data Object]")
    print(pyg_data)
    print(f"\n총 노드 수: {G.number_of_nodes()}  |  총 엣지 수: {G.number_of_edges()}")
    if hasattr(pyg_data, "x"):
        print("\nNode Features (x) Shape:", pyg_data.x.shape)
        print("First 5 node features:\n", pyg_data.x[:5])
    return pyg_data


# =========================================================
# 9. 메인
# =========================================================
def main():
    kfont    = set_korean_font()
    G        = build_taerim_graph()

    print("그래프 시각화를 시작합니다...")
    visualize_graph(G, kfont)

    print("\nNetworkX → PyTorch Geometric 변환 중...")
    pyg_data = convert_to_pyg(G)
    return G, pyg_data


if __name__ == "__main__":
    main()