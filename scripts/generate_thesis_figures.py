from __future__ import annotations

from pathlib import Path
import math

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "figures"
W, H = 2400, 1350

FONT_CANDIDATES = [
    Path(r"C:\Windows\Fonts\NotoSansSC-VF.ttf"),
    Path(r"C:\Windows\Fonts\msyh.ttc"),
    Path(r"C:\Windows\Fonts\simhei.ttf"),
    Path(r"C:\Windows\Fonts\simsun.ttc"),
]
BOLD_CANDIDATES = [
    Path(r"C:\Windows\Fonts\msyhbd.ttc"),
    Path(r"C:\Windows\Fonts\simhei.ttf"),
    Path(r"C:\Windows\Fonts\NotoSansSC-VF.ttf"),
]

COLORS = {
    "blue": "#1455b8",
    "navy": "#0b2a6b",
    "cyan": "#0f8fa8",
    "green": "#237a35",
    "light_green": "#effaf0",
    "purple": "#6b41aa",
    "orange": "#d95f02",
    "red": "#c82e2e",
    "gray": "#607080",
    "line": "#2864c8",
    "panel": "#f7fbff",
    "white": "#ffffff",
    "black": "#172033",
}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = BOLD_CANDIDATES if bold else FONT_CANDIDATES
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


FONTS = {
    "title": font(58, True),
    "subtitle": font(40, True),
    "section": font(34, True),
    "body": font(30),
    "body_bold": font(30, True),
    "small": font(24),
    "small_bold": font(24, True),
    "tiny": font(20),
    "tiny_bold": font(20, True),
    "micro_bold": font(16, True),
}


def canvas() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (W, H), "#ffffff")
    draw = ImageDraw.Draw(image)
    return image, draw


def text_size(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.FreeTypeFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=fnt)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def draw_text_center(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    fnt: ImageFont.FreeTypeFont,
    fill: str = COLORS["black"],
) -> None:
    x, y = xy
    lines = text.split("\n")
    line_heights = [text_size(draw, line, fnt)[1] for line in lines]
    total_h = sum(line_heights) + (len(lines) - 1) * 8
    cursor = y - total_h / 2
    for line, line_h in zip(lines, line_heights):
        line_w, _ = text_size(draw, line, fnt)
        draw.text((x - line_w / 2, cursor), line, font=fnt, fill=fill)
        cursor += line_h + 8


def draw_wrapped_center(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    fnt: ImageFont.FreeTypeFont,
    fill: str = COLORS["black"],
    line_gap: int = 6,
) -> None:
    x, y, w, h = box
    lines = []
    for raw in text.split("\n"):
        if not raw:
            lines.append("")
            continue
        current = ""
        for ch in raw:
            trial = current + ch
            if text_size(draw, trial, fnt)[0] <= w - 20:
                current = trial
            else:
                if current:
                    lines.append(current)
                current = ch
        if current:
            lines.append(current)
    heights = [text_size(draw, line, fnt)[1] for line in lines]
    total_h = sum(heights) + max(len(lines) - 1, 0) * line_gap
    cursor = y + (h - total_h) / 2
    for line, line_h in zip(lines, heights):
        line_w, _ = text_size(draw, line, fnt)
        draw.text((x + (w - line_w) / 2, cursor), line, font=fnt, fill=fill)
        cursor += line_h + line_gap


def rounded(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    fill: str = COLORS["white"],
    outline: str = COLORS["line"],
    radius: int = 18,
    width: int = 3,
) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def header_box(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    header: str,
    body: str | None = None,
    color: str = COLORS["blue"],
    fill: str = COLORS["white"],
    fnt_header: ImageFont.FreeTypeFont = FONTS["body_bold"],
    fnt_body: ImageFont.FreeTypeFont = FONTS["small"],
) -> None:
    x1, y1, x2, y2 = box
    rounded(draw, box, fill=fill, outline=color, radius=18, width=3)
    draw.rounded_rectangle((x1, y1, x2, y1 + 58), radius=18, fill=color, outline=color)
    draw.rectangle((x1, y1 + 30, x2, y1 + 58), fill=color)
    draw_text_center(draw, ((x1 + x2) / 2, y1 + 29), header, fnt_header, "#ffffff")
    if body is not None:
        draw_wrapped_center(draw, (x1 + 8, y1 + 70, x2 - x1 - 16, y2 - y1 - 78), body, fnt_body)


def arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    color: str = COLORS["line"],
    width: int = 5,
) -> None:
    draw.line((start, end), fill=color, width=width)
    x1, y1 = start
    x2, y2 = end
    angle = math.atan2(y2 - y1, x2 - x1)
    size = 18
    pts = [
        (x2, y2),
        (x2 - size * math.cos(angle - 0.45), y2 - size * math.sin(angle - 0.45)),
        (x2 - size * math.cos(angle + 0.45), y2 - size * math.sin(angle + 0.45)),
    ]
    draw.polygon(pts, fill=color)


def title(draw: ImageDraw.ImageDraw, text: str) -> None:
    draw_text_center(draw, (W / 2, 72), text, FONTS["title"], COLORS["navy"])


def save(image: Image.Image, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    image.save(OUT_DIR / name, dpi=(300, 300), quality=95)


def fig1() -> None:
    img, d = canvas()
    title(d, "融合差分隐私的联邦学习医疗影像协同诊断系统总体架构")
    row_y = [170, 310, 450, 590, 730]
    cols = [60, 255, 575, 895]
    widths = [150, 260, 260, 280]
    labels = ["医院", "本地皮肤镜图像", "本地模型训练", "更新裁剪与加噪"]
    row_colors = [COLORS["blue"], COLORS["green"], COLORS["cyan"], COLORS["blue"], COLORS["green"]]
    for i, y in enumerate(row_y, start=1):
        color = row_colors[i - 1]
        for j, x in enumerate(cols):
            box = (x, y, x + widths[j], y + 90)
            rounded(d, box, fill="#f7fbff", outline=color, radius=14)
            text = f"医院 {i}" if j == 0 else labels[j]
            draw_text_center(d, ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2), text, FONTS["small_bold"], COLORS["black"])
            if j < len(cols) - 1:
                arrow(d, (x + widths[j] + 12, y + 45), (cols[j + 1] - 12, y + 45), color)
        arrow(d, (1175, y + 45), (1500, 455), color)

    draw_text_center(d, (1370, 390), "上传隐私保护后的\n模型更新", FONTS["section"], COLORS["blue"])
    rounded(d, (1230, 520, 1470, 675), fill="#fff7f7", outline=COLORS["red"], radius=20, width=4)
    draw_text_center(d, (1350, 585), "不上传\n原始图像", FONTS["section"], COLORS["red"])

    rounded(d, (1535, 170, 1985, 775), fill="#f8fbff", outline=COLORS["blue"], radius=22, width=4)
    draw_text_center(d, (1760, 215), "联邦服务器", FONTS["section"], COLORS["navy"])
    server_steps = ["FedAvg 加权聚合", "全局模型更新", "模型下发"]
    sy = 285
    for step in server_steps:
        rounded(d, (1590, sy, 1930, sy + 95), fill=COLORS["white"], outline=COLORS["blue"], radius=14)
        draw_text_center(d, (1760, sy + 47), step, FONTS["body_bold"], COLORS["black"])
        if sy < 530:
            arrow(d, (1760, sy + 100), (1760, sy + 150), COLORS["blue"])
        sy += 145

    rounded(d, (2050, 170, 2340, 775), fill="#fbfdff", outline="#86a9e8", radius=22, width=3)
    draw_text_center(d, (2195, 215), "差分隐私机制", FONTS["section"], COLORS["navy"])
    dp_items = ["Gaussian DP", "Laplace DP", "Hybrid DP", "Adaptive Hybrid DP"]
    dp_colors = [COLORS["blue"], COLORS["green"], COLORS["cyan"], COLORS["purple"]]
    for idx, (item, color) in enumerate(zip(dp_items, dp_colors)):
        y = 285 + idx * 115
        rounded(d, (2070, y, 2325, y + 75), fill="#ffffff", outline=color, radius=14)
        label = "Adaptive\nHybrid DP" if item == "Adaptive Hybrid DP" else item
        draw_text_center(d, (2197, y + 38), label, FONTS["tiny_bold"] if "\n" in label else FONTS["small_bold"], color)

    rounded(d, (60, 920, 2340, 1250), fill="#fbfdff", outline=COLORS["line"], radius=22, width=4)
    draw_text_center(d, (1200, 970), "实验分析与可视化看板", FONTS["section"], COLORS["navy"])
    metrics = ["Accuracy", "AUC", "Epsilon", "Attack Accuracy"]
    for i, item in enumerate(metrics):
        x = 145 + i * 560
        rounded(d, (x, 1035, x + 450, 1185), fill="#ffffff", outline=dp_colors[i], radius=18)
        draw_text_center(d, (x + 225, 1110), item, FONTS["body_bold"], dp_colors[i])
    save(img, "fig1_system_architecture.png")


def fig2() -> None:
    img, d = canvas()
    title(d, "联邦学习训练流程")
    steps = [
        "数据预处理\n与客户端划分",
        "初始化\n全局模型",
        "服务端\n选择客户端",
        "客户端下载\n全局模型",
        "客户端\n本地训练",
        "计算模型\n更新 Δw",
        "L2 范数\n裁剪",
        "加入差分\n隐私噪声",
        "FedAvg\n加权聚合",
        "验证集/测试集\n评估",
        "判断通信轮次\n是否结束",
    ]
    x0, y0, card_w, card_h, gap = 50, 255, 195, 470, 15
    for i, step in enumerate(steps, start=1):
        x = x0 + (i - 1) * (card_w + gap)
        color = COLORS["blue"] if i <= 6 else COLORS["green"]
        rounded(d, (x, y0, x + card_w, y0 + card_h), fill="#fbfdff", outline=color, radius=20)
        d.ellipse((x + 58, y0 + 32, x + 138, y0 + 112), fill=color)
        draw_text_center(d, (x + 98, y0 + 72), str(i), FONTS["section"], "#ffffff")
        draw_wrapped_center(d, (x + 10, y0 + 150, card_w - 20, 150), step, FONTS["small_bold"])
        if i < len(steps):
            arrow(d, (x + card_w + 2, y0 + 235), (x + card_w + gap - 4, y0 + 235), color)
    rounded(d, (60, 830, 470, 1030), fill="#f7fbff", outline="#86a9e8", radius=20)
    draw_text_center(d, (265, 930), "原始医疗图像\n始终保留在本地", FONTS["section"], COLORS["navy"])
    d.line((2070, 725, 2070, 1040, 650, 1040, 650, 735), fill=COLORS["blue"], width=5)
    arrow(d, (650, 1040), (650, 735), COLORS["blue"])
    draw_text_center(d, (1210, 980), "否，则进入下一轮通信", FONTS["body_bold"], COLORS["navy"])
    save(img, "fig2_federated_training_flow.png")


def fig3() -> None:
    img, d = canvas()
    title(d, "混合差分隐私扰动机制")
    rounded(d, (70, 445, 380, 610), fill="#f7fbff", outline=COLORS["blue"], radius=18)
    draw_text_center(d, (225, 527), "客户端模型更新\nΔw", FONTS["section"], COLORS["navy"])
    arrow(d, (380, 527), (490, 527))
    rounded(d, (490, 445, 790, 610), fill="#f5fff7", outline=COLORS["green"], radius=18)
    draw_text_center(d, (640, 527), "L2 范数裁剪\nClip(Δw, C)", FONTS["section"], COLORS["green"])
    arrow(d, (790, 527), (910, 365))
    arrow(d, (790, 527), (910, 715))
    rounded(d, (910, 230, 1280, 500), fill="#f7fbff", outline=COLORS["blue"], radius=20)
    draw_text_center(d, (1095, 300), "Gaussian 噪声\nN(0, σ²)", FONTS["section"], COLORS["blue"])
    d.arc((985, 350, 1205, 470), 180, 360, fill=COLORS["blue"], width=5)
    rounded(d, (910, 610, 1280, 880), fill="#f6fff7", outline=COLORS["green"], radius=20)
    d.line((1000, 810, 1095, 735, 1190, 810), fill=COLORS["green"], width=5)
    draw_text_center(d, (1095, 680), "Laplace 噪声\nLap(0, b)", FONTS["section"], COLORS["green"])
    arrow(d, (1280, 365), (1435, 527))
    arrow(d, (1280, 745), (1435, 527))
    rounded(d, (1435, 430, 1710, 625), fill="#fff8ef", outline=COLORS["orange"], radius=18)
    draw_text_center(d, (1573, 527), "α · Gaussian\n+\n(1-α) · Laplace", FONTS["body_bold"], COLORS["orange"])
    arrow(d, (1710, 527), (1840, 527))
    rounded(d, (1840, 445, 2090, 610), fill="#f7fbff", outline=COLORS["blue"], radius=18)
    draw_text_center(d, (1965, 527), "隐私保护后的\n更新 Δw'", FONTS["section"], COLORS["navy"])
    arrow(d, (2090, 527), (2240, 527))
    rounded(d, (2240, 445, 2350, 610), fill="#f7fbff", outline=COLORS["blue"], radius=18)
    draw_text_center(d, (2295, 527), "FedAvg\n聚合", FONTS["small_bold"], COLORS["navy"])
    rounded(d, (720, 980, 1680, 1155), fill="#fff8ef", outline=COLORS["orange"], radius=18, width=4)
    draw_text_center(d, (1200, 1025), "自适应 α 调整", FONTS["section"], COLORS["orange"])
    inputs = ["轮次进度", "训练损失", "更新敏感度"]
    for i, text in enumerate(inputs):
        x = 790 + i * 295
        rounded(d, (x, 1070, x + 230, 1135), fill=COLORS["white"], outline=COLORS["green"], radius=14)
        draw_text_center(d, (x + 115, 1102), text, FONTS["small_bold"], COLORS["green"])
    arrow(d, (1570, 980), (1570, 625), COLORS["orange"], width=4)
    save(img, "fig3_hybrid_dp_mechanism.png")


def fig4() -> None:
    img, d = canvas()
    title(d, "HAM10000 数据预处理与客户端划分")
    panels = [
        ("原始数据", "HAM10000_metadata.csv\nHAM10000_images_part_1\nHAM10000_images_part_2"),
        ("二分类标签映射", "mel / bcc / akiec\n→ malignant\nnv / bkl / df / vasc\n→ benign"),
        ("lesion_id 病灶级隔离", "同一 lesion_id 的所有图像\n仅出现在一个 split 中"),
        ("生成数据集划分", "train 80%\nval 10%\ntest 10%"),
        ("数据质量审计", "图像可读性检查\n（PIL verify）\n跨 split 文件名重复检查\n跨 split SHA256 哈希重复检查\nlesion_id 分组泄漏检查"),
    ]
    x, y, pw, ph, gap = 45, 190, 335, 760, 35
    for i, (head, body) in enumerate(panels):
        bx = x + i * (pw + gap)
        header_box(d, (bx, y, bx + pw, y + ph), head, body, COLORS["green"] if i else COLORS["blue"], fnt_body=FONTS["tiny_bold"])
        if i < len(panels) - 1:
            arrow(d, (bx + pw + 5, y + ph // 2), (bx + pw + gap - 8, y + ph // 2), COLORS["green"])
    rounded(d, (1910, 190, 2350, 950), fill="#fffaf5", outline=COLORS["orange"], radius=20, width=4)
    draw_text_center(d, (2130, 240), "Dirichlet Non-IID\n客户端划分", FONTS["section"], COLORS["orange"])
    draw_text_center(d, (2130, 318), "按训练集标签使用 Dirichlet(α) 分配", FONTS["small"], COLORS["black"])
    bar_y = 390
    ratios = [0.82, 0.55, 0.42, 0.30, 0.15]
    for i, r in enumerate(ratios, start=1):
        yy = bar_y + (i - 1) * 95
        draw_text_center(d, (1995, yy + 28), f"客户端 {i}\n(C{i})", FONTS["tiny"], COLORS["orange"])
        d.rounded_rectangle((2075, yy, 2305, yy + 56), radius=10, fill="#fff4ee", outline=COLORS["orange"], width=2)
        benign_w = int(230 * r)
        d.rectangle((2075, yy, 2075 + benign_w, yy + 56), fill="#46a657")
        d.rectangle((2075 + benign_w, yy, 2305, yy + 56), fill="#e45858")
    draw_text_center(d, (2130, 870), "类别比例因 α 不同而变化", FONTS["small_bold"], COLORS["orange"])
    rounded(d, (150, 1045, 2250, 1215), fill="#fbfdff", outline="#86a9e8", radius=22)
    legend = ["标签映射", "病灶隔离", "数据划分", "质量审计", "客户端"]
    for i, item in enumerate(legend):
        draw_text_center(d, (390 + i * 400, 1130), item, FONTS["body_bold"], COLORS["black"])
    save(img, "fig4_ham10000_preprocessing_partition.png")


def fig5() -> None:
    img, d = canvas()
    title(d, "医学图像二分类模型结构")
    x, y, gap = 45, 245, 25
    widths = [285, 325, 500, 260, 230, 270, 310]
    heads = [
        "1 输入皮肤镜图像",
        "2 数据增强",
        "3 预训练主干网络",
        "4 特征聚合",
        "5 正则化",
        "6 分类头",
        "7 输出",
    ]
    bodies = [
        "输入尺寸由配置决定\n224×224（主实验）\n300×300（B3）\n380×380（B4）\nConvNeXt 配置\n256、384、448",
        "RandomResizedCrop\n或 Resize\n水平/垂直翻转\n随机旋转\nColorJitter\nRandAugment\nRandomErasing\n（高精度）",
        "ResNet18（主实验）\nEfficientNet-B3 / B4\n（高精度补充）\nConvNeXt-Tiny / Small\n（高精度补充）",
        "全局池化\n或特征展平",
        "Dropout",
        "Linear 分类层\nlogits",
        "logits\nSoftmax 概率\nbenign（良性）\nmalignant（恶性）",
    ]
    cursor = x
    for i, (w, head, body) in enumerate(zip(widths, heads, bodies)):
        color = COLORS["blue"] if i < 4 else COLORS["green"] if i < 6 else COLORS["red"]
        header_box(d, (cursor, y, cursor + w, y + 780), head, body, color, fnt_header=FONTS["small_bold"], fnt_body=FONTS["tiny_bold"])
        if i < len(widths) - 1:
            arrow(d, (cursor + w + 5, y + 390), (cursor + w + gap - 8, y + 390), color)
        cursor += w + gap
    save(img, "fig5_model_architecture.png")


def fig6() -> None:
    img, d = canvas()
    title(d, "成员推断攻击评估流程")
    left_boxes = [
        (60, 230, 420, 485, "成员样本\n训练集", COLORS["blue"]),
        (60, 610, 420, 865, "非成员样本\n测试集", COLORS["blue"]),
    ]
    for box in left_boxes:
        rounded(d, box[:4], fill="#f7fbff", outline=box[5], radius=18)
        draw_text_center(d, ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2), box[4], FONTS["section"], COLORS["navy"])
        arrow(d, (box[2], (box[1] + box[3]) // 2), (560, 548), COLORS["blue"])
    rounded(d, (560, 410, 860, 685), fill="#f7fbff", outline=COLORS["blue"], radius=18)
    draw_text_center(d, (710, 520), "已训练分类模型\nfinal_model.pt", FONTS["body_bold"], COLORS["navy"])
    arrow(d, (860, 548), (980, 548))
    rounded(d, (980, 395, 1285, 700), fill="#f7fbff", outline=COLORS["blue"], radius=18)
    draw_text_center(d, (1132, 505), "collect_predictions\n输出 Softmax 概率", FONTS["body_bold"], COLORS["navy"])
    arrow(d, (1285, 548), (1405, 548), COLORS["purple"])
    rounded(d, (1405, 260, 1750, 835), fill="#fbf7ff", outline=COLORS["purple"], radius=18)
    draw_text_center(d, (1578, 310), "攻击特征向量", FONTS["section"], COLORS["purple"])
    features = ["最大置信度", "预测 margin", "真实类别置信度", "熵", "交叉熵损失"]
    for i, feat in enumerate(features, start=1):
        yy = 360 + (i - 1) * 85
        rounded(d, (1445, yy, 1710, yy + 58), fill="#ffffff", outline="#b99de0", radius=12, width=2)
        draw_text_center(d, (1578, yy + 29), f"{i}. {feat}", FONTS["small_bold"], COLORS["black"])
    draw_text_center(d, (1578, 790), "成员=1，非成员=0", FONTS["small_bold"], COLORS["purple"])
    arrow(d, (1750, 548), (1870, 548), COLORS["purple"])
    rounded(d, (1870, 395, 2140, 700), fill="#f5fff7", outline=COLORS["green"], radius=18)
    draw_text_center(d, (2005, 475), "train_test_split\n测试比例 0.35", FONTS["small_bold"], COLORS["green"])
    draw_text_center(d, (2005, 590), "GradientBoostingClassifier", FONTS["micro_bold"], COLORS["green"])
    arrow(d, (2140, 548), (2250, 548), COLORS["green"])
    rounded(d, (2250, 330, 2370, 760), fill="#f5fff7", outline=COLORS["green"], radius=18)
    draw_text_center(d, (2310, 430), "Attack\nAccuracy", FONTS["small_bold"], COLORS["green"])
    draw_text_center(d, (2310, 550), "Attack\nAUC", FONTS["small_bold"], COLORS["green"])
    draw_text_center(d, (2310, 690), "50%\n随机参考线", FONTS["tiny"], COLORS["green"])
    rounded(d, (270, 1030, 2130, 1195), fill="#ffffff", outline="#9aa4b5", radius=18, width=2)
    draw_text_center(d, (1200, 1112), "数据流 →  概率输出 →  特征提取 →  攻击分类器 →  攻击评估指标", FONTS["body_bold"], COLORS["gray"])
    save(img, "fig6_membership_inference_attack.png")


def fig7() -> None:
    img, d = canvas()
    title(d, "实验方案设计总览")
    panels = [
        ((70, 190, 820, 520), "1 主实验对比", "Centralized\nNoDP\nGaussian\nLaplace\nHybrid\nAdaptive Hybrid", COLORS["blue"]),
        ((1580, 190, 2330, 520), "2 高精度补充实验", "EfficientNet-B3/B4\nConvNeXt-Tiny / ConvNeXt-Small\n长轮次单客户端无隐私", COLORS["green"]),
        ((70, 610, 820, 940), "3 消融实验", "Non-IID α\n客户端数量\n裁剪阈值\n噪声强度", COLORS["orange"]),
        ((1580, 610, 2330, 940), "4 隐私攻击评估", "成员推断攻击\nAttack Accuracy\nAttack AUC", COLORS["purple"]),
    ]
    for box, head, body, color in panels:
        sx = (box[0] + box[2]) // 2
        sy = (box[1] + box[3]) // 2
        arrow(d, (1200, 530), (sx, sy), color, width=5)
    for box, head, body, color in panels:
        header_box(d, box, head, body, color, fnt_header=FONTS["section"], fnt_body=FONTS["body_bold"])
    rounded(d, (910, 360, 1490, 700), fill="#f7fbff", outline=COLORS["blue"], radius=24, width=4)
    draw_text_center(d, (1200, 475), "FedMedDP 系统", FONTS["subtitle"], COLORS["navy"])
    draw_text_center(d, (1200, 590), "HAM10000 二分类\n联邦学习 + 差分隐私", FONTS["body_bold"], COLORS["navy"])
    rounded(d, (70, 1010, 2330, 1245), fill="#fbfdff", outline=COLORS["line"], radius=22, width=4)
    draw_text_center(d, (1200, 1065), "评价指标", FONTS["section"], COLORS["navy"])
    metrics = ["Accuracy", "AUC", "F1", "Balanced Accuracy", "Sensitivity", "Specificity", "ε（近似）", "Attack Accuracy"]
    for i, item in enumerate(metrics):
        x = 110 + i * 280
        rounded(d, (x, 1125, x + 240, 1205), fill="#ffffff", outline=[COLORS["blue"], COLORS["cyan"], COLORS["green"], COLORS["orange"], COLORS["red"], COLORS["purple"], COLORS["cyan"], "#7a4b4b"][i], radius=14)
        draw_text_center(d, (x + 120, 1165), item, FONTS["small_bold"], COLORS["black"])
    save(img, "fig7_experiment_design.png")


def main() -> None:
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
    fig6()
    fig7()
    print(f"已生成论文图：{OUT_DIR}")


if __name__ == "__main__":
    main()
