from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "figures"
SUMMARY_CSV = ROOT / "outputs" / "experiment_summary.csv"
MULTI_SEED_CSV = ROOT / "outputs" / "multi_seed_summary.csv"
ENSEMBLE_SUMMARY = ROOT / "outputs" / "ham10000_accuracy90_weighted_5models_sens70" / "summary.json"

W, H = 2400, 1350

FONT_CANDIDATES = [
    Path(r"C:\Windows\Fonts\msyh.ttc"),
    Path(r"C:\Windows\Fonts\simhei.ttf"),
    Path(r"C:\Windows\Fonts\simsun.ttc"),
]
BOLD_CANDIDATES = [
    Path(r"C:\Windows\Fonts\msyhbd.ttc"),
    Path(r"C:\Windows\Fonts\simhei.ttf"),
    Path(r"C:\Windows\Fonts\msyh.ttc"),
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
    "grid": "#d8e6ff",
    "panel": "#f7fbff",
    "white": "#ffffff",
    "black": "#172033",
    "bg_top": "#ffffff",
    "text": "#172033",
    "muted": "#607080",
    "accent": "#1455b8",
    "accent_2": "#0f8fa8",
    "warm": "#d95f02",
    "warm_dark": "#c82e2e",
    "success": "#237a35",
    "light": "#f7fbff",
}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    for path in (BOLD_CANDIDATES if bold else FONT_CANDIDATES):
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


FONTS = {
    "title": font(58, True),
    "subtitle": font(30),
    "axis": font(28),
    "label": font(26, True),
    "small": font(23),
    "tiny": font(20),
    "tiny_bold": font(20, True),
}


def text_size(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.FreeTypeFont) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=fnt)
    return box[2] - box[0], box[3] - box[1]


def text_center(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    fnt: ImageFont.FreeTypeFont,
    fill: str = COLORS["text"],
) -> None:
    x, y = xy
    lines = text.split("\n")
    heights = [text_size(draw, line, fnt)[1] for line in lines]
    total_h = sum(heights) + (len(lines) - 1) * 8
    cursor = y - total_h / 2
    for line, line_h in zip(lines, heights):
        line_w, _ = text_size(draw, line, fnt)
        draw.text((x - line_w / 2, cursor), line, font=fnt, fill=fill)
        cursor += line_h + 8


def base_canvas(title: str, subtitle: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (W, H), COLORS["white"])
    draw = ImageDraw.Draw(image)
    _ = title, subtitle
    return image, draw


def rounded(
    draw: ImageDraw.ImageDraw,
    box: tuple[float, float, float, float],
    fill: str,
    outline: str | None = None,
    radius: int = 10,
    width: int = 1,
) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def draw_y_grid(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    ymin: float,
    ymax: float,
    ticks: list[float],
    formatter=lambda value: f"{value:.2f}",
    right: bool = False,
) -> None:
    x1, y1, x2, y2 = box
    draw.rounded_rectangle(
        (x1 - 26, y1 - 28, x2 + 26, y2 + 28),
        radius=20,
        fill=COLORS["panel"],
        outline="#86a9e8",
        width=2,
    )
    draw.line((x1, y2, x2, y2), fill=COLORS["line"], width=3)
    draw.line((x2 if right else x1, y1, x2 if right else x1, y2), fill=COLORS["line"], width=3)
    for tick in ticks:
        y = y2 - (tick - ymin) / (ymax - ymin) * (y2 - y1)
        draw.line((x1, y, x2, y), fill=COLORS["grid"], width=2)
        label = formatter(tick)
        if right:
            draw.text((x2 + 16, y - 16), label, font=FONTS["small"], fill=COLORS["gray"])
        else:
            draw.text((x1 - 82, y - 16), label, font=FONTS["small"], fill=COLORS["gray"])


def y_at(box: tuple[int, int, int, int], ymin: float, ymax: float, value: float) -> float:
    return box[3] - (value - ymin) / (ymax - ymin) * (box[3] - box[1])


def load_experiment_rows() -> dict[str, dict[str, str]]:
    with SUMMARY_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        return {row["experiment_name"]: row for row in csv.DictReader(handle)}


def load_multi_seed_row() -> dict[str, str]:
    with MULTI_SEED_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return rows[0]


def load_ensemble_summary() -> dict[str, object]:
    with ENSEMBLE_SUMMARY.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def val(row: dict[str, str], key: str) -> float:
    return float(row[key])


def draw_legend(draw: ImageDraw.ImageDraw, items: list[tuple[str, str]], x: int, y: int, gap: int = 260) -> None:
    for i, (label, color) in enumerate(items):
        lx = x + i * gap
        rounded(draw, (lx, y, lx + 42, y + 42), fill=color, radius=8)
        draw.text((lx + 54, y + 5), label, font=FONTS["label"], fill=COLORS["text"])


def draw_grouped_bars(
    draw: ImageDraw.ImageDraw,
    chart: tuple[int, int, int, int],
    group_labels: list[str],
    series: list[tuple[str, list[float], str]],
    ymin: float,
    ymax: float,
    ticks: list[float],
    legend_x: int = 1450,
) -> None:
    draw_y_grid(draw, chart, ymin, ymax, ticks)
    draw_legend(draw, [(name, color) for name, _, color in series], legend_x, 115, gap=280)
    group_w = (chart[2] - chart[0]) / len(group_labels)
    bar_w = min(80, max(42, int(group_w / (len(series) + 2))))
    offsets = [
        (i - (len(series) - 1) / 2) * (bar_w + 14)
        for i in range(len(series))
    ]
    for i, label in enumerate(group_labels):
        gx = chart[0] + group_w * i + group_w / 2
        for j, (_, values, color) in enumerate(series):
            value = values[i]
            x1 = gx + offsets[j] - bar_w / 2
            y1 = y_at(chart, ymin, ymax, value)
            rounded(draw, (x1, y1, x1 + bar_w, chart[3]), fill=color, radius=10)
            text_center(draw, (x1 + bar_w / 2, y1 - 26), f"{value:.3f}", FONTS["tiny"], color)
        text_center(draw, (gx, chart[3] + 92), label, FONTS["label"], COLORS["text"])


def draw_line_chart(
    draw: ImageDraw.ImageDraw,
    chart: tuple[int, int, int, int],
    x_labels: list[str],
    series: list[tuple[str, list[float], str]],
    ymin: float,
    ymax: float,
    ticks: list[float],
    legend_x: int = 1440,
    legend_gap: int = 260,
) -> None:
    draw_y_grid(draw, chart, ymin, ymax, ticks)
    draw_legend(draw, [(name, color) for name, _, color in series], legend_x, 115, gap=legend_gap)
    x_count = len(x_labels)
    x_positions = [
        chart[0] + (chart[2] - chart[0]) * i / max(x_count - 1, 1)
        for i in range(x_count)
    ]
    for label, x in zip(x_labels, x_positions):
        draw.line((x, chart[3], x, chart[3] + 10), fill=COLORS["line"], width=3)
        text_center(draw, (x, chart[3] + 58), label, FONTS["label"], COLORS["text"])
    for _, values, color in series:
        points = [(x, y_at(chart, ymin, ymax, value)) for x, value in zip(x_positions, values)]
        draw.line(points, fill=color, width=6)
        for (x, y), value in zip(points, values):
            draw.ellipse((x - 11, y - 11, x + 11, y + 11), fill=color)
            text_center(draw, (x, y - 34), f"{value:.3f}", FONTS["tiny"], color)


def save(image: Image.Image, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    image.save(OUT_DIR / name, dpi=(300, 300), quality=95)


def fig10_client_distribution() -> None:
    image, draw = base_canvas(
        "主实验客户端 Non-IID 类别分布",
        "Dirichlet α = 1.0 下，各客户端恶性样本比例差异明显。",
    )
    chart = (230, 220, 2240, 1030)
    draw_y_grid(draw, chart, 0, 2800, [0, 700, 1400, 2100, 2800], formatter=lambda value: f"{int(value)}")
    clients = [
        ("Client 0", 1469, 841),
        ("Client 1", 1089, 114),
        ("Client 2", 2640, 77),
        ("Client 3", 888, 530),
        ("Client 4", 375, 2),
    ]
    draw_legend(draw, [("benign", COLORS["green"]), ("malignant", COLORS["red"])], 1560, 190, gap=250)
    group_w = (chart[2] - chart[0]) / len(clients)
    bar_w = 175
    for i, (label, benign, malignant) in enumerate(clients):
        gx = chart[0] + group_w * i + group_w / 2
        total = benign + malignant
        y_total = y_at(chart, 0, 2800, total)
        y_benign = y_at(chart, 0, 2800, benign)
        rounded(draw, (gx - bar_w / 2, y_total, gx + bar_w / 2, chart[3]), fill=COLORS["green"], radius=12)
        rounded(draw, (gx - bar_w / 2, y_total, gx + bar_w / 2, y_benign), fill=COLORS["red"], radius=12)
        text_center(draw, (gx, y_total - 28), f"{malignant / total:.1%}", FONTS["tiny_bold"], COLORS["red"])
        text_center(draw, (gx, chart[3] + 60), label, FONTS["label"])
        text_center(draw, (gx, chart[3] + 105), f"n={total}", FONTS["small"], COLORS["gray"])
    save(image, "客户端NonIID类别分布.png")


def fig11_sensitivity_specificity() -> None:
    rows = load_experiment_rows()
    names = [
        ("Centralized", "ham10000_centralized"),
        ("NoDP", "ham10000_nodp"),
        ("Gaussian", "ham10000_gaussian"),
        ("Laplace", "ham10000_laplace"),
        ("Hybrid", "ham10000_hybrid"),
        ("Adaptive\nHybrid", "ham10000_hybrid_adaptive"),
    ]
    labels = [name for name, _ in names]
    sensitivity = [val(rows[key], "test_sensitivity_at_best_val") for _, key in names]
    specificity = [val(rows[key], "test_specificity_at_best_val") for _, key in names]
    image, draw = base_canvas(
        "主实验灵敏度与特异度对比",
        "Sensitivity 反映恶性识别能力，Specificity 反映良性识别能力。",
    )
    draw_grouped_bars(
        draw,
        (190, 220, 2260, 1030),
        labels,
        [
            ("Sensitivity", sensitivity, COLORS["red"]),
            ("Specificity", specificity, COLORS["blue"]),
        ],
        0.45,
        1.00,
        [0.50, 0.60, 0.70, 0.80, 0.90, 1.00],
        legend_x=1510,
    )
    save(image, "灵敏度特异度对比.png")


def fig12_high_accuracy_comparison() -> None:
    rows = load_experiment_rows()
    ensemble = load_ensemble_summary()
    labels = ["Gaussian\n主实验", "ConvNeXt-Small\n多种子均值", "五模型\n加权集成"]
    acc = [
        val(rows["ham10000_gaussian"], "test_accuracy_at_best_val"),
        val(rows["ham10000_literature_target"], "test_accuracy_at_best_val"),
        float(ensemble["test_accuracy_at_best_val"]),
    ]
    auc = [
        val(rows["ham10000_gaussian"], "test_auc_at_best_val"),
        val(rows["ham10000_literature_target"], "test_auc_at_best_val"),
        float(ensemble["test_auc_at_best_val"]),
    ]
    sens = [
        val(rows["ham10000_gaussian"], "test_sensitivity_at_best_val"),
        val(rows["ham10000_literature_target"], "test_sensitivity_at_best_val"),
        float(ensemble["test_sensitivity_at_best_val"]),
    ]
    image, draw = base_canvas(
        "主实验与高精度补充实验性能对比",
        "高精度补充实验展示系统性能上限，不属于隐私机制主实验。",
    )
    draw_grouped_bars(
        draw,
        (230, 220, 2220, 1030),
        labels,
        [
            ("Accuracy", acc, COLORS["blue"]),
            ("AUC", auc, COLORS["green"]),
            ("Sensitivity", sens, COLORS["red"]),
        ],
        0.50,
        1.00,
        [0.50, 0.60, 0.70, 0.80, 0.90, 1.00],
        legend_x=1390,
    )
    save(image, "高精度补充实验性能对比.png")


def fig13_multi_seed_stability() -> None:
    row = load_multi_seed_row()
    metrics = [
        ("Accuracy", "test_accuracy_at_best_val"),
        ("AUC", "test_auc_at_best_val"),
        ("F1", "test_f1_at_best_val"),
        ("Balanced\nAccuracy", "test_balanced_accuracy_at_best_val"),
        ("Sensitivity", "test_sensitivity_at_best_val"),
        ("Specificity", "test_specificity_at_best_val"),
    ]
    means = [float(row[f"{key}_mean"]) for _, key in metrics]
    stds = [float(row[f"{key}_std"]) for _, key in metrics]
    colors = [COLORS["blue"], COLORS["green"], COLORS["orange"], COLORS["purple"], COLORS["red"], COLORS["cyan"]]
    image, draw = base_canvas(
        "多随机种子稳定性分析",
        "柱形为均值，误差线为标准差。",
    )
    chart = (200, 220, 2240, 1030)
    ymin, ymax = 0.50, 1.00
    draw_y_grid(draw, chart, ymin, ymax, [0.50, 0.60, 0.70, 0.80, 0.90, 1.00])
    group_w = (chart[2] - chart[0]) / len(metrics)
    bar_w = 125
    for i, ((label, _), mean, std, color) in enumerate(zip(metrics, means, stds, colors)):
        gx = chart[0] + group_w * i + group_w / 2
        y_mean = y_at(chart, ymin, ymax, mean)
        rounded(draw, (gx - bar_w / 2, y_mean, gx + bar_w / 2, chart[3]), fill=color, radius=12)
        y_low = y_at(chart, ymin, ymax, mean - std)
        y_high = y_at(chart, ymin, ymax, mean + std)
        draw.line((gx, y_high, gx, y_low), fill=COLORS["text"], width=4)
        draw.line((gx - 32, y_high, gx + 32, y_high), fill=COLORS["text"], width=4)
        draw.line((gx - 32, y_low, gx + 32, y_low), fill=COLORS["text"], width=4)
        text_center(draw, (gx, y_high - 36), f"{mean:.3f}\n±{std:.3f}", FONTS["tiny"], color)
        text_center(draw, (gx, chart[3] + 82), label, FONTS["label"])
    save(image, "多随机种子稳定性分析.png")


def fig14_client_ablation() -> None:
    rows = load_experiment_rows()
    labels = ["3", "5", "10", "20"]
    keys = [
        "ham10000_ablation_clients_3",
        "ham10000_ablation_clients_5",
        "ham10000_ablation_clients_10",
        "ham10000_ablation_clients_20",
    ]
    acc = [val(rows[key], "test_accuracy_at_best_val") for key in keys]
    auc = [val(rows[key], "test_auc_at_best_val") for key in keys]
    comm = [val(rows[key], "total_communication_mb") for key in keys]
    image, draw = base_canvas(
        "客户端数量消融趋势",
        "客户端数量增加会显著提高通信量，但 Accuracy 未持续提升。",
    )
    left = (180, 330, 1180, 960)
    right = (1370, 330, 2240, 960)
    draw.text((180, 170), "分类性能", font=FONTS["label"], fill=COLORS["navy"])
    draw_line_chart(
        draw,
        (180, 240, 1180, 1030),
        labels,
        [("Accuracy", acc, COLORS["blue"]), ("AUC", auc, COLORS["green"])],
        0.84,
        0.91,
        [0.84, 0.86, 0.88, 0.90],
        legend_x=600,
        legend_gap=230,
    )
    draw.text((1370, 170), "累计通信量", font=FONTS["label"], fill=COLORS["navy"])
    right = (1370, 240, 2240, 1030)
    draw_y_grid(draw, right, 0, 70000, [0, 20000, 40000, 60000], formatter=lambda value: f"{int(value / 1000)}k")
    group_w = (right[2] - right[0]) / len(labels)
    for i, (label, value) in enumerate(zip(labels, comm)):
        gx = right[0] + group_w * i + group_w / 2
        y = y_at(right, 0, 70000, value)
        rounded(draw, (gx - 75, y, gx + 75, right[3]), fill=COLORS["orange"], radius=12)
        text_center(draw, (gx, y - 28), f"{value / 1000:.1f}k", FONTS["tiny"], COLORS["orange"])
        text_center(draw, (gx, right[3] + 58), label, FONTS["label"])
    save(image, "客户端数量消融趋势.png")


def fig15_noniid_ablation() -> None:
    rows = load_experiment_rows()
    labels = ["0.1", "0.5", "1.0", "5.0", "10.0"]
    keys = [
        "ham10000_ablation_alpha_0_1",
        "ham10000_ablation_alpha_0_5",
        "ham10000_ablation_alpha_1_0",
        "ham10000_ablation_alpha_5_0",
        "ham10000_ablation_alpha_10_0",
    ]
    image, draw = base_canvas(
        "Dirichlet α 对联邦性能的影响",
        "α 越小客户端分布越不均衡，强 Non-IID 下 Accuracy 降低明显。",
    )
    draw_line_chart(
        draw,
        (220, 240, 2240, 1030),
        labels,
        [
            ("Accuracy", [val(rows[key], "test_accuracy_at_best_val") for key in keys], COLORS["blue"]),
            ("AUC", [val(rows[key], "test_auc_at_best_val") for key in keys], COLORS["green"]),
            ("Sensitivity", [val(rows[key], "test_sensitivity_at_best_val") for key in keys], COLORS["red"]),
        ],
        0.48,
        0.92,
        [0.50, 0.60, 0.70, 0.80, 0.90],
        legend_x=1310,
    )
    save(image, "Dirichlet_alpha消融趋势.png")


def fig16_clip_ablation() -> None:
    rows = load_experiment_rows()
    labels = ["1.0", "3.0", "5.0", "10.0"]
    keys = [
        "ham10000_ablation_clip_1_0",
        "ham10000_ablation_clip_3_0",
        "ham10000_ablation_clip_5_0",
        "ham10000_ablation_clip_10_0",
    ]
    image, draw = base_canvas(
        "裁剪阈值对模型性能的影响",
        "裁剪阈值影响有效更新幅度，需要在学习信号和敏感度控制之间折中。",
    )
    draw_line_chart(
        draw,
        (220, 240, 2240, 1030),
        labels,
        [
            ("Accuracy", [val(rows[key], "test_accuracy_at_best_val") for key in keys], COLORS["blue"]),
            ("F1", [val(rows[key], "test_f1_at_best_val") for key in keys], COLORS["orange"]),
            ("Sensitivity", [val(rows[key], "test_sensitivity_at_best_val") for key in keys], COLORS["red"]),
        ],
        0.45,
        0.90,
        [0.50, 0.60, 0.70, 0.80, 0.90],
        legend_x=1310,
    )
    save(image, "裁剪阈值消融趋势.png")


def fig17_noise_ablation() -> None:
    rows = load_experiment_rows()
    labels = ["low\nε=4350.0", "base\nε=2175.0", "high\nε=1087.5"]
    keys = [
        "ham10000_hybrid_noise_low",
        "ham10000_hybrid",
        "ham10000_hybrid_noise_high",
    ]
    image, draw = base_canvas(
        "混合噪声强度消融对比",
        "单次结果不呈严格单调关系，应结合多种子验证。",
    )
    draw_grouped_bars(
        draw,
        (260, 220, 2180, 1030),
        labels,
        [
            ("Accuracy", [val(rows[key], "test_accuracy_at_best_val") for key in keys], COLORS["blue"]),
            ("AUC", [val(rows[key], "test_auc_at_best_val") for key in keys], COLORS["green"]),
            ("F1", [val(rows[key], "test_f1_at_best_val") for key in keys], COLORS["orange"]),
        ],
        0.74,
        0.92,
        [0.75, 0.80, 0.85, 0.90],
        legend_x=1420,
    )
    save(image, "混合噪声强度消融对比.png")


def main() -> None:
    fig10_client_distribution()
    fig11_sensitivity_specificity()
    fig12_high_accuracy_comparison()
    fig13_multi_seed_stability()
    fig14_client_ablation()
    fig15_noniid_ablation()
    fig16_clip_ablation()
    fig17_noise_ablation()
    print(f"第4章新增图已生成：{OUT_DIR}")


if __name__ == "__main__":
    main()
