from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "figures"
SUMMARY_CSV = ROOT / "outputs" / "experiment_summary.csv"
ATTACK_CSV = ROOT / "outputs" / "attack_summary.csv"

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
    for line, h in zip(lines, heights):
        w, _ = text_size(draw, line, fnt)
        draw.text((x - w / 2, cursor), line, font=fnt, fill=fill)
        cursor += h + 8


def load_experiment_rows() -> dict[str, dict[str, str]]:
    with SUMMARY_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    return {row["experiment_name"]: row for row in rows}


def load_attack_rows() -> dict[str, dict[str, str]]:
    with ATTACK_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    return {row["experiment_name"]: row for row in rows}


def base_canvas(title: str, subtitle: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (W, H), COLORS["white"])
    draw = ImageDraw.Draw(img)
    _ = title, subtitle
    return img, draw


def draw_y_grid(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    ymin: float,
    ymax: float,
    ticks: list[float],
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
    draw.line((x1, y1, x1, y2), fill=COLORS["line"], width=3)
    for t in ticks:
        yy = y2 - (t - ymin) / (ymax - ymin) * (y2 - y1)
        draw.line((x1, yy, x2, yy), fill=COLORS["grid"], width=2)
        draw.text((x1 - 72, yy - 16), f"{t:.2f}", font=FONTS["small"], fill=COLORS["muted"])


def fig8_main_metrics() -> None:
    rows = load_experiment_rows()
    names = [
        ("Centralized", "ham10000_centralized", COLORS["muted"]),
        ("NoDP", "ham10000_nodp", COLORS["accent"]),
        ("Gaussian", "ham10000_gaussian", COLORS["accent_2"]),
        ("Laplace", "ham10000_laplace", COLORS["success"]),
        ("Hybrid", "ham10000_hybrid", COLORS["warm"]),
        ("Adaptive\nHybrid", "ham10000_hybrid_adaptive", COLORS["warm_dark"]),
    ]
    metrics = [
        ("Accuracy", "test_accuracy_at_best_val"),
        ("AUC", "test_auc_at_best_val"),
        ("F1", "test_f1_at_best_val"),
    ]
    img, draw = base_canvas(
        "主实验分类指标对比",
        "Gaussian 在 Accuracy 和 F1 上表现较好，各机制 AUC 均接近 0.90。",
    )
    chart = (185, 220, 2260, 1030)
    ymin, ymax = 0.70, 0.92
    draw_y_grid(draw, chart, ymin, ymax, [0.70, 0.75, 0.80, 0.85, 0.90])

    group_w = (chart[2] - chart[0]) / len(names)
    bar_w = 55
    gap = 10
    offsets = [-bar_w - gap, 0, bar_w + gap]
    metric_colors = [COLORS["accent"], COLORS["warm"], COLORS["success"]]
    for i, (label, key, outline) in enumerate(names):
        gx = chart[0] + group_w * i + group_w / 2
        row = rows[key]
        for j, (_, col) in enumerate(metrics):
            val = float(row[col])
            h = (val - ymin) / (ymax - ymin) * (chart[3] - chart[1])
            x1 = gx + offsets[j] - bar_w / 2
            y1 = chart[3] - h
            x2 = x1 + bar_w
            draw.rounded_rectangle((x1, y1, x2, chart[3]), radius=10, fill=metric_colors[j])
            text_center(draw, ((x1 + x2) / 2, y1 - 26), f"{val:.3f}", FONTS["tiny"], metric_colors[j])
        text_center(draw, (gx, chart[3] + 92), label, FONTS["label"], outline)

    legend_x = 1500
    for j, (metric, _) in enumerate(metrics):
        x = legend_x + j * 235
        y = 115
        draw.rounded_rectangle((x, y, x + 42, y + 42), radius=8, fill=metric_colors[j])
        draw.text((x + 54, y + 5), metric, font=FONTS["label"], fill=COLORS["text"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    img.save(OUT_DIR / "主实验分类指标对比.png", dpi=(300, 300), quality=95)


def fig9_attack_metrics() -> None:
    rows = load_attack_rows()
    names = [
        ("NoDP", "ham10000_nodp", COLORS["accent"]),
        ("Gaussian", "ham10000_gaussian", COLORS["accent_2"]),
        ("Laplace", "ham10000_laplace", COLORS["success"]),
        ("Hybrid", "ham10000_hybrid", COLORS["warm"]),
        ("Adaptive\nHybrid", "ham10000_hybrid_adaptive", COLORS["warm_dark"]),
    ]
    img, draw = base_canvas(
        "成员推断攻击指标对比",
        "攻击 Accuracy 和 AUC 越接近 0.50，表示攻击越接近随机猜测。",
    )
    chart = (210, 220, 2240, 1030)
    ymin, ymax = 0.48, 0.58
    draw_y_grid(draw, chart, ymin, ymax, [0.50, 0.52, 0.54, 0.56, 0.58])
    y_random = chart[3] - (0.50 - ymin) / (ymax - ymin) * (chart[3] - chart[1])
    draw.line((chart[0], y_random, chart[2], y_random), fill=COLORS["success"], width=5)

    group_w = (chart[2] - chart[0]) / len(names)
    bar_w = 90
    for i, (label, key, color) in enumerate(names):
        gx = chart[0] + group_w * i + group_w / 2
        row = rows[key]
        vals = [
            ("Attack Accuracy", float(row["attack_accuracy"]), COLORS["accent"]),
            ("Attack AUC", float(row["attack_auc"]), COLORS["warm"]),
        ]
        for j, (_, val, bar_color) in enumerate(vals):
            x1 = gx + (-bar_w - 12 if j == 0 else 12)
            y1 = chart[3] - (val - ymin) / (ymax - ymin) * (chart[3] - chart[1])
            x2 = x1 + bar_w
            draw.rounded_rectangle((x1, y1, x2, chart[3]), radius=12, fill=bar_color)
            text_center(draw, ((x1 + x2) / 2, y1 - 28), f"{val:.3f}", FONTS["tiny"], bar_color)
        text_center(draw, (gx, chart[3] + 92), label, FONTS["label"], color)

    legend_x = 1500
    for j, (label, color) in enumerate([("Attack Accuracy", COLORS["accent"]), ("Attack AUC", COLORS["warm"])]):
        x = legend_x + j * 300
        y = 115
        draw.rounded_rectangle((x, y, x + 42, y + 42), radius=8, fill=color)
        draw.text((x + 54, y + 5), label, font=FONTS["label"], fill=COLORS["text"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    img.save(OUT_DIR / "成员推断攻击指标对比.png", dpi=(300, 300), quality=95)


def main() -> None:
    fig8_main_metrics()
    fig9_attack_metrics()
    print(f"已生成新增结果图：{OUT_DIR / '主实验分类指标对比.png'}，{OUT_DIR / '成员推断攻击指标对比.png'}")


if __name__ == "__main__":
    main()
