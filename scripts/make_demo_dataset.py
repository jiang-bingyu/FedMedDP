from pathlib import Path
import random

from PIL import Image, ImageDraw, ImageFilter


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "demo_medical"


def make_background(size: int, rng: random.Random) -> Image.Image:
    return Image.new(
        "RGB",
        (size, size),
        (
            rng.randint(190, 230),
            rng.randint(150, 200),
            rng.randint(140, 190),
        ),
    )


def draw_normal(size: int, seed: int) -> Image.Image:
    rng = random.Random(seed)
    image = make_background(size, rng)
    draw = ImageDraw.Draw(image)

    for _ in range(8):
        radius = rng.randint(6, 18)
        x = rng.randint(0, size - radius - 1)
        y = rng.randint(0, size - radius - 1)
        color = (
            rng.randint(175, 215),
            rng.randint(130, 180),
            rng.randint(120, 170),
        )
        draw.ellipse((x, y, x + radius, y + radius), fill=color)

    return image.filter(ImageFilter.GaussianBlur(radius=1.2))


def draw_lesion(size: int, seed: int) -> Image.Image:
    rng = random.Random(seed)
    image = make_background(size, rng)
    draw = ImageDraw.Draw(image)

    cx = rng.randint(size // 3, 2 * size // 3)
    cy = rng.randint(size // 3, 2 * size // 3)
    rx = rng.randint(size // 6, size // 4)
    ry = rng.randint(size // 7, size // 4)
    lesion_color = (
        rng.randint(55, 110),
        rng.randint(35, 80),
        rng.randint(25, 65),
    )
    draw.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=lesion_color)

    for _ in range(10):
        radius = rng.randint(3, 8)
        x = rng.randint(cx - rx, cx + rx)
        y = rng.randint(cy - ry, cy + ry)
        dot_color = (
            max(0, lesion_color[0] - rng.randint(0, 20)),
            max(0, lesion_color[1] - rng.randint(0, 15)),
            max(0, lesion_color[2] - rng.randint(0, 10)),
        )
        draw.ellipse((x, y, x + radius, y + radius), fill=dot_color)

    return image.filter(ImageFilter.GaussianBlur(radius=0.6))


def save_split(split: str, class_name: str, count: int, start_seed: int) -> None:
    target_dir = DATA_ROOT / split / class_name
    target_dir.mkdir(parents=True, exist_ok=True)

    for index in range(count):
        seed = start_seed + index
        image = draw_normal(64, seed) if class_name == "normal" else draw_lesion(64, seed)
        image.save(target_dir / f"{class_name}_{index:03d}.png")


def main() -> None:
    splits = {
        "train": 80,
        "val": 20,
        "test": 20,
    }

    for split, count in splits.items():
        save_split(split, "normal", count, start_seed=1000 + hash(split) % 100)
        save_split(split, "lesion", count, start_seed=2000 + hash(split) % 100)

    print(f"演示数据集已生成：{DATA_ROOT}")


if __name__ == "__main__":
    main()
