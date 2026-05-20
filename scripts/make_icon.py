"""Create a simple application icon if missing."""

from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
ASSETS.mkdir(exist_ok=True)
OUT = ASSETS / "icon.ico"

img = Image.new("RGBA", (256, 256), (26, 26, 46, 255))
draw = ImageDraw.Draw(img)
draw.ellipse((48, 40, 208, 220), fill=(0, 200, 255, 255))
draw.rectangle((100, 120, 156, 220), fill=(26, 26, 46, 255))
img.save(OUT, format="ICO", sizes=[(256, 256), (64, 64), (32, 32), (16, 16)])
print(f"Wrote {OUT}")
