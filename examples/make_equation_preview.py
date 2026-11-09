"""Render the four MSD diffusion-model equations to an HTML gallery."""

import base64
import io
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib
matplotlib.use("Agg")

from matplotlib.mathtext import math_to_image
from PIL import Image

from trackart.gui.msd_tab import MODEL_EQUATIONS, MODELS, MODEL_KEY

TITLES = {
    "normal": "Normal (Brownian) diffusion",
    "anomalous": "Anomalous diffusion",
    "confined": "Confined / corralled diffusion",
    "active": "Active (free diffusion with flow)",
}

parts = []
for model_label in MODELS:
    model = MODEL_KEY[model_label]
    parts.append(f"<h2>{TITLES[model]} &mdash; {model_label}</h2>")
    for tex in MODEL_EQUATIONS[model]:
        buf = io.BytesIO()
        math_to_image(tex, buf, dpi=220, format="png")
        buf.seek(0)
        img = Image.open(buf).convert("RGBA")
        # paste on white so the browser shows the glyphs
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        bg.alpha_composite(img)
        out = io.BytesIO()
        bg.convert("RGB").save(out, format="PNG")
        data = base64.b64encode(out.getvalue()).decode()
        parts.append(
            f'<img src="data:image/png;base64,{data}" '
            f'style="display:block;margin:6px 0;background:white;'
            f'border:1px solid #ccc;padding:8px"/>')
    parts.append("<hr/>")

html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>TrackArt MSD model equations</title>
<style>body{{font-family:sans-serif;margin:24px}}h2{{margin-bottom:4px}}</style>
</head><body>
<h1>MSD &ldquo;Fit to&rdquo; equations (as rendered in the tkinter GUI)</h1>
{''.join(parts)}
</body></html>"""

out_path = os.path.join(os.path.dirname(__file__), "out", "msd_equations.html")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    f.write(html)
print("wrote", out_path)
