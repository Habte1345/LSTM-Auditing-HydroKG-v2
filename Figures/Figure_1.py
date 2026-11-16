import sys
import importlib
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image

# ---------------- paths ----------------
root = Path("/bighome/hdagne1/LSTM-Auditing-HydroKG")
img_path = root / "Conceptual_Framework.png"
out_path = root / "Conceptual_Framework_combined.png"
src_dir = root / "scripts"

def whiten(arr, thr=240):
    """Set near-white (light grey) background pixels to pure white."""
    arr = arr.copy()
    arr[(arr >= thr).all(axis=2)] = 255
    return arr

def trim_white(arr, tol=235, pad=10):
    """Crop borders whose pixels are all lighter than `tol`."""
    mask = (arr < tol).any(axis=2)
    rows, cols = np.where(mask.any(axis=1))[0], np.where(mask.any(axis=0))[0]
    if rows.size == 0:
        return arr
    r0, r1 = max(rows[0] - pad, 0), min(rows[-1] + pad + 1, arr.shape[0])
    c0, c1 = max(cols[0] - pad, 0), min(cols[-1] + pad + 1, arr.shape[1])
    return arr[r0:r1, c0:c1]

# ---------------- (a) static image ----------------
arr_a = np.asarray(Image.open(img_path).convert("RGB"))
arr_a = trim_white(whiten(arr_a))

# ---------------- (b) capture the module figure ----------------
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

_show = plt.show
plt.show = lambda *args, **kwargs: None
before = set(plt.get_fignums())
try:
    import conceptual_framework_realtime_audit as framework
    importlib.reload(framework)
finally:
    plt.show = _show

new_figs = [n for n in plt.get_fignums() if n not in before]
if not new_figs:
    raise RuntimeError("Module did not leave an open figure (it may close it after saving).")

fig_b = plt.figure(new_figs[-1])
fig_b.set_dpi(400)
canvas = FigureCanvasAgg(fig_b)
canvas.draw()
arr_b = np.asarray(canvas.buffer_rgba())[..., :3].copy()
for n in new_figs:
    plt.close(n)
arr_b = trim_white(arr_b)

# ---------------- resize both to the same width ----------------
target_w = max(arr_a.shape[1], arr_b.shape[1])

def to_width(arr, w):
    h = int(round(arr.shape[0] * w / arr.shape[1]))
    return np.asarray(Image.fromarray(arr).resize((w, h), Image.LANCZOS))

arr_a = to_width(arr_a, target_w)
arr_b = to_width(arr_b, target_w)

# ---------------- stack with gap and left label margin ----------------
gap = int(0.03 * target_w)
margin = int(0.04 * target_w)
white = lambda h, w: np.full((h, w, 3), 255, dtype=np.uint8)

stack = np.vstack([arr_a, white(gap, target_w), arr_b])
stack = np.hstack([white(stack.shape[0], margin), stack])

H, Wpx = stack.shape[:2]
y_a = 0
y_b = arr_a.shape[0] + gap

# ---------------- plot ----------------
W = 8.0
fig = plt.figure(figsize=(W, W * H / Wpx), dpi=400)
ax = fig.add_axes([0, 0, 1, 1])
ax.imshow(stack)
ax.set_axis_off()

for lab, y in [("a)", y_a), ("b)", y_b)]:
    ax.text(margin * 0.15, y, lab, fontsize=12, fontweight="bold",
            ha="left", va="top")

# fig.savefig(out_path, dpi=400, facecolor="white")
plt.show()