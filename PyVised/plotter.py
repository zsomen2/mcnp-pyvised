"""Slice rendering: classify a pixel grid and turn it into a colored image.

Pure computation + matplotlib drawing helpers; no backend is selected here, so
both the Tk GUI and the headless Agg path can use it.
"""

import colorsys
import time

import numpy as np
from matplotlib.patches import Patch

from geometry import OUTSIDE, UNDEFINED

# MCNP Vised-like material palette (RGB 0-255).  Materials are assigned colors
# in ascending material-number order, so for the course decks M1=blue (fuel),
# M2=yellow (water), M3=green (Al), M4=red (graphite) — same as Vised.
VISED_PALETTE = [
    (0, 0, 255),       # blue
    (255, 255, 0),     # yellow
    (0, 170, 0),       # green
    (255, 0, 0),       # red
    (0, 255, 255),     # cyan
    (255, 165, 0),     # orange
    (75, 0, 130),      # indigo (kept dark so it can't be mistaken for the
                       # magenta "undefined" marker)
    (150, 75, 0),      # brown
    (255, 150, 200),   # pink
    (64, 224, 208),    # turquoise
    (128, 128, 0),     # olive
    (100, 149, 237),   # cornflower blue
    (139, 0, 0),       # dark red
    (144, 238, 144),   # light green
    (0, 0, 128),       # navy
    (218, 165, 32),    # goldenrod
]

COLOR_VOID = (255, 255, 255)
COLOR_OUTSIDE = (80, 80, 80)
COLOR_UNDEFINED = (255, 0, 255)

# view name -> (horizontal axis, vertical axis, fixed axis)
VIEWS = {
    "front": (0, 2, 1),   # X-Z plane, cut at a fixed Y
    "top": (0, 1, 2),     # X-Y plane, cut at a fixed Z
}
_AXIS_NAMES = ("x", "y", "z")


def _extra_color(i):
    """Deterministic extra colors past the fixed palette (never magenta-,
    white-, grey- or black-like, so the special markers stay unambiguous)."""
    hue = (i * 0.618033988749895) % 1.0
    if abs(hue - 0.85) < 0.07:        # keep away from the magenta band
        hue = (hue + 0.14) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.9)
    return (int(r * 255), int(g * 255), int(b * 255))


def material_colors(materials):
    """Map material number -> RGB tuple, stable for a given deck."""
    cmap = {}
    for i, m in enumerate(sorted(materials)):
        if i < len(VISED_PALETTE):
            cmap[m] = VISED_PALETTE[i]
        else:
            cmap[m] = _extra_color(i - len(VISED_PALETTE))
    return cmap


def square_window(window):
    """Expand (hmin,hmax,vmin,vmax) so both spans match (kept centered).

    Both views are always displayed in a square 1:1 region; geometry outside
    the model bounding box simply classifies as whatever is there (usually
    the imp=0 outside world).
    """
    hmin, hmax, vmin, vmax = window
    hspan, vspan = hmax - hmin, vmax - vmin
    if hspan > vspan:
        c = 0.5 * (vmin + vmax)
        vmin, vmax = c - 0.5 * hspan, c + 0.5 * hspan
    elif vspan > hspan:
        c = 0.5 * (hmin + hmax)
        hmin, hmax = c - 0.5 * vspan, c + 0.5 * vspan
    return (hmin, hmax, vmin, vmax)


class RenderResult:
    def __init__(self, view, coord, rgb, extent, n_undefined, elapsed):
        self.view = view              # 'front' / 'top'
        self.coord = coord            # fixed-axis coordinate of the cut
        self.rgb = rgb                # (nv, nh, 3) uint8 image
        self.extent = extent          # (hmin, hmax, vmin, vmax)
        self.n_undefined = n_undefined
        self.elapsed = elapsed
        h_ax, v_ax, f_ax = VIEWS[view]
        self.h_label = f"{_AXIS_NAMES[h_ax]} [cm]"
        self.v_label = f"{_AXIS_NAMES[v_ax]} [cm]"
        self.title = (f"{view.capitalize()} view "
                      f"({_AXIS_NAMES[h_ax].upper()}–{_AXIS_NAMES[v_ax].upper()}) "
                      f"at {_AXIS_NAMES[f_ax].upper()} = {coord:g} cm")


def render_slice(evaluator, view, coord, bbox, resolution=900, window=None):
    """Render one slice; `window` optionally overrides (hmin,hmax,vmin,vmax)."""
    t0 = time.perf_counter()
    h_ax, v_ax, f_ax = VIEWS[view]
    if window is not None:
        hmin, hmax, vmin, vmax = window
    else:
        (hmin, hmax), (vmin, vmax) = bbox[h_ax], bbox[v_ax]
        hmin, hmax, vmin, vmax = square_window((hmin, hmax, vmin, vmax))
    if hmax <= hmin or vmax <= vmin:
        raise ValueError("empty plot window")

    aspect = (vmax - vmin) / (hmax - hmin)
    if aspect >= 1.0:
        nv = int(resolution)
        nh = max(2, int(round(resolution / aspect)))
    else:
        nh = int(resolution)
        nv = max(2, int(round(resolution * aspect)))

    hs = hmin + (np.arange(nh) + 0.5) * (hmax - hmin) / nh
    vs = vmin + (np.arange(nv) + 0.5) * (vmax - vmin) / nv

    pts = np.empty((nv * nh, 3), dtype=np.float64)
    pts[:, h_ax] = np.tile(hs, nv)
    pts[:, v_ax] = np.repeat(vs, nh)
    # nudge the cut plane so pixel centers never sit exactly on a surface
    pts[:, f_ax] = coord + 1e-6

    mat, sig = evaluator.classify(pts)
    mat = mat.reshape(nv, nh)
    sig = sig.reshape(nv, nh)

    cmap = material_colors(evaluator.deck.materials)
    rgb = np.empty((nv, nh, 3), dtype=np.uint8)
    rgb[:] = COLOR_UNDEFINED
    rgb[mat == 0] = COLOR_VOID
    rgb[mat == OUTSIDE] = COLOR_OUTSIDE
    for m, color in cmap.items():
        rgb[mat == m] = color

    # black boundary where the cell/lattice-element path changes; mark both
    # sides of the change (2 px) so the line survives display downsampling
    edge = np.zeros(mat.shape, dtype=bool)
    hdiff = sig[:, 1:] != sig[:, :-1]
    vdiff = sig[1:, :] != sig[:-1, :]
    edge[:, 1:] |= hdiff
    edge[:, :-1] |= hdiff
    edge[1:, :] |= vdiff
    edge[:-1, :] |= vdiff
    rgb[edge] = (0, 0, 0)

    n_undefined = int((mat == UNDEFINED).sum())
    return RenderResult(view, coord, rgb, (hmin, hmax, vmin, vmax),
                        n_undefined, time.perf_counter() - t0)


def draw_result(ax, result):
    ax.clear()
    ax.imshow(result.rgb, origin="lower", extent=result.extent,
              interpolation="nearest", aspect="equal")
    ax.set_xlabel(result.h_label)
    ax.set_ylabel(result.v_label)
    ax.set_title(result.title, fontsize=10)


def legend_handles(deck, results=()):
    cmap = material_colors(deck.materials)
    handles = [Patch(facecolor=_norm(c), edgecolor="black", label=f"M{m}")
               for m, c in sorted(cmap.items())]
    handles.append(Patch(facecolor=_norm(COLOR_VOID), edgecolor="black",
                         label="void"))
    handles.append(Patch(facecolor=_norm(COLOR_OUTSIDE), edgecolor="black",
                         label="outside (imp=0)"))
    if any(r.n_undefined for r in results):
        handles.append(Patch(facecolor=_norm(COLOR_UNDEFINED), edgecolor="black",
                             label="undefined!"))
    return handles


def _norm(rgb):
    return tuple(v / 255.0 for v in rgb)


def render_figure(fig, evaluator, deck, front_coord=0.0, top_coord=0.0,
                  resolution=900, front_window=None, top_window=None):
    """Fill `fig` with the front + top view; returns the two RenderResults."""
    bbox = evaluator.world_bbox()
    front = render_slice(evaluator, "front", front_coord, bbox,
                         resolution, front_window)
    top = render_slice(evaluator, "top", top_coord, bbox,
                       resolution, top_window)

    # size the figure so each view is displayed at >= native raster
    # resolution (downsampling would chop the boundary lines into dashes);
    # each of the two axes columns spans ~41% of the figure width and the
    # axes row ~72% of its height (see subplots_adjust below)
    width_px = 2 * max(front.rgb.shape[1], top.rgb.shape[1]) / 0.82
    height_px = max(front.rgb.shape[0], top.rgb.shape[0]) / 0.72
    fig.set_size_inches(min(36.0, width_px / fig.dpi),
                        min(36.0, height_px / fig.dpi))

    fig.clear()
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    draw_result(ax1, front)
    draw_result(ax2, top)
    handles = legend_handles(deck, (front, top))
    fig.suptitle(deck.title, fontsize=11)
    fig.legend(handles=handles, loc="lower center",
               ncol=min(8, len(handles)), fontsize=9, frameon=False)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.90, bottom=0.16,
                        wspace=0.18)
    return front, top
