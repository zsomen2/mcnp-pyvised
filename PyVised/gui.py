"""tkinter GUI: startup window with a file selector + geometry viewer window."""

import pathlib
import queue
import threading
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)
from matplotlib.figure import Figure

import mcnp_parser
import plotter
from geometry import Evaluator, GeometryError

FILETYPES = [
    ("MCNP input", "*.mcnp *.inp *.txt *.i"),
    ("Text files", "*.txt"),
    ("All files", "*.*"),
]


def _enable_windows_dpi_awareness():
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass


class StartupWindow:
    def __init__(self, root, front_y=0.0, top_z=0.0, resolution=900):
        self.root = root
        self.open_viewers = 0
        self.front_y = front_y
        self.top_z = top_z
        self.resolution = max(100, min(3000, resolution))
        root.title("PyVised")
        root.resizable(False, False)

        frame = ttk.Frame(root, padding=(40, 28, 40, 24))
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="PyVised", font=("Segoe UI", 26, "bold")).pack()
        ttk.Label(frame, text="MCNP input geometry viewer",
                  font=("Segoe UI", 11)).pack(pady=(2, 18))
        ttk.Button(frame, text="Open MCNP input…", command=self.open_file,
                   padding=(16, 8)).pack()
        ttk.Label(frame,
                  text="Shows the front (X–Z @ Y=0) and top (X–Y @ Z=0)\n"
                       "geometry cuts, colored by material.",
                  font=("Segoe UI", 9), foreground="#555",
                  justify="center").pack(pady=(16, 0))

    def open_file(self, path=None, parent=None):
        if path is None:
            path = filedialog.askopenfilename(
                title="Select an MCNP input file", filetypes=FILETYPES,
                parent=parent or self.root)
            if not path:
                return
        try:
            deck = mcnp_parser.parse_file(path)
            evaluator = Evaluator(deck)
        except (mcnp_parser.MCNPParseError, GeometryError, OSError) as exc:
            messagebox.showerror("PyVised — cannot load file",
                                 f"{pathlib.Path(path).name}:\n\n{exc}",
                                 parent=parent or self.root)
            return
        self.open_viewers += 1
        self.root.withdraw()
        ViewerWindow(self, deck, evaluator, path)

    def viewer_closed(self):
        self.open_viewers -= 1
        if self.open_viewers <= 0:
            self.root.deiconify()


class ViewerWindow(tk.Toplevel):
    """Geometry viewer.

    The two views are raster classifications of the cut planes; to keep them
    crisp at any magnification the window re-renders a view in the background
    whenever its axes limits change (toolbar zoom/pan, mouse wheel, home),
    at the axes' actual on-screen pixel resolution.
    """

    ZOOM_DEBOUNCE_MS = 300      # wait for the limits to settle before rendering
    ZOOM_MAX_RES = 2400         # cap for adaptive zoom renders

    def __init__(self, app, deck, evaluator, path):
        super().__init__(app.root)
        self.app = app
        self.deck = deck
        self.evaluator = evaluator
        self.queue = queue.Queue()
        self.rendering = False
        self._axes = {}             # view -> Axes
        self._coords = {}           # view -> cut coordinate of the last render
        self._rendered = {}         # view -> (extent, resolution) last rendered
        self._dirty = set()         # views whose limits changed
        self._zoom_job = None       # pending tk-after id of the debounce
        self._applying = False      # True while we set limits programmatically
        self.title(f"PyVised — {pathlib.Path(path).name}")
        self.geometry("1280x760")
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # --- controls bar ---
        bar = ttk.Frame(self, padding=(8, 6, 8, 0))
        bar.pack(side="top", fill="x")
        ttk.Label(bar, text="Resolution:").pack(side="left")
        self.res_var = tk.StringVar(value=str(app.resolution))
        ttk.Combobox(bar, textvariable=self.res_var, width=6,
                     values=("400", "600", "900", "1200", "1600"),
                     state="normal").pack(side="left", padx=(2, 12))
        self.redraw_btn = ttk.Button(bar, text="Redraw", command=self.start_render)
        self.redraw_btn.pack(side="left")
        ttk.Button(bar, text="Reset view",
                   command=self.reset_view).pack(side="left", padx=(6, 0))
        ttk.Button(bar, text="Open another file…",
                   command=lambda: self.app.open_file(parent=self)
                   ).pack(side="left", padx=(12, 0))
        self.status_var = tk.StringVar(value="")
        ttk.Label(bar, textvariable=self.status_var,
                  foreground="#444").pack(side="right")

        # --- slice position sliders ---
        self._bbox = evaluator.world_bbox()
        self._slice_vars = {}
        self._slice_scales = {}
        self._slice_dirty = set()
        self._slice_job = None
        self._syncing = False
        sbar = ttk.Frame(self, padding=(8, 2, 8, 6))
        sbar.pack(side="top", fill="x")
        self.front_var = self._make_slice_control(
            sbar, "Front cut  Y =", "front", self._bbox[1], app.front_y)
        self.top_var = self._make_slice_control(
            sbar, "Top cut  Z =", "top", self._bbox[2], app.top_z)

        # --- matplotlib canvas ---
        self.figure = Figure(figsize=(12.4, 6.6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        toolbar = NavigationToolbar2Tk(self.canvas, self, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side="bottom", fill="x")
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
        self.canvas.mpl_connect("scroll_event", self._on_scroll)

        self._warn_count = 0
        self._note_warnings()

        self.after(100, self._poll_queue)
        self.start_render()

    def _make_slice_control(self, parent, label, view, axis_range, initial):
        """Label + entry + slider controlling the cut coordinate of a view."""
        lo, hi = axis_range
        frame = ttk.Frame(parent)
        frame.pack(side="left", fill="x", expand=True, padx=(0, 18))
        ttk.Label(frame, text=label).pack(side="left")
        var = tk.StringVar(value=f"{initial:g}")
        entry = ttk.Entry(frame, textvariable=var, width=8)
        entry.pack(side="left", padx=(2, 6))
        scale = ttk.Scale(frame, from_=lo, to=hi,
                          value=min(max(initial, lo), hi),
                          command=lambda val, v=view: self._on_slider(v, val))
        scale.pack(side="left", fill="x", expand=True)
        entry.bind("<Return>", lambda _e, v=view: self._on_slice_entry(v))
        entry.bind("<FocusOut>", lambda _e, v=view: self._on_slice_entry(v))
        self._slice_vars[view] = var
        self._slice_scales[view] = scale
        return var

    def _on_slider(self, view, value):
        if self._syncing:
            return
        self._slice_vars[view].set(f"{float(value):.2f}")
        self._slice_dirty.add(view)
        self._schedule_slice()

    def _on_slice_entry(self, view):
        try:
            coord = float(self._slice_vars[view].get())
        except ValueError:
            return
        if abs(coord - self._coords.get(view, coord)) < 1e-12 \
                and view not in self._slice_dirty:
            return
        self._syncing = True
        try:
            self._slice_scales[view].set(coord)   # tk clamps to the range
        finally:
            self._syncing = False
        self._slice_dirty.add(view)
        self._schedule_slice()

    def _schedule_slice(self):
        if self._slice_job is not None:
            self.after_cancel(self._slice_job)
        self._slice_job = self.after(self.ZOOM_DEBOUNCE_MS, self._slice_now)

    def _slice_now(self):
        self._slice_job = None
        if self.rendering:
            self._schedule_slice()
            return
        requests = []
        for view in tuple(self._slice_dirty):
            ax = self._axes.get(view)
            if ax is None:
                continue
            try:
                coord = float(self._slice_vars[view].get())
            except ValueError:
                continue
            self._coords[view] = coord
            window, res = self._zoom_params(ax)
            requests.append((view, coord, window, res))
        self._slice_dirty.clear()
        if not requests:
            return
        self.rendering = True
        self._set_status("moving cut…")
        threading.Thread(target=self._zoom_worker, args=(requests,),
                         daemon=True).start()

    def _note_warnings(self):
        """Print new deck warnings and keep a persistent status-bar note."""
        warnings = self.deck.warnings
        for w in warnings[self._warn_count:]:
            print(f"PyVised warning: {w}")
        self._warn_count = len(warnings)
        self._warn_note = (f"   ⚠ {len(warnings)} warning(s) — see console"
                           if warnings else "")

    def _set_status(self, text):
        self.status_var.set(text + getattr(self, "_warn_note", ""))

    # --- rendering ---

    def reset_view(self):
        """Cuts back to Y=0 / Z=0 and zoom back to the full model."""
        for view in ("front", "top"):
            self._slice_vars[view].set("0")
            self._syncing = True
            try:
                self._slice_scales[view].set(0.0)
            finally:
                self._syncing = False
        self.start_render()

    def start_render(self):
        if self.rendering:
            return
        try:
            front_y = float(self.front_var.get())
            top_z = float(self.top_var.get())
            resolution = max(100, min(3000, int(self.res_var.get())))
        except ValueError:
            messagebox.showerror("PyVised", "Cut coordinates and resolution "
                                 "must be numbers.", parent=self)
            return
        # a full render supersedes any pending zoom/slice refinements
        for attr in ("_zoom_job", "_slice_job"):
            job = getattr(self, attr)
            if job is not None:
                self.after_cancel(job)
                setattr(self, attr, None)
        self._dirty.clear()
        self._slice_dirty.clear()
        self.rendering = True
        self.redraw_btn.state(["disabled"])
        self._set_status("Rendering…")
        threading.Thread(target=self._render_worker,
                         args=(front_y, top_z, resolution), daemon=True).start()

    def _render_worker(self, front_y, top_z, resolution):
        try:
            bbox = self.evaluator.world_bbox()
            front = plotter.render_slice(self.evaluator, "front", front_y,
                                         bbox, resolution)
            top = plotter.render_slice(self.evaluator, "top", top_z,
                                       bbox, resolution)
            if self.evaluator.max_depth_hit:
                self.deck.warn("maximum universe nesting depth exceeded — "
                               "some regions were left undefined")
            self.queue.put(("ok", (front_y, top_z), front, top))
        except Exception:
            self.queue.put(("error", traceback.format_exc()))

    def _poll_queue(self):
        try:
            item = self.queue.get_nowait()
        except queue.Empty:
            self.after(100, self._poll_queue)
            return
        self.rendering = False
        self.redraw_btn.state(["!disabled"])
        if item[0] == "error":
            self._note_warnings()
            self._set_status("Render failed")
            messagebox.showerror("PyVised — render error", item[1], parent=self)
        elif item[0] == "ok":
            self._show_full_render(item)
        else:
            self._show_zoom_render(item)
        self.after(100, self._poll_queue)

    def _show_full_render(self, item):
        _tag, (front_y, top_z), front, top = item
        self.figure.clear()
        self._axes = {"front": self.figure.add_subplot(1, 2, 1),
                      "top": self.figure.add_subplot(1, 2, 2)}
        self._coords = {"front": front_y, "top": top_z}
        self._dirty.clear()
        self._applying = True
        try:
            for view, rr in (("front", front), ("top", top)):
                ax = self._axes[view]
                plotter.draw_result(ax, rr)
                self._rendered[view] = (rr.extent, max(rr.rgb.shape[:2]))
                callback = lambda _ax, v=view: self._on_limits_changed(v)
                ax.callbacks.connect("xlim_changed", callback)
                ax.callbacks.connect("ylim_changed", callback)
        finally:
            self._applying = False
        handles = plotter.legend_handles(self.deck, (front, top))
        self.figure.suptitle(self.deck.title, fontsize=11)
        self.figure.legend(handles=handles, loc="lower center",
                           ncol=min(8, len(handles)), fontsize=9, frameon=False)
        self.figure.subplots_adjust(left=0.06, right=0.98, top=0.90,
                                    bottom=0.15, wspace=0.15)
        self.canvas.draw_idle()
        status = (f"done in {front.elapsed + top.elapsed:.1f} s   "
                  f"({front.rgb.shape[1]}×{front.rgb.shape[0]} px / view)")
        if front.n_undefined or top.n_undefined:
            status += (f"   ⚠ {front.n_undefined + top.n_undefined} "
                       "undefined pixels (magenta)")
        self._note_warnings()
        self._set_status(status)

    def _show_zoom_render(self, item):
        _tag, results = item
        self._applying = True
        try:
            for view, rr in results:
                ax = self._axes.get(view)
                if ax is None or not ax.images:
                    continue
                ax.images[0].set(data=rr.rgb, extent=rr.extent)
                ax.set_xlim(rr.extent[0], rr.extent[1])
                ax.set_ylim(rr.extent[2], rr.extent[3])
                ax.set_title(rr.title, fontsize=10)
                self._rendered[view] = (rr.extent, max(rr.rgb.shape[:2]))
        finally:
            self._applying = False
        self.canvas.draw_idle()
        self._note_warnings()
        self._set_status("view refined at "
                         + ", ".join(f"{rr.rgb.shape[1]}×{rr.rgb.shape[0]} px"
                                     for _v, rr in results))
        # the user may have kept zooming / sliding while we rendered
        if self._dirty:
            self._schedule_zoom()
        if self._slice_dirty:
            self._schedule_slice()

    # --- adaptive re-rendering on zoom/pan ---

    def _on_limits_changed(self, view):
        if self._applying or view not in self._rendered:
            return
        self._dirty.add(view)
        self._schedule_zoom()

    def _schedule_zoom(self):
        if self._zoom_job is not None:
            self.after_cancel(self._zoom_job)
        self._zoom_job = self.after(self.ZOOM_DEBOUNCE_MS, self._zoom_now)

    def _zoom_now(self):
        self._zoom_job = None
        if self.rendering:
            self._schedule_zoom()
            return
        requests = []
        for view in tuple(self._dirty):
            ax = self._axes.get(view)
            if ax is None:
                self._dirty.discard(view)
                continue
            window, res = self._zoom_params(ax)
            old_extent, old_res = self._rendered.get(view, (None, None))
            if old_extent is not None and self._close(window, old_extent, res, old_res):
                self._dirty.discard(view)
                continue
            requests.append((view, self._coords.get(view, 0.0), window, res))
        self._dirty.clear()
        if not requests:
            return
        self.rendering = True
        self._set_status("refining view…")
        threading.Thread(target=self._zoom_worker, args=(requests,),
                         daemon=True).start()

    def _zoom_params(self, ax):
        """Square 1:1 window + the on-screen pixel size of the square box."""
        x0, x1 = ax.get_xlim()
        y0, y1 = sorted(ax.get_ylim())
        x0, x1, y0, y1 = plotter.square_window((x0, x1, y0, y1))
        fig_w, fig_h = self.canvas.get_width_height()
        pos = ax.get_position()
        # a square window drawn with aspect='equal' occupies the largest
        # square that fits in the layout cell
        px = min(max(100.0, fig_w * pos.width), max(100.0, fig_h * pos.height))
        res = int(min(self.ZOOM_MAX_RES, max(200, round(px))))
        return (x0, x1, y0, y1), res

    @staticmethod
    def _close(window, extent, res, old_res):
        scale = max(abs(extent[1] - extent[0]), abs(extent[3] - extent[2]), 1e-12)
        same_window = all(abs(w - e) < 0.005 * scale
                          for w, e in zip(window, extent))
        return same_window and old_res is not None and abs(res - old_res) < 64

    def _zoom_worker(self, requests):
        try:
            bbox = self.evaluator.world_bbox()
            results = []
            for view, coord, window, res in requests:
                rr = plotter.render_slice(self.evaluator, view, coord, bbox,
                                          res, window)
                results.append((view, rr))
            self.queue.put(("zoom", results))
        except Exception:
            self.queue.put(("error", traceback.format_exc()))

    def _on_scroll(self, event):
        """Mouse-wheel zoom centered on the cursor."""
        ax = event.inaxes
        if ax not in self._axes.values() or event.xdata is None:
            return
        factor = 0.8 if event.button == "up" else 1.25
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        ax.set_xlim(event.xdata + (x0 - event.xdata) * factor,
                    event.xdata + (x1 - event.xdata) * factor)
        ax.set_ylim(event.ydata + (y0 - event.ydata) * factor,
                    event.ydata + (y1 - event.ydata) * factor)
        self.canvas.draw_idle()

    def _on_close(self):
        self.destroy()
        self.app.viewer_closed()


def run_app(initial_file=None, front_y=0.0, top_z=0.0, resolution=900):
    _enable_windows_dpi_awareness()
    root = tk.Tk()
    app = StartupWindow(root, front_y=front_y, top_z=top_z,
                        resolution=resolution)
    if initial_file:
        root.after(50, lambda: app.open_file(initial_file))
    root.mainloop()
