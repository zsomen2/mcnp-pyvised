# PyVised — Implementation Plan

A lightweight MCNP input geometry viewer (a small "MCNP Vised" clone) in Python.
It parses an MCNP input deck, classifies every pixel of two slice planes by the
cell/material it falls into, and shows the result in a tkinter window — **no
transport calculation is run**.

## Goals

* Startup window with a file selector (`.mcnp`, `.inp`, `.txt`, any file).
* One viewer window with two geometry cuts through the middle of the model:
  * **Front view** — X–Z plane at **Y = 0**
  * **Top view** — X–Y plane at **Z = 0**
* Each material gets its own color (MCNP Vised-like palette: M1 blue,
  M2 yellow, M3 green, M4 red, ...). Void is white, zero-importance
  ("outside world") cells are dark grey, undefined regions are magenta.
* Black cell-boundary lines, like the Vised surface plot.

## Architecture

```
PyVised/
├── main.py        entry point: CLI args, headless mode, GUI launch
├── gui.py         tkinter startup window + viewer window (TkAgg canvas)
├── plotter.py     slice rendering -> RGB image, colors, legend, figure layout
├── geometry.py    point-in-cell engine (vectorized numpy), universes/lattices
└── mcnp_parser.py MCNP input deck parser (cells, surfaces, TR, materials)
```

### 1. Parser (`mcnp_parser.py`)

* Handles `message:` block, title line, blank-line block delimiters,
  `c` comment lines, `$` inline comments, continuation by 5+ leading spaces
  or trailing `&`, tabs.
* **Cell cards**: id, material, density, geometry-expression tokens
  (`-1 2 (3 : -4) #5 #(-6 7)`), and the parameters that matter for plotting:
  `u=`, `lat=`, `fill=` (single universe, with optional transform, or a full
  lattice fill array with `i0:i1 j0:j1 k0:k1` ranges and `nR` repeats),
  `trcl=`/`*trcl=` (TR number or inline transform), `imp:?=...`
  (to recognize the zero-importance outside world), `like n but`.
* **Surface cards**: optional `*`/`+` (reflecting/white) prefix, optional
  transformation number, mnemonics:
  `p px py pz | so s sx sy sz | c/x c/y c/z cx cy cz | kx ky kz k/x k/y k/z |
  sq gq | tx ty tz | rpp box sph rcc trc` (others raise a clear error).
* **Data cards**: `TRn`/`*TRn` transformations and `Mn` material numbers.

### 2. Geometry engine (`geometry.py`)

* Every surface becomes a vectorized function `f(points) -> signed value`;
  sense tests are `f < 0` / `f >= 0`.
* Cell geometry tokens are compiled to an AST with MCNP precedence
  (complement > implicit intersection > union `:`), evaluated with numpy
  boolean arrays; `#n` resolves through the referenced cell (with its `trcl`),
  with cycle protection. Surface values are cached per point set.
* `classify(points, universe)` walks the cells of a universe in input order;
  points matched by a `fill`ed cell are transformed (trcl/fill transform) and
  recursively classified in the filled universe.
* **`lat=1` lattices**: element extents and the i/j/k directions are derived
  from the *order in which the bounding planes are listed on the cell card*
  (MCNP convention: the element beyond the first listed surface is (1,0,0)).
  Lattice indices map into the `fill` array (i fastest, then j, then k);
  an element whose universe number equals the lattice cell's own universe is
  filled with the lattice cell's own material.
* Output per point: material number (or `0` void, `OUTSIDE`, `UNDEFINED`) and
  a hash of the full cell/lattice-element path, used to draw boundary lines.

### 3. Renderer (`plotter.py`)

* Builds a pixel grid over the model's bounding box (auto-computed from the
  surfaces referenced by real-world cells, i.e. excluding imp=0 cells),
  classifies all pixel centers (the slice coordinate is nudged by 1e-6 cm so
  pixels never lie exactly on a surface), maps materials to colors and
  overlays black boundaries where the cell-path hash changes between
  neighboring pixels.
* Shared figure builder used by both the GUI and the headless CLI.

### 4. GUI (`gui.py`, `main.py`)

* `python PyVised/main.py` → startup window → "Open MCNP input…" file dialog →
  viewer window (front + top view, navigation toolbar for zoom/pan, editable
  slice coordinates, resolution selector, redraw button, status bar).
* Rendering runs in a worker thread so the window stays responsive.
* `python PyVised/main.py FILE --headless --out img.png` renders without a
  display (used for automated verification), with `--front-y/--top-z/--res/
  --front-window/--top-window` overrides.

## Known limitations

* `lat=2` (hexagonal) lattices, surface facets (`mb.k`), `rec/rhp/wed/arb/ell`
  and point-defined `x/y/z` surfaces are not supported (clear error/warning).
* Partially specified TR rotation matrices (5/6 entries) are not completed.
* Plot-only fidelity: densities, physics and tallies are ignored.

## Verification

* Headless renders of the three decks in `examples/` are compared against the
  reference MCNP Vised picture `reactor.png` (core layout, fuel-pin lattices,
  truncated assemblies, water holes in graphite, Al walls).
* Undefined (magenta) pixels are counted and reported — a non-zero count
  signals a geometry/parser bug for these known-good decks.
