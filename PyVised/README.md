# PyVised

A small MCNP input **geometry viewer** (an "MCNP Vised"-style plotter) in pure
Python — it parses an MCNP input deck and displays two geometry cuts, with one
color per material. **No transport calculation is run.**

* **Front view** — X–Z plane cut at **Y = 0** (the middle)
* **Top view** — X–Y plane cut at **Z = 0** (the middle)

Requirements: Python 3.10+, `numpy`, `matplotlib` (with the Tk backend, which
ships with the standard Python installer on Windows).

## Run it

```bash
python PyVised/main.py
```

A startup window appears — click **Open MCNP input…** and select a `.mcnp` /
`.txt` / `.inp` file. The viewer window shows both cuts with:

* both projections always shown in a **square 1:1 display window** (equal cm
  spans, centered on the model / your zoom region),
* one color per material, MCNP Vised-like palette (M1 blue, M2 yellow,
  M3 green, M4 red, …), void = white, imp=0 outside world = dark grey,
  undefined regions = magenta,
* black cell boundary lines (including lattice element boundaries),
* zoom with the **mouse wheel** or the toolbar, pan with the toolbar — the
  view **re-renders itself at screen resolution after every zoom/pan**, so it
  stays sharp at any magnification (like Vised, which also re-rasterizes),
* **sliders to move the cut positions interactively** (front-view Y and
  top-view Z, spanning the model extent), synced with the text entries —
  the view follows the slider while you drag, keeping your current zoom,
* resolution selector + **Redraw** (re-render, zoom back to full view) +
  **Reset view** (cuts back to Y=0 / Z=0 and zoom back to the full model).

You can also open a file directly:

```bash
python PyVised/main.py examples/HV_oktatoreaktor.mcnp
```

### Headless rendering (no window)

```bash
python PyVised/main.py examples/HV_oktatoreaktor.mcnp --headless --out views.png
```

Options: `--res N` (pixels on the longer side), `--front-y Y`, `--top-z Z`,
`--front-window XMIN XMAX ZMIN ZMAX`, `--top-window XMIN XMAX YMIN YMAX`.

## What is supported

Cells with intersection/union/complement geometry (`-1 2 : (3 -4) #5 #(6 7)`),
universes (`u=`, negative numbers normalized), `fill=` (single universe,
optional transform), **`lat=1` lattices** with full fill arrays
(`fill=-2:1 -2:1 0:0 …` and `nR`/bare `r` repeats), `trcl`/`*trcl`,
`TRn`/`*TRn` cards (3-, 9-, 12/13-entry forms and `j` defaults), `like n but`,
importances on cell cards or as `imp:n` data cards, and the surface types
`p px py pz | so s sx sy sz | cx cy cz c/x c/y c/z | kx ky kz k/x k/y k/z |
sq gq | tx ty tz | rpp box sph rcc trc` (reflecting `*` prefixes are accepted).

Not supported: `lat=2` hexagonal lattices, macrobody facet references
(`12.3`), `rec/rhp/wed/arb/ell` and point-defined `X/Y/Z` surfaces, 5/6-entry
TR rotations (translation is kept, rotation warned about). See `PLAN.md` for
design details, and run `python PyVised/test_regression.py` for the
parser/geometry self-checks.
