"""PyVised — MCNP input geometry viewer.

Usage:
    python main.py                          start with the file-selector window
    python main.py FILE                     open FILE directly in the viewer
    python main.py FILE --headless          render to a PNG without a display
        [--out IMG.png] [--res N] [--front-y Y] [--top-z Z]
        [--front-window XMIN XMAX ZMIN ZMAX] [--top-window XMIN XMAX YMIN YMAX]
"""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))


def _resolution(value):
    res = int(value)
    if not 50 <= res <= 3000:
        raise argparse.ArgumentTypeError("resolution must be between 50 and 3000")
    return res


def build_argparser():
    p = argparse.ArgumentParser(
        prog="PyVised",
        description="Display the front (X-Z @ Y=0) and top (X-Y @ Z=0) "
                    "geometry cuts of an MCNP input file, colored by material.")
    p.add_argument("file", nargs="?", help="MCNP input file (.mcnp/.txt/...)")
    p.add_argument("--headless", action="store_true",
                   help="render to PNG without opening a window")
    p.add_argument("--out", help="output PNG path (headless mode)")
    p.add_argument("--res", type=_resolution, default=900,
                   help="pixels along the longer image side, 50-3000 (default 900)")
    p.add_argument("--front-y", type=float, default=0.0,
                   help="Y coordinate of the front cut (default 0)")
    p.add_argument("--top-z", type=float, default=0.0,
                   help="Z coordinate of the top cut (default 0)")
    p.add_argument("--front-window", nargs=4, type=float,
                   metavar=("XMIN", "XMAX", "ZMIN", "ZMAX"),
                   help="zoom window of the front view")
    p.add_argument("--top-window", nargs=4, type=float,
                   metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
                   help="zoom window of the top view")
    return p


def run_headless(args):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    import mcnp_parser
    import plotter
    from geometry import Evaluator, GeometryError

    out = args.out or str(pathlib.Path(args.file).with_suffix("")) + "_views.png"
    try:
        deck = mcnp_parser.parse_file(args.file)
        evaluator = Evaluator(deck)
        fig = Figure(figsize=(14, 7.5), dpi=110)
        front, top = plotter.render_figure(
            fig, evaluator, deck,
            front_coord=args.front_y, top_coord=args.top_z,
            resolution=args.res,
            front_window=tuple(args.front_window) if args.front_window else None,
            top_window=tuple(args.top_window) if args.top_window else None)
        fig.savefig(out)
    except (mcnp_parser.MCNPParseError, GeometryError, OSError, ValueError) as exc:
        print(f"PyVised error: {exc}", file=sys.stderr)
        return 2
    print(f"title       : {deck.title}")
    print(f"cells       : {len(deck.cells)}   surfaces: {len(deck.surfaces)}   "
          f"materials: {deck.materials}")
    print(f"front view  : {front.rgb.shape[1]}x{front.rgb.shape[0]} px, "
          f"{front.n_undefined} undefined px, {front.elapsed:.2f} s")
    print(f"top view    : {top.rgb.shape[1]}x{top.rgb.shape[0]} px, "
          f"{top.n_undefined} undefined px, {top.elapsed:.2f} s")
    for w in deck.warnings:
        print(f"warning     : {w}")
    print(f"saved       : {out}")
    return 0


def main(argv=None):
    # accented deck titles/warnings must never crash console output on
    # Windows code pages (piped stdout defaults to cp1252)
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    args = build_argparser().parse_args(argv)
    if args.headless:
        if not args.file:
            print("PyVised error: --headless needs an input file", file=sys.stderr)
            return 2
        return run_headless(args)
    from gui import run_app
    run_app(args.file, front_y=args.front_y, top_z=args.top_z,
            resolution=args.res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
