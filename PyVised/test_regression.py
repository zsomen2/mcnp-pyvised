"""Parser/geometry regression spot-checks (run: python PyVised/test_regression.py)."""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np

import mcnp_parser
from geometry import Evaluator, UNDEFINED, OUTSIDE

PASS = []
FAIL = []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name + (f"  [{detail}]" if detail and not cond else ""))


def deck(text):
    return mcnp_parser.parse_string(text)


def classify_one(ev, xyz):
    mat, _ = ev.classify(np.array([xyz], dtype=float))
    return int(mat[0])


# 1. negative universe numbers (u=-n) -----------------------------------------
d = deck("""neg universe
1 1 -1.0 -1 u=-3 imp:n=1
2 0 -2 fill=3 imp:n=1
3 0 2 imp:n=0

1 so 5
2 so 10

m1 1001.50c 1
""")
ev = Evaluator(d)
check("u=-n normalized", classify_one(ev, (0, 0, 0)) == 1)

# 2. IMP given as a data card --------------------------------------------------
d = deck("""imp data card
1 1 -1.0 -1
2 0 1 -3
3 0 3 -2
4 0 2

1 so 5
3 so 6
2 so 5000

m1 1001.50c 1
imp:n 1 1 0 0
""")
ev = Evaluator(d)
check("imp data card: outside", classify_one(ev, (1000, 0, 0)) == OUTSIDE)
bb = ev.world_bbox()
check("imp data card: bbox excludes graveyard", bb[0][1] < 100, str(bb))

# 3. 3-point plane sense is independent of point listing order ----------------
base = """3pt plane {}
1 1 -1.0 -1 2 imp:n=1
2 0 -1 -2 imp:n=1
3 0 1 imp:n=0

1 so 10
2 p {}

m1 1001.50c 1
"""
pts_a = "0 0 5  1 0 5  0 1 5"
pts_b = "0 0 5  0 1 5  1 0 5"   # swapped -> opposite raw normal
ma = classify_one(Evaluator(deck(base.format("a", pts_a))), (0, 0, 7))
mb = classify_one(Evaluator(deck(base.format("b", pts_b))), (0, 0, 7))
check("3-pt plane order independent", ma == mb == 1, f"{ma} vs {mb}")

# 4. lattice direction with a negative-coefficient P plane --------------------
lat = """lat p-plane {}
1 1 -1.0 -10 u=1 imp:n=1
2 0 -10 #1 u=1 imp:n=1
3 0 {} lat=1 fill=0:1 0:0 0:0 1 1 u=2 imp:n=1
4 0 -5 fill=2 imp:n=1
5 0 5 imp:n=0

10 so 0.4
5 rpp -1 3 -1 1 -1 1
1 px 1
2 px -1
3 py 1
4 py -1
11 p -1 0 0 -1

m1 1001.50c 1
"""
m_px = classify_one(Evaluator(deck(lat.format("px", "-1 2 -3 4"))), (2.0, 0, 0))
m_pp = classify_one(Evaluator(deck(lat.format("p", "11 2 -3 4"))), (2.0, 0, 0))
check("lattice dir: P plane == px plane", m_px == m_pp == 1, f"{m_px} vs {m_pp}")

# 5. TR forms ------------------------------------------------------------------
t9 = mcnp_parser.make_transform("30 0 0 1 0 0 0 1 0".split())
check("TR 9-entry keeps shift+rotation",
      np.allclose(t9.origin, [30, 0, 0]) and t9.rot is not None
      and np.allclose(t9.rot, np.eye(3)))
tj = mcnp_parser.make_transform("5 0 0 j j j j j j j j j".split())
check("TR with j jumps", np.allclose(tj.origin, [5, 0, 0])
      and np.allclose(tj.rot, np.eye(3)))
d = deck("""tr fallback
1 0 -1 trcl=1 imp:n=1
2 0 1 imp:n=0

1 so 5

tr1 7 0 0 0.5 0.5
""")
check("unsupported TR keeps translation",
      np.allclose(d.transforms[1].origin, [7, 0, 0]) and d.transforms[1].rot is None
      and any("TR1" in w for w in d.warnings), str(d.warnings))

# 6. bare 'r' repeat in fill arrays --------------------------------------------
d = deck("""bare r
1 1 -1.0 -10 u=1 imp:n=1
2 0 -10 #1 u=1 imp:n=1
3 0 -1 2 -3 4 lat=1 fill=0:1 0:0 0:0 1 r u=2 imp:n=1
4 0 -5 fill=2 imp:n=1
5 0 5 imp:n=0

10 so 0.4
5 rpp -1 3 -1 1 -1 1
1 px 1
2 px -1
3 py 1
4 py -1

m1 1001.50c 1
""")
check("bare r repeat in fill", classify_one(Evaluator(d), (2.0, 0, 0)) == 1)

# 7. density glued to a parenthesis --------------------------------------------
d = deck("""glued density
1 1 -1.0(-1:-2) imp:n=1
2 0 1 2 imp:n=0

1 so 5
2 so 3

m1 1001.50c 1
""")
check("density glued to paren", d.cells[1].density == -1.0
      and classify_one(Evaluator(d), (0, 0, 0)) == 1)

# 8. LIKE n BUT with empty change list -----------------------------------------
d = deck("""like but empty
1 1 -1.0 -1 imp:n=1
2 like 1 but
3 0 1 imp:n=0

1 so 5

m1 1001.50c 1
""")
check("like-but empty list", d.cells[2].material == 1)

# 9. facet reference error message ----------------------------------------------
try:
    deck("""facet
1 0 -1.2 imp:n=1
2 0 1 imp:n=0

1 rpp -1 1 -1 1 -1 1
""")
    check("facet ref message", False, "no error raised")
except mcnp_parser.MCNPParseError as exc:
    check("facet ref message", "facet" in str(exc).lower(), str(exc))

# 10. fill array entry 0 -> undefined -------------------------------------------
d = deck("""fill zero
1 1 -1.0 -10 u=1 imp:n=1
2 0 -10 #1 u=1 imp:n=1
3 0 -1 2 -3 4 lat=1 fill=0:1 0:0 0:0 1 0 u=2 imp:n=1
4 0 -5 fill=2 imp:n=1
5 0 5 imp:n=0

10 so 0.4
5 rpp -1 3 -1 1 -1 1
1 px 1
2 px -1
3 py 1
4 py -1

m1 1001.50c 1
""")
ev = Evaluator(d)
check("fill entry 0 undefined", classify_one(ev, (2.0, 0, 0)) == UNDEFINED
      and any("entry 0" in w for w in d.warnings))

# 11. regression: example decks still classify the same key points --------------
root = pathlib.Path(__file__).resolve().parents[1]
d = mcnp_parser.parse_file(root / "examples" / "HV_oktatoreaktor.mcnp")
ev = Evaluator(d)
check("HV fuel pin at origin", classify_one(ev, (0, 0, 0)) == 1)
check("HV water in tank", classify_one(ev, (-4.55, -0.85, 60)) == 2)
check("HV outside", classify_one(ev, (80, 0, 0)) == OUTSIDE)

print(f"PASS {len(PASS)}  FAIL {len(FAIL)}")
for name in FAIL:
    print("FAIL:", name)
sys.exit(1 if FAIL else 0)
