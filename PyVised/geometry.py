"""Point-in-cell engine for plotting MCNP geometries.

Every surface becomes a vectorized signed function f(points); cell geometry
expressions are compiled into a small AST and evaluated with numpy boolean
arrays.  ``Evaluator.classify`` maps an (N,3) array of points to the material
found at each point, recursing through FILL universes and LAT=1 lattices.
"""

import re

import numpy as np

from mcnp_parser import MCNPParseError

UNDEFINED = -1   # no cell claims the point (geometry error -> magenta)
OUTSIDE = -2     # void cell with importance 0 (the "outside world")

_MAX_DEPTH = 50


class GeometryError(Exception):
    pass


# --------------------------------------------------------------------------
# surface evaluation functions
# --------------------------------------------------------------------------

def _need(surf, n):
    if len(surf.coeffs) < n:
        raise GeometryError(
            f"surface {surf.sid} ({surf.mnemonic}) needs {n} coefficients, "
            f"got {len(surf.coeffs)}")
    return surf.coeffs


def _axis_of(ch):
    return {"x": 0, "y": 1, "z": 2}[ch]


def _plane_fn(axis, value):
    return lambda P: P[:, axis] - value


def _general_plane_fn(a, b, c, d):
    n = np.array([a, b, c], dtype=float)
    return lambda P: P @ n - d


def _sphere_fn(cx, cy, cz, r):
    c = np.array([cx, cy, cz], dtype=float)
    r2 = r * r
    return lambda P: ((P - c) ** 2).sum(axis=1) - r2


def _cyl_fn(axis, c1, c2, r):
    """Infinite cylinder parallel to `axis`; (c1,c2) are the two other coords."""
    o1, o2 = [a for a in (0, 1, 2) if a != axis]
    r2 = r * r
    return lambda P: (P[:, o1] - c1) ** 2 + (P[:, o2] - c2) ** 2 - r2


def _cone_fn(axis, apex, t2, sheet):
    o1, o2 = [a for a in (0, 1, 2) if a != axis]
    ax_apex = apex[axis]

    def f(P):
        d = P[:, axis] - ax_apex
        val = ((P[:, o1] - apex[o1]) ** 2 + (P[:, o2] - apex[o2]) ** 2
               - t2 * d * d)
        if sheet:
            val = np.where(np.sign(d) * sheet < 0, np.abs(val) + 1.0, val)
        return val
    return f


def _sq_fn(c):
    a, b, cc, d, e, f, g, x0, y0, z0 = c[:10]

    def fn(P):
        dx = P[:, 0] - x0
        dy = P[:, 1] - y0
        dz = P[:, 2] - z0
        return (a * dx * dx + b * dy * dy + cc * dz * dz
                + 2 * d * dx + 2 * e * dy + 2 * f * dz + g)
    return fn


def _gq_fn(c):
    a, b, cc, d, e, f, g, h, j, k = c[:10]

    def fn(P):
        x, y, z = P[:, 0], P[:, 1], P[:, 2]
        return (a * x * x + b * y * y + cc * z * z + d * x * y + e * y * z
                + f * z * x + g * x + h * y + j * z + k)
    return fn


def _torus_fn(axis, c):
    x0, y0, z0, a, b, cr = c[:6]
    center = (x0, y0, z0)
    o1, o2 = [ax for ax in (0, 1, 2) if ax != axis]

    def fn(P):
        ring = np.sqrt((P[:, o1] - center[o1]) ** 2
                       + (P[:, o2] - center[o2]) ** 2) - a
        axial = P[:, axis] - center[axis]
        return (axial * axial) / (b * b) + (ring * ring) / (cr * cr) - 1.0
    return fn


# ---- macrobodies: f = max over face functions (negative inside) ----

def _rpp_fn(c):
    x0, x1, y0, y1, z0, z1 = c[:6]

    def fn(P):
        return np.maximum.reduce([
            x0 - P[:, 0], P[:, 0] - x1,
            y0 - P[:, 1], P[:, 1] - y1,
            z0 - P[:, 2], P[:, 2] - z1])
    return fn


def _box_fn(c):
    v = np.array(c[0:3])
    edges = [np.array(c[3:6]), np.array(c[6:9]), np.array(c[9:12])]

    def fn(P):
        q = P - v
        faces = []
        for a in edges:
            ln = np.linalg.norm(a)
            t = q @ (a / ln)
            faces.extend([-t, t - ln])
        return np.maximum.reduce(faces)
    return fn


def _rcc_fn(c):
    v = np.array(c[0:3])
    h = np.array(c[3:6])
    r = c[6]
    hlen = np.linalg.norm(h)
    hu = h / hlen

    def fn(P):
        q = P - v
        t = q @ hu
        perp = np.linalg.norm(q - np.outer(t, hu), axis=1)
        return np.maximum.reduce([perp - r, -t, t - hlen])
    return fn


def _trc_fn(c):
    v = np.array(c[0:3])
    h = np.array(c[3:6])
    r1, r2 = c[6], c[7]
    hlen = np.linalg.norm(h)
    hu = h / hlen

    def fn(P):
        q = P - v
        t = q @ hu
        perp = np.linalg.norm(q - np.outer(t, hu), axis=1)
        return np.maximum.reduce([perp - (r1 + (r2 - r1) * t / hlen), -t, t - hlen])
    return fn


def make_surface_function(surf):
    """Return a vectorized f(P[N,3]) -> values; negative means 'inside'."""
    mn = surf.mnemonic
    c = surf.coeffs

    if mn in ("px", "py", "pz"):
        _need(surf, 1)
        fn = _plane_fn(_axis_of(mn[1]), c[0])
    elif mn == "p":
        if len(c) == 4:
            fn = _general_plane_fn(*c)
        elif len(c) == 9:
            p1, p2, p3 = (np.array(c[0:3]), np.array(c[3:6]), np.array(c[6:9]))
            n = np.cross(p2 - p1, p3 - p1)
            if not np.any(n):
                raise GeometryError(f"surface {surf.sid}: degenerate 3-point plane")
            d = float(n @ p1)
            # MCNP normalizes the sign independently of the point listing
            # order: D > 0, else C > 0, else B > 0, else A > 0.
            for key in (d, n[2], n[1], n[0]):
                if key != 0.0:
                    if key < 0.0:
                        n, d = -n, -d
                    break
            fn = _general_plane_fn(n[0], n[1], n[2], d)
        else:
            raise GeometryError(f"surface {surf.sid}: P needs 4 or 9 coefficients")
    elif mn == "so":
        _need(surf, 1)
        fn = _sphere_fn(0, 0, 0, c[0])
    elif mn == "s":
        _need(surf, 4)
        fn = _sphere_fn(*c[:4])
    elif mn in ("sx", "sy", "sz"):
        _need(surf, 2)
        center = [0.0, 0.0, 0.0]
        center[_axis_of(mn[1])] = c[0]
        fn = _sphere_fn(*center, c[1])
    elif mn in ("cx", "cy", "cz"):
        _need(surf, 1)
        fn = _cyl_fn(_axis_of(mn[1]), 0.0, 0.0, c[0])
    elif mn in ("c/x", "c/y", "c/z"):
        _need(surf, 3)
        fn = _cyl_fn(_axis_of(mn[2]), c[0], c[1], c[2])
    elif mn in ("kx", "ky", "kz"):
        _need(surf, 2)
        apex = [0.0, 0.0, 0.0]
        axis = _axis_of(mn[1])
        apex[axis] = c[0]
        sheet = int(c[2]) if len(c) > 2 else 0
        fn = _cone_fn(axis, apex, c[1], sheet)
    elif mn in ("k/x", "k/y", "k/z"):
        _need(surf, 4)
        axis = _axis_of(mn[2])
        sheet = int(c[4]) if len(c) > 4 else 0
        fn = _cone_fn(axis, c[0:3], c[3], sheet)
    elif mn == "sq":
        _need(surf, 10)
        fn = _sq_fn(c)
    elif mn == "gq":
        _need(surf, 10)
        fn = _gq_fn(c)
    elif mn in ("tx", "ty", "tz"):
        _need(surf, 6)
        fn = _torus_fn(_axis_of(mn[1]), c)
    elif mn == "rpp":
        _need(surf, 6)
        fn = _rpp_fn(c)
    elif mn == "box":
        _need(surf, 12)
        fn = _box_fn(c)
    elif mn == "sph":
        _need(surf, 4)
        fn = _sphere_fn(*c[:4])
    elif mn == "rcc":
        _need(surf, 7)
        fn = _rcc_fn(c)
    elif mn == "trc":
        _need(surf, 8)
        fn = _trc_fn(c)
    else:
        raise GeometryError(
            f"surface {surf.sid}: surface type '{mn}' is not supported by PyVised")

    if surf.transform is not None:
        tr = surf.transform
        base = fn
        fn = lambda P, _b=base, _t=tr: _b(_t.to_aux(P))
    return fn


# --------------------------------------------------------------------------
# region expression AST
# --------------------------------------------------------------------------

class SurfRef:
    __slots__ = ("sid", "sense")

    def __init__(self, sid, sense):
        self.sid = sid
        self.sense = sense


class NotCell:
    __slots__ = ("cid",)

    def __init__(self, cid):
        self.cid = cid


class NotExpr:
    __slots__ = ("node",)

    def __init__(self, node):
        self.node = node


class BoolOp:
    __slots__ = ("op", "children")

    def __init__(self, op, children):
        self.op = op            # '&' or ':'
        self.children = children


_PREC = {"#": 3, "&": 2, ":": 1}


def parse_region(tokens, where=""):
    """Compile geometry tokens into an AST (shunting-yard with implicit AND)."""
    output = []
    ops = []

    def apply(op):
        if op == "#":
            if not output:
                raise GeometryError(f"{where}: misplaced '#'")
            output.append(NotExpr(output.pop()))
        else:
            if len(output) < 2:
                raise GeometryError(f"{where}: malformed geometry expression")
            b = output.pop()
            a = output.pop()
            kids = []
            for n in (a, b):
                if isinstance(n, BoolOp) and n.op == op:
                    kids.extend(n.children)
                else:
                    kids.append(n)
            output.append(BoolOp(op, kids))

    def push_op(op):
        if op == "#":
            ops.append(op)  # unary, right associative: never pop on push
            return
        while ops and ops[-1] != "(" and _PREC.get(ops[-1], 0) >= _PREC[op]:
            apply(ops.pop())
        ops.append(op)

    prev_operand = False
    for tok in tokens:
        if re.fullmatch(r"[+-]?\d+", tok):
            if prev_operand:
                push_op("&")
            sid = int(tok)
            output.append(SurfRef(abs(sid), -1 if tok.lstrip().startswith("-") else 1))
            while ops and ops[-1] == "#":
                apply(ops.pop())
            prev_operand = True
        elif tok.startswith("#") and len(tok) > 1:
            if prev_operand:
                push_op("&")
            output.append(NotCell(int(tok[1:])))
            while ops and ops[-1] == "#":
                apply(ops.pop())
            prev_operand = True
        elif tok == "#":
            if prev_operand:
                push_op("&")
            push_op("#")
            prev_operand = False
        elif tok == "(":
            if prev_operand:
                push_op("&")
            ops.append("(")
            prev_operand = False
        elif tok == ")":
            while ops and ops[-1] != "(":
                apply(ops.pop())
            if not ops:
                raise GeometryError(f"{where}: unbalanced parentheses")
            ops.pop()
            while ops and ops[-1] == "#":
                apply(ops.pop())
            prev_operand = True
        elif tok == ":":
            push_op(":")
            prev_operand = False
        else:
            raise GeometryError(f"{where}: unexpected geometry token {tok!r}")
    while ops:
        op = ops.pop()
        if op == "(":
            raise GeometryError(f"{where}: unbalanced parentheses")
        apply(op)
    if len(output) != 1:
        raise GeometryError(f"{where}: malformed geometry expression")
    return output[0]


def ordered_surface_refs(tokens):
    """Surface ids with senses, in card order (cell complements excluded)."""
    refs = []
    for tok in tokens:
        if re.fullmatch(r"[+-]?\d+", tok):
            sid = int(tok)
            refs.append((abs(sid), -1 if sid < 0 else 1))
    return refs


# --------------------------------------------------------------------------
# lattices
# --------------------------------------------------------------------------

class LatAxis:
    __slots__ = ("axis", "lo", "hi", "direction")

    def __init__(self, axis, lo, hi, direction):
        self.axis = axis            # 0/1/2 spatial axis
        self.lo = lo
        self.hi = hi
        self.direction = direction  # +1 / -1: spatial direction of +index


class LatticeSpec:
    def __init__(self, axes):
        self.axes = axes            # up to three LatAxis, in i, j, k order

    def indices(self, P):
        """Return (idx[N,3] int array, local points[N,3])."""
        idx = np.zeros((len(P), 3), dtype=np.int64)
        local = P.copy()
        for n, ax in enumerate(self.axes):
            coord = P[:, ax.axis]
            pitch = ax.hi - ax.lo
            if ax.direction > 0:
                i = np.floor((coord - ax.lo) / pitch).astype(np.int64)
            else:
                i = np.floor((ax.hi - coord) / pitch).astype(np.int64)
            local[:, ax.axis] = coord - i * pitch * ax.direction
            idx[:, n] = i
        return idx, local


def _plane_axis_value(surf):
    """(axis, position, normal_sign) for axis-aligned planes, else None.

    ``normal_sign`` is the sign of f along +axis: +1 for px/py/pz, the sign
    of the non-zero coefficient for a general axis-aligned P card.
    """
    if surf.transform is not None:
        return None
    if surf.mnemonic in ("px", "py", "pz"):
        return _axis_of(surf.mnemonic[1]), surf.coeffs[0], 1
    if surf.mnemonic == "p" and len(surf.coeffs) == 4:
        a, b, c, d = surf.coeffs
        nz = [i for i, v in enumerate((a, b, c)) if v != 0.0]
        if len(nz) == 1:
            coeff = (a, b, c)[nz[0]]
            return nz[0], d / coeff, (1 if coeff > 0 else -1)
    return None


# --------------------------------------------------------------------------
# evaluator
# --------------------------------------------------------------------------

_PRIME = np.uint64(1099511628211)
_IDX_OFFSET = np.int64(1 << 20)


def _mix(sig, value):
    """FNV-style hash mixing on uint64 arrays (overflow wraps, that is fine)."""
    with np.errstate(over="ignore"):
        return sig * _PRIME + np.asarray(value).astype(np.uint64)


class Evaluator:
    def __init__(self, deck):
        self.deck = deck
        self.by_universe = {}
        for cell in deck.cells.values():
            self.by_universe.setdefault(cell.universe, []).append(cell)
        if 0 not in self.by_universe:
            raise GeometryError("the deck has no cells in universe 0")
        self._surf_fn = {}
        self._region = {}
        self._lattice = {}
        self.max_depth_hit = False

    # -- lazy caches -------------------------------------------------------

    def surface_function(self, sid):
        fn = self._surf_fn.get(sid)
        if fn is None:
            if sid not in self.deck.surfaces:
                raise GeometryError(f"surface {sid} is referenced but not defined")
            fn = make_surface_function(self.deck.surfaces[sid])
            self._surf_fn[sid] = fn
        return fn

    def region(self, cid):
        node = self._region.get(cid)
        if node is None:
            cell = self.deck.cells[cid]
            node = parse_region(cell.geom_tokens, where=f"cell {cid}")
            self._region[cid] = node
        return node

    def lattice_spec(self, cell):
        spec = self._lattice.get(cell.cid)
        if spec is None:
            spec = self._build_lattice_spec(cell)
            self._lattice[cell.cid] = spec
        return spec

    def _build_lattice_spec(self, cell):
        if cell.lat != 1:
            raise GeometryError(
                f"cell {cell.cid}: LAT={cell.lat} is not supported (only LAT=1)")
        refs = ordered_surface_refs(cell.geom_tokens)
        axes = []
        pos = 0
        while pos + 1 < len(refs) and len(axes) < 3:
            (sid_a, sense_a), (sid_b, sense_b) = refs[pos], refs[pos + 1]
            pa = _plane_axis_value(self.deck.surfaces[sid_a])
            pb = _plane_axis_value(self.deck.surfaces[sid_b])
            if pa is None or pb is None or pa[0] != pb[0]:
                if not axes:
                    raise GeometryError(
                        f"cell {cell.cid}: LAT=1 element must be bounded by "
                        "axis-aligned plane pairs")
                break
            lo, hi = sorted((pa[1], pb[1]))
            if hi - lo <= 0:
                raise GeometryError(f"cell {cell.cid}: zero lattice pitch")
            # MCNP: element (1,0,0) lies beyond the FIRST listed surface.
            # The effective sense along +axis also depends on the plane's
            # normal sign (a P card may have a negative coefficient).
            direction = 1 if sense_a * pa[2] < 0 else -1
            axes.append(LatAxis(pa[0], lo, hi, direction))
            pos += 2
        if not axes:
            raise GeometryError(f"cell {cell.cid}: could not derive lattice spacing")
        return LatticeSpec(axes)

    # -- region / containment evaluation ------------------------------------

    def _eval_node(self, node, P, cache, frame, stack):
        if isinstance(node, SurfRef):
            key = (node.sid, frame)
            vals = cache.get(key)
            if vals is None:
                vals = self.surface_function(node.sid)(P)
                cache[key] = vals
            return vals < 0 if node.sense < 0 else vals >= 0
        if isinstance(node, BoolOp):
            result = self._eval_node(node.children[0], P, cache, frame, stack)
            for child in node.children[1:]:
                nxt = self._eval_node(child, P, cache, frame, stack)
                result = (result & nxt) if node.op == "&" else (result | nxt)
            return result
        if isinstance(node, NotExpr):
            return ~self._eval_node(node.node, P, cache, frame, stack)
        if isinstance(node, NotCell):
            if node.cid in stack:
                raise GeometryError(
                    f"circular cell complement involving cell {node.cid}")
            if node.cid not in self.deck.cells:
                raise GeometryError(
                    f"complement references unknown cell {node.cid}")
            target = self.deck.cells[node.cid]
            return ~self.cell_contains(target, P, cache, frame,
                                       stack + (node.cid,))
        raise GeometryError("internal: unknown AST node")

    def cell_contains(self, cell, P, cache, frame=0, stack=()):
        if cell.trcl is not None:
            P = cell.trcl.to_aux(P)
            frame = ("trcl", cell.cid, frame)
        return self._eval_node(self.region(cell.cid), P, cache, frame, stack)

    # -- classification ------------------------------------------------------

    def classify(self, P, universe=0, depth=0):
        """Return (mat[N] int32, sig[N] uint64) for points P in `universe`."""
        n = len(P)
        mat = np.full(n, UNDEFINED, dtype=np.int32)
        sig = np.zeros(n, dtype=np.uint64)
        if depth > _MAX_DEPTH:
            self.max_depth_hit = True
            return mat, sig

        cells = self.by_universe.get(universe)
        if not cells:
            return mat, sig

        cache = {}
        remaining = np.arange(n)
        for cell in cells:
            if remaining.size == 0:
                break
            if cell.lat:
                inside_full = None  # decided below, per remaining point
                Pf = cell.trcl.to_aux(P) if cell.trcl is not None else P
                idx3, local = self.lattice_spec(cell).indices(Pf)
                if cell.fill is None:
                    raise GeometryError(f"cell {cell.cid}: LAT=1 cell without FILL")
                if cell.fill.is_lattice_array:
                    rng = cell.fill.ranges
                    inside_all = ((idx3[:, 0] >= rng[0][0]) & (idx3[:, 0] <= rng[0][1])
                                  & (idx3[:, 1] >= rng[1][0]) & (idx3[:, 1] <= rng[1][1])
                                  & (idx3[:, 2] >= rng[2][0]) & (idx3[:, 2] <= rng[2][1]))
                else:
                    inside_all = np.ones(n, dtype=bool)
                sub = inside_all[remaining]
                idx = remaining[sub]
                remaining = remaining[~sub]
                if idx.size:
                    self._enter_lattice(cell, idx, idx3[idx], local[idx],
                                        mat, sig, depth)
                continue

            inside_all = self.cell_contains(cell, P, cache)
            sub = inside_all[remaining]
            idx = remaining[sub]
            remaining = remaining[~sub]
            if not idx.size:
                continue
            if cell.fill is None:
                if cell.material == 0 and cell.imp_zero:
                    mat[idx] = OUTSIDE
                else:
                    mat[idx] = cell.material
                sig[idx] = _mix(np.zeros(idx.size, np.uint64), cell.cid)
            else:
                self._enter_fill(cell, P[idx], idx, mat, sig, depth)
        return mat, sig

    def _enter_fill(self, cell, pts, idx, mat, sig, depth):
        if cell.trcl is not None:
            pts = cell.trcl.to_aux(pts)
        if cell.fill.transform is not None:
            pts = cell.fill.transform.to_aux(pts)
        sub_mat, sub_sig = self.classify(pts, cell.fill.universe, depth + 1)
        mat[idx] = sub_mat
        base = _mix(np.zeros(idx.size, np.uint64), cell.cid)
        sig[idx] = _mix(base, np.zeros(idx.size, np.uint64)) + sub_sig

    def _enter_lattice(self, cell, idx, idx3, local, mat, sig, depth):
        elem_sig = _mix(np.zeros(idx.size, np.uint64), cell.cid)
        for col in range(3):
            elem_sig = _mix(elem_sig, idx3[:, col] + _IDX_OFFSET)

        if cell.fill.is_lattice_array:
            rng = cell.fill.ranges
            u_pt = cell.fill.array[idx3[:, 2] - rng[2][0],
                                   idx3[:, 1] - rng[1][0],
                                   idx3[:, 0] - rng[0][0]]
        else:
            u_pt = np.full(idx.size, cell.fill.universe, dtype=np.int64)

        for u in np.unique(u_pt):
            sel = u_pt == u
            gidx = idx[sel]
            if u == 0:
                # a zero FILL entry means "element does not exist" in MCNP
                self.deck.warn(
                    f"cell {cell.cid}: FILL array entry 0 treated as undefined")
                mat[gidx] = UNDEFINED
                sig[gidx] = elem_sig[sel]
            elif u == cell.universe:
                # element filled with the lattice cell's own material
                mat[gidx] = cell.material
                sig[gidx] = elem_sig[sel]
            elif int(u) not in self.by_universe:
                self.deck.warn(
                    f"cell {cell.cid}: FILL references unknown universe {int(u)}")
                mat[gidx] = UNDEFINED
            else:
                sub_mat, sub_sig = self.classify(local[sel], int(u), depth + 1)
                mat[gidx] = sub_mat
                with np.errstate(over="ignore"):
                    sig[gidx] = elem_sig[sel] * _PRIME + sub_sig

    # -- bounding box ---------------------------------------------------------

    def world_bbox(self):
        """((xmin,xmax),(ymin,ymax),(zmin,zmax)) of the universe-0 geometry."""
        points = [[], [], []]
        for cell in self.by_universe.get(0, []):
            if cell.imp_zero:
                continue
            shift = np.zeros(3)
            if cell.trcl is not None:
                if not cell.trcl.shift_only:
                    # rotated cell: skip its contribution (conservative,
                    # same policy as _surface_extent for rotated surfaces)
                    continue
                shift = cell.trcl.origin if cell.trcl.m >= 0 else -cell.trcl.origin
            for sid, _sense in ordered_surface_refs(cell.geom_tokens):
                surf = self.deck.surfaces.get(sid)
                if surf is None:
                    continue
                for axis, vals in _surface_extent(surf).items():
                    points[axis].extend(v + shift[axis] for v in vals)
        bbox = []
        for axis in range(3):
            if points[axis]:
                lo, hi = min(points[axis]), max(points[axis])
                if hi - lo < 1e-9:
                    lo, hi = lo - 10.0, hi + 10.0
            else:
                lo, hi = -50.0, 50.0
                self.deck.warn(
                    "could not determine the model extent along axis "
                    f"{'xyz'[axis]}; using +/-50 cm")
            margin = 0.02 * (hi - lo)
            bbox.append((lo - margin, hi + margin))
        return tuple(bbox)


def _surface_extent(surf):
    """Map axis -> list of coordinate values this surface can contribute."""
    mn = surf.mnemonic
    c = surf.coeffs
    tr_shift = np.zeros(3)
    if surf.transform is not None:
        if not surf.transform.shift_only:
            return {}
        tr_shift = (surf.transform.origin if surf.transform.m >= 0
                    else -surf.transform.origin)
    out = {}

    def add(axis, *vals):
        out.setdefault(axis, []).extend(v + tr_shift[axis] for v in vals)

    if mn in ("px", "py", "pz") and len(c) >= 1:
        add(_axis_of(mn[1]), c[0])
    elif mn == "p" and len(c) == 4:
        a, b, cc, d = c
        nz = [i for i, v in enumerate((a, b, cc)) if v != 0.0]
        if len(nz) == 1:
            add(nz[0], d / (a, b, cc)[nz[0]])
    elif mn in ("so",) and len(c) >= 1:
        for axis in range(3):
            add(axis, -c[0], c[0])
    elif mn in ("s", "sph") and len(c) >= 4:
        for axis in range(3):
            add(axis, c[axis] - c[3], c[axis] + c[3])
    elif mn in ("sx", "sy", "sz") and len(c) >= 2:
        center = [0.0, 0.0, 0.0]
        center[_axis_of(mn[1])] = c[0]
        for axis in range(3):
            add(axis, center[axis] - c[1], center[axis] + c[1])
    elif mn in ("cx", "cy", "cz") and len(c) >= 1:
        axis = _axis_of(mn[1])
        for other in (a for a in range(3) if a != axis):
            add(other, -c[0], c[0])
    elif mn in ("c/x", "c/y", "c/z") and len(c) >= 3:
        axis = _axis_of(mn[2])
        others = [a for a in range(3) if a != axis]
        add(others[0], c[0] - c[2], c[0] + c[2])
        add(others[1], c[1] - c[2], c[1] + c[2])
    elif mn == "rpp" and len(c) >= 6:
        add(0, c[0], c[1])
        add(1, c[2], c[3])
        add(2, c[4], c[5])
    elif mn == "box" and len(c) >= 12:
        v = np.array(c[0:3])
        for bits in range(8):
            corner = v.copy()
            for e in range(3):
                if bits >> e & 1:
                    corner = corner + np.array(c[3 + 3 * e: 6 + 3 * e])
            for axis in range(3):
                add(axis, corner[axis])
    elif mn in ("rcc", "trc") and len(c) >= 7:
        v = np.array(c[0:3])
        h = np.array(c[3:6])
        r = max(c[6:8]) if mn == "trc" else c[6]
        for end in (v, v + h):
            for axis in range(3):
                add(axis, end[axis] - r, end[axis] + r)
    return out
