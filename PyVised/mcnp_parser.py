"""Parser for MCNP input decks — geometry-relevant cards only.

Extracts the title, cell cards, surface cards, TR (transformation) cards and
material numbers from an MCNP input file.  Physics/tally data cards are
ignored: the goal is to have everything needed to *plot* the geometry.
"""

import math
import re

import numpy as np


class MCNPParseError(Exception):
    """Raised when the input deck cannot be parsed."""


# --------------------------------------------------------------------------
# coordinate transformations (TR cards, TRCL, FILL transforms)
# --------------------------------------------------------------------------

class Transform:
    """An MCNP TR-style transformation.

    ``origin`` is the displacement vector O1 O2 O3 and ``rot`` is a 3x3
    matrix whose *columns* are the auxiliary-system axes expressed in
    main-system coordinates (built from B1..B9).  ``m`` is the meaning flag
    (13th entry, default +1).
    """

    def __init__(self, origin, rot=None, m=1):
        self.origin = np.asarray(origin, dtype=float)
        self.rot = None if rot is None else np.asarray(rot, dtype=float)
        self.m = m

    def to_aux(self, pts):
        """Map an (N,3) array of main-system points into the auxiliary system."""
        if self.m >= 0:
            q = pts - self.origin
            if self.rot is not None:
                q = q @ self.rot
            return q
        q = pts @ self.rot if self.rot is not None else pts
        return q + self.origin

    @property
    def shift_only(self):
        return self.rot is None

    def __repr__(self):
        return f"Transform(origin={self.origin.tolist()}, rot={'yes' if self.rot is not None else 'no'}, m={self.m})"


# default TR entries (O=0, B=identity, M=+1); 'j' picks the default — note the
# defaults are *cosines* even on a *TR card, so they bypass degree conversion
_TR_DEFAULTS = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0)


def make_transform(values, degrees=False):
    """Build a Transform from the entries of a TR card / inline form."""
    vals = []
    defaulted = []
    for i, v in enumerate(values):
        if isinstance(v, str) and v.strip().lower() == "j":
            if i >= len(_TR_DEFAULTS):
                raise MCNPParseError("too many TR entries")
            vals.append(_TR_DEFAULTS[i])
            defaulted.append(True)
        else:
            vals.append(float(v))
            defaulted.append(False)
    m = 1
    if len(vals) == 13:
        m = int(vals[12])
        vals = vals[:12]
        defaulted = defaulted[:12]
    if len(vals) == 3:
        return Transform(vals)
    if len(vals) in (9, 12):
        b = vals[3:]
        if degrees:
            b = [v if dflt else math.cos(math.radians(v))
                 for v, dflt in zip(b, defaulted[3:])]
        if len(b) == 6:
            # origin + first two rows (aux x' and y' in main coords);
            # the third row follows from right-handed orthogonality.
            r1 = np.array(b[0:3], dtype=float)
            r2 = np.array(b[3:6], dtype=float)
            rows = np.array([r1, r2, np.cross(r1, r2)], dtype=float)
        else:
            rows = np.array(b, dtype=float).reshape(3, 3)
        # rows are the aux axes in main coordinates -> transpose puts them
        # into columns, which is what Transform expects.
        return Transform(vals[:3], rows.T, m)
    raise MCNPParseError(
        f"transformation with {len(vals)} entries is not supported "
        "(use 3 [translation], 9, or 12/13 entries)")


# --------------------------------------------------------------------------
# card data classes
# --------------------------------------------------------------------------

class Surface:
    def __init__(self, sid, mnemonic, coeffs, tr=None, reflecting=False):
        self.sid = sid
        self.mnemonic = mnemonic        # lower case, e.g. 'pz', 'c/z', 'rpp'
        self.coeffs = coeffs            # list[float]
        self.tr = tr                    # TR card number or None
        self.reflecting = reflecting
        self.transform = None           # resolved Transform or None

    def __repr__(self):
        return f"Surface({self.sid} {self.mnemonic} {self.coeffs})"


class FillSpec:
    """Resolved FILL specification of a cell."""

    def __init__(self):
        self.universe = None            # single-fill universe number
        self.transform = None           # Transform applied to the filled universe
        self.ranges = None              # ((i0,i1),(j0,j1),(k0,k1)) for lattices
        self.array = None               # numpy (nk,nj,ni) int array of universes

    @property
    def is_lattice_array(self):
        return self.array is not None


class Cell:
    def __init__(self, cid):
        self.cid = cid
        self.material = 0
        self.density = None
        self.geom_tokens = []           # e.g. ['-1', '2', ':', '(', '#5', ')']
        self.universe = 0
        self.lat = 0
        self.fill = None                # FillSpec or None
        self.trcl = None                # Transform or None (resolved)
        self.has_imp = False
        self.imp_zero = False
        self.like = None                # cell id of LIKE n BUT base
        self._params = {}               # raw param tokens, kept for LIKE/BUT
        self._fill_raw = None           # (tokens, degrees) before resolution
        self._trcl_raw = None           # (tokens, degrees) before resolution

    def __repr__(self):
        return f"Cell({self.cid} mat={self.material} u={self.universe})"


class Deck:
    def __init__(self):
        self.title = ""
        self.cells = {}                 # cid -> Cell, insertion ordered
        self.surfaces = {}              # sid -> Surface
        self.transforms = {}            # tid -> Transform
        self.materials = []             # sorted material numbers present
        self.warnings = []

    def warn(self, msg):
        if msg not in self.warnings:
            self.warnings.append(msg)


# --------------------------------------------------------------------------
# low level line handling
# --------------------------------------------------------------------------

_COMMENT_RE = re.compile(r"^ {0,4}[cC](?:\s|$)")


def _strip_dollar(line):
    return line.split("$", 1)[0]


def _assemble_cards(lines):
    """Join continuation lines into whole logical cards (list of strings)."""
    cards = []
    for raw in lines:
        if _COMMENT_RE.match(raw):
            continue
        line = _strip_dollar(raw)
        if not line.strip():
            continue
        amp_cont = bool(cards) and cards[-1].rstrip().endswith("&")
        blank_cont = bool(cards) and line[:5] == "     "
        if amp_cont or blank_cont:
            prev = cards[-1].rstrip()
            if prev.endswith("&"):
                prev = prev[:-1]
            cards[-1] = prev + " " + line.strip()
        else:
            cards.append(line.strip())
    return cards


def _split_blocks(lines):
    """Split body lines into blank-line separated blocks."""
    blocks = []
    cur = []
    for ln in lines:
        if ln.strip():
            cur.append(ln)
        elif cur:
            blocks.append(cur)
            cur = []
    if cur:
        blocks.append(cur)
    return blocks


# --------------------------------------------------------------------------
# cell card parsing
# --------------------------------------------------------------------------

_NUM_RE = r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?"
_GEOM_TOK_RE = re.compile(r"\s*(\(|\)|:|#|[+-]?\d+(?![\d.eE]))")
_KEYWORD_START_RE = re.compile(r"[A-Za-z*]")


def _tokenize_geometry(text):
    """Split the leading geometry expression off a cell-card remainder.

    Returns (tokens, params_text).  ``#`` directly followed by an integer is
    merged into a single '#<n>' (cell complement) token.
    """
    raw = []
    pos = 0
    while pos < len(text):
        m = _GEOM_TOK_RE.match(text, pos)
        if not m:
            break
        raw.append(m.group(1))
        pos = m.end()
    params_text = text[pos:].strip()
    if params_text and not _KEYWORD_START_RE.match(params_text):
        if re.match(r"[+-]?\d+\.\d+", params_text) or params_text.startswith("."):
            raise MCNPParseError(
                "macrobody facet references (surface.facet) are not supported "
                f"by PyVised: {params_text[:30]!r}")
        raise MCNPParseError(
            f"cannot understand cell card near: {params_text[:40]!r}")
    tokens = []
    i = 0
    while i < len(raw):
        tok = raw[i]
        if tok == "#" and i + 1 < len(raw) and re.fullmatch(r"[+-]?\d+", raw[i + 1]):
            tokens.append("#" + raw[i + 1].lstrip("+"))
            i += 2
        else:
            tokens.append(tok)
            i += 1
    return tokens, params_text


_SHORTHAND_RE = re.compile(r"(?i)^\d*[rijm]$")


def _is_keyword_token(tok):
    """True for tokens that start a new 'key=value' parameter.

    Bare repeat/interpolate shorthand ('r', '3r', ...) starts with a letter
    but belongs to the current value list, not to a new keyword.
    """
    return bool(_KEYWORD_START_RE.match(tok)) and not _SHORTHAND_RE.match(tok)


def _parse_params(text):
    """Parse 'key=value...' cell parameters into an ordered dict of token lists."""
    toks = re.sub(r"([()=])", r" \1 ", text).split()
    out = {}
    i = 0
    while i < len(toks):
        tok = toks[i]
        if not _is_keyword_token(tok):
            raise MCNPParseError(f"unexpected token {tok!r} in cell parameters")
        key = tok.lower()
        i += 1
        if i < len(toks) and toks[i] == "=":
            i += 1
        vals = []
        while i < len(toks) and not _is_keyword_token(toks[i]):
            vals.append(toks[i])
            i += 1
        out[key] = vals
    return out


_IGNORED_CELL_KEYS = (
    "vol", "tmp", "pwt", "ext", "fcl", "wwn", "dxc", "nonu", "pd", "elpt",
    "cosy", "bflcl", "unc", "tmp1",
)


def _apply_params(cell, params, deck):
    for key, vals in params.items():
        base = key.split(":", 1)[0]
        try:
            if base == "u":
                # negative U is only a truncation hint; identity is |n|
                cell.universe = abs(int(float(vals[0])))
            elif base == "lat":
                cell.lat = int(float(vals[0]))
            elif base in ("fill", "*fill"):
                cell._fill_raw = (vals, key.startswith("*"))
            elif base in ("trcl", "*trcl"):
                cell._trcl_raw = (vals, key.startswith("*"))
            elif base == "imp":
                cell.has_imp = True
                nums = [float(v) for v in vals if re.fullmatch(_NUM_RE, v)]
                cell.imp_zero = bool(nums) and all(v == 0.0 for v in nums)
            elif base == "mat":
                cell.material = int(float(vals[0]))
            elif base == "rho":
                cell.density = float(vals[0])
            elif base in _IGNORED_CELL_KEYS:
                pass
            else:
                deck.warn(f"cell {cell.cid}: ignoring unknown parameter '{key}'")
        except (ValueError, IndexError) as exc:
            raise MCNPParseError(
                f"cell {cell.cid}: bad value for parameter '{key}': {vals}") from exc


def _parse_cell_card(card, deck):
    m = re.match(r"\s*(\d+)\s+(.*)$", card, re.S)
    if not m:
        raise MCNPParseError(f"bad cell card: {card[:60]!r}")
    cell = Cell(int(m.group(1)))
    rest = m.group(2)

    like = re.match(r"(?i)like\s+(\d+)\s+but\b\s*(.*)$", rest, re.S)
    if like:
        cell.like = int(like.group(1))
        cell._params = _parse_params(like.group(2))
        return cell

    m = re.match(r"([+-]?\d+)\s+(.*)$", rest, re.S)
    if not m:
        raise MCNPParseError(f"cell {cell.cid}: missing material number")
    cell.material = int(m.group(1))
    rest = m.group(2)
    if cell.material != 0:
        # '(' / '#' right after the density are legal delimiters in MCNP
        m = re.match(rf"({_NUM_RE})(?=\s|\(|#)\s*(.*)$", rest, re.S)
        if not m:
            raise MCNPParseError(f"cell {cell.cid}: missing density")
        cell.density = float(m.group(1))
        rest = m.group(2)

    cell.geom_tokens, params_text = _tokenize_geometry(rest)
    if not cell.geom_tokens:
        raise MCNPParseError(f"cell {cell.cid}: no geometry specification")
    cell._params = _parse_params(params_text)
    _apply_params(cell, cell._params, deck)
    return cell


# --------------------------------------------------------------------------
# FILL / TRCL resolution
# --------------------------------------------------------------------------

def _expand_repeats(tokens, where, deck):
    out = []
    for tok in tokens:
        m = re.fullmatch(r"(\d*)[rR]", tok)
        if m:
            if not out:
                raise MCNPParseError(f"{where}: repeat with no previous entry")
            out.extend([out[-1]] * (int(m.group(1) or 1)))
        elif re.fullmatch(r"\d*[iImMjJ].*", tok):
            raise MCNPParseError(
                f"{where}: '{tok}' shorthand is not supported in fill arrays")
        else:
            out.append(tok)
    return out


def _parse_inline_or_ref_transform(tokens, degrees, deck, where):
    """tokens like ['(', '5', ')'] (TR ref) or ['(', x, y, z, ..., ')']."""
    if not tokens:
        return None
    if tokens[0] != "(" or tokens[-1] != ")":
        raise MCNPParseError(f"{where}: cannot parse transformation {tokens}")
    inner = [t for t in tokens[1:-1] if t not in ("(", ")")]
    if len(inner) == 1 and re.fullmatch(r"\d+", inner[0]):
        tid = int(inner[0])
        if tid not in deck.transforms:
            raise MCNPParseError(f"{where}: TR{tid} is not defined")
        return deck.transforms[tid]
    return make_transform(inner, degrees)


def _resolve_fill(cell, deck):
    tokens, degrees = cell._fill_raw
    if not tokens:
        raise MCNPParseError(f"cell {cell.cid}: empty FILL")
    spec = FillSpec()
    if any(":" in t for t in tokens[:3]):
        # lattice form: i0:i1 j0:j1 k0:k1 followed by the universe array
        if len(tokens) < 3:
            raise MCNPParseError(f"cell {cell.cid}: bad lattice FILL ranges")
        ranges = []
        for t in tokens[:3]:
            m = re.fullmatch(r"([+-]?\d+):([+-]?\d+)", t)
            if not m:
                raise MCNPParseError(f"cell {cell.cid}: bad FILL range '{t}'")
            lo, hi = int(m.group(1)), int(m.group(2))
            if hi < lo:
                raise MCNPParseError(f"cell {cell.cid}: FILL range '{t}' is reversed")
            ranges.append((lo, hi))
        entries = _expand_repeats(tokens[3:], f"cell {cell.cid} FILL", deck)
        if any(t in ("(", ")") for t in entries):
            raise MCNPParseError(
                f"cell {cell.cid}: per-element fill transformations are not supported")
        ni = ranges[0][1] - ranges[0][0] + 1
        nj = ranges[1][1] - ranges[1][0] + 1
        nk = ranges[2][1] - ranges[2][0] + 1
        try:
            arr = np.abs(np.array([int(float(t)) for t in entries], dtype=np.int64))
        except ValueError as exc:
            raise MCNPParseError(f"cell {cell.cid}: bad FILL array entry") from exc
        if arr.size != ni * nj * nk:
            raise MCNPParseError(
                f"cell {cell.cid}: FILL array has {arr.size} entries, "
                f"expected {ni}*{nj}*{nk} = {ni * nj * nk}")
        spec.ranges = tuple(ranges)
        spec.array = arr.reshape(nk, nj, ni)
    else:
        spec.universe = abs(int(float(tokens[0])))
        spec.transform = _parse_inline_or_ref_transform(
            tokens[1:], degrees, deck, f"cell {cell.cid} FILL")
    cell.fill = spec


def _resolve_trcl(cell, deck):
    tokens, degrees = cell._trcl_raw
    if not tokens:
        raise MCNPParseError(f"cell {cell.cid}: empty TRCL")
    if tokens[0] == "(":
        cell.trcl = _parse_inline_or_ref_transform(
            tokens, degrees, deck, f"cell {cell.cid} TRCL")
    else:
        tid = int(float(tokens[0]))
        if tid not in deck.transforms:
            raise MCNPParseError(f"cell {cell.cid}: TR{tid} is not defined")
        cell.trcl = deck.transforms[tid]


def _resolve_like(deck):
    """Expand LIKE n BUT cells (chains allowed, cycles rejected)."""
    for cell in deck.cells.values():
        if cell.like is None:
            continue
        seen = {cell.cid}
        base = cell
        chain = [cell]
        while base.like is not None:
            if base.like not in deck.cells:
                raise MCNPParseError(
                    f"cell {cell.cid}: LIKE references unknown cell {base.like}")
            base = deck.cells[base.like]
            if base.cid in seen:
                raise MCNPParseError(f"cell {cell.cid}: circular LIKE reference")
            seen.add(base.cid)
            chain.append(base)
        # walk from the concrete base back down, applying each BUT layer
        for c in reversed(chain):
            if c.like is None:
                continue
            parent = deck.cells[c.like]
            c.material = parent.material
            c.density = parent.density
            c.geom_tokens = list(parent.geom_tokens)
            c.universe = parent.universe
            c.lat = parent.lat
            c._fill_raw = parent._fill_raw
            c._trcl_raw = parent._trcl_raw
            c.has_imp = parent.has_imp
            c.imp_zero = parent.imp_zero
            _apply_params(c, c._params, deck)
            c.like = None


# --------------------------------------------------------------------------
# surface / data card parsing
# --------------------------------------------------------------------------

_SURFACE_RE = re.compile(
    r"\s*([*+]?)(\d+)\s+(?:([+-]?\d+)\s+)?([A-Za-z][A-Za-z0-9/]*)\s*(.*)$", re.S)


def _parse_surface_card(card, deck):
    m = _SURFACE_RE.match(card)
    if not m:
        raise MCNPParseError(f"bad surface card: {card[:60]!r}")
    prefix, sid, tr, mnemonic, rest = m.groups()
    sid = int(sid)
    coeffs = []
    for tok in rest.split():
        if not re.fullmatch(_NUM_RE, tok):
            raise MCNPParseError(
                f"surface {sid}: bad coefficient '{tok}'")
        coeffs.append(float(tok))
    trn = None
    if tr is not None:
        trn = int(tr)
        if trn < 0:
            deck.warn(f"surface {sid}: periodic surface (negative TR) treated as plain")
            trn = None
    return Surface(sid, mnemonic.lower(), coeffs, trn, reflecting=(prefix == "*"))


_TR_RE = re.compile(r"(?i)^\s*(\*?)tr(\d+)\s+(.*)$", re.S)
_MAT_RE = re.compile(r"(?i)^\s*m(\d+)(?:\s|$)")
_IMP_RE = re.compile(r"(?i)^\s*imp:([a-z,]+)\s+")


def _apply_imp_data_cards(imp_cards, deck):
    """IMP:n data cards list one importance per cell, in cell input order."""
    per_cell = {cid: [] for cid in deck.cells}
    for particles, rest in imp_cards:
        try:
            entries = _expand_repeats(rest.split(), f"IMP:{particles} data card", deck)
            values = [float(v) for v in entries]
        except (ValueError, MCNPParseError) as exc:
            deck.warn(f"IMP:{particles} data card not understood ({exc}); ignored")
            continue
        if len(values) != len(deck.cells):
            deck.warn(f"IMP:{particles} data card has {len(values)} entries for "
                      f"{len(deck.cells)} cells; ignored")
            continue
        for cid, val in zip(deck.cells, values):
            per_cell[cid].append(val)
    for cid, values in per_cell.items():
        cell = deck.cells[cid]
        if values and not cell.has_imp:   # cell-card IMP wins if both given
            cell.has_imp = True
            cell.imp_zero = all(v == 0.0 for v in values)


def _parse_data_cards(cards, deck):
    imp_cards = []
    for card in cards:
        m = _TR_RE.match(card)
        if m:
            star, tid, rest = m.groups()
            vals = rest.split()
            try:
                deck.transforms[int(tid)] = make_transform(vals, degrees=bool(star))
            except (ValueError, MCNPParseError) as exc:
                # keep at least the translation rather than dropping the card
                try:
                    shift = [float(v) for v in vals[:3]]
                    deck.warn(f"TR{tid}: rotation not understood ({exc}); "
                              "keeping the translation only")
                except ValueError:
                    shift = [0.0, 0.0, 0.0]
                    deck.warn(f"TR{tid} could not be parsed ({exc}); "
                              "treated as identity")
                deck.transforms[int(tid)] = Transform(shift)
            continue
        m = _MAT_RE.match(card)
        if m:
            mid = int(m.group(1))
            if mid not in deck.materials:
                deck.materials.append(mid)
            continue
        m = _IMP_RE.match(card)
        if m:
            imp_cards.append((m.group(1), card[m.end():]))
    if imp_cards:
        _apply_imp_data_cards(imp_cards, deck)


# --------------------------------------------------------------------------
# top level
# --------------------------------------------------------------------------

def parse_string(text):
    lines = text.expandtabs(8).splitlines()

    i = 0
    while i < len(lines) and not lines[i].strip():
        i += 1
    if i < len(lines) and lines[i].lstrip().lower().startswith("message:"):
        while i < len(lines) and lines[i].strip():
            i += 1
        while i < len(lines) and not lines[i].strip():
            i += 1
    if i >= len(lines):
        raise MCNPParseError("input file is empty")

    deck = Deck()
    deck.title = lines[i].strip()
    blocks = _split_blocks(lines[i + 1:])
    if len(blocks) < 2:
        raise MCNPParseError(
            "could not find the surface block — an MCNP deck needs "
            "cell, surface and data blocks separated by blank lines")

    cell_cards = _assemble_cards(blocks[0])
    surf_cards = _assemble_cards(blocks[1])
    data_lines = [ln for blk in blocks[2:] for ln in blk]
    data_cards = _assemble_cards(data_lines)

    if not cell_cards:
        raise MCNPParseError("no cell cards found")
    if not surf_cards:
        raise MCNPParseError("no surface cards found")

    for card in cell_cards:
        cell = _parse_cell_card(card, deck)
        if cell.cid in deck.cells:
            raise MCNPParseError(f"cell {cell.cid} defined twice")
        deck.cells[cell.cid] = cell

    for card in surf_cards:
        surf = _parse_surface_card(card, deck)
        if surf.sid in deck.surfaces:
            raise MCNPParseError(f"surface {surf.sid} defined twice")
        deck.surfaces[surf.sid] = surf

    _parse_data_cards(data_cards, deck)

    # resolution passes (TR table is now known)
    _resolve_like(deck)
    for cell in deck.cells.values():
        if cell._fill_raw is not None:
            _resolve_fill(cell, deck)
        if cell._trcl_raw is not None:
            _resolve_trcl(cell, deck)
    for surf in deck.surfaces.values():
        if surf.tr is not None:
            if surf.tr not in deck.transforms:
                raise MCNPParseError(
                    f"surface {surf.sid}: TR{surf.tr} is not defined")
            surf.transform = deck.transforms[surf.tr]

    # make sure every material referenced by a cell shows up in the legend
    for cell in deck.cells.values():
        if cell.material and cell.material not in deck.materials:
            deck.materials.append(cell.material)
    deck.materials.sort()
    return deck


def parse_file(path):
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return parse_string(fh.read())
