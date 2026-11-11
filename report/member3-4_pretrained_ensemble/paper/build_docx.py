#!/usr/bin/env python3
"""Convert main.tex (+ references.bib, figures/*.png) to paper.docx.

Only the standard library and python-docx are used.  The parser is general:
it does not hard-code paper content.  Usage:

    python build_docx.py [main.tex] [paper.docx]
"""
import collections
import re
import sys
import unicodedata
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

HERE = Path(__file__).resolve().parent
BODY_FONT = "Times New Roman"
MONO_FONT = "Consolas"
BODY_PT = 10.5

# ----------------------------------------------------------------------------
# symbol tables
# ----------------------------------------------------------------------------
GREEK_LOWER = dict(
    alpha="α", beta="β", gamma="γ", delta="δ", epsilon="ε", varepsilon="ε",
    zeta="ζ", eta="η", theta="θ", vartheta="ϑ", iota="ι", kappa="κ",
    lambda_="λ", mu="μ", nu="ν", xi="ξ", pi="π", varpi="ϖ", rho="ρ",
    varrho="ϱ", sigma="σ", varsigma="ς", tau="τ", upsilon="υ", phi="φ",
    varphi="φ", chi="χ", psi="ψ", omega="ω")
GREEK_LOWER["lambda"] = "λ"
GREEK_UPPER = dict(
    Gamma="Γ", Delta="Δ", Theta="Θ", Lambda="Λ", Xi="Ξ", Pi="Π", Sigma="Σ",
    Upsilon="Υ", Phi="Φ", Psi="Ψ", Omega="Ω")
MATH_SYM = dict(
    pm="±", mp="∓", times="×", div="÷", cdot="·", ast="∗", star="⋆",
    circ="∘", bullet="•", le="≤", leq="≤", ge="≥", geq="≥", ll="≪", gg="≫",
    ne="≠", neq="≠", approx="≈", sim="∼", simeq="≃", equiv="≡", propto="∝",
    to="→", rightarrow="→", leftarrow="←", gets="←", Rightarrow="⇒",
    Leftarrow="⇐", leftrightarrow="↔", Leftrightarrow="⇔", mapsto="↦",
    implies="⇒", iff="⇔", uparrow="↑", downarrow="↓",
    **{"in": "∈"}, notin="∉", subset="⊂", subseteq="⊆", supset="⊃",
    supseteq="⊇", cup="∪", cap="∩", emptyset="∅", varnothing="∅",
    forall="∀", exists="∃", infty="∞", partial="∂", nabla="∇", sum="Σ",
    prod="Π", int="∫", bot="⊥", ldots="…", cdots="⋯", dots="…", vdots="⋮",
    ddots="⋱", prime="′", angle="∠", mid="|", vert="|", lvert="|", rvert="|",
    Vert="‖", lVert="‖", rVert="‖", parallel="∥", langle="⟨", rangle="⟩",
    lfloor="⌊", rfloor="⌋", lceil="⌈", rceil="⌉", degree="°", hbar="ℏ",
    ell="ℓ", Re="ℜ", Im="ℑ", neg="¬", lnot="¬", land="∧", lor="∨",
    wedge="∧", vee="∨", oplus="⊕", otimes="⊗", odot="⊙", setminus="∖",
    backslash="∖", lbrace="{", rbrace="}", lbrack="[", rbrack="]",
    textbackslash="∖", dagger="†", ddagger="‡", checkmark="✓",
    sqcup="⊔", triangleq="≜", coloneqq=":=", colon=":", cdotp="·")
MATH_SYM.update(GREEK_UPPER)
MATH_RELATIONS = set("≤≥≪≫≠≈∼≃≡∝→←⇒⇐↔⇔↦∈∉⊂⊆⊃⊇≜")
MATH_OPNAMES = {"ln", "log", "exp", "sin", "cos", "tan", "max", "min", "arg",
                "lim", "sup", "inf", "det", "Pr", "deg", "dim", "sec", "csc",
                "cot", "sinh", "cosh", "tanh", "argmax", "argmin", "gcd",
                "ker", "lg", "mod", "var", "cov"}
MATH_IGNORE = {"big", "Big", "bigg", "Bigg", "bigl", "bigr", "Bigl", "Bigr",
               "biggl", "biggr", "Biggl", "Biggr", "!", "nonumber", "notag",
               "displaystyle", "textstyle", "scriptstyle", "limits",
               "nolimits", "mathstrut", "allowbreak", "relax", "protect"}
MATH_SPACE = {",", ";", ":", " ", "quad", "qquad", ">", "hspace"}
MATH_TEXTLIKE = {"mathrm", "text", "textrm", "operatorname", "mbox", "hbox",
                 "textnormal", "mathsf", "mathcal", "mathbb", "mathscr",
                 "mathfrak", "textsf", "mathnormal"}
MATH_ACCENT = {"hat": "\u0302", "widehat": "\u0302", "bar": "\u0304",
               "overline": "\u0305", "tilde": "\u0303", "widetilde": "\u0303",
               "vec": "\u20d7", "dot": "\u0307", "ddot": "\u0308",
               "check": "\u030c", "breve": "\u0306", "acute": "\u0301",
               "grave": "\u0300", "underline": "\u0332"}

ACCENTS = {"'": "\u0301", "`": "\u0300", "^": "\u0302", '"': "\u0308",
           "~": "\u0303", "=": "\u0304", ".": "\u0307", "c": "\u0327",
           "v": "\u030c", "u": "\u0306", "H": "\u030b", "k": "\u0328",
           "r": "\u030a"}
TEXT_SYMBOLS = dict(
    ldots="…", dots="…", textellipsis="…", LaTeX="LaTeX", TeX="TeX", S="§",
    P="¶", dag="†", ddag="‡", textbar="|", copyright="©", textregistered="®",
    texttrademark="™", textdegree="°", textasciitilde="∼",
    textasciicircum="^", textunderscore="_", ss="ß", o="ø", O="Ø", ae="æ",
    AE="Æ", oe="œ", OE="Œ", aa="å", AA="Å", l="ł", L="Ł", i="ı", j="ȷ",
    textendash="–", textemdash="—", texttimes="×", textminus="−",
    textpm="±", textless="<", textgreater=">", textbackslash="\\",
    textbullet="•", textquotedblleft="“", textquotedblright="”",
    textquoteleft="‘", textquoteright="’", checkmark="✓", pounds="£",
    euro="€", textperiodcentered="·", guillemotleft="«", guillemotright="»")
SYMBOLS_NO_SKIP_SPACE = set(TEXT_SYMBOLS)

# commands that take N brace arguments and produce nothing
DROP_ARGS = dict(
    label=1, vspace=1, hspace=1, vskip=1, hskip=1, setlength=2,
    addtolength=2, setcounter=2, captionsetup=1, bibliographystyle=1,
    bibliography=1, pagestyle=1, thispagestyle=1, nocite=1, cline=1,
    includegraphics=1, pdfinfo=1, newcommand=2, renewcommand=2,
    usepackage=1, urlstyle=1, phantom=1, hphantom=1, vphantom=1,
    linespread=1, graphicspath=1)
NOOP = {
    "noindent", "centering", "small", "footnotesize", "scriptsize",
    "normalsize", "large", "Large", "LARGE", "huge", "Huge", "tiny",
    "raggedright", "raggedleft", "frenchspacing", "nonfrenchspacing",
    "maketitle", "rm", "sf", "sl", "normalfont", "bigskip", "medskip",
    "smallskip", "relax", "protect", "hfill", "indent", "clearpage",
    "newpage", "pagebreak", "linebreak", "nobreak", "ignorespaces", "null",
    "strut", "mdseries", "upshape", "selectfont", "unskip", "par",
    "toprule", "midrule", "bottomrule", "hline", "sloppy", "allowbreak",
    "rmfamily", "sffamily", "fussy", "vfill", "break", "nolinebreak",
    "UrlFont", "centerline", "FloatBarrier", "floatplacement"}
SPACE_CMDS = {"quad", "qquad", "enspace", "thinspace", "space"}
SWITCH_FMT = {"bf": "b", "bfseries": "b", "it": "i", "em": "i", "itshape": "i",
              "slshape": "i", "tt": "tt", "ttfamily": "tt"}
ARG_FMT = {"emph": "i", "textit": "i", "textsl": "i", "textbf": "b",
           "texttt": "tt", "textsuperscript": "sup", "textsubscript": "sub",
           "mathbf": "b", "mathit": "i", "mathtt": "tt"}
ARG_PLAIN = {"textrm", "textsf", "textnormal", "text", "mbox", "hbox",
             "textsc", "underline", "mathrm", "mathsf", "uppercase",
             "lowercase", "MakeUppercase", "MakeLowercase", "textup",
             "textmd", "mathcal", "mathbb", "bm", "boldsymbol", "makebox",
             "acronym", "ac", "acs", "hyperref", "nolinkurl", "textnormal"}
CITE_CMDS = {"cite", "citep", "citet", "citealp", "citealt", "citeauthor",
             "citeyear", "citeyearpar", "parencite", "textcite", "Citep",
             "Citet", "Cite", "citenum", "citetext", "citeA", "autocite"}
REF_CMDS = {"ref", "eqref", "autoref", "cref", "Cref", "pageref", "nameref",
            "vref"}
PARTICLES = {"van", "von", "der", "den", "de", "del", "della", "di", "da",
             "dos", "du", "la", "le", "ter", "ten", "bin", "al", "el", "st"}

STRUCT_RE = re.compile(
    r"\\((?:sub){0,2}section)(\*?)\{"
    r"|\\((?:sub)?paragraph)\*?\{"
    r"|\\begin\{([A-Za-z]+\*?)\}"
    r"|\n[ \t]*\n\s*"
    r"|\\\[")


def fz(*a):
    return frozenset(a)


# ----------------------------------------------------------------------------
# small text helpers
# ----------------------------------------------------------------------------
def strip_comments(text):
    out = []
    for line in text.split("\n"):
        k = 0
        cut = None
        while k < len(line):
            ch = line[k]
            if ch == "\\":
                k += 2
                continue
            if ch == "%":
                cut = k
                break
            k += 1
        if cut is None:
            out.append(line)
        elif line[:cut].strip() == "":
            continue  # whole-line comment: drop line
        else:
            out.append(line[:cut])
    return "\n".join(out)


def match_brace(s, i, open_="{", close="}"):
    """s[i] == open_; return index of the matching close (or len(s)-1)."""
    depth = 0
    j = i
    n = len(s)
    while j < n:
        c = s[j]
        if c == "\\":
            j += 2
            continue
        if c == open_:
            depth += 1
        elif c == close:
            depth -= 1
            if depth == 0:
                return j
        j += 1
    return n - 1


def brace_arg_after(s, name_end):
    """Find '{' after position name_end (skipping spaces) and return
    (content, index_after) or (None, name_end)."""
    j = name_end
    while j < len(s) and s[j].isspace():
        j += 1
    if j < len(s) and s[j] == "{":
        k = match_brace(s, j)
        return s[j + 1:k], k + 1
    return None, name_end


def find_env_end(text, start, name):
    """start = index just after \\begin{name}; return (body_end, after_end)."""
    pat = re.compile(r"\\(begin|end)\{" + re.escape(name) + r"\}")
    depth = 1
    for m in pat.finditer(text, start):
        depth += 1 if m.group(1) == "begin" else -1
        if depth == 0:
            return m.start(), m.end()
    return len(text), len(text)


def split_depth0(s, sep_re):
    """Split s at regex matches that occur at brace depth 0."""
    parts = []
    depth = 0
    last = 0
    i = 0
    n = len(s)
    while i < n:
        c = s[i]
        if c == "\\":
            i += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        elif depth == 0:
            m = sep_re.match(s, i)
            if m and m.end() > i:
                parts.append(s[last:i])
                last = m.end()
                i = m.end()
                continue
        i += 1
    parts.append(s[last:])
    return parts


# ----------------------------------------------------------------------------
# the converter
# ----------------------------------------------------------------------------
class Converter:
    def __init__(self, tex_path, bib_path, out_path):
        self.tex_path = Path(tex_path)
        self.dir = self.tex_path.parent
        self.bib_path = Path(bib_path)
        self.out_path = Path(out_path)
        self.unknown = collections.Counter()
        self.warnings = []
        self.labels = {}
        self.cited = []
        self.cite_ok = 0
        self.cite_bad = 0
        self.stats = collections.Counter()
        self.bib = {}
        self.names_cache = {}
        self.fig_n = 0
        self.tab_n = 0
        self.eq_n = 0
        self.pending = None
        self.doc = None

    # ---- warnings ----------------------------------------------------------
    def warn(self, msg):
        if msg not in self.warnings:
            self.warnings.append(msg)

    # ======================================================================
    # inline (text mode) parsing -> runs [(text, fmt)]
    # ======================================================================
    def parse_inline(self, s, fmt=frozenset()):
        runs = []
        buf = []
        cur = fmt
        i = 0
        n = len(s)

        def flush():
            if buf:
                runs.append(("".join(buf), cur))
                buf.clear()

        def emit(runs_):
            flush()
            runs.extend(runs_)

        while i < n:
            c = s[i]
            if c.isspace():
                buf.append(" ")
                i += 1
            elif c == "-":
                if s.startswith("---", i):
                    buf.append("—"); i += 3
                elif s.startswith("--", i):
                    buf.append("–"); i += 2
                else:
                    buf.append("-"); i += 1
            elif c == "`":
                if s.startswith("``", i):
                    buf.append("“"); i += 2
                else:
                    buf.append("‘"); i += 1
            elif c == "'":
                if s.startswith("''", i):
                    buf.append("”"); i += 2
                else:
                    buf.append("’"); i += 1
            elif c == "~":
                buf.append(" "); i += 1
            elif c == "{":
                j = match_brace(s, i)
                emit(self.parse_inline(s[i + 1:j], cur))
                i = j + 1
            elif c == "}":
                i += 1
            elif c == "$":
                if s.startswith("$$", i):
                    j = s.find("$$", i + 2)
                    L = 2
                else:
                    j = i + 1
                    while j < len(s) and not (s[j] == "$" and s[j - 1] != "\\"):
                        j += 1
                    j = j if j < len(s) else -1
                    L = 1
                if j < 0:
                    self.warn("unmatched $ in text")
                    buf.append("$"); i += 1
                else:
                    emit(self.math_runs(s[i + L:j], False, cur))
                    i = j + L
            elif c == "\\":
                i = self._command_text(s, i, runs, buf, flush, emit,
                                       lambda: cur)
                if isinstance(i, tuple):  # (new_i, new_cur)
                    i, cur = i
            else:
                buf.append(c)
                i += 1
        flush()
        return runs

    # returns int or (int, new_fmt)
    def _command_text(self, s, i, runs, buf, flush, emit, getcur):
        cur = getcur()
        n = len(s)
        j = i + 1
        if j >= n:
            return n
        if s[j].isalpha():
            k = j
            while k < n and s[k].isalpha():
                k += 1
            name = s[j:k]
            i2 = k
            if name not in SYMBOLS_NO_SKIP_SPACE:
                while i2 < n and s[i2] in " \t\n":
                    i2 += 1
        else:
            name = s[j]
            i2 = j + 1

        # ---- accents -------------------------------------------------------
        if name in ACCENTS:
            if True:
                arg, i3 = self._read_token_arg(s, i2)
                arg = re.sub(r"\\([ij])(?![A-Za-z])", r"\1", arg)
                arg = arg.replace("{", "").replace("}", "").strip()
                if arg:
                    buf.append(unicodedata.normalize(
                        "NFC", arg[0] + ACCENTS[name]) + arg[1:])
                else:
                    buf.append(ACCENTS[name])
                return i3
        if name == "\\":
            flush()
            runs.append(("\n", cur))
            if i2 < n and s[i2] == "[":
                e = s.find("]", i2)
                if e > 0:
                    i2 = e + 1
            return i2
        if not name.isalpha():
            if name in "%&_#${}":
                buf.append(name)
            elif name in ", ;:":
                buf.append(" ")
            elif name in "!/-@>":
                pass
            elif name == "(":
                e = s.find("\\)", i2)
                emit(self.math_runs(s[i2:e], False, cur))
                return e + 2
            elif name == "[":
                e = s.find("\\]", i2)
                emit(self.math_runs(s[i2:e], False, cur))
                return e + 2
            else:
                self.unknown["\\" + name] += 1
            return i2

        # ---- alphabetic commands ------------------------------------------
        if name in ARG_FMT:
            arg, i3 = self._read_arg(s, i2)
            emit(self.parse_inline(arg, cur | {ARG_FMT[name]}))
            return i3
        if name in ARG_PLAIN:
            arg, i3 = self._read_arg(s, i2)
            emit(self.parse_inline(arg, cur))
            return i3
        if name in ("url", "path"):
            arg, i3 = self._read_arg(s, i2)
            arg = re.sub(r"\\([_%#&$])", r"\1", arg)
            flush()
            runs.append((arg.strip(), cur))
            return i3
        if name == "href":
            _, i3 = self._read_arg(s, i2)
            arg, i4 = self._read_arg(s, i3)
            emit(self.parse_inline(arg, cur))
            return i4
        if name in ("footnote", "footnotetext", "endnote"):
            if s[i2:i2 + 1] == "[":
                e = s.find("]", i2)
                i2 = e + 1
            arg, i3 = self._read_arg(s, i2)
            flush()
            runs.append((" (Note: ", cur))
            fr = self.parse_inline(arg.strip(), cur)
            if fr and fr[-1][0].endswith("."):
                fr[-1] = (fr[-1][0][:-1], fr[-1][1])
            runs.extend(fr)
            runs.append((")", cur))
            return i3
        if name in CITE_CMDS:
            return self._cite(s, i2, name, runs, flush, cur)
        if name in REF_CMDS:
            arg, i3 = self._read_arg(s, i2)
            flush()
            runs.append((self._ref_text(arg, name), cur))
            return i3
        if name == "ensuremath":
            arg, i3 = self._read_arg(s, i2)
            emit(self.math_runs(arg, False, cur))
            return i3
        if name in SWITCH_FMT:
            flush()
            return i2, cur | {SWITCH_FMT[name]}
        if name in TEXT_SYMBOLS:
            buf.append(TEXT_SYMBOLS[name])
            # swallow an empty "{}" used to terminate the command
            if s.startswith("{}", i2):
                i2 += 2
            return i2
        if name in DROP_ARGS:
            k = i2
            if s[k:k + 1] == "*":
                k += 1
            if s[k:k + 1] == "[":
                e = s.find("]", k)
                k = e + 1
            for _ in range(DROP_ARGS[name]):
                a, k2 = brace_arg_after(s, k)
                if a is None:
                    break
                k = k2
            return k
        if name in SPACE_CMDS:
            buf.append(" ")
            return i2
        if name == "newline":
            flush()
            runs.append(("\n", cur))
            return i2
        if name in NOOP:
            return i2
        if name in ("item",):
            return i2
        # unknown: drop name, keep braced argument text (parsed next loop)
        self.unknown["\\" + name] += 1
        return i2

    @staticmethod
    def _read_arg(s, i):
        j = i
        while j < len(s) and s[j].isspace():
            j += 1
        if j < len(s) and s[j] == "{":
            k = match_brace(s, j)
            return s[j + 1:k], k + 1
        return "", i

    @staticmethod
    def _read_token_arg(s, i):
        """argument for an accent: {x} or a single char / \\i."""
        j = i
        while j < len(s) and s[j] == " ":
            j += 1
        if j >= len(s):
            return "", j
        if s[j] == "{":
            k = match_brace(s, j)
            return s[j + 1:k], k + 1
        if s[j] == "\\":
            k = j + 1
            while k < len(s) and s[k].isalpha():
                k += 1
            return s[j:k], k
        return s[j], j + 1

    # ======================================================================
    # math
    # ======================================================================
    def math_runs(self, s, nopad=False, fmt=frozenset(), roman=False):
        out = []
        i = 0
        n = len(s)

        def put(t, f=None):
            if t:
                out.append((t, fmt if f is None else f))

        def last_char():
            for t, _ in reversed(out):
                t2 = t.rstrip()
                if t2:
                    return t2[-1]
            return ""

        def getarg(i):
            while i < n and s[i].isspace():
                i += 1
            if i >= n:
                return "", i
            if s[i] == "{":
                j = match_brace(s, i)
                return s[i + 1:j], j + 1
            if s[i] == "\\":
                k = i + 1
                if k < n and s[k].isalpha():
                    while k < n and s[k].isalpha():
                        k += 1
                else:
                    k += 1
                return s[i:k], k
            return s[i], i + 1

        def plain(rs):
            return "".join(t for t, _ in rs)

        while i < n:
            c = s[i]
            if c.isspace():
                i += 1
            elif c == "&":
                put(" "); i += 1
            elif c == "{":
                j = match_brace(s, i)
                out.extend(self.math_runs(s[i + 1:j], nopad, fmt, roman))
                i = j + 1
            elif c == "}":
                i += 1
            elif c in "_^":
                arg, i = getarg(i + 1)
                tag = "sub" if c == "_" else "sup"
                for t, f in self.math_runs(arg, True, fmt, roman):
                    put(t, (f - {"sub", "sup"}) | {tag})
            elif c == "\\":
                k = i + 1
                if k < n and s[k].isalpha():
                    while k < n and s[k].isalpha():
                        k += 1
                    name = s[i + 1:k]
                else:
                    name = s[k] if k < n else ""
                    k += 1
                i = k
                if name in ("left", "right"):
                    if i < n and s[i] == ".":
                        i += 1
                elif name in MATH_IGNORE:
                    pass
                elif name in MATH_SPACE:
                    put(" ")
                elif name == "\\":
                    put(" ")
                elif name in ("{", "}", "%", "#", "$", "&", "_"):
                    put(name)
                elif name == "|":
                    put("‖")
                elif name in GREEK_LOWER:
                    put(GREEK_LOWER[name], fmt | {"i"})
                elif name == "top":
                    put("T" if nopad else "ᵀ")
                elif name in MATH_SYM:
                    sym = MATH_SYM[name]
                    if sym in MATH_RELATIONS and not nopad:
                        put(" " + sym + " ")
                    else:
                        put(sym)
                elif name in MATH_OPNAMES:
                    put(name)
                    if i < n and s[i] not in "_^":
                        put(" ")
                elif name in ("frac", "dfrac", "tfrac"):
                    a, i = getarg(i)
                    b, i = getarg(i)
                    ra = self.math_runs(a, nopad, fmt, roman)
                    rb = self.math_runs(b, nopad, fmt, roman)
                    for part, last in ((ra, False), (rb, True)):
                        simple = re.fullmatch(r"[\w.]+", plain(part)) is not None
                        if not last:
                            put("" if simple else "(")
                            out.extend(part)
                            put("/" if simple else ")/")
                        else:
                            put("" if simple else "(")
                            out.extend(part)
                            put("" if simple else ")")
                elif name == "sqrt":
                    if i < n and s[i] == "[":
                        i = s.find("]", i) + 1
                    a, i = getarg(i)
                    put("√(")
                    out.extend(self.math_runs(a, nopad, fmt, roman))
                    put(")")
                elif name in MATH_TEXTLIKE or name in ("textbf", "textit",
                                                       "emph", "mathbf",
                                                       "boldsymbol", "bm",
                                                       "mathit", "mathtt"):
                    a, i = getarg(i)
                    f2 = fmt
                    if name in ("textbf", "mathbf", "boldsymbol", "bm"):
                        f2 = fmt | {"b"}
                    elif name in ("textit", "emph", "mathit"):
                        f2 = fmt | {"i"}
                    elif name == "mathtt":
                        f2 = fmt | {"tt"}
                    if name in ("mathbf", "boldsymbol", "bm", "mathit"):
                        out.extend(self.math_runs(a, nopad, f2, True))
                    else:
                        out.extend(self.parse_inline(a, f2))
                elif name in MATH_ACCENT:
                    a, i = getarg(i)
                    rs = self.math_runs(a, nopad, fmt, roman)
                    for t, f in rs:
                        put("".join(ch + MATH_ACCENT[name] if ch.strip() else ch
                                    for ch in t), f)
                elif name == "label":
                    _, i = getarg(i)
                elif name == "prime":
                    put("′")
                else:
                    self.unknown["\\" + name] += 1
            elif c.isalpha():
                put(c, fmt if roman else fmt | {"i"})
                i += 1
            elif c == "-":
                if nopad:
                    put("−")
                else:
                    lc = last_char()
                    if lc and (lc.isalnum() or lc in ")]}"):
                        put(" − ")
                    else:
                        put("−")
                i += 1
            elif c == "+":
                put("+" if nopad else " + ")
                i += 1
            elif c in "=<>":
                put(c if nopad else " " + c + " ")
                i += 1
            elif c == "'":
                put("′"); i += 1
            elif c == "*":
                put("∗"); i += 1
            elif c == "~":
                put(" "); i += 1
            elif c == "|":
                put("|"); i += 1
            else:
                put(c); i += 1
        return out

    # ======================================================================
    # references / citations
    # ======================================================================
    def _ref_text(self, key, cmd):
        key = key.strip()
        if key in self.labels:
            kind, num = self.labels[key]
            if cmd == "eqref":
                return "(%s)" % num
            if cmd in ("autoref", "cref", "Cref", "vref"):
                return "%s %s" % (kind, num)
            return str(num)
        self.warn("unknown label: " + key)
        return "??"

    def _cite(self, s, i2, name, runs, flush, cur):
        opts = []
        j = i2
        while True:
            k = j
            while k < len(s) and s[k] == " ":
                k += 1
            if k < len(s) and s[k] == "[":
                e = s.find("]", k)
                opts.append(s[k + 1:e].strip())
                j = e + 1
            else:
                break
        keys_s, j = self._read_arg(s, j)
        keys = [k.strip() for k in keys_s.split(",") if k.strip()]
        pre = post = ""
        if len(opts) == 1:
            post = opts[0]
        elif len(opts) >= 2:
            pre, post = opts[0], opts[1]
        items = []
        for key in keys:
            ent = self.bib.get(key)
            if ent is None:
                self.cite_bad += 1
                self.warn("unresolved citation key: " + key)
                items.append(("?" + key, "?"))
                continue
            self.cite_ok += 1
            if key not in self.cited:
                self.cited.append(key)
            items.append((self.cite_authors(ent), ent.get("year", "n.d.")))
        mode = name.lower()
        if mode in ("citet", "textcite"):
            txt = "; ".join("%s (%s)" % it for it in items)
            if post:
                txt = txt[:-1] + ", " + post + ")" if txt.endswith(")") else txt
        elif mode == "citeauthor":
            txt = "; ".join(a for a, _ in items)
        elif mode == "citeyear":
            txt = "; ".join(y for _, y in items)
        elif mode == "citeyearpar":
            txt = "(" + "; ".join(y for _, y in items) + ")"
        else:
            body = "; ".join("%s %s" % it for it in items)
            if pre:
                body = pre + " " + body
            if post:
                body += ", " + post
            txt = body if mode in ("citealp", "citealt", "citenum") \
                else "(" + body + ")"
        flush()
        runs.append((txt, cur))
        return j

    def cite_authors(self, ent):
        names, more = self.entry_names(ent)
        if not names:
            return ent.get("title", "?")[:20]
        sur = [n[1] for n in names]
        if len(sur) == 1 and not more:
            return sur[0]
        if len(sur) == 2 and not more:
            return "%s and %s" % (sur[0], sur[1])
        return sur[0] + " et al."

    # ---- bibliography ------------------------------------------------------
    def load_bib(self):
        if not self.bib_path.exists():
            self.warn("references.bib not found")
            return
        txt = self.bib_path.read_text(encoding="utf-8")
        txt = "\n".join(l for l in txt.split("\n")
                        if not l.lstrip().startswith("%"))
        for m in re.finditer(r"@(\w+)\s*\{", txt):
            typ = m.group(1).lower()
            if typ in ("comment", "string", "preamble"):
                continue
            end = match_brace(txt, m.end() - 1)
            body = txt[m.end():end]
            km = re.match(r"\s*([^,\s]+)\s*,", body)
            if not km:
                continue
            key = km.group(1)
            fields = {"_type": typ}
            rest = body[km.end():]
            p = 0
            while True:
                fm = re.compile(r"\s*,?\s*([A-Za-z_\-]+)\s*=\s*").match(rest, p)
                if not fm:
                    break
                p = fm.end()
                if p < len(rest) and rest[p] == "{":
                    e = match_brace(rest, p)
                    val = rest[p + 1:e]
                    p = e + 1
                elif p < len(rest) and rest[p] == '"':
                    e = p + 1
                    while e < len(rest) and not (rest[e] == '"' and rest[e - 1] != "\\"):
                        e += 1
                    val = rest[p + 1:e]
                    p = e + 1
                else:
                    vm = re.compile(r"[^,\s]+").match(rest, p)
                    val = vm.group(0) if vm else ""
                    p = vm.end() if vm else p
                fields[fm.group(1).lower()] = re.sub(r"\s+", " ", val).strip()
            self.bib[key] = fields

    def plain(self, text):
        return "".join(t for t, _ in self.parse_inline(text)).strip()

    def entry_names(self, ent):
        """-> ([(first, surname)], has_others)"""
        key = id(ent)
        if key in self.names_cache:
            return self.names_cache[key]
        raw = ent.get("author") or ent.get("editor") or ""
        names = []
        more = False
        for a in split_depth0(raw, re.compile(r"\s+and\s+")):
            a = a.strip()
            if not a:
                continue
            if a == "others":
                more = True
                continue
            if a.startswith("{") and match_brace(a, 0) == len(a) - 1:
                names.append(("", self.plain(a[1:-1])))
                continue
            comma = split_depth0(a, re.compile(r","))
            if len(comma) >= 2:
                sur, first = comma[0], comma[-1]
            else:
                words = [w for w in split_depth0(a, re.compile(r"\s+")) if w]
                if len(words) == 1:
                    sur, first = words[0], ""
                else:
                    k = len(words) - 1
                    while k > 1 and words[k - 1][:1].islower() \
                            and words[k - 1].lower() in PARTICLES:
                        k -= 1
                    sur = " ".join(words[k:])
                    first = " ".join(words[:k])
            names.append((self.plain(first), self.plain(sur)))
        res = (names, more)
        self.names_cache[key] = res
        return res

    def format_reference(self, ent):
        runs = []
        names, more = self.entry_names(ent)
        full = [(f + " " + s).strip() for f, s in names]
        if more:
            auth = ", ".join(full) + ", et al."
        elif len(full) <= 1:
            auth = "".join(full)
        elif len(full) == 2:
            auth = " and ".join(full)
        else:
            auth = ", ".join(full[:-1]) + ", and " + full[-1]
        if auth:
            runs.append((auth + ". ", fz()))
        runs.append((ent.get("year", "n.d.") + ". ", fz()))
        title = ent.get("title", "")
        if title:
            t = "".join(x for x, _ in self.parse_inline(title)).strip()
            runs.append((t + ("" if t[-1:] in ".?!" else ".") + " ", fz()))
        venue = ""
        if ent.get("booktitle"):
            venue = "In " + self.plain(ent["booktitle"])
        elif ent.get("journal"):
            venue = self.plain(ent["journal"])
        elif ent.get("publisher") and not ent.get("eprint"):
            venue = self.plain(ent["publisher"])
        if venue:
            runs.append((venue, fz("i")))
        extra = []
        if ent.get("series") and self.plain(ent["series"]) not in venue:
            extra.append(self.plain(ent["series"]))
        if ent.get("volume"):
            extra.append("vol. " + self.plain(ent["volume"]))
        if ent.get("number"):
            extra.append("no. " + self.plain(ent["number"]))
        if ent.get("pages"):
            extra.append("pp. " + self.plain(ent["pages"]).replace("--", "–")
                         .replace("-", "–") if True else "")
        if ent.get("eprint") and not venue:
            pre = ent.get("archiveprefix", "arXiv")
            extra.append("%s preprint %s:%s" % (pre, pre, ent["eprint"]))
        if venue and extra:
            runs.append((", " + ", ".join(extra), fz()))
        elif extra:
            runs.append((", ".join(extra), fz()))
        if venue or extra:
            runs.append((". ", fz()))
        if ent.get("note"):
            nt = self.plain(ent["note"])
            runs.append((nt + ("" if nt[-1:] in ".?!" else ".") + " ", fz()))
        link = ""
        if ent.get("doi"):
            d = ent["doi"]
            link = d if d.startswith("http") else "https://doi.org/" + d
        elif ent.get("url"):
            link = ent["url"]
        elif ent.get("eprint") and ent.get("archiveprefix", "arXiv").lower() == "arxiv":
            link = "https://arxiv.org/abs/" + ent["eprint"]
        if link:
            runs.append((re.sub(r"\\([_%#&$])", r"\1", link), fz()))
        return runs

    # ======================================================================
    # pre-pass: numbering of floats and equations
    # ======================================================================
    def eq_rows(self, env, body):
        """-> list of (math_source, numbered, [labels])"""
        star = env.endswith("*")
        base = env.rstrip("*")
        rows = [body] if base == "equation" else re.split(r"\\\\(?:\[[^\]]*\])?", body)
        res = []
        for r in rows:
            if not r.strip():
                continue
            labs = re.findall(r"\\label\{([^}]*)\}", r)
            numbered = not star and "\\nonumber" not in r and "\\notag" not in r
            r2 = re.sub(r"\\label\{[^}]*\}", "", r)
            res.append((r2, numbered, labs))
        return res

    def prepass(self, body):
        f = t = e = 0
        pat = re.compile(r"\\begin\{(figure\*?|table\*?|equation\*?|align\*?|"
                         r"gather\*?|eqnarray\*?)\}")
        pos = 0
        while True:
            m = pat.search(body, pos)
            if not m:
                break
            env = m.group(1)
            b_end, a_end = find_env_end(body, m.end(), env)
            inner = body[m.end():b_end]
            if env.startswith("figure"):
                f += 1
                for lab in re.findall(r"\\label\{([^}]*)\}", inner):
                    self.labels[lab] = ("Figure", f)
            elif env.startswith("table"):
                t += 1
                for lab in re.findall(r"\\label\{([^}]*)\}", inner):
                    self.labels[lab] = ("Table", t)
            else:
                for _, numbered, labs in self.eq_rows(env, inner):
                    if numbered:
                        e += 1
                    for lab in labs:
                        self.labels[lab] = ("Equation", e)
            pos = a_end

    # ======================================================================
    # docx helpers
    # ======================================================================
    @staticmethod
    def set_font(style_or_run, name, size=None, bold=None, color=None):
        f = style_or_run.font
        f.name = name
        el = style_or_run.element
        rpr = el.get_or_add_rPr() if hasattr(el, "get_or_add_rPr") else el.rPr
        rfonts = rpr.find(qn("w:rFonts"))
        if rfonts is None:
            rfonts = OxmlElement("w:rFonts")
            rpr.insert(0, rfonts)
        for a in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
            rfonts.set(qn(a), name)
        for a in ("w:asciiTheme", "w:hAnsiTheme", "w:cstheme", "w:eastAsiaTheme"):
            if rfonts.get(qn(a)) is not None:
                del rfonts.attrib[qn(a)]
        if size:
            f.size = Pt(size)
        if bold is not None:
            f.bold = bold
        if color is not None:
            f.color.rgb = color

    def setup_doc(self):
        d = Document()
        sec = d.sections[0]
        sec.page_width = Inches(8.5)
        sec.page_height = Inches(11)
        for a in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
            setattr(sec, a, Inches(1))
        st = d.styles
        self.set_font(st["Normal"], BODY_FONT, BODY_PT)
        pf = st["Normal"].paragraph_format
        pf.space_after = Pt(6)
        pf.space_before = Pt(0)
        pf.line_spacing = 1.08
        for name, size in (("Title", 16), ("Heading 1", 13),
                           ("Heading 2", 11.5), ("Heading 3", 10.5)):
            s = st[name]
            self.set_font(s, BODY_FONT, size, True, RGBColor(0, 0, 0))
            if name != "Title":
                s.paragraph_format.space_before = Pt(12 if name == "Heading 1" else 8)
                s.paragraph_format.space_after = Pt(4)
                s.paragraph_format.keep_with_next = True
        ppr = st["Title"].element.get_or_add_pPr()
        bd = ppr.find(qn("w:pBdr"))
        if bd is not None:
            ppr.remove(bd)
        st["Title"].paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for name in ("List Bullet", "List Bullet 2", "List Number"):
            try:
                self.set_font(st[name], BODY_FONT, BODY_PT)
                st[name].paragraph_format.space_after = Pt(3)
            except KeyError:
                pass
        # footer page number
        p = sec.footer.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run()
        for typ, txt in (("begin", None), (None, "PAGE"), ("end", None)):
            if typ:
                fc = OxmlElement("w:fldChar")
                fc.set(qn("w:fldCharType"), typ)
                r._r.append(fc)
            else:
                it = OxmlElement("w:instrText")
                it.set(qn("xml:space"), "preserve")
                it.text = txt
                r._r.append(it)
        self.doc = d

    @staticmethod
    def normalize_runs(runs):
        out = []
        prev_space = True
        for t, f in runs:
            if not t:
                continue
            parts = re.split(r"(\n)", t)
            for part in parts:
                if part == "\n":
                    if out and out[-1][0].endswith(" "):
                        out[-1] = (out[-1][0].rstrip(" "), out[-1][1])
                    out.append(("\n", f))
                    prev_space = True
                    continue
                part = re.sub(r"[ \t]+", " ", part)
                if prev_space:
                    part = part.lstrip(" ")
                if not part:
                    continue
                prev_space = part.endswith(" ")
                out.append((part, f))
        if out and out[-1][0].endswith(" "):
            out[-1] = (out[-1][0].rstrip(" "), out[-1][1])
        merged = []
        for t, f in out:
            if merged and merged[-1][1] == f and t != "\n" and merged[-1][0] != "\n":
                merged[-1] = (merged[-1][0] + t, f)
            else:
                merged.append((t, f))
        return [(t, f) for t, f in merged if t]

    def add_runs(self, par, runs, size=None):
        for t, f in self.normalize_runs(runs):
            r = par.add_run(t)
            if "b" in f:
                r.bold = True
            if "i" in f:
                r.italic = True
            if "sup" in f:
                r.font.superscript = True
            if "sub" in f:
                r.font.subscript = True
            if "tt" in f:
                self.set_font(r, MONO_FONT, (size or BODY_PT) - 1)
            elif size:
                r.font.size = Pt(size)

    def new_par(self, runs, style=None, align=None, indent=None, hanging=None,
                keep_next=False, space_after=None, size=None):
        p = self.doc.add_paragraph(style=style)
        if align is not None:
            p.alignment = align
        if indent is not None:
            p.paragraph_format.left_indent = Inches(indent)
        if hanging is not None:
            p.paragraph_format.first_line_indent = Inches(-hanging)
        if keep_next:
            p.paragraph_format.keep_with_next = True
        if space_after is not None:
            p.paragraph_format.space_after = Pt(space_after)
        self.add_runs(p, runs, size)
        return p

    def heading(self, runs, level):
        p = self.doc.add_paragraph(style="Heading %d" % min(level, 3))
        self.add_runs(p, runs)
        self.stats["headings"] += 1
        return p

    # ======================================================================
    # block-level processing
    # ======================================================================
    def process(self, text, lst=None):
        """lst: None or dict(style, prefix, indent, cont_indent, used)."""
        buf = []
        pos = 0
        state = {"lst": lst}

        def emit_text(txt):
            txt = txt.strip()
            if not txt:
                return
            runs = self.parse_inline(txt)
            if not self.normalize_runs(runs) and not self.pending:
                return
            self.emit_body_par(runs, state["lst"])

        def flush():
            emit_text("".join(buf))
            buf.clear()

        def ensure_prefix():
            """Emit a pending run-in heading / list prefix as its own paragraph."""
            L = state["lst"]
            if self.pending:
                self.new_par(self.pending, keep_next=True)
                self.stats["paragraphs"] += 1
                self.pending = None
            if L and not L["used"]:
                self.emit_body_par([], L)

        while True:
            m = STRUCT_RE.search(text, pos)
            if not m:
                buf.append(text[pos:])
                break
            buf.append(text[pos:m.start()])
            tok = m.group(0)
            if m.group(1):  # section
                flush()
                ensure_prefix()
                title, after = brace_arg_after(text, m.end() - 1)
                level = 1 + m.group(1).count("sub")
                self.heading(self.parse_inline(title or ""), level)
                pos = after
            elif m.group(3):  # paragraph
                flush()
                ensure_prefix()
                title, after = brace_arg_after(text, m.end() - 1)
                runs = self.parse_inline(title or "", fz("b"))
                runs.append((" ", fz("b")))
                self.pending = runs
                t = "".join(x for x, _ in runs).strip()
                if t and t[-1] not in ".?!:":
                    self.pending = self.parse_inline(title or "", fz("b")) + [(". ", fz("b"))]
                pos = after
            elif m.group(4):  # environment
                name = m.group(4)
                flush()
                b_end, a_end = find_env_end(text, m.end(), name)
                body = text[m.end():b_end]
                if name in ("itemize", "enumerate", "description"):
                    ensure_prefix()
                    self.do_list(name, body, state["lst"])
                elif name in ("figure", "figure*", "table", "table*"):
                    ensure_prefix()
                    self.do_float(name, body)
                elif name == "abstract":
                    self.heading([("Abstract", fz())], 1)
                    self.process(body)
                elif re.match(r"(equation|align|gather|eqnarray|multline|"
                              r"displaymath|math)\*?$", name):
                    ensure_prefix()
                    self.do_equation(name, body)
                elif name in ("tabular", "tabularx", "tabular*"):
                    ensure_prefix()
                    self.do_tabular(body)
                else:
                    if name not in ("center", "centering", "flushleft",
                                    "flushright", "quote", "quotation",
                                    "minipage", "small", "footnotesize",
                                    "sloppypar", "samepage"):
                        self.unknown["\\begin{%s}" % name] += 1
                    self.process(body, state["lst"])
                pos = a_end
            elif tok == "\\[":
                flush()
                ensure_prefix()
                e = text.find("\\]", m.end())
                if e < 0:
                    e = len(text)
                self.do_equation("equation*", text[m.end():e])
                pos = e + 2
            else:  # blank line
                flush()
                pos = m.end()
        flush()
        if self.pending and lst is None:
            pass  # carried to the next paragraph

    def emit_body_par(self, runs, L):
        if self.pending:
            runs = self.pending + runs
            self.pending = None
        if L is None:
            self.new_par(runs, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
        else:
            if not L["used"]:
                L["used"] = True
                runs = (L["prefix"] or []) + runs
                self.new_par(runs, style=L["style"], indent=L["indent"],
                             hanging=L["hanging"], space_after=3)
            else:
                self.new_par(runs, indent=L["cont_indent"], space_after=3)
        self.stats["paragraphs"] += 1

    # ---- lists -------------------------------------------------------------
    def do_list(self, kind, body, outer):
        level = (outer["level"] + 1) if outer else 1
        items = []
        depth = 0
        marks = []
        for m in re.finditer(r"\\(begin|end)\{(?:itemize|enumerate|description)\}|\\item\b", body):
            if m.group(1) == "begin":
                depth += 1
            elif m.group(1) == "end":
                depth -= 1
            elif depth == 0:
                marks.append(m)
        for idx, m in enumerate(marks):
            end = marks[idx + 1].start() if idx + 1 < len(marks) else len(body)
            txt = body[m.end():end]
            label = None
            mm = re.match(r"\s*\[", txt)
            if mm:
                e = match_brace(txt, mm.end() - 1, "[", "]")
                label = txt[mm.end():e]
                txt = txt[e + 1:]
            items.append((label, txt))
        base = 0.25 * level
        for n, (label, txt) in enumerate(items, 1):
            if kind == "itemize":
                style = "List Bullet" if level == 1 else "List Bullet 2"
                L = dict(style=style, prefix=[], indent=None, hanging=None,
                         cont_indent=0.25 + 0.25 * level, used=False, level=level)
            elif kind == "enumerate":
                L = dict(style=None, prefix=[("%d.\t" % n, fz())],
                         indent=base + 0.25, hanging=0.25,
                         cont_indent=base + 0.25, used=False, level=level)
            else:
                lab = self.parse_inline(label or "", fz("b"))
                L = dict(style=None, prefix=lab + [(" ", fz())],
                         indent=base, hanging=None,
                         cont_indent=base, used=False, level=level)
            self.process(txt, L)
            if not L["used"]:
                self.emit_body_par([], L)

    # ---- equations ---------------------------------------------------------
    def do_equation(self, env, body):
        rows = self.eq_rows(env, body) if not env.startswith(("displaymath", "math")) \
            else [(body, False, [])]
        for src, numbered, _ in rows:
            runs = self.math_runs(src)
            if numbered:
                self.eq_n += 1
                runs = runs + [("      (%d)" % self.eq_n, fz())]
            self.new_par(runs, align=WD_ALIGN_PARAGRAPH.CENTER)
            self.stats["equations"] += 1

    # ---- floats ------------------------------------------------------------
    def do_float(self, name, body):
        is_fig = name.startswith("figure")
        star = name.endswith("*")
        cap_runs = []
        m = re.search(r"\\caption\*?", body)
        if m:
            k = m.end()
            if body[k:k + 1] == "[":
                k = match_brace(body, k, "[", "]") + 1
            cap, after = brace_arg_after(body, k)
            if cap is not None:
                cap_runs = self.parse_inline(cap)
                body = body[:m.start()] + body[after:]
        if is_fig:
            self.fig_n += 1
            num = self.fig_n
            label = "Figure %d. " % num
        else:
            self.tab_n += 1
            num = self.tab_n
            label = "Table %d. " % num
        cap_par_runs = [(label, fz("b"))] + cap_runs

        def caption(keep):
            self.new_par(cap_par_runs, align=WD_ALIGN_PARAGRAPH.JUSTIFY,
                         keep_next=keep, size=BODY_PT - 1)
            self.stats["captions"] += 1

        if is_fig:
            imgs = re.findall(r"\\includegraphics\s*(?:\[[^\]]*\])?\s*\{([^}]*)\}", body)
            if not imgs:
                self.warn("figure %d has no \\includegraphics" % num)
            for path in imgs:
                p = Path(path.strip())
                cands = [self.dir / p.with_suffix(".png"),
                         self.dir / "figures" / (p.stem + ".png")]
                found = next((c for c in cands if c.exists()), None)
                par = self.doc.add_paragraph()
                par.alignment = WD_ALIGN_PARAGRAPH.CENTER
                par.paragraph_format.keep_with_next = True
                par.paragraph_format.space_after = Pt(3)
                if found is None:
                    self.warn("missing PNG for " + path)
                    par.add_run("[missing figure: %s]" % p.stem)
                else:
                    par.add_run().add_picture(str(found),
                                              width=Inches(6.5 if star else 3.3))
                    self.stats["images"] += 1
            caption(False)
        else:
            caption(True)
            pos = 0
            found_tab = False
            for tm in re.finditer(r"\\begin\{(tabular\*?|tabularx)\}", body):
                if tm.start() < pos:
                    continue
                found_tab = True
                b_end, a_end = find_env_end(body, tm.end(), tm.group(1))
                self.do_tabular(body[tm.end():b_end])
                pos = a_end
            if not found_tab:
                self.warn("table %d has no tabular" % num)
            sp = self.doc.add_paragraph()
            sp.paragraph_format.space_after = Pt(2)

    def do_tabular(self, body):
        # column spec
        if body.lstrip().startswith("{"):
            st = body.index("{")
            e = match_brace(body, st)
            spec = body[st + 1:e]
            body = body[e + 1:]
        else:
            spec = ""
        aligns = []
        k = 0
        while k < len(spec):
            ch = spec[k]
            if ch in "lcr":
                aligns.append(ch)
            elif ch in "pmbX":
                aligns.append("l")
                if k + 1 < len(spec) and spec[k + 1] == "{":
                    k = match_brace(spec, k + 1)
            elif ch in "@>!<":
                if k + 1 < len(spec) and spec[k + 1] == "{":
                    k = match_brace(spec, k + 1)
            k += 1
        body = re.sub(r"\\(toprule|midrule|bottomrule|hline)\b", "", body)
        body = re.sub(r"\\cmidrule\s*(\([^)]*\))?\s*\{[^}]*\}", "", body)
        rows = [r for r in re.split(r"\\\\(?:\[[^\]]*\])?", body) if r.strip()]
        grid = []
        for r in rows:
            cells = []
            for c in re.split(r"(?<!\\)&", r):
                c = c.strip()
                span = 1
                al = None
                mm = re.match(r"\\multicolumn\s*\{(\d+)\}\s*\{([^}]*)\}\s*\{", c)
                if mm:
                    span = int(mm.group(1))
                    al = next((x for x in mm.group(2) if x in "lcr"), None)
                    e = match_brace(c, mm.end() - 1)
                    c = c[mm.end():e]
                cells.append((c, span, al))
            grid.append(cells)
        ncols = max((sum(sp for _, sp, _ in row) for row in grid), default=0)
        if not grid or ncols == 0:
            self.warn("empty tabular")
            return
        tbl = self.doc.add_table(rows=len(grid), cols=ncols)
        try:
            tbl.style = "Table Grid"
        except KeyError:
            pass
        tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
        amap = {"l": WD_ALIGN_PARAGRAPH.LEFT, "c": WD_ALIGN_PARAGRAPH.CENTER,
                "r": WD_ALIGN_PARAGRAPH.RIGHT}
        for ri, row in enumerate(grid):
            ci = 0
            for text, span, al in row:
                cell = tbl.cell(ri, ci)
                if span > 1:
                    cell = cell.merge(tbl.cell(ri, ci + span - 1))
                a = al or (aligns[ci] if ci < len(aligns) else "l")
                if span > 1 and al is None:
                    a = "c"
                fmt = fz("b") if ri == 0 else fz()
                runs = self.parse_inline(text, fmt)
                par = cell.paragraphs[0]
                par.alignment = amap.get(a, WD_ALIGN_PARAGRAPH.LEFT)
                par.paragraph_format.space_after = Pt(0)
                par.paragraph_format.line_spacing = 1.0
                self.add_runs(par, runs, size=8.5)
                if ri == 0:
                    tcPr = cell._tc.get_or_add_tcPr()
                    shd = OxmlElement("w:shd")
                    shd.set(qn("w:val"), "clear")
                    shd.set(qn("w:color"), "auto")
                    shd.set(qn("w:fill"), "E7E6E6")
                    tcPr.append(shd)
                ci += span
        self.stats["tables"] += 1

    # ======================================================================
    # driver
    # ======================================================================
    def run(self):
        raw = self.tex_path.read_text(encoding="utf-8")
        tex = strip_comments(raw)
        self.load_bib()
        self.setup_doc()

        def grab(cmd):
            m = re.search(r"\\" + cmd + r"\s*\{", tex)
            if not m:
                return ""
            e = match_brace(tex, m.end() - 1)
            return tex[m.end():e]

        title_src = grab("title")
        author_src = grab("author")
        aff_src = grab("affiliations") or grab("affil")
        bm = re.search(r"\\begin\{document\}", tex)
        em = tex.rfind("\\end{document}")
        body = tex[bm.end():em if em > 0 else len(tex)] if bm else tex

        # title block
        title_runs = self.parse_inline(title_src.replace("\\\\", " "))
        title_plain = "".join(t for t, _ in self.normalize_runs(title_runs))
        p = self.doc.add_paragraph(style="Title")
        self.add_runs(p, title_runs)
        # authors: strip \textsuperscript and \rm
        a = author_src
        while True:
            m = re.search(r"\\textsuperscript\s*\{", a)
            if not m:
                break
            e = match_brace(a, m.end() - 1)
            a = a[:m.start()] + a[e + 1:]
        a = re.sub(r"\\rm\b\s*", "", a)
        a = re.sub(r"\\(And|and)\b", ",", a)
        self.new_par(self.parse_inline(a), align=WD_ALIGN_PARAGRAPH.CENTER,
                     space_after=2)
        # affiliations
        f = aff_src
        while True:
            m = re.search(r"\\textsuperscript\s*\{", f)
            if not m:
                break
            e = match_brace(f, m.end() - 1)
            f = f[:m.start()] + f[e + 1:]
        f = re.sub(r"\\rm\b\s*", "", f)
        f = re.sub(r"\\\{([^{}]*?)\\\}@([\w.\-]+)",
                   lambda mm: ", ".join(x.strip() + "@" + mm.group(2)
                                        for x in mm.group(1).split(",")), f)
        self.new_par(self.parse_inline(f), align=WD_ALIGN_PARAGRAPH.CENTER,
                     space_after=10)

        # remove commands that must not be parsed as content
        body = re.sub(r"\\maketitle\b", "", body)
        bib_pos = re.search(r"\\bibliography\{", body)
        self.prepass(body)
        self.process(body)
        if self.pending:
            self.new_par(self.pending)
            self.pending = None

        # references
        keys = [k for k in self.cited if k in self.bib]
        if keys:
            self.heading([("References", fz())], 1)

            def sortkey(k):
                ent = self.bib[k]
                names, _ = self.entry_names(ent)
                s = names[0][1] if names else ent.get("title", "")
                s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
                return (s, ent.get("year", ""))
            for k in sorted(keys, key=sortkey):
                self.new_par(self.format_reference(self.bib[k]), indent=0.3,
                             hanging=0.3, space_after=3, size=BODY_PT - 0.5)
                self.stats["references"] += 1

        # core properties
        cp = self.doc.core_properties
        cp.author = "Author One et al."
        cp.last_modified_by = "Author One et al."
        cp.title = title_plain
        cp.comments = ""
        cp.subject = ""
        cp.keywords = ""
        cp.category = ""
        self.doc.save(str(self.out_path))
        return title_plain


# ----------------------------------------------------------------------------
def verify(path):
    d = Document(str(path))
    texts = [p.text for p in d.paragraphs]
    for t in d.tables:
        for row in t.rows:
            for c in row.cells:
                texts.extend(p.text for p in c.paragraphs)
    for s in d.sections:
        texts.extend(p.text for p in s.footer.paragraphs)
    bad = []
    for t in texts:
        for m in re.finditer(r"[\\${}~]", t):
            bad.append(t[max(0, m.start() - 30):m.end() + 30])
    return texts, bad


def main():
    tex = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "main.tex"
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "paper.docx"
    bib = tex.parent / "references.bib"
    conv = Converter(tex, bib, out)
    conv.run()
    texts, bad = verify(out)
    print("Wrote", out)
    st = conv.stats
    print("headings=%d paragraphs=%d equations=%d tables=%d images=%d "
          "captions=%d references=%d" % (
              st["headings"], st["paragraphs"], st["equations"], st["tables"],
              st["images"], st["captions"], st["references"]))
    print("citations resolved=%d unresolved=%d (unique cited keys: %d)" % (
        conv.cite_ok, conv.cite_bad, len(conv.cited)))
    if conv.unknown:
        print("UNKNOWN commands:", dict(conv.unknown))
    else:
        print("unknown commands: none")
    for w in conv.warnings:
        print("WARNING:", w)
    if bad:
        print("LEFTOVER special characters (%d):" % len(bad))
        for b in bad[:40]:
            print("   ...%s..." % b)
    else:
        print("check passed: no backslash, $, {, } or ~ in docx text")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
