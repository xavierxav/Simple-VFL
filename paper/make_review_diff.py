"""Build a 'what changed' HTML page between two reading copies made by make_review_html.py.

    python paper/make_review_diff.py OLD_DIR NEW_DIR OUT.html "label of the old version"

Block elements (headings, paragraphs, list items, figures, tables) are aligned with difflib; changed blocks get a
word-level diff (<del>/<ins>, MathML kept whole), unchanged runs are folded, headings always stay visible.
Figures whose PNG changed are flagged. Links point to the full paper (index.html).
"""

import difflib
import hashlib
import html
import os
import re
import sys

from bs4 import BeautifulSoup

OLD, NEW, OUT, LABEL = sys.argv[1:5]
HEAD = {"h1", "h2", "h3", "h4"}


def load(d):
    soup = BeautifulSoup(open(os.path.join(d, "index.html"), encoding="utf-8").read(), "html.parser")
    out = []

    def rec(n):
        for c in n.children:
            if not getattr(c, "name", None):
                continue
            cls = c.get("class") or []
            if c.name == "details" or (c.name == "div" and "note" in cls):
                continue
            if c.name in HEAD | {"p", "figure", "li"} or (c.name == "div" and "tbl" in cls):
                out.append(c)
            else:
                rec(c)

    rec(soup.find("main"))
    return soup, out


def text(b):
    return re.sub(r"\s+", " ", b.get_text(" ")).strip()


def img_hashes(d, b):
    hs = []
    for im in b.find_all("img"):
        p = os.path.join(d, im.get("src", ""))
        hs.append(hashlib.md5(open(p, "rb").read()).hexdigest() if os.path.exists(p) else im.get("src"))
    return hs


def key(d, b):
    return b.name + "|" + text(b) + "|" + ",".join(img_hashes(d, b))


TOK = re.compile(r"<math\b.*?</math>|<[^>]+>|&[#\w]+;|\s+|\w+(?:[-.,'\u2019]\w+)*|[^\s\w<&]", re.S)


def inner(b):
    return "".join(str(c) for c in b.contents)


def is_tag(t):
    return t.startswith("<") and not t.startswith("<math")


def wdiff(a, b):
    ta, tb = TOK.findall(a), TOK.findall(b)
    ka = [" " if t.isspace() else t for t in ta]
    kb = [" " if t.isspace() else t for t in tb]
    ops = difflib.SequenceMatcher(None, ka, kb, autojunk=False).get_opcodes()
    # semantic cleanup: short equal islands (<= 2 words, no tag or math) between two changes become part of the change
    for n in range(1, len(ops) - 1):
        op, i1, i2, j1, j2 = ops[n]
        words = [t for t in tb[j1:j2] if not t.isspace()]
        if op == "equal" and ops[n - 1][0] != "equal" and ops[n + 1][0] != "equal" and len(words) <= 2 \
                and not any(t.startswith("<") for t in words):
            ops[n] = ("replace", i1, i2, j1, j2)
    merged = []
    for o in ops:
        if merged and o[0] != "equal" and merged[-1][0] != "equal":
            p = merged[-1]
            merged[-1] = ("replace", p[1], o[2], p[3], o[4])
        else:
            merged.append(o)
    out = []
    for op, i1, i2, j1, j2 in merged:
        if op == "equal":
            out += tb[j1:j2]
            continue
        if op in ("delete", "replace"):
            d = "".join(t for t in ta[i1:i2] if not is_tag(t))
            if d.strip():
                out.append("<del>" + d + "</del>")
        if op in ("insert", "replace"):
            run = []
            for t in tb[j1:j2] + ["<END>"]:
                if is_tag(t):
                    if "".join(run).strip():
                        out.append("<ins>" + "".join(run) + "</ins>")
                    elif run:
                        out += run
                    run = []
                    if t != "<END>":
                        out.append(t)
                else:
                    run.append(t)
    return "".join(out)


def shell(b, body, cls):
    """Re-wrap new inner HTML in the block's own tag (ids dropped, local links sent to the full paper)."""
    attrs = {k: v for k, v in b.attrs.items() if k != "id"}
    a = "".join(f' {k}="{html.escape(" ".join(v) if isinstance(v, list) else v)}"' for k, v in attrs.items())
    return f'<div class="blk {cls}"><{b.name}{a}>{body}</{b.name}></div>'


def links(s):
    return re.sub(r'href="#', 'href="index.html#', s)


osoup, ob = load(OLD)
nsoup, nb = load(NEW)
ok, nk = [key(OLD, b) for b in ob], [key(NEW, b) for b in nb]
otx, ntx = [text(b) for b in ob], [text(b) for b in nb]

# 1) exact alignment, 2) inside changed chunks pair similar blocks in order, 3) pair what is left as moves
pair, used_old = {}, set()
chunks = []
for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, ok, nk, autojunk=False).get_opcodes():
    if op == "equal":
        for i, j in zip(range(i1, i2), range(j1, j2)):
            pair[j] = (i, "same")
            used_old.add(i)
    else:
        chunks.append((i1, i2, j1, j2))


def sim(i, j):
    if ob[i].name != nb[j].name and not (ob[i].name in HEAD and nb[j].name in HEAD):
        return 0.0
    return difflib.SequenceMatcher(None, otx[i].split(), ntx[j].split(), autojunk=False).ratio()


for i1, i2, j1, j2 in chunks:
    # monotonic DP maximising total similarity (pairs need ratio > 0.4)
    A, B = list(range(i1, i2)), list(range(j1, j2))
    S = {(i, j): sim(i, j) for i in A for j in B}
    best = {}

    def f(x, y):
        if x == len(A) or y == len(B):
            return 0.0, []
        if (x, y) in best:
            return best[(x, y)]
        r = max(f(x + 1, y), f(x, y + 1), key=lambda t: t[0])
        s = S[(A[x], B[y])]
        if s > 0.4:
            t = f(x + 1, y + 1)
            if t[0] + s > r[0]:
                r = (t[0] + s, [(A[x], B[y])] + t[1])
        best[(x, y)] = r
        return r

    sys.setrecursionlimit(100000)
    for i, j in f(0, 0)[1]:
        pair[j] = (i, "mod")
        used_old.add(i)

left_new = [j for j in range(len(nb)) if j not in pair]
for j in left_new:
    cands = [(sim(i, j), i) for i in range(len(ob)) if i not in used_old]
    if cands:
        s, i = max(cands)
        if s > 0.5:
            pair[j] = (i, "moved")
            used_old.add(i)
moved_old = {i: j for j, (i, k) in pair.items() if k == "moved"}

# emit in new order; deleted old blocks are placed after the new block that follows their old predecessor
old_after = {}
last_new_for_old = {}
for j, (i, k) in pair.items():
    if k != "moved":
        last_new_for_old[i] = j
anchor = -1
for i in range(len(ob)):
    if i in last_new_for_old:
        anchor = last_new_for_old[i]
    elif i not in used_old or i in moved_old:
        old_after.setdefault(anchor, []).append(i)

parts, folded = [], []
counts = {"mod": 0, "new": 0, "del": 0, "moved": 0}


def flush():
    if folded:
        n = len(folded)
        parts.append(f'<details class="same"><summary>{n} bloc{"s" if n > 1 else ""} inchangé{"s" if n > 1 else ""}</summary>'
                     + "".join(folded) + "</details>")
        folded.clear()


def emit_deleted(j):
    for i in old_after.get(j, []):
        b = ob[i]
        if i in moved_old:
            flush()
            parts.append(f'<div class="blk gone"><p class="tag">Déplacé plus loin : « {html.escape(otx[i][:90])}… »</p></div>')
            continue
        if not otx[i]:
            continue
        counts["del"] += 1
        flush()
        what = "Figure supprimée" if b.name == "figure" else "Tableau supprimé" if "tbl" in (b.get("class") or []) else "Supprimé"
        body = html.escape(otx[i])
        parts.append(f'<div class="blk del"><p class="tag">{what}</p><p><del>{body}</del></p></div>')


emit_deleted(-1)
for j, b in enumerate(nb):
    i, k = pair.get(j, (None, "new"))
    if k == "same":
        if b.name == "h1":
            continue
        if b.name in HEAD:
            flush()
            parts.append(links(shell(b, inner(b), "head")))
        else:
            folded.append(links(shell(b, inner(b), "same")))
    else:
        flush()
        tag = ""
        if k == "new":
            counts["new"] += 1
            body = "<ins>" + inner(b) + "</ins>" if not b.find(["img", "table", "math"]) else inner(b)
            tag = "Nouveau"
            cls = "new"
        else:
            counts["mod" if k == "mod" else "moved"] += 1
            body = wdiff(inner(ob[i]), inner(b))
            tag = "Déplacé et modifié" if k == "moved" else "Modifié"
            cls = "mod"
            if img_hashes(OLD, ob[i]) != img_hashes(NEW, b):
                tag += " · image modifiée"
        if b.name not in HEAD:
            parts.append(f'<p class="tag {cls}">{tag}</p>')
        parts.append(links(shell(b, body, cls)))
    emit_deleted(j)
flush()

style = nsoup.find("style").string + """
.blk.same, details.same > .blk { opacity: .55; }
details.same { margin: .4rem 0 1rem; font: .8rem var(--sans); color: var(--muted); }
details.same summary { cursor: pointer; padding: .2rem .5rem; border-left: 3px solid var(--rule); }
details.same[open] summary { margin-bottom: .5rem; }
details.same > .blk { font: 18px/1.62 var(--serif); color: var(--fg); }
.blk.mod, .blk.new, .blk.del { border-left: 3px solid var(--accent); padding-left: .8rem; margin-left: -.95rem; }
.blk.new { border-left-color: #2e8540; }
.blk.del { border-left-color: #c0392b; }
.blk.gone .tag { font-style: italic; }
p.tag { font: 600 .72rem var(--sans); letter-spacing: .04em; text-transform: uppercase; color: var(--muted); margin: 1.4rem 0 .2rem; }
ins { background: rgba(46,133,64,.16); color: inherit; text-decoration: none; border-radius: 2px; }
del { background: rgba(192,57,43,.13); color: var(--muted); text-decoration: line-through; text-decoration-color: rgba(192,57,43,.6); }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) ins { background: rgba(80,190,110,.22); }
  :root:not([data-theme="light"]) del { background: rgba(230,90,80,.2); } }
:root[data-theme="dark"] ins { background: rgba(80,190,110,.22); }
:root[data-theme="dark"] del { background: rgba(230,90,80,.2); }
"""
title = nsoup.find("header").find("h1").get_text(" ").strip()
head = "".join(str(t) for t in nsoup.find_all(["meta", "link"]))
note = (f'<div class="note"><b>Modifications depuis {html.escape(LABEL)}</b> : {counts["mod"]} blocs modifiés, '
        f'{counts["new"]} ajoutés, {counts["del"]} supprimés, {counts["moved"]} déplacés. '
        '<ins>Ajouts en vert</ins>, <del>suppressions en rouge</del> ; les blocs inchangés sont repliés. '
        'Les liens mènent à la <a href="index.html">version complète</a>.</div>')
page = ('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>Diff: {html.escape(LABEL)}</title>{head}<style>{style}</style></head><body><main>"
        f'<header class="paper"><h1>{html.escape(title)}</h1></header>{note}' + "".join(parts) + "</main></body></html>")
open(OUT, "w", encoding="utf-8").write(page)
print(OUT, counts, len(page), "bytes")
