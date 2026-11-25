"""Build a commentable HTML reading copy of the paper (for review on claude.ai) from the LaTeX sources.

Run after a full pdflatex + bibtex build (it reads main.aux for section/figure/table numbers and main.bbl for the
numbered references):   python paper/make_review_html.py OUT_DIR
Writes OUT_DIR/paper.html (an HTML fragment: title, styles, header, body) and OUT_DIR/fig/*.png.
"""

import os
import re
import subprocess
import sys

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
OUT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "review_html")
os.makedirs(os.path.join(OUT, "fig"), exist_ok=True)


def read(p):
    return open(os.path.join(SRC, p), encoding="utf-8").read()


def braced(s, i):
    """s[i] == '{': return (content, index after the matching brace)."""
    depth, j = 0, i
    while j < len(s):
        c = s[j]
        if c == "\\":
            j += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return s[i + 1 : j], j + 1
        j += 1
    raise ValueError("unbalanced braces at %d" % i)


def strip_comments(s):
    return re.sub(r"(?<!\\)%.*", "", s)


# ---------------------------------------------------------------- flatten the sources
def expand(s):
    s = strip_comments(s)
    s = re.sub(r"\\input\{([^}]+)\}", lambda m: expand(read(m.group(1) if m.group(1).endswith(".tex") else m.group(1) + ".tex")), s)
    s = re.sub(r"\\inputtab\{([^}]+)\}", lambda m: expand(read(m.group(1))), s)
    s = re.sub(r"\\InputIfFileExists\{([^}]+)\}\{\}\{\}", lambda m: expand(read(m.group(1))) if os.path.exists(os.path.join(SRC, m.group(1))) else "", s)
    return s


main = read("main.tex")
preamble, body = main.split("\\begin{document}")
body = body.split("\\end{document}")[0]
macros = expand(read("gen/macros.tex")) + expand(read("gen/macros_frontier.tex")) if os.path.exists(os.path.join(SRC, "gen/macros_frontier.tex")) else expand(read("gen/macros.tex"))
macros += "\n\\newcommand{\\pmstd}[1]{{\\scriptsize$\\pm$#1}}\n"
body = expand(body)

# \IfFileExists{A}{B}{C} -> B
out, i = [], 0
while True:
    k = body.find("\\IfFileExists{", i)
    if k < 0:
        out.append(body[i:])
        break
    out.append(body[i:k])
    _, j = braced(body, k + len("\\IfFileExists"))
    b, j = braced(body, j)
    _, j = braced(body, j)
    out.append(b)
    i = j
body = "".join(out)

# ---------------------------------------------------------------- numbers from the LaTeX build
aux = open(os.path.join(SRC, "main.aux"), encoding="utf-8").read()
labels = {m.group(1): m.group(2) for m in re.finditer(r"\\newlabel\{([^}]+)\}\{\{([^}]*)\}", aux)}
bbl = open(os.path.join(SRC, "main.bbl"), encoding="utf-8").read()
keys = re.findall(r"\\bibitem(?:\[[^]]*\])?\{([^}]+)\}", bbl)
num = {k: i + 1 for i, k in enumerate(keys)}

body = re.sub(r"\\(?:ref|autoref)\{([^}]+)\}", lambda m: f"\\hyperref[{m.group(1)}]{{{labels.get(m.group(1), '?')}}}", body)


def cite(m):
    ks = [k.strip() for k in m.group(1).split(",")]
    return "[" + ", ".join(f"\\hyperref[ref-{k}]{{{num.get(k, '?')}}}" for k in ks) + "]"


body = re.sub(r"\\cite(?:\[[^]]*\])?\{([^}]+)\}", cite, body)


# caption numbers: the label inside each float
def number_floats(s, env, prefix):
    def f(m):
        blk = m.group(0)
        lab = re.search(r"\\label\{([^}]+)\}", blk)
        n = labels.get(lab.group(1), "") if lab else ""
        return blk.replace("\\caption{", f"\\caption{{\\textbf{{{prefix}~{n}.}} ", 1)

    return re.sub(r"\\begin\{%s\*?\}.*?\\end\{%s\*?\}" % (env, env), f, s, flags=re.S)


body = number_floats(body, "figure", "Fig.")
body = number_floats(body, "table", "Table")

# section numbers (llncs: numbered sections and subsections; letters after \appendix)
sec, sub, app = 0, 0, False


def heading(m):
    global sec, sub, app
    kind, star, title = m.group(1), m.group(2), m.group(3)
    if star:
        return m.group(0)
    if kind == "section":
        sec += 1
        sub = 0
        n = chr(64 + sec) if app else str(sec)
    else:
        sub += 1
        n = f"{chr(64 + sec) if app else sec}.{sub}"
    return f"\\{kind}*{{{n}~~{title}"


parts = body.split("\\appendix")
parts[0] = re.sub(r"\\(section|subsection)(\*?)\{(.)", heading, parts[0])
if len(parts) > 1:
    sec, sub, app = 0, 0, True
    parts[1] = re.sub(r"\\(section|subsection)(\*?)\{(.)", heading, parts[1])
    body = parts[0] + "\n\\section*{Appendix}\n" + parts[1]

# ---------------------------------------------------------------- llncs front matter and float wrappers
title = re.search(r"\\title\{(.+?)\}\n", body).group(1)
body = re.sub(r"\\(title|titlerunning|author|authorrunning)\{.*?\}\n", "", body)
body = re.sub(r"\\institute\{.*?\}\n", "", body, flags=re.S)
body = body.replace("\\maketitle", "")
body = re.sub(r"\\keywords\{(.*?)\}", lambda m: "\n\n\\textbf{Keywords:} " + m.group(1).replace("\\and", "·"), body, flags=re.S)
body = body.replace("\\begin{abstract}", "\\section*{Abstract}\n").replace("\\end{abstract}", "")
for cmd in ("FloatBarrier", "centering", "small", "footnotesize", "scriptsize", "clearpage", "newpage"):
    body = body.replace("\\" + cmd, "")
body = re.sub(r"\\begin\{adjustbox\}\{[^}]*\}|\\end\{adjustbox\}", "", body)
body = re.sub(r"\\resizebox\{[^}]*\}\{[^}]*\}", "", body)
body = re.sub(r"\\fbox\{\\parbox\{[^}]*\}\{[^}]*\}\}", "", body)
body = body.replace("\\bibliographystyle{splncs04}", "")
body = re.sub(r"\\multicolumn\{(\d+)\}\{(?:@\{\})?\|?([lcr])\|?(?:@\{\})?\}", r"\\multicolumn{\1}{\2}", body)


# figures -> PNG
def fig(m):
    name = os.path.splitext(os.path.basename(m.group(1)))[0]
    png = os.path.join(OUT, "fig", name + ".png")
    if not os.path.exists(png):
        subprocess.run(["pdftoppm", "-r", "170", "-png", "-singlefile", os.path.join(SRC, m.group(1)), png[:-4]], check=True)
    return "\\includegraphics{fig/%s.png}" % name


body = re.sub(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+\.pdf)\}", fig, body)

# references from the .bbl, numbered as in the PDF
items = re.split(r"\\bibitem(?:\[[^]]*\])?\{([^}]+)\}", bbl.split("\\end{thebibliography}")[0])[1:]
refs = ["\\section*{References}", "\\begin{enumerate}"]
for k, txt in zip(items[0::2], items[1::2]):
    txt = txt.replace("\\newblock", " ").strip()
    txt = re.sub(r"\\doi\{([^}]+)\}", r"\\url{https://doi.org/\1}", txt)
    txt = txt.replace("\\urlprefix", "")
    refs.append(f"\\item \\label{{ref-{k}}}{txt}")
refs.append("\\end{enumerate}")
body = body.replace("\\bibliography{refs}", "\n".join(refs))

tex = "\\documentclass{article}\n\\usepackage{amsmath}\n" + macros + "\n\\begin{document}\n" + body + "\n\\end{document}\n"
open(os.path.join(OUT, "paper_flat.tex"), "w", encoding="utf-8").write(tex)
r = subprocess.run(["pandoc", "-f", "latex", "-t", "html5", "--mathml", "--wrap=none", "--section-divs", os.path.join(OUT, "paper_flat.tex")],
                   capture_output=True, text=True, cwd=OUT)
if r.returncode:
    sys.exit(r.stderr)
html = r.stdout
title_txt = subprocess.run(["pandoc", "-f", "latex", "-t", "plain", "--wrap=none"], input=title, capture_output=True, text=True).stdout.strip()
rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=SRC).stdout.strip()

# ---------------------------------------------------------------- post-process the pandoc fragment
seen = set()


def dedup(m):
    i = m.group(1)
    if i in seen:
        return "<div>"
    seen.add(i)
    return m.group(0)


html = re.sub(r'<div id="([^"]+)">', dedup, html)
html = re.sub(r"(<div>\s*<table>)\s*<caption>.*?</caption>", r"\1", html, flags=re.S)  # repeated caption of a split table
html = html.replace("<table>", '<div class="scroll"><table>').replace("</table>", "</table></div>")
html = re.sub(r'<div class="scroll"><table>\s*<caption>(.*?)</caption>', r'<p class="caption">\1</p><div class="scroll"><table>', html, flags=re.S)
html = re.sub(r'(<div id="tab:[^"]+")>', r'\1 class="tbl">', html)
html = re.sub(r' label="[^"]*"', "", html)
html = html.replace(' class="unnumbered"', "").replace(" unnumbered", "")
for a, b in (("h2", "h3"), ("h1", "h2")):
    html = html.replace(f"<{a}>", f"<{b}>").replace(f"<{a} ", f"<{b} ").replace(f"</{a}>", f"</{b}>")
html = html.replace('<img src=', '<img loading="lazy" alt="" src=')

toc = "".join(f'<li><a href="#{i}">{re.sub("<[^>]+>", "", t)}</a></li>'
              for i, t in re.findall(r'<section id="([^"]+)" class="level1">\s*<h2>(.*?)</h2>', html, flags=re.S)
              if i not in ("abstract", "appendix"))

authors = "Xavier Desprez · Oudom Kem"
page = f"""<title>VFL Paper Review</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;1,8..60,400&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root {{
  --bg: #fbfaf7; --fg: #1d1d1b; --muted: #66635c; --rule: #dedad1; --accent: #2a5db0;
  --panel: #f2efe8; --stripe: #f6f4ee; --fig-bg: #ffffff;
  --serif: "Source Serif 4", "Iowan Old Style", Georgia, serif;
  --sans: Inter, system-ui, -apple-system, "Segoe UI", sans-serif;
  color-scheme: light;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --bg: #17181a; --fg: #e6e3dc; --muted: #a29e95; --rule: #34363a; --accent: #8db4f0;
    --panel: #212327; --stripe: #1d1f22; --fig-bg: #f4f2ec; color-scheme: dark;
  }}
}}
:root[data-theme="dark"] {{
  --bg: #17181a; --fg: #e6e3dc; --muted: #a29e95; --rule: #34363a; --accent: #8db4f0;
  --panel: #212327; --stripe: #1d1f22; --fig-bg: #f4f2ec; color-scheme: dark;
}}
* {{ box-sizing: border-box; }}
html {{ -webkit-text-size-adjust: 100%; }}
body {{ margin: 0; background: var(--bg); color: var(--fg); font: 18px/1.62 var(--serif); }}
main {{ max-width: 44rem; margin: 0 auto;
  padding: 2.5rem max(16px, env(safe-area-inset-right)) 5rem max(16px, env(safe-area-inset-left)); }}
header.paper h1 {{ font: 600 1.75rem/1.25 var(--serif); margin: 0 0 .75rem; text-wrap: balance; }}
header.paper .authors {{ font: 500 .95rem var(--sans); color: var(--muted); margin: 0; }}
.note {{ font: .85rem/1.5 var(--sans); color: var(--muted); background: var(--panel); border-radius: 8px;
  padding: .75rem 1rem; margin: 1.5rem 0; }}
.note b {{ color: var(--fg); font-weight: 600; }}
details.toc {{ font: .9rem/1.5 var(--sans); margin: 0 0 2rem; }}
details.toc summary {{ cursor: pointer; color: var(--muted); }}
details.toc ol {{ list-style: none; padding-left: 0; margin: .5rem 0 0; columns: 2 14rem; }}
details.toc li {{ break-inside: avoid; margin: .15rem 0; }}
a {{ color: var(--accent); text-decoration: none; }}
a:hover {{ text-decoration: underline; }}
h2 {{ font: 600 1.35rem/1.3 var(--serif); margin: 2.75rem 0 .75rem; padding-top: .5rem; border-top: 1px solid var(--rule); }}
h3 {{ font: 600 1.1rem/1.35 var(--serif); margin: 2rem 0 .5rem; }}
h4 {{ font: 600 1rem/1.5 var(--serif); margin: 1.5rem 0 .25rem; }}
section#abstract h2 {{ border-top: 0; font-size: 1rem; font-family: var(--sans); text-transform: uppercase; letter-spacing: .06em; color: var(--muted); }}
section#abstract p {{ font-size: .97rem; }}
p {{ margin: 0 0 1rem; hyphens: auto; }}
figure {{ margin: 2rem 0; }}
figure img {{ display: block; width: 100%; height: auto; background: var(--fig-bg); border-radius: 6px; padding: 6px; }}
figcaption, p.caption {{ font: .85rem/1.5 var(--sans); color: var(--muted); margin: .6rem 0 0; }}
figcaption strong, p.caption strong {{ color: var(--fg); }}
.tbl {{ margin: 2rem 0; }}
.tbl p.caption {{ margin: 0 0 .6rem; }}
.scroll {{ overflow-x: auto; -webkit-overflow-scrolling: touch; }}
table {{ border-collapse: collapse; font: .8rem/1.4 var(--sans); font-variant-numeric: tabular-nums; margin: 0 0 .5rem; }}
th, td {{ padding: .3rem .55rem; border-bottom: 1px solid var(--rule); vertical-align: top; }}
thead th {{ border-bottom: 1.5px solid var(--fg); font-weight: 600; white-space: nowrap; }}
tbody tr:nth-child(even) {{ background: var(--stripe); }}
td {{ white-space: nowrap; }}
td[style*="left"] {{ white-space: normal; min-width: 9rem; }}
math {{ font-size: 1.02em; }}
code {{ font-size: .85em; background: var(--panel); padding: .05em .3em; border-radius: 4px; overflow-wrap: anywhere; }}
section#references ol {{ font-size: .88rem; line-height: 1.5; padding-left: 2.2rem; }}
section#references li {{ margin: .35rem 0; }}
section#references p {{ margin: 0; overflow-wrap: anywhere; }}
section.footnotes, aside.footnotes {{ font-size: .88rem; color: var(--muted); }}
:target {{ scroll-margin-top: 1rem; }}
li:target, div:target, figure:target {{ outline: 2px solid var(--accent); outline-offset: 4px; border-radius: 4px; }}
@media (max-width: 560px) {{
  body {{ font-size: 17px; }}
  header.paper h1 {{ font-size: 1.45rem; }}
  main {{ padding-top: 1.5rem; }}
}}
</style>
<main>
<header class="paper">
<h1>{title_txt}</h1>
<p class="authors">{authors}</p>
</header>
<div class="note"><b>Brouillon pour relecture</b> (version {rev}). Sélectionnez un passage pour le commenter&nbsp;: texte ou dictée vocale du clavier. Numéros de sections, figures, tableaux et références identiques au PDF.</div>
<details class="toc"><summary>Sommaire</summary><ol>{toc}</ol></details>
{html}
</main>
"""
open(os.path.join(OUT, "index.html"), "w", encoding="utf-8").write(page)
print("warnings:", len(r.stderr.splitlines()), "| bytes:", len(page), "| rev:", rev)
