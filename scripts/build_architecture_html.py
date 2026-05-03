"""
Regenerate docs/ARCHITECTURE.html from docs/ARCHITECTURE.md (Markdown → HTML).

Run from project root:
  python3 scripts/build_architecture_html.py

Requires: pip install markdown>=3.5
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MD_PATH = ROOT / "docs" / "ARCHITECTURE.md"
OUT_PATH = ROOT / "docs" / "ARCHITECTURE.html"

_MERMAID_BLOCK = re.compile(
    r'<pre><code class="language-mermaid">(.*?)</code></pre>',
    re.DOTALL,
)


def _promote_mermaid_blocks(html: str) -> str:
    def _repl(m: re.Match[str]) -> str:
        return f'<div class="mermaid">\n{m.group(1).strip()}\n</div>'

    return _MERMAID_BLOCK.sub(_repl, html)


SHELL_HEAD = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Architecture — Transcript Intelligence</title>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght@0,400..800;1,600&display=swap" rel="stylesheet" />
  <style>
    :root {
      --text: #f4f1ff;
      --muted: #9d94b8;
      --accent: #5eead4;
      --accent-2: #f0abfc;
      --border: rgba(167, 139, 250, 0.24);
    }
    * { box-sizing: border-box; }
    html { scroll-behavior: smooth; }
    body {
      margin: 0;
      font-family: "Plus Jakarta Sans", system-ui, sans-serif;
      background-color: #08070f;
      background-image:
        radial-gradient(ellipse 100% 80% at 0% -20%, rgba(139, 92, 246, 0.34), transparent 55%),
        radial-gradient(ellipse 90% 55% at 100% 8%, rgba(20, 184, 166, 0.18), transparent 52%),
        radial-gradient(ellipse 70% 45% at 85% 100%, rgba(251, 113, 133, 0.11), transparent 42%),
        linear-gradient(180deg, #0a0912 0%, #14122a 38%, #0b0a12 100%);
      color: var(--text);
      line-height: 1.62;
      min-height: 100vh;
    }
    .doc-top {
      position: sticky;
      top: 0;
      z-index: 20;
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
      gap: 0.75rem;
      width: 100%;
      padding: 0.9rem clamp(1rem, 3vw, 2.5rem);
      background: rgba(8, 7, 15, 0.9);
      backdrop-filter: blur(14px);
      border-bottom: 1px solid rgba(94, 234, 212, 0.22);
    }
    .doc-top a {
      color: var(--accent);
      text-decoration: none;
      font-weight: 650;
      font-size: 0.9rem;
    }
    .doc-top a:hover { color: var(--accent-2); text-decoration: underline; }
    .doc-top span { font-size: 0.78rem; color: var(--muted); font-weight: 500; }
    .doc-wrap {
      width: 100%;
      max-width: none;
      margin: 0;
      padding: clamp(1rem, 1.8vw, 1.75rem) clamp(1rem, 4vw, 3.5rem) clamp(2.5rem, 6vw, 5rem);
    }
    .doc-article {
      width: 100%;
      max-width: none;
      background: transparent;
      border: none;
      border-radius: 0;
      padding: 0;
      box-shadow: none;
    }
    .doc-article h1 {
      font-size: clamp(1.55rem, 3.2vw, 2.25rem);
      margin: 0 0 1rem;
      letter-spacing: -0.03em;
      line-height: 1.15;
      background: linear-gradient(105deg, #fdf4ff 0%, #a5f3fc 42%, #c4b5fd 100%);
      -webkit-background-clip: text;
      background-clip: text;
      color: transparent;
    }
    .doc-article h2 {
      font-size: clamp(1.05rem, 1.9vw, 1.35rem);
      margin: 2.35rem 0 0.85rem;
      padding: 0.5rem 0 0.6rem 1rem;
      border-left: 4px solid #a78bfa;
      border-bottom: 1px solid var(--border);
      color: #ede9fe;
    }
    .doc-article h2:first-of-type { margin-top: 0; }
    .doc-article h2:nth-of-type(4n+2) {
      border-left-color: #2dd4bf;
      color: #ccfbf1;
    }
    .doc-article h2:nth-of-type(4n+3) {
      border-left-color: #fbbf24;
      color: #fef9c3;
    }
    .doc-article h2:nth-of-type(4n+4) {
      border-left-color: #fb7185;
      color: #ffe4e6;
    }
    .doc-article h3 {
      font-size: clamp(0.98rem, 1.4vw, 1.08rem);
      margin: 1.45rem 0 0.55rem;
      color: #7ee8d8;
    }
    .doc-article h4 {
      font-size: 0.95rem;
      margin: 1rem 0 0.4rem;
      color: #c4b5fd;
    }
    .doc-article p {
      margin: 0.65rem 0;
      color: #dcd7ec;
      font-size: clamp(0.88rem, 1.2vw, 0.98rem);
    }
    .doc-article a { color: #7dd3fc; font-weight: 500; }
    .doc-article a:hover { color: #f0abfc; }
    .doc-article hr {
      border: none;
      height: 1px;
      background: linear-gradient(90deg, transparent, rgba(167, 139, 250, 0.45), transparent);
      margin: 2.25rem 0;
    }
    .doc-article ul, .doc-article ol {
      margin: 0.6rem 0 0.6rem 1.25rem;
      color: #dcd7ec;
      font-size: clamp(0.88rem, 1.2vw, 0.98rem);
    }
    .doc-article li { margin: 0.35rem 0; }
    .doc-article strong { color: #fff; }
    .doc-article code {
      font-size: 0.86em;
      padding: 0.12rem 0.4rem;
      border-radius: 6px;
      background: rgba(30, 27, 55, 0.95);
      border: 1px solid rgba(94, 234, 212, 0.22);
      color: #a5f3fc;
    }
    .doc-article blockquote {
      margin: 0.9rem 0;
      padding: 0.65rem 1rem;
      border-left: 4px solid #f0abfc;
      background: rgba(139, 92, 246, 0.1);
      color: #c4bcd9;
      font-size: 0.92rem;
      border-radius: 0 10px 10px 0;
    }
    .doc-article table {
      width: 100%;
      border-collapse: collapse;
      font-size: clamp(0.8rem, 1.1vw, 0.9rem);
      margin: 1.15rem 0;
    }
    .doc-article th, .doc-article td {
      border: 1px solid var(--border);
      padding: 0.55rem 0.65rem;
      text-align: left;
      vertical-align: top;
    }
    .doc-article thead th {
      background: linear-gradient(180deg, rgba(139, 92, 246, 0.38) 0%, rgba(45, 212, 191, 0.16) 100%);
      color: #faf5ff;
      font-weight: 650;
    }
    .doc-article tbody tr:nth-child(even) td { background: rgba(255, 255, 255, 0.025); }
    .doc-article td { color: #e8e4f6; }
    .doc-article pre:not(.mermaid) {
      overflow-x: auto;
      padding: 1rem 1.15rem;
      border-radius: 12px;
      background: rgba(12, 10, 22, 0.92);
      border: 1px solid rgba(167, 139, 250, 0.28);
      font-size: 0.82rem;
    }
    .doc-article pre code { background: none; border: none; padding: 0; color: #d8d4e8; }
    .mermaid {
      margin: 1.5rem 0;
      padding: 1.15rem 1.25rem;
      border-radius: 14px;
      background: rgba(18, 16, 35, 0.78);
      border: 1px solid rgba(45, 212, 191, 0.24);
      overflow-x: auto;
    }
    .kw-risk {
      color: #fecaca;
      font-weight: 700;
      background: rgba(248, 113, 113, 0.2);
      border: 1px solid rgba(248, 113, 113, 0.45);
      padding: 0.06rem 0.28rem;
      border-radius: 4px;
      white-space: nowrap;
    }
    .kw-cadence {
      color: #fef08a;
      font-weight: 650;
      background: rgba(250, 204, 21, 0.12);
      border: 1px solid rgba(250, 204, 21, 0.38);
      padding: 0.06rem 0.28rem;
      border-radius: 4px;
      white-space: nowrap;
    }
    .doc-footer {
      font-size: 0.82rem;
      color: var(--muted);
      margin-top: 3rem;
      padding-top: 1.25rem;
      border-top: 1px dashed rgba(167, 139, 250, 0.35);
    }
  </style>
</head>
<body>
  <header class="doc-top">
    <a href="/">← Dashboard</a>
    <span>Generated from docs/ARCHITECTURE.md</span>
  </header>
  <div class="doc-wrap">
    <article class="doc-article">
"""

SHELL_TAIL = """
      <p class="doc-footer">Regenerate: <code>python3 scripts/build_architecture_html.py</code> · Source: <code>docs/ARCHITECTURE.md</code></p>
    </article>
  </div>
  <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
  <script>
    mermaid.initialize({ startOnLoad: false, theme: "dark", securityLevel: "loose" });
    mermaid.run({ querySelector: ".mermaid" });
  </script>
</body>
</html>
"""


def main() -> None:
    try:
        import markdown as markdown_lib
    except ImportError:
        print("Install markdown: pip install markdown>=3.5", file=sys.stderr)
        sys.exit(1)
    if not MD_PATH.is_file():
        print("Missing", MD_PATH, file=sys.stderr)
        sys.exit(1)
    md_text = MD_PATH.read_text(encoding="utf-8")
    inner = markdown_lib.markdown(
        md_text,
        extensions=["fenced_code", "tables", "extra"],
    )
    inner = _promote_mermaid_blocks(inner)
    OUT_PATH.write_text(SHELL_HEAD + inner + SHELL_TAIL, encoding="utf-8")
    print("Wrote", OUT_PATH)


if __name__ == "__main__":
    main()
