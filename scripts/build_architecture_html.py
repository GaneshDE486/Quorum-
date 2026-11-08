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
  <title>Quorum — Architecture</title>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=Figtree:wght@400;500;600;700&family=Fraunces:opsz,wght@9..144,520;9..144,650&display=swap" rel="stylesheet" />
  <style>
    :root {
      --text: #1c1612;
      --muted: #6d6258;
      --accent: #8d3b24;
      --accent-2: #2c4638;
      --border: #d8ccba;
    }
    * { box-sizing: border-box; }
    html { scroll-behavior: smooth; }
    body {
      margin: 0;
      font-family: "Figtree", "Segoe UI", sans-serif;
      background: #f3ecdf;
      color: var(--text);
      line-height: 1.65;
      min-height: 100vh;
    }
    .doc-top {
      position: sticky; top: 0; z-index: 20;
      display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 0.75rem;
      padding: 0.85rem clamp(1rem, 3vw, 2rem);
      background: #f3ecdf;
      color: var(--text);
      border-bottom: 1px solid #1c1612;
    }
    .doc-top a { color: #1c1612; text-decoration: none; font-family: Fraunces, Georgia, serif; font-weight: 560; font-size: 1.35rem; }
    .doc-top a:hover { color: var(--accent); }
    .doc-top span { font-size: 0.75rem; letter-spacing: 0.1em; text-transform: uppercase; color: var(--muted); }
    .doc-wrap { width: min(860px, 100%); margin: 0 auto; padding: 1.5rem 1.15rem 3rem; }
    .doc-article { background: transparent; border: none; padding: 0; box-shadow: none; }
    .doc-article h1 { font-family: Fraunces, Georgia, serif; font-weight: 560; font-size: clamp(1.8rem, 4vw, 2.5rem); letter-spacing: -0.03em; line-height: 1.1; margin: 0 0 0.8rem; color: #1c1612; }
    .doc-article h2 { font-family: Fraunces, Georgia, serif; font-weight: 560; font-size: 1.35rem; margin: 2rem 0 0.6rem; padding-bottom: 0.35rem; border-bottom: 1px solid #1c1612; border-left: none; color: #1c1612; }
    .doc-article h3 { font-size: 1.02rem; margin: 1.3rem 0 0.35rem; color: var(--accent); }
    .doc-article h4 { margin: 1rem 0 0.3rem; color: var(--accent-2); }
    .doc-article p, .doc-article li { color: #3a322c; }
    .doc-article a { color: var(--accent); font-weight: 650; }
    .doc-article hr { border: none; border-top: 1px solid var(--border); margin: 1.6rem 0; }
    .doc-article strong { color: #1c1612; }
    .doc-article code { font-size: 0.86em; background: #efe4d4; padding: 0.05rem 0.3rem; border: none; color: #1c1612; }
    .doc-article blockquote { margin: 0.8rem 0; padding: 0.2rem 0 0.2rem 0.8rem; border-left: 2px solid var(--accent); background: none; color: #3a322c; }
    .doc-article table { width: 100%; border-collapse: collapse; font-size: 0.88rem; margin: 1rem 0; }
    .doc-article th, .doc-article td { border: 1px solid var(--border); padding: 0.5rem 0.6rem; text-align: left; vertical-align: top; }
    .doc-article thead th { background: #efe4d4; color: #1c1612; }
    .doc-article tbody tr:nth-child(even) td { background: rgba(255,255,255,0.35); }
    .doc-article td { color: #3a322c; }
    .doc-article pre:not(.mermaid) { overflow-x: auto; padding: 0.9rem 1rem; background: #1c1612; color: #f6efe4; border-radius: 0; font-size: 0.8rem; }
    .doc-article pre code { background: none; color: inherit; padding: 0; }
    .mermaid { margin: 1rem 0; padding: 0.8rem; background: #faf6ee; border: 1px solid var(--border); overflow-x: auto; }
    .kw-risk, .doc-article .kw-risk { color: #8f2d2d; font-weight: 750; border-bottom: 1px solid #8f2d2d; background: none; }
    .kw-cadence, .doc-article .kw-cadence { color: #8a5a12; font-weight: 700; border-bottom: 1px solid #8a5a12; background: none; }
    .md-missing { color: #8d3b24; }
    .md-fallback { white-space: pre-wrap; background: #1c1612; color: #f6efe4; padding: 1rem; overflow-x: auto; }
    .doc-footer { margin-top: 2rem; padding-top: 0.8rem; border-top: 1px solid #1c1612; color: var(--muted); font-size: 0.82rem; }
    @media (max-width: 720px) {
      .doc-article table { display: block; overflow-x: auto; }
    }

  </style>
</head>
<body>
  <header class="doc-top">
    <a href="/">Quorum</a>
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
    mermaid.initialize({ startOnLoad: false, theme: "neutral", securityLevel: "loose" });
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
