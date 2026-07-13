#!/usr/bin/env python3
"""Validate Hexo Markdown, generated HTML, assets, and MathJax DOM."""

import argparse
import re
from pathlib import Path
from typing import Optional


FENCED_CODE_RE = re.compile(r"```.*?```", re.DOTALL)
INLINE_CODE_RE = re.compile(r"`[^`\n]*`")
DISPLAY_MATH_RE = re.compile(r"(?ms)^\s*\$\$\s*\n(.*?)^\s*\$\$\s*$")
WRAPPED_DISPLAY_MATH_RE = re.compile(
    r"(?ms)\{%\s*raw\s*%\}\s*\n\s*\$\$\s*\n.*?^\s*\$\$\s*\n\s*\{%\s*endraw\s*%\}"
)
INLINE_MATH_RE = re.compile(r"(?<!\\)(?<!\$)\$(?!\$)(.+?)(?<!\\)\$(?!\$)")
MALFORMED_HTML_PATTERNS = {
    '$$=""': "display math was parsed as an HTML attribute",
    '<p="">': "paragraph tag has an impossible empty attribute",
    "y_{<": "raw less-than sign from TeX leaked into generated HTML",
}


def line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def has_mathjax_front_matter(text: str) -> bool:
    front_matter = text.split("---", 2)[1] if text.startswith("---") else ""
    return bool(re.search(r"(?m)^mathjax:\s*true\s*$", front_matter))


def is_stellar_site(site: Path) -> bool:
    config = (site / "_config.yml").read_text(encoding="utf-8")
    return bool(re.search(r"(?m)^theme:\s*stellar\s*$", config))


def validate_math_source(text: str, require_raw_math: bool) -> tuple[list[str], int]:
    errors = []
    prose = INLINE_CODE_RE.sub("", FENCED_CODE_RE.sub("", text))
    delimiters = re.findall(r"(?<!\\)\$\$", prose)
    blocks = list(DISPLAY_MATH_RE.finditer(prose))
    if len(delimiters) != len(blocks) * 2:
        errors.append(
            "display math must use balanced $$ delimiters on their own lines; "
            f"found {len(delimiters)} delimiters but {len(blocks)} complete blocks"
        )

    for match in blocks:
        body = match.group(1)
        if "<" in body or ">" in body:
            errors.append(
                f"display math at line {line_number(prose, match.start())} contains raw < or >; "
                r"use \lt, \gt, \le, \ge, \langle, or \rangle because browsers parse angle "
                "brackets as HTML even inside Hexo raw tags"
            )

    without_display = DISPLAY_MATH_RE.sub("", prose)
    for match in INLINE_MATH_RE.finditer(without_display):
        if "<" in match.group(1) or ">" in match.group(1):
            errors.append(
                f"inline math at line {line_number(without_display, match.start())} contains raw < or >; "
                r"replace them with HTML-safe TeX commands such as \lt or \gt"
            )

    if blocks and not has_mathjax_front_matter(text):
        errors.append("article contains math but front matter does not set mathjax: true")

    wrapped_count = len(WRAPPED_DISPLAY_MATH_RE.findall(prose))
    if require_raw_math and wrapped_count != len(blocks):
        errors.append(
            "Stellar display math must be wrapped one block at a time with "
            "{% raw %} and {% endraw %}; "
            f"found {len(blocks)} blocks and {wrapped_count} wrapped blocks"
        )
    return errors, len(blocks)


def validate_generated_html(html: str, math_blocks: int) -> list[str]:
    errors = []
    for marker, explanation in MALFORMED_HTML_PATTERNS.items():
        if marker in html:
            errors.append(f"generated HTML contains {marker!r}: {explanation}")
    if "{% raw %}" in html or "{% endraw %}" in html:
        errors.append("generated HTML still contains unprocessed Hexo raw tags")
    if math_blocks and "MathJax" not in html:
        errors.append("generated HTML has formulas but does not load MathJax")
    return errors


def validate_rendered_dom(dom: str, math_blocks: int) -> tuple[list[str], int]:
    errors = validate_generated_html(dom, math_blocks)
    rendered_count = dom.count("<mjx-container")
    if math_blocks and rendered_count < math_blocks:
        errors.append(
            "browser-rendered DOM has fewer MathJax containers than source display blocks: "
            f"{rendered_count} < {math_blocks}"
        )
    return errors, rendered_count


def validate_article(
    site: Path,
    article: Path,
    html_path: Optional[Path] = None,
    rendered_dom_path: Optional[Path] = None,
) -> tuple[list[str], dict]:
    errors = []
    metrics = {"images": 0, "math_blocks": 0, "code_blocks": 0, "mjx_nodes": 0}
    if not (site / "_config.yml").is_file():
        return [f"not a Hexo root: {site}"], metrics
    if not article.is_file():
        return [f"article not found: {article}"], metrics

    text = article.read_text(encoding="utf-8")
    if not text.startswith("---\n") or text.count("---\n", 0, 2000) < 2:
        errors.append("missing or malformed YAML front matter")
    for field in ("title:", "date:", "description:", "categories:", "tags:"):
        if field not in text[:2000]:
            errors.append(f"front matter missing {field}")
    if "<!-- more -->" not in text:
        errors.append("missing <!-- more --> excerpt marker")
    if text.count("```") % 2:
        errors.append("unbalanced code fences")
    if text.count("{% raw %}") != text.count("{% endraw %}"):
        errors.append("unbalanced Hexo raw tags")

    math_errors, math_blocks = validate_math_source(text, is_stellar_site(site))
    errors.extend(math_errors)
    metrics["math_blocks"] = math_blocks
    metrics["code_blocks"] = text.count("```") // 2

    image_urls = re.findall(r"!\[[^]]*\]\((/images/[^)]+)\)", text)
    metrics["images"] = len(image_urls)
    for url in image_urls:
        asset = site / "source" / url.lstrip("/")
        if not asset.is_file():
            errors.append(f"missing image asset for {url}: {asset}")

    if html_path is not None:
        if not html_path.is_file():
            errors.append(f"generated HTML not found: {html_path}")
        else:
            errors.extend(validate_generated_html(html_path.read_text(encoding="utf-8"), math_blocks))

    if rendered_dom_path is not None:
        if not rendered_dom_path.is_file():
            errors.append(f"rendered DOM not found: {rendered_dom_path}")
        else:
            dom_errors, rendered_count = validate_rendered_dom(
                rendered_dom_path.read_text(encoding="utf-8"), math_blocks
            )
            errors.extend(dom_errors)
            metrics["mjx_nodes"] = rendered_count
    return errors, metrics


def fail(errors: list[str]) -> None:
    for error in errors:
        print(f"ERROR: {error}")
    raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", type=Path, required=True)
    parser.add_argument("--article", type=Path, required=True)
    parser.add_argument("--html", type=Path, help="generated public/.../index.html")
    parser.add_argument("--rendered-dom", type=Path, help="DOM dumped after MathJax runs in a browser")
    args = parser.parse_args()

    site = args.site.resolve()
    article = args.article if args.article.is_absolute() else site / args.article
    html_path = args.html if args.html is None or args.html.is_absolute() else site / args.html
    dom_path = (
        args.rendered_dom
        if args.rendered_dom is None or args.rendered_dom.is_absolute()
        else site / args.rendered_dom
    )
    errors, metrics = validate_article(site, article, html_path, dom_path)
    if errors:
        fail(errors)
    print(
        f"PASS: {article.name}; lines={len(article.read_text(encoding='utf-8').splitlines())}; "
        f"images={metrics['images']}; math_blocks={metrics['math_blocks']}; "
        f"code_blocks={metrics['code_blocks']}; mjx_nodes={metrics['mjx_nodes']}"
    )


if __name__ == "__main__":
    main()
