#!/usr/bin/env python3
"""Validate a Hexo post and its local static assets before deployment."""

import argparse
import re
from pathlib import Path


def fail(errors):
    for error in errors:
        print(f"ERROR: {error}")
    raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", type=Path, required=True)
    parser.add_argument("--article", type=Path, required=True)
    args = parser.parse_args()

    site = args.site.resolve()
    article = args.article if args.article.is_absolute() else site / args.article
    errors = []
    if not (site / "_config.yml").is_file():
        errors.append(f"not a Hexo root: {site}")
    if not article.is_file():
        errors.append(f"article not found: {article}")
    if errors:
        fail(errors)

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
    if text.count("$$") % 2:
        errors.append("unbalanced display math markers")
    if text.count("{% raw %}") != text.count("{% endraw %}"):
        errors.append("unbalanced Hexo raw tags")

    image_urls = re.findall(r"!\[[^]]*\]\((/images/[^)]+)\)", text)
    for url in image_urls:
        asset = site / "source" / url.lstrip("/")
        if not asset.is_file():
            errors.append(f"missing image asset for {url}: {asset}")

    if errors:
        fail(errors)
    print(
        f"PASS: {article.name}; lines={len(text.splitlines())}; "
        f"images={len(image_urls)}; math_blocks={text.count('$$') // 2}; "
        f"code_blocks={text.count('```') // 2}"
    )


if __name__ == "__main__":
    main()
