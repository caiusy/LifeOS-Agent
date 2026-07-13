#!/usr/bin/env python3
"""Regression tests for Hexo and MathJax validation."""

import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("check_hexo_article.py")
SPEC = importlib.util.spec_from_file_location("check_hexo_article", SCRIPT)
CHECKER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(CHECKER)


def article(formula: str) -> str:
    return f"""---
title: Formula Test
date: 2026-07-13 00:00:00
description: test
categories:
  - Test
tags:
  - Math
mathjax: true
---
<!-- more -->

{{% raw %}}
$$
{formula}
$$
{{% endraw %}}
"""


class MathSafetyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "_config.yml").write_text("theme: stellar\n", encoding="utf-8")
        self.post = self.root / "source" / "_posts" / "formula.md"
        self.post.parent.mkdir(parents=True)

    def tearDown(self):
        self.temp.cleanup()

    def validate(self, html=None, dom=None):
        return CHECKER.validate_article(self.root, self.post, html, dom)

    def test_html_safe_tex_passes(self):
        self.post.write_text(article(r"y_t \mid y_{\lt t}"), encoding="utf-8")
        errors, metrics = self.validate()
        self.assertEqual(errors, [])
        self.assertEqual(metrics["math_blocks"], 1)

    def test_raw_less_than_in_display_math_fails(self):
        self.post.write_text(article(r"y_t \mid y_{<t}"), encoding="utf-8")
        errors, _ = self.validate()
        self.assertTrue(any("contains raw < or >" in error for error in errors))

    def test_raw_less_than_in_inline_math_fails(self):
        self.post.write_text(article(r"y_t") + "\nBad $y_{<t}$ formula.\n", encoding="utf-8")
        errors, _ = self.validate()
        self.assertTrue(any("inline math" in error for error in errors))

    def test_unsafe_formula_shown_as_code_is_not_treated_as_math(self):
        self.post.write_text(
            article(r"y_t \mid y_{\lt t}") + "\nFor example, do not write `$y_{<t}$`.\n",
            encoding="utf-8",
        )
        errors, _ = self.validate()
        self.assertEqual(errors, [])

    def test_malformed_generated_html_fails(self):
        self.post.write_text(article(r"y_t \mid y_{\lt t}"), encoding="utf-8")
        html = self.root / "bad.html"
        html.write_text('<script>MathJax</script>$$="" <p="">', encoding="utf-8")
        errors, _ = self.validate(html=html)
        self.assertTrue(any("generated HTML contains" in error for error in errors))

    def test_rendered_mathjax_dom_passes(self):
        self.post.write_text(article(r"y_t \mid y_{\lt t}"), encoding="utf-8")
        html = self.root / "good.html"
        html.write_text("<script>MathJax</script><p>formula</p>", encoding="utf-8")
        dom = self.root / "dom.html"
        dom.write_text(
            '<script>MathJax</script><mjx-container jax="CHTML"></mjx-container>',
            encoding="utf-8",
        )
        errors, metrics = self.validate(html=html, dom=dom)
        self.assertEqual(errors, [])
        self.assertEqual(metrics["mjx_nodes"], 1)

    def test_missing_rendered_mathjax_node_fails(self):
        self.post.write_text(article(r"y_t \mid y_{\lt t}"), encoding="utf-8")
        dom = self.root / "dom.html"
        dom.write_text("<script>MathJax</script><p>formula never rendered</p>", encoding="utf-8")
        errors, _ = self.validate(dom=dom)
        self.assertTrue(any("fewer MathJax containers" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
