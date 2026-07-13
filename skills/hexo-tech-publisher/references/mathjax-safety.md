# MathJax and Hexo Formula Safety

Use this reference whenever an article contains LaTeX or MathJax.

## The failure mode

Hexo `{% raw %}` prevents the Markdown renderer from changing LaTeX, but it does not escape the final HTML. A browser still interprets raw angle brackets as tags.

Unsafe:

```latex
$$
L=-\log \pi_\theta(y_t\mid y_{<t},x)
$$
```

The `<t` fragment can become an HTML tag and produce malformed output such as `$$=""` or `<p="">`. MathJax then cannot recognize the complete formula.

Safe:

```latex
{% raw %}
$$
L=-\log \pi_\theta(y_t\mid y_{\lt t},x)
$$
{% endraw %}
```

## Required substitutions

| Unsafe literal | Safe TeX |
|---|---|
| `<` | `\lt` |
| `>` | `\gt` |
| `<=` | `\le` |
| `>=` | `\ge` |
| angle brackets | `\langle`, `\rangle` |

Apply the same rule to inline math. Do not write `$i < t$`; write `$i \lt t$`.

## Three validation layers

1. Source Markdown: balanced delimiters, `mathjax: true`, raw wrappers on Stellar, no raw angle brackets in math.
2. Generated HTML: MathJax script exists and malformed signatures such as `$$=""`, `<p="">`, and `y_{<` are absent.
3. Rendered DOM: after browser JavaScript runs, `<mjx-container>` nodes exist for every display block.

The bundled checker supports all three layers:

```bash
python scripts/check_hexo_article.py \
  --site /path/to/site \
  --article source/_posts/article.md \
  --html public/year/month/day/slug/index.html \
  --rendered-dom /tmp/article-rendered-dom.html
```

Run the source-only check before `hexo generate`, then run it again with `--html` after generation. When Chrome is available, dump the rendered DOM after MathJax has loaded and pass it with `--rendered-dom`.

## Browser verification

Static HTML checks are necessary but not sufficient. For math-heavy posts, inspect at least one formula containing subscripts, comparison operators, fractions, and sums in a real browser. Search the rendered DOM for `<mjx-container`; searching only for the word `MathJax` proves that the library loaded, not that formulas rendered.
