---
name: hexo-tech-publisher
description: Optimize Markdown technical documentation into a polished Hexo long-form article, safely render MathJax formulas, manage diagrams and static assets, build locally, back up source, deploy, and verify the public URL. Use when the user asks to publish or update a Hexo technical article, especially math-heavy posts or requests like "发到博客", "整理为 Hexo 文章", "发布并备份", or "把这篇文档做成长文".
---

# Hexo Tech Publisher

Turn an existing technical document or topic into a readable, illustrated Hexo article and own the workflow through public verification. Do not make the user run commands.

## Workflow

### 1. Discover the real site

1. Inspect the supplied directory before editing.
2. Locate the directory containing `package.json`, `_config.yml`, `source/_posts`, and its own `.git`.
3. Read `package.json`, deployment configuration, theme configuration, one recent long-form post, branch tracking, remotes, and `git status`.
4. Preserve unrelated user changes. Never stage unrelated files.
5. If the site is Caius's known site, read [references/caius-site-profile.md](references/caius-site-profile.md).

Do not assume the directory named `HEXO` is the repository. Nested source and deployment repositories are common.

### 2. Design the article

Create a publication-oriented version rather than blindly copying the source:

- Add valid YAML front matter matching the site's existing conventions.
- Write a concise opening promise and add `<!-- more -->` after the introduction.
- Preserve technical correctness, equations, code, data provenance, and caveats.
- Replace report-like repetition with a reader path: intuition, concrete example, data flow, equations, code, tradeoffs, summary.
- Clearly label synthetic numbers, hypothetical rollouts, measured logs, and real dataset samples.
- For long technical articles, include at least one end-to-end worked example.
- Use code-native SVG diagrams for architecture, tensor flow, and loss flow when visual explanation helps.

Never claim a hypothetical reward, metric, or generated answer came from a real log.

### 3. Adapt assets and mathematics

1. Put static assets under `source/images/<article-slug>/` when `post_asset_folder: false`.
2. Reference them as `/images/<article-slug>/<file>`.
3. Inspect an existing math-heavy article before choosing syntax.
4. For the known Stellar site, set `mathjax: true` and wrap each display block in `{% raw %}` / `{% endraw %}`.
5. Never put literal `<` or `>` inside inline or display math. Use `\lt`, `\gt`, `\le`, `\ge`, `\langle`, and `\rangle`; Hexo raw tags do not protect the final HTML parser.
6. Keep every `{% raw %}` paired and every `$$`/code fence balanced.
7. Read [references/mathjax-safety.md](references/mathjax-safety.md) for formula syntax and the three-layer validation procedure.
8. Use descriptive image alt text.

### 4. Back up before deployment

Maintain at least two recoverable copies:

- The canonical Markdown in the project or source repository.
- A dated copy under the Hexo repository's `backups/posts/` or an equivalent tracked backup directory.

Commit and push article source and assets to the source branch before running deployment. Report both source and deployment commits.

### 5. Validate locally

Run the bundled checker:

```bash
CHECKER="${CODEX_HOME:-$HOME/.codex}/skills/hexo-tech-publisher/scripts/check_hexo_article.py"
python "$CHECKER" \
  --site /path/to/hexo-site \
  --article source/_posts/article.md
```

Then use the site's scripts, typically:

```bash
npm run clean
npm run build
```

Confirm the generated HTML contains:

- Article title and description.
- A distinctive worked-example value.
- Every expected image URL.
- MathJax when equations are present.
- The expected permalink.

For every math-heavy article, rerun the checker against generated HTML:

```bash
python "$CHECKER" \
  --site /path/to/hexo-site \
  --article source/_posts/article.md \
  --html public/year/month/day/slug/index.html
```

Reject deployment if generated HTML contains `$$=""`, `<p="">`, `y_{<`, raw Hexo tags, or lacks MathJax. When Chrome is available, dump the post-render DOM and pass `--rendered-dom`; require `<mjx-container>` nodes, not merely a MathJax script tag. Visually inspect at least one representative formula.

Do not deploy after a failed build.

### 6. Deploy without destroying source

Inspect `_config.yml` and branch topology first.

- If source and generated site use different branches, push source first, then run the configured deploy command.
- If deploy targets the same branch that stores source, pause and choose a safe Pages workflow or deployment branch. Do not force-overwrite source.
- Treat `hexo deploy`, force pushes, branch changes, and public publishing as consequential actions requiring the normal approval path.

### 7. Verify online

After deployment:

1. Check GitHub Pages build state when available.
2. Allow for deployment propagation instead of treating the first 404 as failure.
3. Require HTTP 200 from the final public URL.
4. Search downloaded HTML for title, worked example, image path, and MathJax marker.
5. Report the public URL, source file, backup file, source commit, deployment commit, and verification result.

## Quality Gate

The task is complete only when all applicable items pass:

- Article is written for readers, not copied as an internal report.
- Front matter parses.
- Equations and fences are balanced.
- Math source contains no raw angle brackets and Stellar display blocks are raw-wrapped.
- Generated HTML has no malformed formula signatures.
- A browser-rendered DOM contains MathJax containers for math-heavy posts.
- Assets exist and are generated into `public`.
- Local Hexo build succeeds.
- Source and backup are committed before deployment.
- Public URL returns 200 and contains the new content.
- Unrelated worktree changes remain untouched.

## Trigger Examples

- “把这个教程整理成博客并发布。”
- “将这份 Markdown 优化后放到我的 Hexo，公式和配图也处理好。”
- “更新上一篇技术文章，备份源码后重新部署。”
- “把项目复盘写成长文，发布到 caiusy.github.io。”
