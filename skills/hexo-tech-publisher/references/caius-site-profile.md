# Caius Hexo Site Profile

## Paths and branches

- User-facing root: `/Users/caius/Documents/alma/HEXO`
- Actual Hexo repository: `/Users/caius/Documents/alma/HEXO/caiusy.github.io`
- Posts: `source/_posts`
- Static images: `source/images`
- Tracked backups: `backups/posts`
- Repository: `git@github.com:caiusy/caiusy.github.io.git`
- `source` branch: Hexo source, Markdown, configuration, and assets
- `main` branch: generated GitHub Pages site
- Public root: `https://caiusy.github.io`

Always rediscover and verify these values because configuration can change.

## Site conventions

- Theme: Stellar
- `post_asset_folder: false`
- Static image URL: `/images/<slug>/<asset>`
- Long-form front matter commonly uses `mathjax`, `description`, `categories`, `tags`, `type`, `difficulty`, and `review_status`.
- Add `<!-- more -->` after the introductory promise.
- Math-heavy existing posts wrap display equations in Hexo raw tags.

## Safe release sequence

1. Inspect `git status`; do not absorb unrelated changes.
2. Create/update post, assets, and dated backup.
3. Run `npm run clean` and `npm run build`.
4. Inspect generated article HTML under `public/<year>/<month>/<day>/<slug>/index.html`.
5. Commit only the post, its assets, and backup on `source`.
6. Push `origin source`.
7. Run `npm run deploy`; configured deployment writes generated files to `main`.
8. Check Pages build status and public URL.

The outer `/Users/caius/Documents/alma/HEXO` directory belongs to a larger parent repository and contains legacy copies. Do not publish from its `source` directory.
