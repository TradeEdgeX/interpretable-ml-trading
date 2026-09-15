# Teaching site

Human-facing docs (English default + Chinese at `/zh/`). Fine print stays in repo `docs/`.
Plain-language Lab Q&A: `docs/zh/qa/index.md` / `docs/en/qa/index.md`.

```bash
cd website
pip install -r requirements.txt
mkdocs serve
# build: mkdocs build --strict
```

GitHub Pages: [`.github/workflows/pages.yml`](../.github/workflows/pages.yml). Enable Pages on the publishing repo (Actions source). Do not push this extract’s old history to `TradeEdgeX/interpretable-ml-trading` unless that is the intentional publish path.
