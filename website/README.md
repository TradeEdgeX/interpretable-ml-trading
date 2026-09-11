# Teaching site

Human-facing docs (Chinese default + English). Fine print stays in repo `docs/`.

```bash
cd website
pip install -r requirements.txt
mkdocs serve
# build: mkdocs build --strict
```

GitHub Pages: [`.github/workflows/pages.yml`](../.github/workflows/pages.yml). Enable Pages on the publishing repo (Actions source). Do not push this extract’s old history to `TradeEdgeX/interpretable-ml-trading` unless that is the intentional publish path.
