from __future__ import annotations

from pathlib import Path

from src.research.rd_qa import (
    REQUIRED_IDS,
    load_rd_qa,
    qa_payload,
    render_qa_markdown,
)

REPO = Path(__file__).resolve().parents[2]


def test_rd_qa_has_required_ids() -> None:
    data = load_rd_qa(REPO)
    ids = [item["id"] for item in data["items"]]
    assert list(REQUIRED_IDS) == ids


def test_rd_qa_payload_bilingual() -> None:
    payload = qa_payload(REPO)
    assert payload["lab_path"] == "/rd/qa"
    assert payload["title"]["zh"]
    assert payload["title"]["en"]
    first = payload["items"][0]
    assert first["q"]["zh"]
    assert first["q"]["en"]
    assert first["a"]["zh"]
    assert first["a"]["en"]


def test_rd_qa_markdown_mirrors_yaml() -> None:
    data = load_rd_qa(REPO)
    assert (REPO / "docs/agent/rd_qa_CN.md").read_text(
        encoding="utf-8"
    ) == render_qa_markdown(data, "zh")
    assert (REPO / "docs/agent/rd_qa_EN.md").read_text(
        encoding="utf-8"
    ) == render_qa_markdown(data, "en")
