# Research extract — no deploy / SSH / live hosts.

PYTHON ?= python3

.PHONY: help install test test-court

help:
	@echo "install     pip install -e .[dev]"
	@echo "test-court  research + closed-bar contract tests"
	@echo "test        alias of test-court"

install:
	$(PYTHON) -m pip install -e ".[dev]"

test-court:
	PYTHONPATH=src $(PYTHON) -m pytest \
		tests/research \
		tests/unit/test_backtest_no_future_data_contracts.py \
		tests/unit/test_timeline_closed_bar_decision.py \
		tests/unit/test_event_backtest_time_alignment.py \
		-q

test: test-court
