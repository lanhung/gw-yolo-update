.PHONY: test audit pipeline

PYTHON ?= python
CONFIG ?= configs/legacy_remote.yaml

test:
	PYTHONPATH=src $(PYTHON) -m pytest

audit:
	PYTHONPATH=src $(PYTHON) -m gwyolo.cli audit --config $(CONFIG)

pipeline:
	PYTHONPATH=src $(PYTHON) -m gwyolo.cli pipeline --config $(CONFIG)
