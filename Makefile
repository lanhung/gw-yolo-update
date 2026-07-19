.PHONY: test audit pipeline factory-pilot

PYTHON ?= python
CONFIG ?= configs/legacy_remote.yaml

test:
	PYTHONPATH=src $(PYTHON) -m pytest

audit:
	PYTHONPATH=src $(PYTHON) -m gwyolo.cli audit --config $(CONFIG)

pipeline:
	PYTHONPATH=src $(PYTHON) -m gwyolo.cli pipeline --config $(CONFIG)

factory-pilot:
	PYTHONPATH=src $(PYTHON) -m gwyolo.cli data-factory --config configs/data_factory_pilot.yaml --output-dir artifacts/data_factory_pilot
