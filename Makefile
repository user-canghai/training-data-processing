PYTHON ?= python3
VENV ?= .venv
PIP := $(VENV)/bin/pip
PY := $(VENV)/bin/python

.PHONY: venv install data cleaning dedup multimodal all clean

venv:
	$(PYTHON) -m venv $(VENV)

install: venv
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

data:
	$(PY) scripts/download_data.py

cleaning:
	$(PY) scripts/run_cleaning.py

dedup:
	$(PY) scripts/run_dedup.py

multimodal:
	HF_ENDPOINT=$${HF_ENDPOINT:-https://hf-mirror.com} $(PY) scripts/run_multimodal.py

all: data cleaning dedup multimodal

clean:
	rm -rf $(VENV) outputs data/processed
