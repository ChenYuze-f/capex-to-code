PYTHON ?= .venv/bin/python
STREAMLIT ?= .venv/bin/streamlit

.PHONY: help venv install test run list export

help:
	@echo "CapEx-to-Code Development & Research Commands:"
	@echo "  make venv     - Create Python virtual environment (.venv)"
	@echo "  make install  - Install project dependencies into .venv"
	@echo "  make test     - Run the full 38-test unit & integration suite"
	@echo "  make run      - Launch the interactive Streamlit web terminal"
	@echo "  make list     - Print ranked cross-sectional peer table in terminal"
	@echo "  make export   - Export recomputed peer valuation table to CSV"

venv:
	python3 -m venv .venv

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m unittest discover -s tests -v

run:
	$(STREAMLIT) run app.py

list:
	$(PYTHON) cli.py --list

export:
	$(PYTHON) cli.py --export-csv peer_valuation_matrix.csv
