.PHONY: install test lint demo tune clean

PY ?= python

install:
	$(PY) -m pip install -r requirements.txt

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check . && $(PY) -m ruff format --check .

demo:
	$(PY) examples/run_demo.py --out outputs/benchmark.json

tune:
	$(PY) -m semiforge.cli tune

clean:
	rm -rf .pytest_cache .ruff_cache outputs __pycache__ */__pycache__ */*/__pycache__
