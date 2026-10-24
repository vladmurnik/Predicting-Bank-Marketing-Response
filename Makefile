.PHONY: setup download repro test lint serve clean

setup:
	uv sync

download:
	uv run python -m bank_marketing.cli download

repro:
	uv run python -m bank_marketing.cli all

test:
	uv run pytest -q

lint:
	uv run ruff check .

serve:
	uv run uvicorn bank_marketing.serve:app --host 127.0.0.1 --port 8000

clean:
	uv run python make.py clean
