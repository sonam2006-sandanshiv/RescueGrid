.PHONY: setup download train run dashboard eval test demo-cache report

setup:
	uv venv
	uv pip install -r requirements.txt

download:
	python -m src.cli download

train:
	python -m src.ai.train

run:
	python -m src.cli run

dashboard:
	streamlit run app/dashboard.py

eval:
	python -m evaluation.emsr927_comparison

test:
	pytest tests/

demo-cache:
	python -m src.cli demo-cache

report:
	python -m src.report.sitrep
