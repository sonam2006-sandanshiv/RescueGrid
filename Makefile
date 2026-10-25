.PHONY: setup download train run dashboard eval eval-ai test demo-cache report
setup:
	python -m venv .venv
	.venv/Scripts/python -m pip install -r requirements.txt
download:
	python -m src.cli download
train:
	python -m src.ai.train
run:
	python -m src.cli run --bbox 85.0,27.8,85.5,28.3 --date 2026-08-26
dashboard:
	streamlit run app/dashboard.py
eval:
	python -m evaluation.metrics evaluation/benchmark_manifest.json
eval-ai:
	python -m src.ai.eval_unseen
test:
	python -m pytest -q
demo-cache:
	python -m src.cli demo-cache
report:
	python -m src.report.sitrep
