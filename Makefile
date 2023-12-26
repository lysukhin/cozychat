SHELL:=/usr/bin/env bash

.PHONY: run
run:
	@cd examples && poetry run jupyter-lab --ServerApp.iopub_data_rate_limit=1.0e10

.PHONY: install
install:
	poetry install --sync
	poetry run dostoevsky download fasttext-social-network-model
	poetry run playwright install chromium

.PHONY: update
update:
	poetry self lock
	poetry self install --sync
	poetry self update
	poetry run pip install --upgrade pip setuptools wheel
	poetry update
	poetry run dostoevsky download fasttext-social-network-model
	poetry run playwright install chromium
	poetry export --without-hashes -f requirements.txt -o requirements.txt
	poetry export -f requirements.txt -o requirements-hashes.txt
