# https://gist.github.com/prwhite/8168133
.DEFAULT_GOAL := help

.PHONY: help test badge release install_dev install

help:                                 ## show this help
	@fgrep -h "##" $(MAKEFILE_LIST) | fgrep -v fgrep | sed -e "s/\\$$//" | sed -e "s/##//"

test:                                 ## run tests
	pytest tests

badge:
	coverage-badge -o documentation/imgs/coverage.svg -f

doc:
	sphinx-build -b html documentation/source documentation/build

release:
	python setup.py upload

install_dev:                          ## install millipede in development mode
	pip install --dev -e .

install:                              ## install local millipede package
	pip install -e . --no-deps

clean:                                ## clean up - remove docs, dist and build
	rm -r docs/build
	rm -r dist
	rm -r build

sample:
	xirt -i DSS_xisearch_fdr_CSM50percent_minimal.csv -o out_dir -x parameter_examples//xirt_params_rp.yaml -l parameter_examples//learning_params_training_cv.yaml

env:
	conda env update -f environment.yml

pip_me:
	pip install -e . --no-deps