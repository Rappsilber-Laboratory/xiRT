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
	rm -r htmlcov

sample:
	xirt -i sample_data/DSS_xisearch_fdr_CSM50percent_minimal.csv -o sample_data/rt_test -x sample_data//parameter_examples//xirt_params_rp.yaml -l sample_data//parameter_examples//learning_params_training_cv.yaml

sample_fast:
	xirt -i sample_data/DSS_xisearch_fdr_CSM50percent_minimal.csv -o sample_data/rt_test_fast -x sample_data//parameter_examples//xirt_params_rp_fast.yaml -l sample_data//parameter_examples//learning_params_training_cv_fast.yaml

env:
	conda env update -f environment.yml --prune

pip_me:
	pip install -e . --no-deps