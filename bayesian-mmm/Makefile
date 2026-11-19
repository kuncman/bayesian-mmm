.PHONY: install data run quick test test-all clean

install:        ## create the package in editable mode with dev tools
	pip install -e ".[dev]"

data:           ## regenerate the synthetic dataset
	bayesian-mmm simulate --out data

run:            ## full model fit + all outputs (~5-15 min)
	bayesian-mmm run --config config/config.yaml

quick:          ## fast smoke run (results NOT reliable)
	bayesian-mmm run --config config/config.yaml --quick --output-dir outputs_quick

test:           ## fast unit tests (no MCMC)
	pytest -q

test-all:       ## includes a short MCMC fit
	pytest -q --runslow

clean:
	rm -rf outputs outputs_quick .pytest_cache **/__pycache__
