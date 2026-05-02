# Makefile for the TTD cosmogenesis publication package
# Usage:
#   make all         - reproduce everything (simulation + figures + paper)
#   make sim         - run the headline simulation
#   make robustness  - run the ensemble of 10 realizations (~ 5 min)
#   make figures     - regenerate publication figures
#   make paper       - recompile the manuscript (requires pdflatex)
#   make clean       - remove generated files

PYTHON := python3

.PHONY: all sim robustness figures paper clean scaling

all: sim figures robustness paper

sim:
	cd code && $(PYTHON) 01_simulation.py

analysis:
	cd code && $(PYTHON) 02_analysis.py

figures:
	cd code && $(PYTHON) 03_make_figures.py

robustness:
	cd code && $(PYTHON) 04_robustness.py
	cd code && $(PYTHON) 05_robustness_figure.py

scaling:
	cd code && $(PYTHON) 06_finite_size_scaling.py

paper:
	pdflatex -interaction=nonstopmode paper.tex
	bibtex paper
	pdflatex -interaction=nonstopmode paper.tex
	pdflatex -interaction=nonstopmode paper.tex
	rm -f paper.aux paper.log paper.out paper.bbl paper.blg

clean:
	rm -f paper.aux paper.log paper.out paper.bbl paper.blg
	rm -f data/simulation_best.npz data/ensemble.npz data/ensemble_summary.json
	rm -f figures/*.pdf figures/*.png
