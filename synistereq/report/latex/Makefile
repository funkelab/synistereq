.PHONY: default

default:
	latexmk -auxdir=.latexmk -outdir=.latexmk -pdf -halt-on-error -shell-escape main.tex
	cp .latexmk/main.pdf .

clean:
	-rm -rf .latexmk
