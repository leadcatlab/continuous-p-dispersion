# Paper and results pipeline.
#
#   make figures   rebuild generated figures from results/sweep.csv
#   make tables    rebuild the summary table
#   make fmt       format the manuscript with latexindent
#   make paper     tables + figures + fmt + build the PDF
#   make watch     continuous rebuild on save
#   make clean     remove LaTeX build artifacts
#
# The manuscript is the only hand-edited file. Figures and tables are build
# products regenerated from results/sweep.csv, so a number in the paper can
# always be traced back to the runs that produced it.

PY       ?= .venv/bin
PAPER    ?= continuous-p-dispersion-in-three-dimensions
PAPERDIR ?= paper
SWEEP    ?= results/sweep.csv
LATEXMK  ?= latexmk -pdf -interaction=nonstopmode -halt-on-error

.PHONY: all paper figures tables fmt watch clean distclean check help

help:
	@grep -E '^#   make' $(MAKEFILE_LIST) | sed 's/^#   /  /'

all: paper

figures:
	$(PY)/tsgo-figures --sweep $(SWEEP) --figures $(PAPERDIR)/figures

tables:
	$(PY)/tsgo-stats --sweep $(SWEEP) --csv results/summary_stats.csv \
	                 --latex $(PAPERDIR)/generated/summary.tex
	$(PY)/tsgo-tables --sweep $(SWEEP) --out $(PAPERDIR)/generated

# tex-fmt rather than latexindent: MacTeX ships latexindent's script but not
# its CPAN dependencies (File::HomeDir, Log::Log4perl, Log::Dispatch,
# Unicode::GCString), and one of those needs a compiler. tex-fmt is a single
# binary. Configured not to rewrap prose, so diffs stay reviewable.
fmt:
	cd $(PAPERDIR) && tex-fmt --config tex-fmt.toml $(PAPER).tex

paper: tables figures fmt
	cd $(PAPERDIR) && $(LATEXMK) $(PAPER).tex
	@echo "built $(PAPERDIR)/$(PAPER).pdf"

# Rebuild on save. Leaves figures alone; run `make figures` when results change.
watch:
	cd $(PAPERDIR) && latexmk -pdf -pvc -interaction=nonstopmode $(PAPER).tex

# Check the manuscript against the results without building anything.
check:
	$(PY)/tsgo-figures --sweep $(SWEEP) --list
	@cd $(PAPERDIR) && grep -o '\\includegraphics\[[^]]*\]{[^}]*}' $(PAPER).tex \
	  | sed 's/.*{\(.*\)}/\1/' | sort -u | while read f; do \
	    [ -f "$$f" ] || [ -f "$$f.png" ] || echo "  MISSING FIGURE: $$f"; done; true

clean:
	cd $(PAPERDIR) && latexmk -C $(PAPER).tex 2>/dev/null || true
	cd $(PAPERDIR) && rm -f *.aux *.log *.out *.blg *.fls *.fdb_latexmk *.synctex.gz *.bak*

distclean: clean
	rm -rf $(PAPERDIR)/generated
