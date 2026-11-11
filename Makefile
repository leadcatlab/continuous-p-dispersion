# Paper and results pipeline.
#
#   make figures   rebuild generated figures from supplement/tables/sweep.csv
#   make tables    rebuild the summary table
#   make fmt       format the manuscript: tex-fmt, then refill prose paragraphs to
#                  80 columns (skipped where tex-fmt is not installed)
#   make paper     tables + figures + fmt + build the PDF (before a sweep has
#                  finished: committed figures, placeholder tables)
#   make submission  flat LaTeX source zip for Editorial Manager, test-compiled
#                  (paper/submission.zip; build the PDF and check it first)
#   make watch     continuous rebuild on save
#   make clean     remove LaTeX build artifacts
#   make format    format the Python code and fix what ruff can (ruff.toml)
#   make lint      check formatting and lint, change nothing
#
# The manuscript is the only hand-edited file. Figures and tables are build
# products regenerated from supplement/tables/sweep.csv, so a number in the paper can
# always be traced back to the runs that produced it.

PY       ?= .venv/bin
PAPER    ?= continuous-p-dispersion-in-three-dimensions
PAPERDIR ?= paper
SWEEP    ?= supplement/tables/sweep.csv
LATEXMK  ?= latexmk -pdf -interaction=nonstopmode -halt-on-error

# On the campus cluster latexmk comes from `module load texlive/2024`; use
# that installation when latexmk is not already on the PATH.
TEXLIVE  ?= /sw/apps/texlive/2024/bin/x86_64-linux
ifeq ($(shell command -v latexmk 2>/dev/null),)
  export PATH := $(if $(wildcard $(TEXLIVE)/latexmk),$(TEXLIVE):)$(PATH)
endif

.PHONY: all paper figures tables fmt submission watch clean distclean check help format lint

help:
	@grep -E '^#   make' $(MAKEFILE_LIST) | sed 's/^#   /  /'

all: paper

# Without $(SWEEP) (a sweep not yet finished) the committed figures stay and
# each missing generated table becomes a placeholder, so the PDF still builds.
figures:
	@if [ -f $(SWEEP) ]; then \
	  $(PY)/tsgo-figures --sweep $(SWEEP) --figures $(PAPERDIR)/figures \
	    --only-in $(PAPERDIR)/$(PAPER).tex; \
	else echo "figures: no $(SWEEP) yet; keeping the committed figures"; fi

tables:
	@if [ -f $(SWEEP) ]; then \
	  $(PY)/tsgo-stats --sweep $(SWEEP) --csv supplement/tables/summary_stats.csv \
	                   --latex $(PAPERDIR)/generated/summary.tex && \
	  $(PY)/tsgo-tables --sweep $(SWEEP) --out $(PAPERDIR)/generated; \
	else \
	  echo "tables: no $(SWEEP) yet; placeholders for missing tables"; \
	  mkdir -p $(PAPERDIR)/generated; \
	  for t in $$(grep -o 'input{generated/[^}]*}' $(PAPERDIR)/$(PAPER).tex | sed 's/input{generated\/\(.*\)}/\1/'); do \
	    [ -f $(PAPERDIR)/generated/$$t.tex ] || \
	      echo '\par\noindent\textit{[Table pending: $(SWEEP) has not been built yet.]}\par' \
	        > $(PAPERDIR)/generated/$$t.tex; \
	  done; \
	fi

# tex-fmt rather than latexindent: MacTeX ships latexindent's script but not
# its CPAN dependencies (File::HomeDir, Log::Log4perl, Log::Dispatch,
# Unicode::GCString), and one of those needs a compiler. tex-fmt is a single
# binary. It indents and breaks long lines but never joins short ones, so
# texfill.py then refills each prose paragraph to an even 80 columns (only
# whitespace changes) and a last tex-fmt pass checks the indentation. Run
# `make fmt` as its own commit: refilling moves whole paragraphs in a diff.
fmt:
	@if command -v tex-fmt >/dev/null; then \
	  cd $(PAPERDIR) && tex-fmt --config tex-fmt.toml $(PAPER).tex && \
	  ../$(PY)/python texfill.py $(PAPER).tex && \
	  tex-fmt --config tex-fmt.toml $(PAPER).tex; \
	else echo "fmt: tex-fmt not installed here; manuscript left as is"; fi

paper: tables figures fmt
	cd $(PAPERDIR) && $(LATEXMK) $(PAPER).tex
	@echo "built $(PAPERDIR)/$(PAPER).pdf"

# Editorial Manager rejects LaTeX uploads with subfolders, so this flattens the
# figure and table paths into paper/submission/, test-compiles a copy there,
# and zips the sources. Tables come from supplement/tables/latex/ and must
# match paper/generated/. Upload the highlights (paper/highlights.txt) as
# their own item, not in the zip.
submission:
	cd $(PAPERDIR) && ../$(PY)/python submission.py --paper $(PAPER)

# Rebuild on save. Leaves figures alone; run `make figures` when results change.
watch:
	cd $(PAPERDIR) && latexmk -pdf -pvc -interaction=nonstopmode $(PAPER).tex

# Check the manuscript against the results without building anything.
check:
	$(PY)/tsgo-figures --sweep $(SWEEP) --only-in $(PAPERDIR)/$(PAPER).tex --list
	@cd $(PAPERDIR) && grep -o '\\includegraphics\[[^]]*\]{[^}]*}' $(PAPER).tex \
	  | sed 's/.*{\(.*\)}/\1/' | sort -u | while read f; do \
	    [ -f "$$f" ] || [ -f "$$f.png" ] || echo "  MISSING FIGURE: $$f"; done; true

clean:
	cd $(PAPERDIR) && latexmk -C $(PAPER).tex 2>/dev/null || true
	cd $(PAPERDIR) && rm -f *.aux *.log *.out *.blg *.fls *.fdb_latexmk *.synctex.gz *.bak* *Notes.bib

distclean: clean
	rm -rf $(PAPERDIR)/generated

# Python code: ruff, configured in ruff.toml (single quotes, sorted imports,
# pyupgrade to 3.11, bugbear). `format` rewrites files; `lint` only reports.
format:
	$(PY)/ruff check --fix --exit-zero .
	$(PY)/ruff format .

lint:
	$(PY)/ruff format --check .
	$(PY)/ruff check .
