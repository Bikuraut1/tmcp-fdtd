# make install   -> install the Python packages
# make all       -> regenerate every table and figure from the shipped data and verify the tables (~1 min)
# make test      -> fast regression tests (~2 min)
# make simulate  -> re-run every full-wave simulation into ./rerun (about a day), then rebuild from it
PY ?= python3

.PHONY: install all tables figures analysis verify test list simulate clean
install:  ; $(PY) -m pip install -r requirements.txt
all:      ; $(PY) reproduce.py all
tables:   ; $(PY) reproduce.py tables
figures:  ; $(PY) reproduce.py figures
analysis: ; $(PY) reproduce.py analysis
verify:   ; $(PY) reproduce.py verify
test:     ; $(PY) -m pytest -q tests
list:     ; $(PY) reproduce.py simulate --list
simulate:
	TMCP_DATA=./rerun $(PY) reproduce.py simulate --all
	TMCP_DATA=./rerun TMCP_OUT=./outputs_rerun $(PY) reproduce.py all
clean:    ; rm -rf outputs outputs_rerun .pytest_cache
