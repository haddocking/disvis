# Build the DisVis documentation site (../site/), with the JupyterLite demo
# (lite/) and the WebAssembly wheel of DisVis it runs.
#
# Run from docs/, with the docs extra installed (pip install -e '.[docs]'):
#   make          the whole site
#   make serve    serve it at http://localhost:8000
#   make clean    remove the build outputs

ROOT := ..
SITE := $(ROOT)/site
WHEELS := lite/pypi
SRC := wheel-src

.DEFAULT_GOAL := site
.PHONY: site wheel mkdocs lite serve clean

# The wheel is built from the committed source (git archive), as a local
# build/ directory of a native build breaks the WebAssembly build.
wheel:
	rm -rf $(SRC) $(WHEELS)/*.whl && mkdir -p $(SRC)
	git -C $(ROOT) archive HEAD | tar -x -C $(SRC)
	CIBW_BUILD=cp314-pyodide_wasm32 CIBW_ARCHS=wasm32 CIBW_TEST_SKIP='*' \
	  python -m cibuildwheel --platform pyodide --output-dir $(WHEELS) $(SRC)
	rm -rf $(SRC)

mkdocs:
	cd $(ROOT) && mkdocs build --strict

lite:
	jupyter lite build --lite-dir lite --contents content \
	  --output-dir $(SITE)/lite

site: wheel mkdocs lite

serve:
	python -m http.server -d $(SITE) 8000

clean:
	rm -rf $(SRC) $(SITE) $(WHEELS)/*.whl .jupyterlite.doit.db lite/.cache
