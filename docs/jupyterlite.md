# Try it in your browser

This notebook runs DisVis entirely in your browser, with
[JupyterLite](https://jupyterlite.readthedocs.io) and
[Pyodide](https://pyodide.org) (Python compiled to WebAssembly): nothing is
installed and no data leave your computer. It runs the PRE5-PUP2 test case and
shows the report of the results.

<!-- HTML link: lite/ is built after MkDocs (make lite), so mkdocs --strict
would not find it. -->
<a href="lite/lab/index.html?path=disvis_demo.ipynb">Launch the demo notebook</a>

The browser runs DisVis on a single core, without pyFFTW: use it to try
DisVis, and the [command line tool](installation.md) or the
[web service](manual.md#web-service) for real searches.
