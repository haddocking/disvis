"""HTML report of a DisVis run, with a Mol* view (MolViewSpec).

The report is built from the files DisVis writes in the output directory.
It refers to them by relative names, so the directory can be served as is,
e.g. with `python -m http.server -d <directory>`.
"""

import shutil
import warnings
from functools import partial
from html import escape
from os.path import basename, exists, join, samefile
from string import Template

with warnings.catch_warnings():
    # molviewspec 1.8 passes a deprecated keyword to pydantic's Field.
    warnings.filterwarnings(
        'ignore', 'Using extra keyword arguments on `Field`'
    )
    from molviewspec import MVSJ, GlobalMetadata, States, create_builder

MOLSTAR_VERSION = '4.18.0'

TEMPLATE = Template("""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>DisVis Report</title>
<script src="https://cdn.jsdelivr.net/npm/molstar@${molstar}/build/mvs-stories/mvs-stories.js"></script>
<link rel="stylesheet" type="text/css"
  href="https://cdn.jsdelivr.net/npm/molstar@${molstar}/build/mvs-stories/mvs-stories.css">
<style>
  body { margin: 0; font-family: sans-serif; }
  header, section { padding: 0 1rem; }
  .container { display: flex; height: 600px; }
  #viewer { flex: 2 1 0; min-width: 0; position: relative; }
  #controls { flex: 1 1 0; min-width: 300px; padding: 1rem;
              border-left: 1px solid #ccc; background: #f6f5f3; }
  table { border-collapse: collapse; margin: 0.5rem 0 1rem; }
  th, td { border: 1px solid #a0a0a0; padding: 4px 8px; text-align: right; }
  tr.high { font-weight: bold; background: #f8d7da; }
  #file-hint { background: #fff3cd; padding: 0.5rem; }
  @media (max-width: 900px) {
    .container { flex-direction: column; height: auto; }
    #viewer { height: 50vh; }
    #controls { border-left: none; border-top: 1px solid #ccc; }
  }
</style>
</head>
<body>
<header>
<h2>DisVis Report</h2>
<details><summary>Options</summary><ul>${options}</ul></details>
<p>Download: ${downloads}</p>
<p id="file-hint" hidden>The browser does not let a page opened from disk load
the files of the viewer. Serve this directory with
<code>python -m http.server -d &lt;directory&gt;</code> and open
<a href="http://localhost:8000/report.html">http://localhost:8000/report.html</a>.</p>
</header>
<div class="container">
  <div id="viewer"><mvs-stories-viewer /></div>
  <div id="controls"><mvs-stories-snapshot-markdown /></div>
</div>
<section>${tables}</section>
<script>
if (location.protocol === 'file:') {
  document.getElementById('file-hint').hidden = false;
} else {
  mvsStories.loadFromURL('state.mvsj', { format: 'mvsj' });
}
</script>
</body>
</html>
""")  # noqa: E501


def _read_rows(fname):
    """Return the rows of a whitespace-separated table, without comments."""
    with open(fname, encoding='utf-8') as f:
        return [line.split() for line in f if not line.startswith('#')]


def _restraint_labels(restraints):
    """Return labels such as A27(CA)-B18(CA) for the restraints file."""
    labels = []
    for row in _read_rows(restraints):
        if row:
            labels.append('{}{}({})-{}{}({})'.format(*row[:6]))
    return labels


def _table(title, header, rows, highlight=()):
    """Return an HTML table; rows whose index is in highlight are bold."""
    html = ['<h3>{}</h3>'.format(escape(title)), '<table><tr>']
    html += ['<th>{}</th>'.format(escape(h)) for h in header]
    html.append('</tr>')
    for i, row in enumerate(rows):
        html.append('<tr class="high">' if i in highlight else '<tr>')
        html += ['<td>{}</td>'.format(escape(str(c))) for c in row]
        html.append('</tr>')
    html.append('</table>')
    return ''.join(html)


def _snapshot(n, nrestraints, complexes, percentage, receptor, occupancy):
    """Return the Mol* snapshot of the space consistent with n restraints."""
    builder = create_builder()
    (
        builder.download(url=receptor)
        .parse(format='pdb')
        .model_structure()
        .component()
        .representation()
    )
    (
        builder.download(url='accessible_interaction_space.mrc')
        .parse(format='map')
        .volume()
        .representation(type='isosurface', absolute_isovalue=n - 0.5)
        .color(color='darkslategray')
        .opacity(opacity=0.7)
    )
    description = (
        'Accessible interaction space (grey) of the center of mass of the '
        'scanning chain, for complexes consistent with at least '
        '**{:d} of {:d}** restraints: {:d} complexes ({:s}).'
    ).format(n, nrestraints, complexes, percentage)
    if occupancy:
        (
            builder.download(url=occupancy)
            .parse(format='map')
            .volume()
            .representation(type='isosurface', relative_isovalue=2)
            .color(color='firebrick')
            .opacity(opacity=0.3)
        )
        description += (
            '\n\nOccupancy (red, 2 sigma): where the scanning chain is most '
            'often found in these complexes.'
        )
    return builder.get_snapshot(
        key='n{:d}'.format(n),
        title='At least {:d} restraints'.format(n),
        description=description,
        description_format='markdown',
    )


def generate_report(directory, receptor, restraints, options):
    """Write report.html and state.mvsj in the DisVis output directory.

    receptor is the fixed chain (copied to the directory, so the report can
    load it), restraints the restraints file, and options the command-line
    options shown in the report. Returns the path of report.html.
    """
    joiner = partial(join, directory)
    receptor_name = basename(receptor)
    target = joiner(receptor_name)
    if not (exists(target) and samefile(receptor, target)):
        shutil.copyfile(receptor, target)

    labels = _restraint_labels(restraints)
    nrestraints = len(labels)

    # Accessible complexes, up to the first N without any; the percentages
    # from the counts, as the fractions in the file have only 6 decimals.
    rows = _read_rows(joiner('accessible_complexes.out'))
    total = max(int(rows[0][3]), 1)
    complexes = []
    for row in rows:
        n, cumulative = int(row[0]), int(row[3])
        percentage = '{:.3g} %'.format(100 * cumulative / total)
        complexes.append((n, cumulative, percentage))
        if cumulative == 0:
            break

    snapshots = []
    for n, cumulative, percentage in reversed(complexes):
        if n == 0 or cumulative == 0:
            continue
        occupancy = 'occupancy_{:d}.mrc'.format(n)
        if not exists(joiner(occupancy)):
            occupancy = None
        snapshots.append(
            _snapshot(
                n,
                nrestraints,
                cumulative,
                percentage,
                receptor_name,
                occupancy,
            )
        )
    state = MVSJ(
        data=States(
            snapshots=snapshots,
            metadata=GlobalMetadata(description='DisVis results'),
        )
    )
    with open(joiner('state.mvsj'), 'w', encoding='utf-8') as f:
        f.write(state.dumps(indent=2))

    tables = [
        _table(
            'Accessible complexes',
            [
                'Consistent restraints (N)',
                'Complexes with at least N',
                'Fraction',
            ],
            complexes,
        )
    ]
    violations = [
        row
        for row in _read_rows(joiner('violations.out'))[1:]
        if any(float(x) for x in row[1:])
    ]
    tables.append(
        _table(
            'Fraction of complexes violating each restraint',
            ['N'] + labels,
            violations,
        )
    )
    zscores = [
        [row[0], label] + row[1:]
        for row, label in zip(
            _read_rows(joiner('z-score.out')), labels, strict=True
        )
    ]
    tables.append(
        _table(
            'Z-scores (above 1: possible false positive)',
            [
                '#',
                'Restraint',
                'Average violated fraction',
                'Standard deviation',
                'Z-score',
            ],
            zscores,
            {i for i, row in enumerate(zscores) if float(row[4]) >= 1},
        )
    )
    downloads = [
        'accessible_complexes.out',
        'violations.out',
        'z-score.out',
        'accessible_interaction_space.mrc',
        receptor_name,
    ]
    for chain in ('receptor', 'ligand'):
        fname = '{}_interactions.txt'.format(chain)
        if exists(joiner(fname)):
            rows = _read_rows(joiner(fname))
            with open(joiner(fname), encoding='utf-8') as f:
                header = f.readline().split()[1:]
            tables.append(
                _table(
                    'Average interactions per complex, {} residues'.format(
                        chain
                    ),
                    ['Residue'] + ['N ≥ {}'.format(h) for h in header],
                    rows,
                )
            )
            downloads.append(fname)

    html = TEMPLATE.substitute(
        molstar=MOLSTAR_VERSION,
        options=''.join(
            '<li>{}: {}</li>'.format(escape(k), escape(str(v)))
            for k, v in sorted(options.items())
        ),
        downloads=' '.join(
            '<a href="{0}">{0}</a>'.format(escape(f)) for f in downloads
        ),
        tables=''.join(tables),
    )
    fname = joiner('report.html')
    with open(fname, 'w', encoding='utf-8') as f:
        f.write(html)
    return fname
