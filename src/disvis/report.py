"""HTML report of a DisVis run, with a Mol* view (MolViewSpec).

The report is built from the files DisVis writes in the output directory.
It refers to them by relative names, so the directory can be served as is,
e.g. with `python -m http.server -d <directory>`.
"""

import csv
import shutil
from functools import partial
from html import escape
from os.path import basename, exists, join, samefile
from string import Template

from molviewspec import (
    MVSJ,
    ComponentExpression,
    GlobalMetadata,
    States,
    create_builder,
)

from disvis.helpers import parse_interactions

MOLSTAR_VERSION = '5.13.1'
# Number of residues in the lists of the most contacted residues.
TOP = 10

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
  table.sortable th { cursor: pointer; }
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
<section>
<p><i>Click a column header to sort a table by that column; click again to
reverse the order.</i></p>
${tables}</section>
<script>
// Sort a table by a column on a click on its header, again to reverse.
document.querySelectorAll('table.sortable th').forEach(th => {
  th.title = 'Click to sort';
  th.addEventListener('click', () => {
    const table = th.closest('table');
    const col = th.cellIndex;
    const rows = [...table.rows].slice(1);
    const key = r => {
      const text = r.cells[col].textContent;
      const x = parseFloat(text);
      return isNaN(x) ? text : x;
    };
    const dir = th.dataset.dir === 'desc' ? 1 : -1;
    th.dataset.dir = dir === 1 ? 'asc' : 'desc';
    // Show the sorted column and its direction.
    for (const h of table.rows[0].cells) {
      h.textContent = h.textContent.replace(/ [▲▼]$$/, '');
    }
    th.textContent += dir === 1 ? ' ▲' : ' ▼';
    rows.sort((a, b) => {
      const x = key(a), y = key(b);
      return dir * (x < y ? -1 : x > y ? 1 : 0);
    });
    rows.forEach(r => table.tBodies[0].appendChild(r));
  });
});
if (location.protocol === 'file:') {
  document.getElementById('file-hint').hidden = false;
} else {
  mvsStories.loadFromURL('state.mvsj', { format: 'mvsj' });
}
</script>
</body>
</html>
""")  # noqa: E501


def _read_rows(fname, delimiter=None):
    """Return the rows of an output table, without its header.

    Without delimiter, the columns have a fixed width and the header lines
    start with #; with a delimiter (--delimiter), the header is the first
    line.
    """
    with open(fname, encoding='utf-8', newline='') as f:
        if delimiter:
            return list(csv.reader(f, delimiter=delimiter))[1:]
        return [line.split() for line in f if not line.startswith('#')]


def _restraint_labels(restraints):
    """Return labels such as A27(CA)-B18(CA) for the restraints file."""
    # Comments and empty lines as in helpers.parse_restraints.
    with open(restraints, encoding='utf-8') as f:
        rows = [line.split() for line in f]
    return [
        '{}{}({})-{}{}({})'.format(*row[:6])
        for row in rows
        if row and not row[0].startswith('#')
    ]


def _vmax(interactions):
    """Return the maximum of parse_interactions data, or 1 if it is 0."""
    ns = [n for n in interactions if n != 'residues']
    return max(max(interactions[n]) for n in ns) or 1


def _shade(t, rgb):
    """Return the hex color at fraction t from white to rgb."""
    return '#{:02x}{:02x}{:02x}'.format(
        *[round(255 + t * (c - 255)) for c in rgb]
    )


def _table(title, header, rows, highlight=(), sortable=True, shade=None):
    """Return an HTML table; rows whose index is in highlight are bold.

    A sortable table is sorted by a column when its header is clicked. With
    shade, an RGB color and a maximum, the cells after the first column are
    colored from white (0) to that color (the maximum).
    """
    html = [
        '<h3>{}</h3>'.format(escape(title)),
        '<table class="sortable"><tr>' if sortable else '<table><tr>',
    ]
    html += ['<th>{}</th>'.format(escape(h)) for h in header]
    html.append('</tr>')
    for i, row in enumerate(rows):
        html.append('<tr class="high">' if i in highlight else '<tr>')
        for j, c in enumerate(row):
            style = ''
            if shade and j:
                rgb, vmax = shade
                style = ' style="background: {}"'.format(
                    _shade(float(c) / vmax, rgb)
                )
            html.append('<td{}>{}</td>'.format(style, escape(str(c))))
        html.append('</tr>')
    html.append('</table>')
    return ''.join(html)


def _receptor_colors(interactions):
    """Return the colors of the receptor residues for each N, and the maximum.

    interactions is parse_interactions data, or None (no colors). Residues
    are colored from white (no interactions) to red (the most
    interactions per complex); residues without interactions are left out,
    as white is the base color.
    """
    if interactions is None:
        return {}, None
    vmax = _vmax(interactions)
    residues = interactions['residues']
    colors = {}
    for n, values in interactions.items():
        if n != 'residues':
            colors[n] = [
                (resi, _shade(x / vmax, (178, 24, 43)))
                for resi, x in zip(residues, values, strict=True)
                if x > 0
            ]
    return colors, vmax


def _interaction_tables(chain, interactions):
    """Return the tables of the residue interactions of a chain.

    The first lists, for each N, the most contacted residues, most first;
    the second gives the average interactions per complex of all residues,
    colored from white to green (most contacted).
    """
    residues = interactions['residues']
    ns = [n for n in interactions if n != 'residues']
    top = []
    for n in ns:
        ranked = sorted(
            zip(interactions[n], residues, strict=True), key=lambda p: -p[0]
        )[:TOP]
        top.append([n, ', '.join(str(r) for x, r in ranked if x > 0)])
    rows = [
        [resi] + ['{:.3f}'.format(interactions[n][i]) for n in ns]
        for i, resi in enumerate(residues)
    ]
    return [
        _table(
            'Most contacted {} residues (top {:d}, most first)'.format(
                chain, TOP
            ),
            ['N ≥', 'Residues'],
            top,
            sortable=False,
        ),
        _table(
            'Average interactions per complex, {} residues'.format(chain),
            ['Residue'] + ['N ≥ {}'.format(n) for n in ns],
            rows,
            shade=((35, 139, 69), _vmax(interactions)),
        ),
    ]


def _snapshot(
    n, nrestraints, complexes, percentage, receptor, occupancy, colors
):
    """Return the Mol* snapshot of the space consistent with n restraints.

    colors is None, or the residue colors of the receptor and their maximum.
    """
    builder = create_builder()
    representation = (
        builder.download(url=receptor)
        .parse(format='pdb')
        .model_structure()
        .component()
        # Mol* would fit the whole map grid, far larger than the complex.
        .focus(radius_factor=2)
        .representation()
    )
    if colors:
        representation.color(color='white')
        for resi, color in colors[0]:
            representation.color(
                color=color, selector=ComponentExpression(auth_seq_id=resi)
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
            .color(color='forestgreen')
            .opacity(opacity=0.3)
        )
        description += (
            '\n\nOccupancy (green, 2 sigma): where the scanning chain is most '
            'often found in these complexes.'
        )
    if colors:
        description += (
            '\n\nFixed chain: the selected residues colored by their average '
            'number of interactions per complex, from white (0) to red '
            '({:.3g}).'
        ).format(colors[1])
    return builder.get_snapshot(
        key='n{:d}'.format(n),
        title='At least {:d} restraints'.format(n),
        description=description,
        description_format='markdown',
    )


def _copy_inputs(directory, fnames):
    """Copy the input files (None is skipped) to directory; return names."""
    names = []
    for fname in fnames:
        if fname:
            target = join(directory, basename(fname))
            if not (exists(target) and samefile(fname, target)):
                shutil.copyfile(fname, target)
            names.append(basename(fname))
    return names


def generate_report(
    directory,
    receptor,
    ligand,
    restraints,
    options,
    selection=None,
    delimiter=None,
):
    """Write report.html and state.mvsj in the DisVis output directory.

    receptor (the fixed chain), ligand (the scanning chain), restraints and
    the optional interaction selection are the input files: they are copied
    to the directory, where the report links them and loads the receptor.
    options are the command-line options shown in the report, delimiter
    that of the .out files (None: fixed width). Returns the path of
    report.html.
    """
    joiner = partial(join, directory)
    inputs = _copy_inputs(directory, (receptor, ligand, restraints, selection))
    receptor_name = inputs[0]

    labels = _restraint_labels(restraints)
    nrestraints = len(labels)

    # Accessible complexes, up to the first N without any; the percentages
    # from the counts, as the fractions in the file have only 6 decimals.
    rows = _read_rows(joiner('accessible_complexes.out'), delimiter)
    total = int(rows[0][3])
    complexes = []
    for row in rows:
        n, cumulative = int(row[0]), int(row[3])
        percentage = '{:.3g} %'.format(100 * cumulative / total)
        complexes.append((n, cumulative, percentage))
        if cumulative == 0:
            break

    interactions = {}
    for chain in ('receptor', 'ligand'):
        fname = joiner('{}_interactions.txt'.format(chain))
        if exists(fname):
            interactions[chain] = parse_interactions(fname)
    colors, vmax = _receptor_colors(interactions.get('receptor'))

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
                (colors[n], vmax) if n in colors else None,
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
            # The most restraints first, as the snapshots.
            complexes[::-1],
        )
    ]
    violations = _read_rows(joiner('violations.out'), delimiter)
    if not delimiter:
        # The first row numbers the restraints.
        violations = violations[1:]
    # Only the N with complexes; all N if all restraints are consistent,
    # whose row is 0 as no complex violates a restraint.
    accessible = {n for n, cumulative, _ in complexes if cumulative}
    violations = [r for r in violations if int(r[0]) in accessible]
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
            _read_rows(joiner('z-score.out'), delimiter), labels, strict=True
        )
    ]
    # The Z-scores are relative: one is above 1 even when all restraints
    # are consistent, which leaves no false positive to flag.
    consistent = nrestraints in accessible
    tables.append(
        _table(
            'Z-scores (all restraints are consistent)'
            if consistent
            else 'Z-scores (above 1.0: possible false positive)',
            [
                '#',
                'Restraint',
                'Average violated fraction',
                'Standard deviation',
                'Z-score',
            ],
            zscores,
            ()
            if consistent
            else {i for i, row in enumerate(zscores) if float(row[4]) >= 1},
        )
    )
    downloads = [
        'accessible_complexes.out',
        'violations.out',
        'z-score.out',
        'accessible_interaction_space.mrc',
        *inputs,
    ]
    for chain, data in interactions.items():
        tables += _interaction_tables(chain, data)
        downloads.append('{}_interactions.txt'.format(chain))

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
