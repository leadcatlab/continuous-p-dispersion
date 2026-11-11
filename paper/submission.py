#!/usr/bin/env python3
"""Build the LaTeX source upload for Editorial Manager as one flat folder.

EM cannot process LaTeX submissions with subfolders: every file must sit at
the same level as the .tex. This copies the manuscript into OUT/ with its
figure and table paths flattened, adds the .bib, the .bbl and the elsarticle
class and bibliography style, test-compiles a copy in OUT/build/, and zips
the sources to OUT.zip. Run from paper/ (`make submission` does).

The tables come from ../supplement/tables/latex/, the committed copies; the
build stops if paper/generated/ disagrees with them, since the PDF you have
been reading was built from paper/generated/.

    python submission.py [--out submission]
"""

import argparse
import filecmp
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

PAPER = 'continuous-p-dispersion-in-three-dimensions'
TABLES = Path('../supplement/tables/latex')
GRAPHIC = re.compile(r'(\\includegraphics(?:\[[^\]]*\])?\{)figures/([^}]*)\}')
INPUT = re.compile(r'\\input\{generated/([^}]*)\}')


def kpsewhich(name: str) -> Path:
    path = subprocess.run(['kpsewhich', name], capture_output=True, text=True).stdout.strip()
    if not path:
        sys.exit(f'submission: kpsewhich cannot find {name}')
    return Path(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--out', default='submission')
    ap.add_argument('--paper', default=PAPER, help='manuscript name without .tex')
    args = ap.parse_args()
    out, paper = Path(args.out), args.paper

    tex = Path(f'{paper}.tex').read_text(encoding='utf-8')
    figures = sorted(set(m.group(2) for m in GRAPHIC.finditer(tex)))
    tables = sorted(set(INPUT.findall(tex)))

    for t in tables:
        src, used = TABLES / f'{t}.tex', Path('generated') / f'{t}.tex'
        if not src.exists():
            sys.exit(f'submission: {src} is missing')
        if used.exists() and not filecmp.cmp(src, used, shallow=False):
            sys.exit(f'submission: {used} differs from {src}; copy it over and rebuild the PDF first')

    if out.exists():
        shutil.rmtree(out)
    out.mkdir()
    flat = INPUT.sub(r'\\input{\1}', GRAPHIC.sub(r'\1\2}', tex))
    if 'figures/' in flat or 'generated/' in flat:
        sys.exit('submission: a figures/ or generated/ path survived flattening')
    (out / f'{paper}.tex').write_text(flat, encoding='utf-8')
    for f in figures:
        shutil.copy2(Path('figures') / f, out / f)
    for t in tables:
        shutil.copy2(TABLES / f'{t}.tex', out / f'{t}.tex')
    shutil.copy2(f'{paper}.bib', out)
    shutil.copy2(kpsewhich('elsarticle.cls'), out)
    shutil.copy2(kpsewhich('elsarticle-num-names.bst'), out)

    # Test-compile a copy, the way EM will: from the flat folder alone.
    build = out / 'build'
    build.mkdir()
    for f in out.iterdir():
        if f.is_file():
            shutil.copy2(f, build)
    run = subprocess.run(
        ['latexmk', '-pdf', '-interaction=nonstopmode', '-halt-on-error', f'{paper}.tex'],
        cwd=build,
        capture_output=True,
        text=True,
    )
    log = (build / f'{paper}.log').read_text(encoding='utf-8', errors='replace')
    if run.returncode != 0:
        sys.exit(f'submission: the flat folder does not compile; see {build}/{paper}.log')
    undefined = len(re.findall(r'undefined', log))
    overfull = len(re.findall(r'Overfull', log))
    pages = re.search(r'Output written on .*?\((\d+) pages', log)
    shutil.copy2(build / f'{paper}.bbl', out)

    zpath = out.with_suffix('.zip')
    with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_DEFLATED) as z:
        for f in sorted(out.iterdir()):
            if f.is_file():
                z.write(f, f.name)
    n = len([f for f in out.iterdir() if f.is_file()])
    print(
        f'submission: {zpath} ({n} files: tex, bib, bbl, cls, bst, {len(tables)} tables, '
        f'{len(figures)} figures); test build {pages.group(1) if pages else "?"} pages, '
        f'{undefined} undefined, {overfull} overfull; its PDF is {build}/{paper}.pdf'
    )
    if undefined or overfull:
        sys.exit(1)


if __name__ == '__main__':
    main()
