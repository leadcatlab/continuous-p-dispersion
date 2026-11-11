#!/usr/bin/env python3
"""Refill the prose paragraphs of a LaTeX file to an even width, in place.

tex-fmt (run first by `make fmt`) indents and breaks lines that are too
long, but never joins short ones, so edited paragraphs end up ragged. This
joins each run of prose lines and refills it greedily to WIDTH columns.

Only whitespace changes, and only inside plain prose: lines in the
environments of PROSE_ENVS, outside math, tables, figures and algorithms.
A line is left exactly as it is, and ends the run it would have joined, if
it is blank, holds a comment (%), ends in a forced break (\\\\), starts a
structural command (\\begin, \\section, \\label, \\item, \\caption, ...) or
opens or closes display math. A line break and a space are the same token
to TeX, so the output typesets identically.

    python texfill.py paper.tex [--width 80]
"""

import argparse
import re

# Judged by the innermost open environment, so the title block (frontmatter)
# is left alone but the abstract inside it is filled.
PROSE_ENVS = {'document', 'abstract', 'lemma', 'proof'}

STRUCTURAL = re.compile(
    r'^\s*(?:\\(?:begin|end|section|subsection|subsubsection|label|item|'
    r'caption|centering|includegraphics|input|include|bibliography|'
    r'bibliographystyle|maketitle|documentclass|usepackage|newcommand|'
    r'renewcommand|def|hline|title|author|affiliation|cortext|ead|'
    r'State|If|ElsIf|Else|EndIf|For|EndFor|While|EndWhile|Function|'
    r'EndFunction|Procedure|EndProcedure|Return|Require|Ensure|Comment)\b'
    r'|\\\[|\\\]|\$\$)'
)
COMMENT = re.compile(r'(?<!\\)%')
ENV = re.compile(r'\\(begin|end)\{([^}]*)\}')


def is_prose(line: str, stack: list[str]) -> bool:
    s = line.strip()
    return bool(
        s
        and stack
        and stack[-1] in PROSE_ENVS
        and not STRUCTURAL.match(line)
        and not COMMENT.search(line)
        and not ENV.search(line)
        and '\\[' not in line
        and '\\]' not in line
        and not s.endswith('\\\\')
    )


BRACKET = re.compile(r'(?<!\\)[(\[{]|(?<!\\)[)\]}]')


def depth_change(word: str) -> int:
    return sum(1 if b in '([{' else -1 for b in BRACKET.findall(word))


def fill(lines: list[str], width: int, tabsize: int = 2) -> list[str]:
    """Greedy fill, indenting each line the way tex-fmt will.

    tex-fmt indents a line by tabsize for each bracket ( [ { still open when
    it starts, less any it closes before opening another. A line broken
    inside a bracket is therefore shifted right, and if that pushes it past
    the width tex-fmt breaks it again, leaving a ragged stub. Predicting the
    indentation here, and filling each line to the width it will really
    have, makes the output a fixed point of tex-fmt.
    """
    base = len(re.match(r'\s*', lines[0]).group(0))
    words = ' '.join(lines).split()
    out: list[str] = []

    def render(ws: list[str], start: int) -> str:
        low = run = 0
        for w in ws:
            for b in BRACKET.findall(w):
                run += 1 if b in '([{' else -1
                low = min(low, run)
        return ' ' * max(0, base + tabsize * (start + low)) + ' '.join(ws)

    line: list[str] = []
    start = 0  # bracket depth, relative to the run, where `line` begins
    for word in words:
        if line and len(render(line + [word], start)) > width:
            out.append(render(line, start))
            start += sum(depth_change(w) for w in line)
            line = []
        line.append(word)
    if line:
        out.append(render(line, start))
    return out


def refill(text: str, width: int) -> str:
    stack: list[str] = []
    out: list[str] = []
    run: list[str] = []
    for line in text.split('\n'):
        if is_prose(line, stack):
            run.append(line)
            continue
        if run:
            out += fill(run, width)
            run = []
        out.append(line)
        for kind, env in ENV.findall(COMMENT.split(line)[0]):
            if kind == 'begin':
                stack.append(env)
            elif stack and stack[-1] == env:
                stack.pop()
    if run:
        out += fill(run, width)
    return '\n'.join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('file')
    ap.add_argument('--width', type=int, default=80)
    args = ap.parse_args()
    with open(args.file, encoding='utf-8') as f:
        text = f.read()
    new = refill(text, args.width)
    if new != text:
        with open(args.file, 'w', encoding='utf-8') as f:
            f.write(new)


if __name__ == '__main__':
    main()
