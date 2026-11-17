#!/bin/sh
set -eu
repo=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
python3 - "$repo" "$@" <<'PY'
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
allowed_literals = set((root / 'tools/lint/coordinates_allowlist.txt').read_text().splitlines())
allowed_raw = {'src/Omega_h_coords_view.hpp', 'src/Omega_h_mesh.cpp',
               'src/Omega_h_vtk.cpp', 'src/PyOmega_h_mesh.cpp'}
legacy = {'src/Omega_h_mesh.hpp', 'src/Omega_h_mesh.cpp', 'src/unit_embedded.cpp'}
errors = []
backlog = {}
paths = sorted([*root.joinpath('src').glob('*.cpp'), *root.joinpath('src').glob('*.hpp'),
                *root.joinpath('example').rglob('*.cpp'), *root.joinpath('example').rglob('*.hpp')])
for path in paths:
    rel = path.relative_to(root).as_posix()
    text = path.read_text()
    code = re.sub(r'//[^\n]*|/\*.*?\*/', '', text, flags=re.S)
    if rel not in legacy:
        if re.search(r'(?:\.|->)\s*coords\s*\(', code) or re.search(r'&\s*(?:Omega_h::)?Mesh::coords\b', code):
            errors.append(f'L1 {rel}: deprecated coordinate access')
        # A textual check cannot infer argument types. Require explicit typed
        # wrappers at setters; the compiler also enforces deprecation errors.
        for call in re.finditer(r'(?:\.|->)\s*set_coords\s*\(\s*', code):
            if not re.match(r'(?:Omega_h::)?CoordsView\s*<', code[call.end():]):
                errors.append(f'L1 {rel}: setter requires an explicit CoordsView')
    if '"coordinates"' in code and rel not in allowed_literals:
        errors.append(f'L2 {rel}: new direct coordinates-tag access')
    raw_code = code
    if rel == 'src/unit_embedded.cpp':
        raw_code = re.sub(r'OMEGA_H_ALLOW_DEPRECATED_BEGIN.*?OMEGA_H_ALLOW_DEPRECATED_END', '', code, flags=re.S)
    if re.search(r'\.\s*raw\s*\(', raw_code) and rel not in allowed_raw:
        errors.append(f'L3 {rel}: coordinate raw access outside approved files')
    if 'OMEGA_H_ALLOW_DEPRECATED_BEGIN' in code and rel not in ('src/Omega_h_macros.h', 'src/unit_embedded.cpp'):
        errors.append(f'L4 {rel}: deprecation suppression outside compatibility test')
    if rel != 'src/Omega_h_coords_view.hpp':
        calls = re.sub(r'\bReals\s+coords_tdim_only\s*\([^)]*\)\s*\{', '', code)
        count = len(re.findall(r'\bcoords_tdim_only\s*\(', calls))
        if count:
            backlog[rel] = count

count = sum(backlog.values())
baseline = int((root / 'tools/lint/tdim_only_baseline.txt').read_text())
if count > baseline:
    errors.append(f'L5 gated coordinate reads increased: {count} > {baseline}')
for error in errors:
    print(error, file=sys.stderr)
print(f'Gated coordinate reads remaining: {count} (baseline {baseline})')
if '--list' in sys.argv:
    for path, n in backlog.items():
        print(f'{n:3} {path}')
sys.exit(bool(errors))
PY
