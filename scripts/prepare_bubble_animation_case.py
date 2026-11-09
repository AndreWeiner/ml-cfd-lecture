"""Copy the tested source case and apply the recorded animation mesh/settings."""
from pathlib import Path
import argparse
import json
import re
import shutil

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--models", type=Path, required=True)
parser.add_argument("--case", type=Path, required=True)
parser.add_argument("--build", type=Path, help="Reuse an already validated example build directory")
args = parser.parse_args()
config = json.loads((root/'experiments/bubble_transport/animation-case.json').read_text())
if args.case.exists():
    raise FileExistsError("Use a fresh output directory")
shutil.copytree(root/'test_cases/single_phase_approach', args.case,
                ignore=shutil.ignore_patterns('build', 'linux*', 'lnInclude', '*.so', '.vscode'))
# Reuse the binaries already compiled and tested inside the selected container.
if args.build:
    (args.case/'build').symlink_to(args.build.resolve(), target_is_directory=True)
sim = args.case/'simulation'
for name in ('ub', 'r', 'ut'):
    (sim/f'{name}.pt2').symlink_to((args.models/f'{name}.pt2').resolve())
p = sim/'system/blockMeshDict'; text = p.read_text()
for key, value in config['mesh'].items():
    text = re.sub(rf'({key}\s+)[\d.]+;',rf'\g<1>{value};',text)
p.write_text(text)
p = sim/'system/controlDict'; text = p.read_text()
for key, value in {'endTime': config['pilot_end'], 'writeInterval': config['write_interval'],
                   'maxDeltaT': config['max_delta_t'], 'writeFormat': 'ascii',
                   'timePrecision': 10}.items():
    text = re.sub(rf'({key}\s+)[^;]+;',rf'\g<1>{value};',text)
# Inspect scalar overshoots directly; do not clip the fields to make prettier movies.
text = text.replace('#include "limitFields"','// No scalar clipping in this visualization run.')
p.write_text(text)
shutil.copytree(sim/'0.org', sim/'0')
(sim/'post.foam').touch()
(args.case/'animation-case.json').write_text(json.dumps(config, indent=2)+'\n')
print(sim)
