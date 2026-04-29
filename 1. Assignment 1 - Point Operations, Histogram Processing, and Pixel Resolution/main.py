from section4 import run_section4
from section3 import run_section3
from section2 import run_section2
from section1_fast import run_section1
from section0 import run_section0
import argparse
import os
import sys
import time

# ── Ensure the working directory is the script's own directory ──────────────
# This makes all relative image paths (e.g. 'Images/Section 1/cameraman.tif')
# resolve correctly regardless of where python is invoked from.
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def _timed(label: str, fn):
    t0 = time.time()
    fn()
    elapsed = time.time() - t0
    print(f'\n {label} completed in {elapsed:.1f} s\n')


def main():
    parser = argparse.ArgumentParser(
        description='DIP Homework 1 - run all sections')
    parser.add_argument('--section', type=int, default=None,
                        choices=[0, 1, 2, 3, 4],
                        help='Run only this section (default: run all)')
    args = parser.parse_args()

    sections = {
        0: ('Section 0 - Diamond Pattern Embedding', run_section0),
        1: ('Section 1 - Pixel Resolution & Interpolation',  run_section1),
        2: ('Section 2 - Point Operations & Bit-Plane Slicing', run_section2),
        3: ('Section 3 - Histogram Equalization (GHE + LHE)', run_section3),
        4: ('Section 4 - Histogram Matching & Color Spaces',  run_section4),
    }

    if args.section is not None:
        label, fn = sections[args.section]
        _timed(label, fn)
    else:
        print('=' * 65)
        print('  DIP Homework 1 - Running all sections')
        print('=' * 65)
        t_total = time.time()
        for sec_id, (label, fn) in sections.items():
            _timed(label, fn)
        total = time.time() - t_total
        print('=' * 65)
        print(f'  All sections done.  Total time: {total:.1f} s')
        print('  Outputs saved under  output/section{{0..4}}/')
        print('=' * 65)


if __name__ == '__main__':
    main()
