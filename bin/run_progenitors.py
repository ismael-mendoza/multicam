#!/usr/bin/env python3
from pathlib import Path
import shutil
import argparse
import warnings

from relaxed.frames.catalogs import catalog_properties
from relaxed.progenitors.save_progenitors import write_main_line_progenitors, merge_progenitors


def setup_paths(args):
    tree_path = Path(args.tree_path)

    paths = {
        'trees': tree_path,
        'progenitors': tree_path.joinpath("progenitors")
    }

    return paths


def write(args, paths):
    assert args.cpu is not None, "Need to specify cpus"
    # Bolshoi
    Mcut = 1e3 * catalog_properties['Bolshoi'][0]

    tree_path = Path(paths['trees'])
    progenitor_path = tree_path.joinpath("progenitors")
    if progenitor_path.exists() and args.overwrite:
        warnings.warn("Overwriting current progenitor directory")
        shutil.rmtree(progenitor_path)

    progenitor_path.mkdir(exist_ok=False)

    write_main_line_progenitors(tree_path, progenitor_path.joinpath("mline"), Mcut, cpus=args.cpus)


def merge(args, paths):
    merge_progenitors(paths['trees'], paths['progenitors'])


def summarize(args, paths):
    pass


def main(args):
    paths = setup_paths(args)
    if args.write:
        write(args, paths)

    elif args.merge:
        merge(args, paths)

    elif args.summarize:
        summarize(args, paths)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Write main line progenitors from tree files')
    parser.add_argument('--cpus', type=int, default=None)
    parser.add_argument('--write', action='store_true')
    parser.add_argument('--merge', action='store_true')
    parser.add_argument('--overwrite', action='store_true')

    parser.add_argument('--tree-path', type=str,
                        default="/home/imendoza/alcca/nbody-relaxed/data/trees_bolshoi")

    pargs = parser.parse_args()

    main(pargs)
