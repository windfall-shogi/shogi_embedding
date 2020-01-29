#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import re
from pathlib import Path

import cshogi
import numpy as np
import h5py
import click
from tqdm import tqdm

__author__ = 'Yasuhiro'
__date__ = '2020/01/27'


@click.command()
@click.option('--data-dir', type=click.Path(exists=True))
@click.option('--output-path', type=click.Path())
def cmd(data_dir, output_path):
    data_dir = Path(data_dir)
    p = data_dir.glob('wdoor+floodgate-*.csa')

    year = data_dir.stem

    output_path = Path(output_path)
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True)

    r = re.compile(r'wdoor\+floodgate-(\d+-\d+F?)\+(.+)\.csa')
    with h5py.File(str(output_path), 'a') as f:
        group = f.create_group('/{}'.format(year))

        parser = cshogi.Parser()
        for path in tqdm(p):
            m = r.search(str(path))
            if m is None:
                print(path)

            parser.parse_csa_file(str(path))
            group.create_dataset(
                '/{}/{}/{}'.format(year, m.group(1), m.group(2)),
                shape=(len(parser.moves),), dtype=np.int32, data=parser.moves
            )


def main():
    cmd()


if __name__ == '__main__':
    main()
