#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import click
import tensorflow as tf
import h5py
import cshogi
import numpy as np

from model import ShogiEmbedding, convert_state

__author__ = 'Yasuhiro'
__date__ = '2020/02/04'


def sample_actions(data_list, f):
    index = np.random.choice(len(data_list))
    k1, k2, k3 = data_list[index]
    actions = f[k1][k2][k3]
    return actions


def get_positive_data(data_list, f, board):
    actions = sample_actions(data_list=data_list, f=f)
    n = len(actions)

    # 系列の中から異なる2個を選ぶ
    p = np.random.choice(n, size=2, replace=False)
    p_max = np.max(p)

    anchor, positive = None, None
    board.reset()
    for i, action in enumerate(actions[:p_max + 1]):
        if i == p[0]:
            anchor = convert_state(board=board)
        elif i == p[1]:
            positive = convert_state(board=board)

        board.push(action)
    if anchor is None:
        anchor = convert_state(board=board)
    elif positive is None:
        positive = convert_state(board=board)

    return anchor, positive


def get_negative_data(data_list, f, board, n_data):
    negative_list = []
    for i in range(n_data):
        actions = sample_actions(data_list=data_list, f=f)
        size = len(actions)

        n = np.random.choice(size)
        board.reset()
        for action in actions[:n]:
            board.push(action)
        negative = convert_state(board=board)
        negative_list.append(negative)

    return negative_list


def make_dataset(path, batch_size, n_negative_data):
    def generator():
        data_list = []
        with h5py.File(path, 'r') as f:
            for key1, value1 in f.items():
                for key2, value2 in value1.items():
                    for key3 in value2.keys():
                        data_list.append((key1, key2, key3))

            board = cshogi.Board()
            while True:
                # positive dataの系列
                anchor, positive = get_positive_data(data_list=data_list,
                                                     f=f, board=board)
                # negativeの配列
                negative_list = get_negative_data(
                    data_list=data_list, f=f, board=board,
                    n_data=n_negative_data
                )

                yield anchor, positive, negative_list

    dataset = tf.data.Dataset.from_generator(
        generator=generator,
        output_types=(tf.int32, tf.int32, tf.int32),
        output_shapes=(tf.TensorShape([41]), tf.TensorShape([41]),
                       tf.TensorShape([n_negative_data, 41]))
    ).batch(batch_size=batch_size)

    return dataset


@click.command()
def cmd():
    def generator():
        while True:
            yield 5, 10, [11, 12, 13]

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_shapes=(tf.TensorShape([]), tf.TensorShape([]),
                       tf.TensorShape([3])),
        output_types=(tf.int32, tf.int32, tf.int32)
    ).batch(3)
    for a, b, c in dataset:
        print(a + 1)
        print(a, b, c)
        break


def main():
    cmd()


if __name__ == '__main__':
    main()
