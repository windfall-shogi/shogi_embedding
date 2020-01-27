#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Poincare Embedding
https://arxiv.org/pdf/1705.08039.pdf

上記論文を参考に将棋の局面の低次元空間への埋め込みを学習する
その埋め込みによって戦型や進行度の判定を狙う

word2vecではなく、doc2vecになるので、gensimなどのライブラリは利用できない
また、論文の式(5)のprojectionで問題となる可能性がある


"""

import tensorflow as tf

__author__ = 'Yasuhiro'
__date__ = '2020/01/26'

# 学習が難しいと思うので、
# fは先手、eは後手で固定
# 範囲を拡張して王も他の駒と同様変換する
BONA_PIECE_ZERO = 0
BonaPiece = [
    BONA_PIECE_ZERO + 1,  # //0//0+1
    20,  # //f_hand_pawn + 19,//19+1
    39,  # //e_hand_pawn + 19,//38+1
    44,  # //f_hand_lance + 5,//43+1
    49,  # //e_hand_lance + 5,//48+1
    54,  # //f_hand_knight + 5,//53+1
    59,  # //e_hand_knight + 5,//58+1
    64,  # //f_hand_silver + 5,//63+1
    69,  # //e_hand_silver + 5,//68+1
    74,  # //f_hand_gold + 5,//73+1
    79,  # //e_hand_gold + 5,//78+1
    82,  # //f_hand_bishop + 3,//81+1
    85,  # //e_hand_bishop + 3,//84+1
    88,  # //f_hand_rook + 3,//87+1
    90,  # //e_hand_rook + 3,//90
    # f_pawn = fe_hand_end,
    90 + 81,  # e_pawn = f_pawn + 81,
    90 + 81 * 2,  # f_lance = e_pawn + 81,
    90 + 81 * 3,  # e_lance = f_lance + 81,
    90 + 81 * 4,  # f_knight = e_lance + 81,
    90 + 81 * 5,  # e_knight = f_knight + 81,
    90 + 81 * 6,  # f_silver = e_knight + 81,
    90 + 81 * 7,  # e_silver = f_silver + 81,
    90 + 81 * 8,  # f_gold = e_silver + 81,
    90 + 81 * 9,  # e_gold = f_gold + 81,
    90 + 81 * 10,  # f_bishop = e_gold + 81,
    90 + 81 * 11,  # e_bishop = f_bishop + 81,
    90 + 81 * 12,  # f_horse = e_bishop + 81,
    90 + 81 * 13,  # e_horse = f_horse + 81,
    90 + 81 * 14,  # f_rook = e_horse + 81,
    90 + 81 * 15,  # e_rook = f_rook + 81,
    90 + 81 * 16,  # f_dragon = e_rook + 81,
    90 + 81 * 17,  # e_dragon = f_dragon + 81,
    90 + 81 * 18,  # f_king = e_dragon + 81,
    90 + 81 * 19,  # e_king = f_king + 81,
    90 + 81 * 20,  # fe_old_end = e_king + 81,
]
# None, pawn, lance, knight, silver, bishop, rook, gold, king,
# gold, gold, gold, gold, horse, dragon
PIECE_OFFSET = [0, 2, 4, 6, 8, 12, 16, 10, 20,
                10, 10, 10, 10, 14, 18]


def convert_state(board):
    """
    cshogi.BoardをBona Pieceの配列に変換する

    1駒関係
    王も他の駒と同様に場所と駒の種類で一意にインデックスを与える
    :param board:
    :return:
    """
    values = [BonaPiece[-1] + board.turn]   # 手番

    for square in range(81):
        piece = board.piece(square)
        if piece == 0:
            continue

        piece_type = piece & 0
        piece_color = piece >> 4
        index = PIECE_OFFSET[piece_type] + 12 + piece_color
        values.append(index + square)

    pieces_in_hand = board.pieces_in_hand
    for color in range(2):
        for piece_type in range(7):
            hand = BonaPiece[color + piece_type * 2]

            piece_count = pieces_in_hand[color][piece_type]
            for i in range(piece_count):
                values.append(hand + i)

    return values


class ShogiEmbedding(tf.keras.Model):
    def __init__(self, output_dim):
        super().__init__()

        # bona piece + 先手と後手それぞれの手番
        self.embedding = tf.keras.layers.Embedding(
            input_dim=BonaPiece[-1] + 2, output_dim=output_dim,
            embeddings_initializer=tf.keras.initializers.uniform(
                minval=-0.001, maxval=0.001
            )
        )

    def call(self, inputs, training=None, mask=None):
        h = self.embedding(inputs)
        outputs = tf.reduce_sum(h, axis=1)
        return outputs
