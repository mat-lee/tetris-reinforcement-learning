from const import *

import torch
from torch import nn
from dataclasses import dataclass


@dataclass
class AlphaSameConfig:
    blocks: int = 10
    pooling_blocks: int = 2   # number of GlobalPooling blocks (Keras only)
    filters: int = 16
    cpool: int = 4            # channels reserved for global pooling (Keras only)
    dropout: float = 0.25
    kernels: int = 1
    o_side_neurons: int = 16
    value_head_neurons: int = 16


# Derived sizes — named so the architecture code reads as intent, not arithmetic
_PIECES_PER_PLAYER = (2 + PREVIEWS) * len(MINOS)  # one-hot piece encoding size per player
_SCALARS_PER_PLAYER = 3                             # b2b, combo, garbage


# ── PyTorch ──────────────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    def __init__(self, model_config: AlphaSameConfig):
        super().__init__()

        self.conv_block1 = nn.Sequential(
            nn.BatchNorm2d(num_features=model_config.filters),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=model_config.filters,
                out_channels=model_config.filters,
                kernel_size=3,
                padding='same',
                bias=False,
            ),
        )

        self.conv_block2 = nn.Sequential(
            nn.BatchNorm2d(num_features=model_config.filters),
            nn.Dropout(p=model_config.dropout),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=model_config.filters,
                out_channels=model_config.filters,
                kernel_size=3,
                padding='same',
                bias=False,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_block2(self.conv_block1(x)) + x


class AlphaSame(nn.Module):
    def __init__(self, model_config: AlphaSameConfig, use_tanh: bool = False):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=model_config.filters,
            kernel_size=5,
            padding='same',
            bias=False,
        )

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(model_config) for _ in range(model_config.blocks)]
        )

        self.batchnorm1 = nn.BatchNorm2d(num_features=model_config.filters)
        self.relu1 = nn.ReLU()

        self.kernel1 = nn.Conv2d(
            in_channels=model_config.filters,
            out_channels=model_config.kernels,
            kernel_size=1,
            padding='same',
            bias=False,
        )

        self.batchnorm2 = nn.BatchNorm2d(num_features=model_config.kernels)
        self.relu2 = nn.ReLU()

        self.flatten1 = nn.Flatten()

        grid_out_size = ROWS * COLS * model_config.kernels

        self.osidedense = nn.Sequential(
            nn.Linear(grid_out_size, model_config.o_side_neurons),
            nn.BatchNorm1d(num_features=model_config.o_side_neurons),
            nn.ReLU(),
        )

        head_inputs = (
            grid_out_size                   # active player grid (after 1x1 kernel + flatten)
            + model_config.o_side_neurons   # opponent grid (compressed)
            + 2 * _PIECES_PER_PLAYER        # both players' piece queues
            + 2 * _SCALARS_PER_PLAYER       # both players' b2b, combo, garbage
            + 1                             # color (who has first move)
        )

        self.policy_head = nn.Linear(head_inputs, POLICY_SIZE)

        self.value_head = nn.Sequential(
            nn.Linear(head_inputs, model_config.value_head_neurons),
            nn.BatchNorm1d(num_features=model_config.value_head_neurons),
            nn.ReLU(),
            nn.Linear(model_config.value_head_neurons, 1),
            nn.Dropout(model_config.dropout),
            nn.Tanh() if use_tanh else nn.Sigmoid(),
        )

    def forward(self, a_grid, a_pieces, a_b2b, a_combo, a_garbage,
                o_grid, o_pieces, o_b2b, o_combo, o_garbage, color):
        def process_grid(grid):
            x = self.conv1(grid)
            x = self.res_blocks(x)
            x = self.relu1(self.batchnorm1(x))
            x = self.kernel1(x)
            x = self.relu2(self.batchnorm2(x))
            return self.flatten1(x)

        a_grid_out = process_grid(a_grid)
        o_grid_out = self.osidedense(process_grid(o_grid))

        x = torch.cat([
            a_grid_out,
            self.flatten1(a_pieces),
            a_b2b.unsqueeze(1), a_combo.unsqueeze(1), a_garbage.unsqueeze(1),
            o_grid_out,
            self.flatten1(o_pieces),
            o_b2b.unsqueeze(1), o_combo.unsqueeze(1), o_garbage.unsqueeze(1),
            color.unsqueeze(1),
        ], dim=1)

        return self.value_head(x), self.policy_head(x)


# ── Keras ─────────────────────────────────────────────────────────────────────

from tensorflow import keras
import tensorflow as tf


def _create_input_layers():
    piece_shape = (2 + PREVIEWS, len(MINOS))
    grid_shape = (ROWS, COLS, 1)
    scalar_shape = (1,)

    shapes = [
        grid_shape, piece_shape, scalar_shape, scalar_shape, scalar_shape,  # active player
        grid_shape, piece_shape, scalar_shape, scalar_shape, scalar_shape,  # opponent
        scalar_shape,                                                         # color
    ]

    inputs, active_grid, active_features = [], None, []
    opponent_grid, opponent_features, non_player_features = None, [], []

    n_player_inputs = (len(shapes) - 1) // 2

    for i, shape in enumerate(shapes):
        inp = keras.Input(shape=shape, name=f"{i}")
        inputs.append(inp)

        if i < n_player_inputs:
            if shape == grid_shape:
                active_grid = inp
            else:
                active_features.append(keras.layers.Flatten()(inp))
        elif i < len(shapes) - 1:
            if shape == grid_shape:
                opponent_grid = inp
            else:
                opponent_features.append(keras.layers.Flatten()(inp))
        else:
            non_player_features.append(keras.layers.Flatten()(inp))

    return inputs, active_grid, active_features, opponent_grid, opponent_features, non_player_features


def gen_alphasame_nn(model_config: AlphaSameConfig, use_tanh: bool = False) -> keras.Model:
    f = model_config.filters

    def value_head(x):
        x = keras.layers.Dense(model_config.value_head_neurons)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.Dropout(model_config.dropout)(x)
        x = keras.layers.Dense(1, activation='tanh' if use_tanh else 'sigmoid')(x)
        return x

    def policy_head(x):
        return keras.layers.Dense(POLICY_SIZE, activation='softmax')(x)

    def residual_layer():
        def apply(in_1, in_2):
            bn1   = keras.layers.BatchNormalization()
            act1  = keras.layers.Activation('relu')
            conv1 = keras.layers.Conv2D(f, (3, 3), padding='same')
            bn2   = keras.layers.BatchNormalization()
            drop  = keras.layers.Dropout(model_config.dropout)
            act2  = keras.layers.Activation('relu')
            conv2 = keras.layers.Conv2D(f, (3, 3), padding='same')

            out_1 = conv2(act2(drop(bn2(conv1(act1(bn1(in_1)))))))
            out_2 = conv2(act2(drop(bn2(conv1(act1(bn1(in_2)))))))

            return keras.layers.Add()([in_1, out_1]), keras.layers.Add()([in_2, out_2])
        return apply

    def global_pooling_layer():
        def apply(in_1, in_2):
            bn1   = keras.layers.BatchNormalization()
            act1  = keras.layers.Activation('relu')
            conv1 = keras.layers.Conv2D(f, (3, 3), padding='same')

            out_1 = conv1(act1(bn1(in_1)))
            out_2 = conv1(act1(bn1(in_2)))

            cp = model_config.cpool
            rest = f - cp

            def pool_and_bias(out, in_orig):
                pool_ch = out[:, :, :, :cp]
                rest_ch = out[:, :, :, cp:]

                pool_act = keras.layers.Activation('relu')(keras.layers.BatchNormalization()(pool_ch))
                avg = keras.layers.GlobalAveragePooling2D()(pool_act)
                mx  = keras.layers.GlobalMaxPooling2D()(pool_ch)
                pooled = keras.layers.Concatenate()([avg, mx])

                bias = keras.layers.Reshape((1, 1, rest))(keras.layers.Dense(rest)(pooled))
                biased = keras.layers.Add()([rest_ch, bias])
                combined = keras.layers.Concatenate(axis=-1)([pool_ch, biased])

                bn2   = keras.layers.BatchNormalization()
                drop  = keras.layers.Dropout(model_config.dropout)
                act2  = keras.layers.Activation('relu')
                conv2 = keras.layers.Conv2D(f, (3, 3), padding='same')
                out_final = conv2(act2(drop(bn2(combined))))
                return keras.layers.Add()([in_orig, out_final])

            return pool_and_bias(out_1, in_1), pool_and_bias(out_2, in_2)
        return apply

    inputs, a_grid, a_features, o_grid, o_features, non_player = _create_input_layers()

    conv1 = keras.layers.Conv2D(f, (5, 5), padding='same')
    a_grid = conv1(a_grid)
    o_grid = conv1(o_grid)

    # Inject piece features as channel biases before the residual trunk
    dense_feat = keras.layers.Dense(f)
    a_feat_vec = dense_feat(keras.layers.Concatenate()(a_features))
    o_feat_vec = dense_feat(keras.layers.Concatenate()(o_features))
    a_grid = keras.layers.Add()([a_grid, keras.layers.Reshape((1, 1, f))(a_feat_vec)])
    o_grid = keras.layers.Add()([o_grid, keras.layers.Reshape((1, 1, f))(o_feat_vec)])

    pool_indices = {
        round((i + 1) / (model_config.pooling_blocks + 1) * model_config.blocks) - 1
        for i in range(model_config.pooling_blocks)
    }
    for i in range(model_config.blocks):
        layer = global_pooling_layer() if i in pool_indices else residual_layer()
        a_grid, o_grid = layer(a_grid, o_grid)

    bn_final = keras.layers.BatchNormalization()
    act_final = keras.layers.Activation('relu')
    a_grid = act_final(bn_final(a_grid))
    o_grid = act_final(bn_final(o_grid))

    kernel = keras.layers.Conv2D(model_config.kernels, (1, 1))
    a_grid = kernel(a_grid)
    o_grid = kernel(o_grid)

    flatten = keras.layers.Flatten()
    bn2 = keras.layers.BatchNormalization()
    act2 = keras.layers.Activation('relu')
    a_grid = act2(bn2(flatten(a_grid)))
    o_grid = act2(bn2(flatten(o_grid)))

    o_grid = keras.layers.Activation('relu')(
        keras.layers.BatchNormalization()(
            keras.layers.Dense(model_config.o_side_neurons)(o_grid)
        )
    )

    x = keras.layers.Concatenate()([a_grid, o_grid, *non_player])

    return keras.Model(inputs=inputs, outputs=[value_head(x), policy_head(x)])