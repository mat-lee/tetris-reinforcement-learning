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


# ── BaseResNet (PyTorch) ──────────────────────────────────────────────────────
# Simple post-activation ResNet trunk shared across both boards. No 1x1
# bottleneck, no separate opponent compression, no dropout. Designed to pair
# with AdamW + weight_decay rather than dropout for regularization.

@dataclass
class BaseResNetConfig:
    blocks: int = 10
    filters: int = 16
    opp_hidden: int = 16   # width of the opponent encoding before it is combined with own extras
    own_kernels: int = 1    # channels after the own-side 1x1 collapse before flatten -> heads
    value_hidden: int = 16  # width of the value-head hidden layer
    # 'original'                     -> flat Linear(POLICY_SIZE), raw logits (existing behavior).
    # 'spatial_list'                 -> 1x1 conv on FiLM-biased map -> (B, ROWS, COLS, 27) logits.
    # 'heatmap_AM' / 'heatmap_FFT'   -> same spatial 1x1 conv head, used as a heatmap (linear output).
    policy_format: str = 'original'


class _BaseResBlock(nn.Module):
    def __init__(self, filters: int):
        super().__init__()
        self.conv1 = nn.Conv2d(filters, filters, kernel_size=3, padding='same', bias=False)
        self.bn1 = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, padding='same', bias=False)
        self.bn2 = nn.BatchNorm2d(filters)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + x)


class BaseResNet(nn.Module):
    """
    Shared ResNet trunk runs on both boards. The opponent feature map is
    collapsed with a 1x1 conv and combined with opp extras into an opp_repr;
    opp_repr is then combined with own extras + color and projected to a
    (filters,) bias vector. That bias is added channel-wise (broadcast over
    spatial) to the own feature map — FiLM-add conditioning. The biased map is
    collapsed, flattened, and concatenated with own pieces/scalars/color so the
    heads see them as direct features (FiLM-add stays as an additional
    conditioning signal but is no longer the only path for own-side info).
    Value head has a hidden Linear+BN+ReLU layer so it can learn nonlinear
    evaluations.
    """

    def __init__(self, model_config: BaseResNetConfig, use_tanh: bool = False):
        super().__init__()
        f = model_config.filters

        self.stem = nn.Sequential(
            nn.Conv2d(1, f, kernel_size=3, padding='same', bias=False),
            nn.BatchNorm2d(f),
            nn.ReLU(),
        )
        self.trunk = nn.Sequential(*[_BaseResBlock(f) for _ in range(model_config.blocks)])

        # Opponent collapse: f channels → 1
        self.opp_collapse = nn.Sequential(
            nn.Conv2d(f, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
        )

        # Encode opponent: flattened collapsed grid + opp extras → opp_repr
        opp_in = ROWS * COLS + _PIECES_PER_PLAYER + _SCALARS_PER_PLAYER
        self.opp_encode = nn.Sequential(
            nn.Linear(opp_in, model_config.opp_hidden),
            nn.BatchNorm1d(model_config.opp_hidden),
            nn.ReLU(),
        )

        # Project (opp_repr + own extras + color) → channel-wise bias for own feature map
        bias_in = model_config.opp_hidden + _PIECES_PER_PLAYER + _SCALARS_PER_PLAYER + 1
        self.bias_project = nn.Linear(bias_in, f)

        # Own collapse: 1x1 conv shrinks channels to own_kernels (spatial dims preserved)
        self.own_collapse = nn.Sequential(
            nn.Conv2d(f, model_config.own_kernels, kernel_size=1, bias=False),
            nn.BatchNorm2d(model_config.own_kernels),
            nn.ReLU(),
        )

        # Heads see flattened collapsed map + own pieces + own scalars + color directly
        head_in = (
            model_config.own_kernels * ROWS * COLS
            + _PIECES_PER_PLAYER
            + _SCALARS_PER_PLAYER
            + 1
        )
        self.policy_format = model_config.policy_format
        if self.policy_format == 'original':
            self.policy_head = nn.Linear(head_in, POLICY_SIZE)
        else:
            # 1x1 conv on the FiLM-biased trunk output -> (B, 27, ROWS, COLS).
            # Scalar/piece context already enters via the FiLM bias; no extra tiling needed.
            self.policy_head = nn.Conv2d(f, POLICY_SHAPE[0], kernel_size=1)
        self.value_head = nn.Sequential(
            nn.Linear(head_in, model_config.value_hidden),
            nn.BatchNorm1d(model_config.value_hidden),
            nn.ReLU(),
            nn.Linear(model_config.value_hidden, 1),
            nn.Tanh() if use_tanh else nn.Sigmoid(),
        )

    def _process_grid(self, grid):
        return self.trunk(self.stem(grid))

    def forward(self, a_grid, a_pieces, a_b2b, a_combo, a_garbage,
                o_grid, o_pieces, o_b2b, o_combo, o_garbage, color):
        a_feat = self._process_grid(a_grid)                   # (B, f, ROWS, COLS)
        o_feat = self._process_grid(o_grid)                   # (B, f, ROWS, COLS)

        o_collapsed = self.opp_collapse(o_feat).flatten(1)    # (B, ROWS*COLS)
        opp_repr = self.opp_encode(torch.cat([
            o_collapsed,
            o_pieces.flatten(1),
            o_b2b.unsqueeze(1), o_combo.unsqueeze(1), o_garbage.unsqueeze(1),
        ], dim=1))                                            # (B, opp_hidden)

        bias_vec = self.bias_project(torch.cat([
            opp_repr,
            a_pieces.flatten(1),
            a_b2b.unsqueeze(1), a_combo.unsqueeze(1), a_garbage.unsqueeze(1),
            color.unsqueeze(1),
        ], dim=1))                                            # (B, f)

        biased = a_feat + bias_vec.view(-1, bias_vec.size(1), 1, 1)
        flat = self.own_collapse(biased).flatten(1)

        head_x = torch.cat([
            flat,
            a_pieces.flatten(1),
            a_b2b.unsqueeze(1), a_combo.unsqueeze(1), a_garbage.unsqueeze(1),
            color.unsqueeze(1),
        ], dim=1)
        if self.policy_format == 'original':
            policy = self.policy_head(head_x)
        else:
            policy = self.policy_head(biased).permute(0, 2, 3, 1).contiguous()
        return self.value_head(head_x), policy


# ── AuxBaseResNet (PyTorch) ───────────────────────────────────────────────────
# BaseResNet trunk + 2-output sigmoid aux head predicting normalized holes and
# aggregate height of the active player's board. Aux targets are computed
# on-the-fly from the own grid; see compute_aux_targets below.

@dataclass
class AuxBaseResNetConfig(BaseResNetConfig):
    aux_hidden: int = 16
    aux_weight: float = 1.5


def compute_aux_targets(own_grid_batch: torch.Tensor) -> torch.Tensor:
    """own_grid_batch: (B, 1, ROWS, COLS) float in {0, 1}.
    Returns (B, 2) tensor [holes_norm, height_norm] in [0, 1].

    holes_norm  = (# zero cells with any 1 above in same column) / (ROWS * COLS)
    height_norm = sum_c (ROWS - top_filled_row(c)) / (ROWS * COLS); 0 for empty cols.
    """
    g = own_grid_batch.squeeze(1)
    filled = (g > 0.5).float()

    # cumsum > 0 stands in for cummax (which lacks an MPS kernel)
    has_filled_above = (torch.cumsum(filled, dim=1) > 0).float()
    holes = ((1.0 - filled) * has_filled_above).sum(dim=(1, 2))

    any_filled = filled.sum(dim=1) > 0
    top_idx = filled.argmax(dim=1)
    heights = torch.where(any_filled, (ROWS - top_idx).float(), torch.zeros_like(top_idx, dtype=torch.float))
    height_sum = heights.sum(dim=1)

    denom = float(ROWS * COLS)
    return torch.stack([holes / denom, height_sum / denom], dim=1)


class AuxBaseResNet(BaseResNet):
    def __init__(self, model_config: AuxBaseResNetConfig, use_tanh: bool = False):
        super().__init__(model_config, use_tanh=use_tanh)
        head_in = (
            model_config.own_kernels * ROWS * COLS
            + _PIECES_PER_PLAYER
            + _SCALARS_PER_PLAYER
            + 1
        )
        self.aux_head = nn.Sequential(
            nn.Linear(head_in, model_config.aux_hidden),
            nn.BatchNorm1d(model_config.aux_hidden),
            nn.ReLU(),
            nn.Linear(model_config.aux_hidden, 2),
            nn.Sigmoid(),
        )

    def forward(self, a_grid, a_pieces, a_b2b, a_combo, a_garbage,
                o_grid, o_pieces, o_b2b, o_combo, o_garbage, color):
        a_feat = self._process_grid(a_grid)
        o_feat = self._process_grid(o_grid)

        o_collapsed = self.opp_collapse(o_feat).flatten(1)
        opp_repr = self.opp_encode(torch.cat([
            o_collapsed,
            o_pieces.flatten(1),
            o_b2b.unsqueeze(1), o_combo.unsqueeze(1), o_garbage.unsqueeze(1),
        ], dim=1))

        bias_vec = self.bias_project(torch.cat([
            opp_repr,
            a_pieces.flatten(1),
            a_b2b.unsqueeze(1), a_combo.unsqueeze(1), a_garbage.unsqueeze(1),
            color.unsqueeze(1),
        ], dim=1))

        biased = a_feat + bias_vec.view(-1, bias_vec.size(1), 1, 1)
        flat = self.own_collapse(biased).flatten(1)

        head_x = torch.cat([
            flat,
            a_pieces.flatten(1),
            a_b2b.unsqueeze(1), a_combo.unsqueeze(1), a_garbage.unsqueeze(1),
            color.unsqueeze(1),
        ], dim=1)
        if self.policy_format == 'original':
            policy = self.policy_head(head_x)
        else:
            policy = self.policy_head(biased).permute(0, 2, 3, 1).contiguous()
        return self.value_head(head_x), policy, self.aux_head(head_x)


# ── AuxBaseResNetV2 (PyTorch) ─────────────────────────────────────────────────
# Symmetric pre-trunk channel-bias conditioning + joint value/policy bottleneck.
# Per player: project (pieces ⊕ scalars ⊕ color/1-color) → (B, f) bias, add to
# 5x5 stem output, run shared residual trunk. Both trunks then go through a
# shared 1x1 collapse. The active collapsed map feeds the aux head directly;
# both collapsed maps are flattened+concatenated into a Dense+BN+ReLU bottleneck
# of width value_head_neurons. That vector drives the value head AND projects
# (value_head_neurons → f) into a channel bias added to the active trunk before
# a 1x1 conv emits the spatial policy. Independent dataclass — does NOT subclass
# AuxBaseResNetConfig, so ai.py dispatch must match v2 explicitly.

@dataclass
class AuxBaseResNetV2Config:
    blocks: int = 8
    filters: int = 32
    value_head_neurons: int = 64
    own_kernels: int = 4
    aux_hidden: int = 16
    aux_weight: float = 1.5
    l2_reg: float = 1e-4              # Keras kernel_regularizer; PyTorch uses AdamW weight_decay.
    policy_format: str = 'spatial_list'  # v2 is natively spatial; kept for ai.py introspection.


class AuxBaseResNetV2(nn.Module):
    def __init__(self, model_config: AuxBaseResNetV2Config, use_tanh: bool = False):
        super().__init__()
        f = model_config.filters
        vhn = model_config.value_head_neurons

        # Shared pre-trunk piece/scalar/color → channel bias
        feat_in = _PIECES_PER_PLAYER + _SCALARS_PER_PLAYER + 1
        self.feat_bias = nn.Linear(feat_in, f)

        # 5x5 stem (shared across both players)
        self.stem = nn.Sequential(
            nn.Conv2d(1, f, kernel_size=5, padding='same', bias=False),
            nn.BatchNorm2d(f),
            nn.ReLU(),
        )
        # Residual trunk (shared) — same shape as AuxBaseResNet.
        self.trunk = nn.Sequential(*[_BaseResBlock(f) for _ in range(model_config.blocks)])

        # Shared 1x1 collapse to own_kernels channels
        self.collapse = nn.Sequential(
            nn.Conv2d(f, model_config.own_kernels, kernel_size=1, bias=False),
            nn.BatchNorm2d(model_config.own_kernels),
            nn.ReLU(),
        )
        collapsed_flat = model_config.own_kernels * ROWS * COLS

        # Aux head: from active player's collapsed map only
        self.aux_head = nn.Sequential(
            nn.Linear(collapsed_flat, model_config.aux_hidden),
            nn.BatchNorm1d(model_config.aux_hidden),
            nn.ReLU(),
            nn.Linear(model_config.aux_hidden, 2),
            nn.Sigmoid(),
        )

        # Joint bottleneck: both collapsed maps → value_head_neurons
        self.joint = nn.Sequential(
            nn.Linear(2 * collapsed_flat, vhn),
            nn.BatchNorm1d(vhn),
            nn.ReLU(),
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(vhn, 1),
            nn.Tanh() if use_tanh else nn.Sigmoid(),
        )

        # Policy head: vhn → f channel bias, broadcast-added to a_feat, then 1x1 conv.
        self.policy_bias = nn.Linear(vhn, f)
        self.policy_conv = nn.Conv2d(f, POLICY_SHAPE[0], kernel_size=1)

        self.policy_format = model_config.policy_format

    def _make_bias(self, pieces, b2b, combo, garbage, color_signal):
        return self.feat_bias(torch.cat([
            pieces.flatten(1),
            b2b.unsqueeze(1), combo.unsqueeze(1), garbage.unsqueeze(1),
            color_signal.unsqueeze(1),
        ], dim=1))

    def _run_trunk(self, grid, bias_vec):
        x = self.stem(grid)
        x = x + bias_vec.view(-1, bias_vec.size(1), 1, 1)
        return self.trunk(x)

    def forward(self, a_grid, a_pieces, a_b2b, a_combo, a_garbage,
                o_grid, o_pieces, o_b2b, o_combo, o_garbage, color):
        a_bias = self._make_bias(a_pieces, a_b2b, a_combo, a_garbage, color)
        o_bias = self._make_bias(o_pieces, o_b2b, o_combo, o_garbage, 1.0 - color)

        a_feat = self._run_trunk(a_grid, a_bias)
        o_feat = self._run_trunk(o_grid, o_bias)

        a_flat = self.collapse(a_feat).flatten(1)
        o_flat = self.collapse(o_feat).flatten(1)

        aux = self.aux_head(a_flat)
        vhn_vec = self.joint(torch.cat([a_flat, o_flat], dim=1))

        value = self.value_head(vhn_vec)

        policy_bias = self.policy_bias(vhn_vec)
        policy_biased = a_feat + policy_bias.view(-1, policy_bias.size(1), 1, 1)
        policy = self.policy_conv(policy_biased).permute(0, 2, 3, 1).contiguous()

        return value, policy, aux


# ── Keras ─────────────────────────────────────────────────────────────────────

from tensorflow import keras
import tensorflow as tf


def gen_baseresnet_keras(model_config: BaseResNetConfig, use_tanh: bool = False) -> keras.Model:
    """Keras (channels-last) port of BaseResNet. Same data flow as the PyTorch version.

    Forward inputs ordered to match game_to_X / build_batch order, with numeric names
    "0".."10" so it slots into the existing TFLite path.
    """
    f = model_config.filters
    grid_shape = (ROWS, COLS, 1)
    piece_shape = (2 + PREVIEWS, len(MINOS))

    # Inputs (named "0".."10" to match downstream conventions)
    a_grid    = keras.Input(shape=grid_shape,  name='0')
    a_pieces  = keras.Input(shape=piece_shape, name='1')
    a_b2b     = keras.Input(shape=(1,),        name='2')
    a_combo   = keras.Input(shape=(1,),        name='3')
    a_garbage = keras.Input(shape=(1,),        name='4')
    o_grid    = keras.Input(shape=grid_shape,  name='5')
    o_pieces  = keras.Input(shape=piece_shape, name='6')
    o_b2b     = keras.Input(shape=(1,),        name='7')
    o_combo   = keras.Input(shape=(1,),        name='8')
    o_garbage = keras.Input(shape=(1,),        name='9')
    color     = keras.Input(shape=(1,),        name='10')
    inputs = [a_grid, a_pieces, a_b2b, a_combo, a_garbage,
              o_grid, o_pieces, o_b2b, o_combo, o_garbage, color]

    # ── Shared trunk (one set of layers, called on both grids) ──
    stem_conv = keras.layers.Conv2D(f, 3, padding='same', use_bias=False)
    stem_bn   = keras.layers.BatchNormalization()
    blocks = []
    for _ in range(model_config.blocks):
        blocks.append((
            keras.layers.Conv2D(f, 3, padding='same', use_bias=False),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(f, 3, padding='same', use_bias=False),
            keras.layers.BatchNormalization(),
        ))

    def trunk(x):
        x = keras.layers.ReLU()(stem_bn(stem_conv(x)))
        for conv1, bn1, conv2, bn2 in blocks:
            y = keras.layers.ReLU()(bn1(conv1(x)))
            y = bn2(conv2(y))
            x = keras.layers.ReLU()(keras.layers.Add()([x, y]))
        return x

    a_feat = trunk(a_grid)
    o_feat = trunk(o_grid)

    # ── Opponent collapse: 1x1 conv -> 1 channel, flatten ──
    o_coll = keras.layers.ReLU()(keras.layers.BatchNormalization()(
        keras.layers.Conv2D(1, 1, use_bias=False)(o_feat)))
    o_flat = keras.layers.Flatten()(o_coll)

    # ── Opponent encode ──
    opp_in = keras.layers.Concatenate()([
        o_flat,
        keras.layers.Flatten()(o_pieces),
        o_b2b, o_combo, o_garbage,
    ])
    opp_repr = keras.layers.ReLU()(keras.layers.BatchNormalization()(
        keras.layers.Dense(model_config.opp_hidden)(opp_in)))

    # ── Bias projection: combine opp_repr + own extras + color → (B, f) ──
    bias_in = keras.layers.Concatenate()([
        opp_repr,
        keras.layers.Flatten()(a_pieces),
        a_b2b, a_combo, a_garbage,
        color,
    ])
    bias_vec = keras.layers.Dense(f)(bias_in)

    # FiLM-add: broadcast bias_vec (B, f) → (B, 1, 1, f), add to a_feat (B, ROWS, COLS, f)
    bias_map = keras.layers.Reshape((1, 1, f))(bias_vec)
    biased = keras.layers.Add()([a_feat, bias_map])

    # ── Own collapse: 1x1 conv -> own_kernels, flatten ──
    own_coll = keras.layers.ReLU()(keras.layers.BatchNormalization()(
        keras.layers.Conv2D(model_config.own_kernels, 1, use_bias=False)(biased)))
    flat = keras.layers.Flatten()(own_coll)

    # Heads see flattened collapsed map + own pieces + own scalars + color directly
    head_x = keras.layers.Concatenate()([
        flat,
        keras.layers.Flatten()(a_pieces),
        a_b2b, a_combo, a_garbage,
        color,
    ])

    # ── Heads ──
    v = keras.layers.Dense(model_config.value_hidden)(head_x)
    v = keras.layers.BatchNormalization()(v)
    v = keras.layers.ReLU()(v)
    value = keras.layers.Dense(1, activation='tanh' if use_tanh else 'sigmoid', name='value')(v)
    policy = _build_keras_policy_head(model_config.policy_format, head_x, biased)

    return keras.Model(inputs=inputs, outputs=[value, policy])


def _build_keras_policy_head(policy_format: str, head_x, biased):
    """Build the right Keras policy head for the given format.

    'original'     -> Dense(POLICY_SIZE, softmax) on head_x; (B, 7722) probabilities.
    'spatial_list' -> 1x1 Conv2D(27) on biased + spatial softmax; (B, ROWS, COLS, 27) probs.
    'heatmap_*'    -> 1x1 Conv2D(27, linear) on biased; (B, ROWS, COLS, 27) raw heatmap.
    """
    if policy_format == 'original':
        return keras.layers.Dense(POLICY_SIZE, activation='softmax', name='policy')(head_x)
    if policy_format == 'spatial_list':
        spatial = keras.layers.Conv2D(POLICY_SHAPE[0], 1, name='policy_conv')(biased)
        flat = keras.layers.Flatten()(spatial)
        softmaxed = keras.layers.Softmax(name='policy_softmax')(flat)
        return keras.layers.Reshape((ROWS, COLS, POLICY_SHAPE[0]), name='policy')(softmaxed)
    if policy_format in ('heatmap_AM', 'heatmap_FFT'):
        return keras.layers.Conv2D(POLICY_SHAPE[0], 1, name='policy')(biased)
    raise ValueError(f"unknown policy_format: {policy_format!r}")


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


def gen_auxbaseresnet_keras(model_config: AuxBaseResNetConfig, use_tanh: bool = False) -> keras.Model:
    """Keras (channels-last) port of AuxBaseResNet. Every weight-bearing layer is
    explicitly named so port_keras_auxbaseresnet_to_pytorch can copy weights by name.

    Outputs: [value, policy, aux] (matches PyTorch AuxBaseResNet 3-tuple order).
    BN epsilon=1e-5 to match torch.nn.BatchNorm default (Keras default 1e-3 would
    introduce a small mismatch when porting weights).
    """
    f = model_config.filters
    eps = 1e-5
    grid_shape = (ROWS, COLS, 1)
    piece_shape = (2 + PREVIEWS, len(MINOS))

    a_grid    = keras.Input(shape=grid_shape,  name='0')
    a_pieces  = keras.Input(shape=piece_shape, name='1')
    a_b2b     = keras.Input(shape=(1,),        name='2')
    a_combo   = keras.Input(shape=(1,),        name='3')
    a_garbage = keras.Input(shape=(1,),        name='4')
    o_grid    = keras.Input(shape=grid_shape,  name='5')
    o_pieces  = keras.Input(shape=piece_shape, name='6')
    o_b2b     = keras.Input(shape=(1,),        name='7')
    o_combo   = keras.Input(shape=(1,),        name='8')
    o_garbage = keras.Input(shape=(1,),        name='9')
    color     = keras.Input(shape=(1,),        name='10')
    inputs = [a_grid, a_pieces, a_b2b, a_combo, a_garbage,
              o_grid, o_pieces, o_b2b, o_combo, o_garbage, color]

    stem_conv = keras.layers.Conv2D(f, 3, padding='same', use_bias=False, name='stem_conv')
    stem_bn   = keras.layers.BatchNormalization(epsilon=eps, name='stem_bn')

    blocks = []
    for i in range(model_config.blocks):
        blocks.append((
            keras.layers.Conv2D(f, 3, padding='same', use_bias=False, name=f'block{i}_conv1'),
            keras.layers.BatchNormalization(epsilon=eps, name=f'block{i}_bn1'),
            keras.layers.Conv2D(f, 3, padding='same', use_bias=False, name=f'block{i}_conv2'),
            keras.layers.BatchNormalization(epsilon=eps, name=f'block{i}_bn2'),
        ))

    def trunk(x):
        x = keras.layers.ReLU()(stem_bn(stem_conv(x)))
        for conv1, bn1, conv2, bn2 in blocks:
            y = keras.layers.ReLU()(bn1(conv1(x)))
            y = bn2(conv2(y))
            x = keras.layers.ReLU()(keras.layers.Add()([x, y]))
        return x

    a_feat = trunk(a_grid)
    o_feat = trunk(o_grid)

    # Opp collapse: 1x1 conv -> 1 channel
    o_coll = keras.layers.ReLU()(
        keras.layers.BatchNormalization(epsilon=eps, name='opp_collapse_bn')(
            keras.layers.Conv2D(1, 1, use_bias=False, name='opp_collapse_conv')(o_feat)
        )
    )
    o_flat = keras.layers.Flatten()(o_coll)

    # Opp encode
    opp_in = keras.layers.Concatenate()([
        o_flat,
        keras.layers.Flatten()(o_pieces),
        o_b2b, o_combo, o_garbage,
    ])
    opp_repr = keras.layers.ReLU()(
        keras.layers.BatchNormalization(epsilon=eps, name='opp_encode_bn')(
            keras.layers.Dense(model_config.opp_hidden, name='opp_encode_dense')(opp_in)
        )
    )

    # Bias projection -> FiLM-add
    bias_in = keras.layers.Concatenate()([
        opp_repr,
        keras.layers.Flatten()(a_pieces),
        a_b2b, a_combo, a_garbage,
        color,
    ])
    bias_vec = keras.layers.Dense(f, name='bias_project')(bias_in)
    bias_map = keras.layers.Reshape((1, 1, f))(bias_vec)
    biased = keras.layers.Add()([a_feat, bias_map])

    # Own collapse
    own_coll = keras.layers.ReLU()(
        keras.layers.BatchNormalization(epsilon=eps, name='own_collapse_bn')(
            keras.layers.Conv2D(model_config.own_kernels, 1, use_bias=False, name='own_collapse_conv')(biased)
        )
    )
    flat = keras.layers.Flatten()(own_coll)

    head_x = keras.layers.Concatenate()([
        flat,
        keras.layers.Flatten()(a_pieces),
        a_b2b, a_combo, a_garbage,
        color,
    ])

    # Value head
    v = keras.layers.Dense(model_config.value_hidden, name='value_hidden')(head_x)
    v = keras.layers.BatchNormalization(epsilon=eps, name='value_hidden_bn')(v)
    v = keras.layers.ReLU()(v)
    value = keras.layers.Dense(1, activation='tanh' if use_tanh else 'sigmoid', name='value')(v)

    # Policy head — flat softmax / spatial softmax / spatial heatmap per policy_format.
    policy = _build_keras_policy_head(model_config.policy_format, head_x, biased)

    # Aux head (2 outputs: holes_norm, height_norm)
    a = keras.layers.Dense(model_config.aux_hidden, name='aux_hidden')(head_x)
    a = keras.layers.BatchNormalization(epsilon=eps, name='aux_hidden_bn')(a)
    a = keras.layers.ReLU()(a)
    aux = keras.layers.Dense(2, activation='sigmoid', name='aux')(a)

    return keras.Model(inputs=inputs, outputs=[value, policy, aux])


def gen_auxbaseresnetv2_keras(model_config: AuxBaseResNetV2Config, use_tanh: bool = False) -> keras.Model:
    """Keras (channels-last) port of AuxBaseResNetV2.

    Outputs: [value, policy, aux] matching the PyTorch class. Every weight-bearing
    layer is named and carries an L2 kernel_regularizer (l2_reg). BN epsilon=1e-5
    to match torch.nn.BatchNorm default for potential cross-backend porting.
    """
    f = model_config.filters
    vhn = model_config.value_head_neurons
    eps = 1e-5
    reg = keras.regularizers.l2(model_config.l2_reg) if model_config.l2_reg else None
    grid_shape = (ROWS, COLS, 1)
    piece_shape = (2 + PREVIEWS, len(MINOS))

    a_grid    = keras.Input(shape=grid_shape,  name='0')
    a_pieces  = keras.Input(shape=piece_shape, name='1')
    a_b2b     = keras.Input(shape=(1,),        name='2')
    a_combo   = keras.Input(shape=(1,),        name='3')
    a_garbage = keras.Input(shape=(1,),        name='4')
    o_grid    = keras.Input(shape=grid_shape,  name='5')
    o_pieces  = keras.Input(shape=piece_shape, name='6')
    o_b2b     = keras.Input(shape=(1,),        name='7')
    o_combo   = keras.Input(shape=(1,),        name='8')
    o_garbage = keras.Input(shape=(1,),        name='9')
    color     = keras.Input(shape=(1,),        name='10')
    inputs = [a_grid, a_pieces, a_b2b, a_combo, a_garbage,
              o_grid, o_pieces, o_b2b, o_combo, o_garbage, color]

    # Opponent sees (1 - color). Rescaling(scale=-1, offset=1) is safely serializable
    # (unlike a bare python-lambda Lambda layer, which blocks .keras load without
    # enable_unsafe_deserialization).
    opp_color = keras.layers.Rescaling(scale=-1.0, offset=1.0, name='opp_color')(color)

    # ── Shared pre-trunk piece/scalar/color → (B, f) channel bias ──
    feat_bias = keras.layers.Dense(f, name='feat_bias', kernel_regularizer=reg)
    a_feat_in = keras.layers.Concatenate()([
        keras.layers.Flatten()(a_pieces), a_b2b, a_combo, a_garbage, color,
    ])
    o_feat_in = keras.layers.Concatenate()([
        keras.layers.Flatten()(o_pieces), o_b2b, o_combo, o_garbage, opp_color,
    ])
    a_bias_map = keras.layers.Reshape((1, 1, f))(feat_bias(a_feat_in))
    o_bias_map = keras.layers.Reshape((1, 1, f))(feat_bias(o_feat_in))

    # ── Shared 5x5 stem + residual trunk (weights shared across both players) ──
    stem_conv = keras.layers.Conv2D(f, 5, padding='same', use_bias=False,
                                    name='stem_conv', kernel_regularizer=reg)
    stem_bn   = keras.layers.BatchNormalization(epsilon=eps, name='stem_bn')

    blocks = []
    for i in range(model_config.blocks):
        blocks.append((
            keras.layers.Conv2D(f, 3, padding='same', use_bias=False,
                                name=f'block{i}_conv1', kernel_regularizer=reg),
            keras.layers.BatchNormalization(epsilon=eps, name=f'block{i}_bn1'),
            keras.layers.Conv2D(f, 3, padding='same', use_bias=False,
                                name=f'block{i}_conv2', kernel_regularizer=reg),
            keras.layers.BatchNormalization(epsilon=eps, name=f'block{i}_bn2'),
        ))

    def run_trunk(grid, bias_map):
        x = keras.layers.ReLU()(stem_bn(stem_conv(grid)))
        x = keras.layers.Add()([x, bias_map])
        for conv1, bn1, conv2, bn2 in blocks:
            y = keras.layers.ReLU()(bn1(conv1(x)))
            y = bn2(conv2(y))
            x = keras.layers.ReLU()(keras.layers.Add()([x, y]))
        return x

    a_feat = run_trunk(a_grid, a_bias_map)
    o_feat = run_trunk(o_grid, o_bias_map)

    # ── Shared 1x1 collapse → own_kernels channels ──
    collapse_conv = keras.layers.Conv2D(model_config.own_kernels, 1, use_bias=False,
                                        name='collapse_conv', kernel_regularizer=reg)
    collapse_bn   = keras.layers.BatchNormalization(epsilon=eps, name='collapse_bn')

    def collapse(t):
        return keras.layers.ReLU()(collapse_bn(collapse_conv(t)))

    a_flat = keras.layers.Flatten()(collapse(a_feat))
    o_flat = keras.layers.Flatten()(collapse(o_feat))

    # ── Aux head — fed from active player's collapsed map only ──
    a_h = keras.layers.Dense(model_config.aux_hidden, name='aux_hidden',
                             kernel_regularizer=reg)(a_flat)
    a_h = keras.layers.BatchNormalization(epsilon=eps, name='aux_hidden_bn')(a_h)
    a_h = keras.layers.ReLU()(a_h)
    aux = keras.layers.Dense(2, activation='sigmoid', name='aux',
                             kernel_regularizer=reg)(a_h)

    # ── Joint value bottleneck ──
    joint_in = keras.layers.Concatenate()([a_flat, o_flat])
    j = keras.layers.Dense(vhn, name='joint_dense', kernel_regularizer=reg)(joint_in)
    j = keras.layers.BatchNormalization(epsilon=eps, name='joint_bn')(j)
    vhn_vec = keras.layers.ReLU()(j)

    # ── Value head ──
    value = keras.layers.Dense(1, activation='tanh' if use_tanh else 'sigmoid',
                               name='value', kernel_regularizer=reg)(vhn_vec)

    # ── Policy head: vhn → (B, f) channel bias → add to a_feat → 1x1 conv → spatial softmax ──
    policy_bias = keras.layers.Dense(f, name='policy_bias', kernel_regularizer=reg)(vhn_vec)
    policy_bias_map = keras.layers.Reshape((1, 1, f))(policy_bias)
    policy_biased = keras.layers.Add()([a_feat, policy_bias_map])
    spatial = keras.layers.Conv2D(POLICY_SHAPE[0], 1, name='policy_conv',
                                  kernel_regularizer=reg)(policy_biased)
    flat = keras.layers.Flatten()(spatial)
    softmaxed = keras.layers.Softmax(name='policy_softmax')(flat)
    policy = keras.layers.Reshape((ROWS, COLS, POLICY_SHAPE[0]), name='policy')(softmaxed)

    return keras.Model(inputs=inputs, outputs=[value, policy, aux])