# File for managing different neural networks architectures.
from const import *

# import torch
# from torch import nn
# import torch.nn.functional as F

"""
class ResidualBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.conv_block1 = nn.Sequential(
            nn.BatchNorm2d(num_features=config.filters),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=config.filters,
                out_channels=config.filters,
                kernel_size=3,
                stride=1,
                padding='same',
                bias=False,
            ),
        )

        self.conv_block2 = nn.Sequential(
            nn.BatchNorm2d(num_features=config.filters),
            nn.Dropout(p=config.dropout),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=config.filters,
                out_channels=config.filters,
                kernel_size=3,
                stride=1,
                padding='same',
                bias=False,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv_block1(x)
        out = self.conv_block2(out)
        out += residual
        return out


class AlphaSame(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=config.filters,
            kernel_size=5,
            stride=1,
            padding='same',
            bias=False
        )

        # Residual blocks
        res_blocks = []
        for _ in range(config.blocks):
            res_blocks.append(ResidualBlock(config))
        self.res_blocks = nn.Sequential(*res_blocks)

        self.batchnorm1 = nn.BatchNorm2d(num_features=config.filters)
        self.relu1 = nn.ReLU()

        self.kernel1 = nn.Conv2d(
            in_channels=config.filters,
            out_channels=config.kernels,
            kernel_size=1,
            stride=1,
            padding='same',
            bias=False
        )

        self.flatten1 = nn.Flatten()

        self.batchnorm2 = nn.BatchNorm2d(num_features=config.kernels)
        self.relu2 = nn.ReLU()

        self.osidedense = nn.Sequential(
            nn.Linear(ROWS * COLS * config.kernels, config.o_side_neurons),
            nn.BatchNorm1d(num_features=config.o_side_neurons),
            nn.ReLU(),
        )

        head_inputs = ROWS * COLS * config.kernels + config.o_side_neurons + 2 * ((2 + PREVIEWS) * len(MINOS)) + 2 * 3 + 1

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(head_inputs, POLICY_SIZE),
            nn.Sigmoid()
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(head_inputs, config.value_head_neurons),
            nn.BatchNorm1d(num_features=config.value_head_neurons),
            nn.ReLU(),
            nn.Linear(config.value_head_neurons, 1),
            nn.Dropout(config.dropout),
            (nn.Tanh() if config.use_tanh else nn.Sigmoid()),
        )

    def forward(self, a_grid, a_pieces, a_b2b, a_combo, a_garbage, o_grid, o_pieces, o_b2b, o_combo, o_garbage, color):
        # 5x5 Convolution
        a_grid_out = self.conv1(a_grid)
        o_grid_out = self.conv1(o_grid)

        # Residual blocks
        a_grid_out = self.res_blocks(a_grid_out)
        o_grid_out = self.res_blocks(o_grid_out)

        # Batch norm -> Relu
        a_grid_out = self.batchnorm1(a_grid_out)
        o_grid_out = self.batchnorm1(o_grid_out)
        a_grid_out = self.relu1(a_grid_out)
        o_grid_out = self.relu1(o_grid_out)

        # 1x1 Kernel
        a_grid_out = self.kernel1(a_grid_out)
        o_grid_out = self.kernel1(o_grid_out)

        a_grid_out = self.batchnorm2(a_grid_out)
        o_grid_out = self.batchnorm2(o_grid_out)
        a_grid_out = self.relu2(a_grid_out)
        o_grid_out = self.relu2(o_grid_out)

        # Flatten
        a_grid_out = self.flatten1(a_grid_out)
        o_grid_out = self.flatten1(o_grid_out)

        # Shrink opponent grid info
        o_grid_out = self.osidedense(o_grid_out)

        # Concatenate all features
        a_pieces = self.flatten1(a_pieces)
        o_pieces = self.flatten1(o_pieces)

        a_b2b = a_b2b.unsqueeze(1)
        a_combo = a_combo.unsqueeze(1)
        a_garbage = a_garbage.unsqueeze(1)
        o_b2b = o_b2b.unsqueeze(1)
        o_combo = o_combo.unsqueeze(1)
        o_garbage = o_garbage.unsqueeze(1)
        color = color.unsqueeze(1)
        # pieces_placed = pieces_placed.unsqueeze(1)

        x = torch.concat((a_grid_out, a_pieces, a_b2b, a_combo, a_garbage, o_grid_out, o_pieces, o_b2b, o_combo, o_garbage, color), dim=1)

        # Value and policy head
        value_output = self.value_head(x)
        policy_output = self.policy_head(x)

        return value_output, policy_output
        



"""



import tensorflow as tf


def create_input_layers():
    shapes = [(ROWS, COLS, 1), # Grid
              (2 + PREVIEWS, len(MINOS)), # Pieces
              (1,), # B2B
              (1,), # Combo
              (1,), # Garbage
              (ROWS, COLS, 1), 
              (2 + PREVIEWS, len(MINOS)), 
              (1,), 
              (1,), 
              (1,),
              (1,)] # Color (Whether you had first move or not)

    inputs = []
    active_features = []
    active_grid = None
    opponent_features = []
    opponent_grid = None
    
    non_player_features = []

    for i, shape in enumerate(shapes):
        # Add input
        input = tf.keras.Input(shape=shape, name=f"{i}")
        inputs.append(input)

        num_inputs = len(shapes)
        # Active player's features
        if i < (num_inputs - 1) / 2: # Ignore last input, and take the first half
            if shape == shapes[0]:
                active_grid = input
            else:
                active_features.append(tf.keras.layers.Flatten()(input))
        # Other player's features
        elif i < (num_inputs - 1): # Ignore last input, take remaining half
            if shape == shapes[0]:
                opponent_grid = input
            else:
                opponent_features.append(tf.keras.layers.Flatten()(input))
        # Other features
        else:
            non_player_features.append(tf.keras.layers.Flatten()(input))
    
    return inputs, active_grid, active_features, opponent_grid, opponent_features, non_player_features

def gen_alphasame_nn(config) -> tf.keras.Model:
    def ValueHead():
        # Returns value; found at the end of the network
        def inside(x):
            x = tf.keras.layers.Dense(config.value_head_neurons)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation('relu')(x)
            x = tf.keras.layers.Dropout(config.dropout)(x) # Dropout
            x = tf.keras.layers.Dense(1, activation=('tanh' if config.use_tanh else 'sigmoid'))(x)

            return x
        return inside

    def PolicyHead():
        # Returns policy list; found at the end of the network
        def inside(x):
            # Generate probability distribution
            x = tf.keras.layers.Dense(POLICY_SIZE, activation="softmax")(x)

            return x
        return inside

    # The network uses the same neural network to apply convolutions to both grids
    def ResidualLayer():
        # Uses skip conections
        def inside(in_1, in_2):
            batch_1 = tf.keras.layers.BatchNormalization()
            relu_1 = tf.keras.layers.Activation('relu')
            conv_1 = tf.keras.layers.Conv2D(config.filters, (3, 3), padding="same")
            batch_2 = tf.keras.layers.BatchNormalization()
            dropout_1 = tf.keras.layers.Dropout(config.dropout)
            relu_2 = tf.keras.layers.Activation('relu')
            conv_2 = tf.keras.layers.Conv2D(config.filters, (3, 3), padding="same")

            out_1 = conv_2(relu_2(dropout_1(batch_2(conv_1(relu_1(batch_1(in_1)))))))
            out_2 = conv_2(relu_2(dropout_1(batch_2(conv_1(relu_1(batch_1(in_2)))))))

            out_1 = tf.keras.layers.Add()([in_1, out_1])
            out_2 = tf.keras.layers.Add()([in_2, out_2])

            return out_1, out_2
        return inside
    
    def GlobalPoolingLayer():
        # Uses skip conections and global pooling
        def inside(in_1, in_2):
            batch_1 = tf.keras.layers.BatchNormalization()
            relu_1 = tf.keras.layers.Activation('relu')
            conv_1 = tf.keras.layers.Conv2D(config.filters, (3, 3), padding="same")

            out_1 = conv_1(relu_1(batch_1(in_1)))
            out_2 = conv_1(relu_1(batch_1(in_2)))

            # lambda_pool_1 = tf.keras.layers.Lambda(lambda x: x[..., :config.cpool])
            # lambda_rest_1 = tf.keras.layers.Lambda(lambda x: x[..., config.cpool:])

            # Split channels
            out_1_pool = out_1[:, :, :, :config.cpool]
            out_1_rest = out_1[:, :, :, config.cpool:]

            out_2_pool = out_2[:, :, :, :config.cpool]
            out_2_rest = out_2[:, :, :, config.cpool:]

            # Use batch norm and relu on pooling layers
            pool_batch_1 = tf.keras.layers.BatchNormalization()
            pool_relu_1 = tf.keras.layers.Activation('relu')

            out_1_pool_act = pool_relu_1(pool_batch_1(out_1_pool))
            out_2_pool_act = pool_relu_1(pool_batch_1(out_2_pool))

            # Global average and global pooling on the first cpool channels
            avg_pool_1 = tf.keras.layers.GlobalAveragePooling2D()
            max_pool_1 = tf.keras.layers.GlobalMaxPooling2D()

            out_1_average_pooled = avg_pool_1(out_1_pool_act)
            out_2_average_pooled = avg_pool_1(out_2_pool_act)

            out_1_max_pooled = max_pool_1(out_1_pool)
            out_2_max_pooled = max_pool_1(out_2_pool)

            out_1_pooled = tf.keras.layers.Concatenate()([out_1_average_pooled, out_1_max_pooled])
            out_2_pooled = tf.keras.layers.Concatenate()([out_2_average_pooled, out_2_max_pooled])

            # Fully connected layer
            dense_1 = tf.keras.layers.Dense(config.filters - config.cpool)

            out_1_dense = dense_1(out_1_pooled)
            out_2_dense = dense_1(out_2_pooled)

            # Reshape to match the remaining channels
            out_1_biases = tf.keras.layers.Reshape((1, 1, config.filters - config.cpool))(out_1_dense)
            out_2_biases = tf.keras.layers.Reshape((1, 1, config.filters - config.cpool))(out_2_dense)

            # Add the biases to the remaining channels
            out_1_biased = tf.keras.layers.Add()([out_1_rest, out_1_biases])
            out_2_biased = tf.keras.layers.Add()([out_2_rest, out_2_biases])

            # Concatenate the pooled and biased channels
            out_1 = tf.keras.layers.Concatenate(axis=-1)([out_1_pool, out_1_biased])
            out_2 = tf.keras.layers.Concatenate(axis=-1)([out_2_pool, out_2_biased])

            # Resume second half of the standard residual block
            batch_2 = tf.keras.layers.BatchNormalization()
            dropout_1 = tf.keras.layers.Dropout(config.dropout)
            relu_2 = tf.keras.layers.Activation('relu')
            conv_2 = tf.keras.layers.Conv2D(config.filters, (3, 3), padding="same")

            out_1 = conv_2(relu_2(dropout_1(batch_2(out_1))))
            out_2 = conv_2(relu_2(dropout_1(batch_2(out_2))))

            out_1 = tf.keras.layers.Add()([in_1, out_1])
            out_2 = tf.keras.layers.Add()([in_2, out_2])

            return out_1, out_2
        return inside

    inputs, a_grid, a_features, o_grid, o_features, non_player_features = create_input_layers()

    # Start with a convolutional layer
    # Because each grid needs the same network, use the same layers for each side
    conv_1 = tf.keras.layers.Conv2D(config.filters, (5, 5), padding="same")
    
    a_grid = conv_1(a_grid)
    o_grid = conv_1(o_grid)

    # Take other features and use a dense layer to add channelwise
    dense_1 = tf.keras.layers.Dense(config.filters)

    a_features = tf.keras.layers.Concatenate()(a_features)
    o_features = tf.keras.layers.Concatenate()(o_features)

    a_features = dense_1(a_features)
    o_features = dense_1(o_features)

    a_biases = tf.keras.layers.Reshape((1, 1, config.filters))(a_features)
    o_biases = tf.keras.layers.Reshape((1, 1, config.filters))(o_features)

    a_grid = tf.keras.layers.Add()([a_grid, a_biases])
    o_grid = tf.keras.layers.Add()([o_grid, o_biases])

    # 10 layers: 8 residual, 2 global pooling
    # Evenly space blocks
    global_pool_indices = [round((i + 1) / (config.pooling_blocks + 1) * config.blocks) - 1 for i in range(config.pooling_blocks)]

    for i in range(config.blocks):
        if i in global_pool_indices:
            layer = GlobalPoolingLayer()
        else:
            layer = ResidualLayer()
        
        a_grid, o_grid = layer(a_grid, o_grid)
    
    batch_1 = tf.keras.layers.BatchNormalization()
    relu_1 = tf.keras.layers.Activation('relu')

    a_grid = relu_1(batch_1(a_grid))
    o_grid = relu_1(batch_1(o_grid))
    
    # 1x1 Kernel
    kernel_1 = tf.keras.layers.Conv2D(config.kernels, (1, 1))
    a_grid = kernel_1(a_grid)
    o_grid = kernel_1(o_grid)

    flatten_1 = tf.keras.layers.Flatten()
    a_grid = flatten_1(a_grid)
    o_grid = flatten_1(o_grid)

    batch_2 = tf.keras.layers.BatchNormalization()
    relu_2 = tf.keras.layers.Activation('relu')
    a_grid = relu_2(batch_2(a_grid))
    o_grid = relu_2(batch_2(o_grid))

    # Shrink opponent grid info
    o_grid = tf.keras.layers.Dense(config.o_side_neurons)(o_grid)
    o_grid = tf.keras.layers.BatchNormalization()(o_grid)
    o_grid = tf.keras.layers.Activation('relu')(o_grid)

    # Combine with other features
    x = tf.keras.layers.Concatenate()([a_grid, o_grid, *non_player_features])

    value_output = ValueHead()(x)
    policy_output = PolicyHead()(x)

    model = tf.keras.Model(inputs=inputs, outputs=[value_output, policy_output])

    return model

def gen_model_aux(config) -> tf.keras.Model:
    """
    Neural network that separates spatial board understanding from piece information.
    Auxiliary heads only predict spatial board properties (height, holes).
    """
    def ValueHead():
        def inside(x):
            x = tf.keras.layers.Dense(config.value_head_neurons)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation('relu')(x)
            x = tf.keras.layers.Dropout(config.dropout)(x)
            x = tf.keras.layers.Dense(1, activation=('tanh' if config.use_tanh else 'sigmoid'), name="value")(x)
            return x
        return inside

    def PolicyHead():
        def inside(x):
            x = tf.keras.layers.Dense(POLICY_SIZE, activation="softmax", name="policy")(x)
            return x
        return inside

    def ResidualLayer():
        def inside(in_1, in_2):
            batch_1 = tf.keras.layers.BatchNormalization()
            relu_1 = tf.keras.layers.Activation('relu')
            conv_1 = tf.keras.layers.Conv2D(config.filters, (3, 3), padding="same")
            batch_2 = tf.keras.layers.BatchNormalization()
            dropout_1 = tf.keras.layers.Dropout(config.dropout)
            relu_2 = tf.keras.layers.Activation('relu')
            conv_2 = tf.keras.layers.Conv2D(config.filters, (3, 3), padding="same")

            out_1 = conv_2(relu_2(dropout_1(batch_2(conv_1(relu_1(batch_1(in_1)))))))
            out_2 = conv_2(relu_2(dropout_1(batch_2(conv_1(relu_1(batch_1(in_2)))))))

            out_1 = tf.keras.layers.Add()([in_1, out_1])
            out_2 = tf.keras.layers.Add()([in_2, out_2])

            return out_1, out_2
        return inside
    
    def GlobalPoolingLayer():
        def inside(in_1, in_2):
            batch_1 = tf.keras.layers.BatchNormalization()
            relu_1 = tf.keras.layers.Activation('relu')
            conv_1 = tf.keras.layers.Conv2D(config.filters, (3, 3), padding="same")

            out_1 = conv_1(relu_1(batch_1(in_1)))
            out_2 = conv_1(relu_1(batch_1(in_2)))

            # Split channels for pooling
            out_1_pool = out_1[:, :, :, :config.cpool]
            out_1_rest = out_1[:, :, :, config.cpool:]
            out_2_pool = out_2[:, :, :, :config.cpool]
            out_2_rest = out_2[:, :, :, config.cpool:]

            # Apply batch norm and relu to pooling channels
            pool_batch_1 = tf.keras.layers.BatchNormalization()
            pool_relu_1 = tf.keras.layers.Activation('relu')

            out_1_pool_act = pool_relu_1(pool_batch_1(out_1_pool))
            out_2_pool_act = pool_relu_1(pool_batch_1(out_2_pool))

            # Global pooling
            avg_pool_1 = tf.keras.layers.GlobalAveragePooling2D()
            max_pool_1 = tf.keras.layers.GlobalMaxPooling2D()

            out_1_average_pooled = avg_pool_1(out_1_pool_act)
            out_2_average_pooled = avg_pool_1(out_2_pool_act)
            out_1_max_pooled = max_pool_1(out_1_pool)
            out_2_max_pooled = max_pool_1(out_2_pool)

            out_1_pooled = tf.keras.layers.Concatenate()([out_1_average_pooled, out_1_max_pooled])
            out_2_pooled = tf.keras.layers.Concatenate()([out_2_average_pooled, out_2_max_pooled])

            # Dense layer
            dense_1 = tf.keras.layers.Dense(config.filters - config.cpool)
            out_1_dense = dense_1(out_1_pooled)
            out_2_dense = dense_1(out_2_pooled)

            # Reshape and add biases
            out_1_biases = tf.keras.layers.Reshape((1, 1, config.filters - config.cpool))(out_1_dense)
            out_2_biases = tf.keras.layers.Reshape((1, 1, config.filters - config.cpool))(out_2_dense)

            out_1_biased = tf.keras.layers.Add()([out_1_rest, out_1_biases])
            out_2_biased = tf.keras.layers.Add()([out_2_rest, out_2_biases])

            # Concatenate channels
            out_1 = tf.keras.layers.Concatenate(axis=-1)([out_1_pool, out_1_biased])
            out_2 = tf.keras.layers.Concatenate(axis=-1)([out_2_pool, out_2_biased])

            # Continue residual block
            batch_2 = tf.keras.layers.BatchNormalization()
            dropout_1 = tf.keras.layers.Dropout(config.dropout)
            relu_2 = tf.keras.layers.Activation('relu')
            conv_2 = tf.keras.layers.Conv2D(config.filters, (3, 3), padding="same")

            out_1 = conv_2(relu_2(dropout_1(batch_2(out_1))))
            out_2 = conv_2(relu_2(dropout_1(batch_2(out_2))))

            out_1 = tf.keras.layers.Add()([in_1, out_1])
            out_2 = tf.keras.layers.Add()([in_2, out_2])

            return out_1, out_2
        return inside

    inputs, a_grid, a_features, o_grid, o_features, non_player_features = create_input_layers()

    # Initial convolution for spatial features only
    conv_1 = tf.keras.layers.Conv2D(config.filters, (5, 5), padding="same")
    
    a_grid = conv_1(a_grid)
    o_grid = conv_1(o_grid)

    # Process residual/pooling layers for spatial understanding
    global_pool_indices = [round((i + 1) / (config.pooling_blocks + 1) * config.blocks) - 1 for i in range(config.pooling_blocks)]

    for i in range(config.blocks):
        if i in global_pool_indices:
            layer = GlobalPoolingLayer()
        else:
            layer = ResidualLayer()
        
        a_grid, o_grid = layer(a_grid, o_grid)
    
    # Final spatial processing
    batch_1 = tf.keras.layers.BatchNormalization()
    relu_1 = tf.keras.layers.Activation('relu')

    a_grid = relu_1(batch_1(a_grid))
    o_grid = relu_1(batch_1(o_grid))
    
    # Extract spatial features
    kernel_1 = tf.keras.layers.Conv2D(config.kernels, (1, 1))
    a_grid_spatial = kernel_1(a_grid)
    o_grid_spatial = kernel_1(o_grid)

    flatten_1 = tf.keras.layers.Flatten()
    a_grid_spatial = flatten_1(a_grid_spatial)
    o_grid_spatial = flatten_1(o_grid_spatial)

    batch_2 = tf.keras.layers.BatchNormalization()
    relu_2 = tf.keras.layers.Activation('relu')
    a_grid_spatial = relu_2(batch_2(a_grid_spatial))
    o_grid_spatial = relu_2(batch_2(o_grid_spatial))

    # Compress opponent spatial features
    o_grid_spatial = tf.keras.layers.Dense(config.o_side_neurons)(o_grid_spatial)
    o_grid_spatial = tf.keras.layers.BatchNormalization()(o_grid_spatial)
    o_grid_spatial = tf.keras.layers.Activation('relu')(o_grid_spatial)

    # Process piece information separately (no shared layers with spatial features)
    a_pieces_flat = tf.keras.layers.Flatten()(a_features[0])  # First element is pieces
    o_pieces_flat = tf.keras.layers.Flatten()(o_features[0])

    piece_dense = tf.keras.layers.Dense(64)
    a_piece_processed = piece_dense(a_pieces_flat)
    o_piece_processed = piece_dense(o_pieces_flat)

    a_piece_processed = tf.keras.layers.BatchNormalization()(a_piece_processed)
    o_piece_processed = tf.keras.layers.BatchNormalization()(o_piece_processed)
    a_piece_processed = tf.keras.layers.Activation('relu')(a_piece_processed)
    o_piece_processed = tf.keras.layers.Activation('relu')(o_piece_processed)

    # Combine remaining non-spatial features
    other_a_features = a_features[1:]  # Skip pieces, keep b2b, combo, garbage
    other_o_features = o_features[1:]

    # Auxiliary heads - only spatial board properties
    aux_height = tf.keras.layers.Dense(1, activation="sigmoid", name="aux_height")(a_grid_spatial)
    aux_holes = tf.keras.layers.Dense(1, activation="sigmoid", name="aux_holes")(a_grid_spatial)

    # Main heads - combine spatial and piece information
    combined_features = tf.keras.layers.Concatenate()([
        a_grid_spatial, 
        a_piece_processed,
        *other_a_features,
        o_grid_spatial,
        o_piece_processed, 
        *other_o_features,
        *non_player_features
    ])

    value_output = ValueHead()(combined_features)
    policy_output = PolicyHead()(combined_features)

    model = tf.keras.Model(
        inputs=inputs, 
        outputs={
            'value': value_output,
            'policy': policy_output, 
            'aux_height': aux_height,
            'aux_holes': aux_holes
        }
    )

    return model