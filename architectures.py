# File for managing different neural networks architectures.
from const import *

from tensorflow import keras


def create_input_layers():
    shapes = [(ROWS, COLS, 1), # Grid
              (2 + PREVIEWS, len(MINOS)), # Pieces
              (1,), # B2B
              (1,), # Combo
              (1,), # Lines cleared
              (1,), # Lines sent
              (ROWS, COLS, 1), 
              (2 + PREVIEWS, len(MINOS)), 
              (1,), 
              (1,), 
              (1,),
              (1,),
              (1,), # Color (Whether you had first move or not)
              (1,)] # Total pieces placed

    inputs = []
    active_features = []
    active_grid = None
    opponent_features = []
    opponent_grid = None
    
    non_player_features = []

    for i, shape in enumerate(shapes):
        # Add input
        input = keras.Input(shape=shape, name=f"{i}")
        inputs.append(input)

        num_inputs = len(shapes)
        # Active player's features
        if i < (num_inputs - 2) / 2: # Ignore last two inputs, and take the first half
            if shape == shapes[0]:
                active_grid = input
            else:
                active_features.append(keras.layers.Flatten()(input))
        # Other player's features
        elif i < (num_inputs - 2): # Ignore last two inputs, take remaining half
            if shape == shapes[0]:
                opponent_grid = input
            else:
                opponent_features.append(keras.layers.Flatten()(input))
        # Other features
        else:
            non_player_features.append(keras.layers.Flatten()(input))
    
    return inputs, active_grid, active_features, opponent_grid, opponent_features, non_player_features

def ValueHead(config):
    # Returns value; found at the end of the network
    def inside(x):
        x = keras.layers.Dense(config.value_head_neurons)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.Dropout(config.dropout)(x) # Dropout
        x = keras.layers.Dense(1, activation=('tanh' if config.use_tanh else 'sigmoid'))(x)

        return x
    return inside

def PolicyHead():
    # Returns policy list; found at the end of the network
    def inside(x):
        # Generate probability distribution
        x = keras.layers.Dense(POLICY_SIZE, activation="softmax")(x)

        return x
    return inside

def gen_alphasame_nn(config) -> keras.Model:
    # The network uses the same neural network to apply convolutions to both grids
    def ResidualLayer():
        # Uses skip conections
        def inside(in_1, in_2):
            batch_1 = keras.layers.BatchNormalization()
            relu_1 = keras.layers.Activation('relu')
            conv_1 = keras.layers.Conv2D(config.filters, (3, 3), padding="same")
            batch_2 = keras.layers.BatchNormalization()
            dropout_1 = keras.layers.Dropout(config.dropout)
            relu_2 = keras.layers.Activation('relu')
            conv_2 = keras.layers.Conv2D(config.filters, (3, 3), padding="same")

            out_1 = conv_2(relu_2(dropout_1(batch_2(conv_1(relu_1(batch_1(in_1)))))))
            out_2 = conv_2(relu_2(dropout_1(batch_2(conv_1(relu_1(batch_1(in_2)))))))

            out_1 = keras.layers.Add()([in_1, out_1])
            out_2 = keras.layers.Add()([in_2, out_2])

            return out_1, out_2
        return inside

    inputs, a_grid, a_features, o_grid, o_features, non_player_features = create_input_layers()

    # Start with a convolutional layer
    # Because each grid needs the same network, use the same layers for each side
    conv_1 = keras.layers.Conv2D(config.filters, (5, 5), padding="same")
    
    a_grid = conv_1(a_grid)
    o_grid = conv_1(o_grid)

    # 10 residual layers
    for _ in range(config.blocks):
        residual_layer = ResidualLayer()
        a_grid, o_grid = residual_layer(a_grid, o_grid)
    
    batch_1 = keras.layers.BatchNormalization()
    relu_1 = keras.layers.Activation('relu')

    a_grid = relu_1(batch_1(a_grid))
    o_grid = relu_1(batch_1(o_grid))
    
    # 1x1 Kernel
    kernel_1 = keras.layers.Conv2D(config.kernels, (1, 1))
    a_grid = kernel_1(a_grid)
    o_grid = kernel_1(o_grid)

    flatten_1 = keras.layers.Flatten()
    a_grid = flatten_1(a_grid)
    o_grid = flatten_1(o_grid)

    batch_2 = keras.layers.BatchNormalization()
    relu_2 = keras.layers.Activation('relu')
    a_grid = relu_2(batch_2(a_grid))
    o_grid = relu_2(batch_2(o_grid))

    # Shrink opponent grid info
    o_grid = keras.layers.Dense(config.o_side_neurons)(o_grid)
    o_grid = keras.layers.BatchNormalization()(o_grid)
    o_grid = keras.layers.Activation('relu')(o_grid)

    # Combine with other features
    x = keras.layers.Concatenate()([a_grid, o_grid, *a_features, *o_features, *non_player_features])

    value_output = ValueHead(config)(x)
    policy_output = PolicyHead()(x)

    model = keras.Model(inputs=inputs, outputs=[value_output, policy_output])

    return model

def test_3(config) -> keras.Model: # Using channel wise addition
    # The network uses the same neural network to apply convolutions to both grids
    def ResidualLayer():
        # Uses skip conections
        def inside(in_1, in_2):
            conv_1 = keras.layers.Conv2D(config.filters, (3, 3), padding="same")
            batch_1 = keras.layers.BatchNormalization()
            relu_1 = keras.layers.Activation('relu')
            conv_2 = keras.layers.Conv2D(config.filters, (3, 3), padding="same")
            batch_2 = keras.layers.BatchNormalization()
            relu_2 = keras.layers.Activation('relu')

            out_1 = relu_2(batch_2(conv_2(relu_1(batch_1(conv_1(in_1))))))
            out_2 = relu_2(batch_2(conv_2(relu_1(batch_1(conv_1(in_2))))))

            out_1 = keras.layers.Add()([in_1, out_1])
            out_2 = keras.layers.Add()([in_2, out_2])

            dropout_1 = keras.layers.Dropout(config.dropout)

            out_1 = dropout_1(out_1)
            out_2 = dropout_1(out_2)

            return out_1, out_2
        return inside

    inputs, a_grid, a_features, o_grid, o_features, non_player_features = create_input_layers()

    # Start with a convolutional layer
    # Because each grid needs the same network, use the same layers for each side
    conv_1 = keras.layers.Conv2D(config.filters, (3,3 ), padding="same")

    game_state = keras.layers.Concatenate()([*a_features, *o_features, *non_player_features])
    game_state = keras.layers.Dense(config.filters)(game_state)
    game_state = keras.layers.Reshape((1, 1, config.filters))





    batch_1 = keras.layers.BatchNormalization()
    relu_1 = keras.layers.Activation('relu')
    dropout_1 = keras.layers.Dropout(config.dropout)
    
    a_grid = dropout_1(relu_1(batch_1(conv_1(a_grid))))
    o_grid = dropout_1(relu_1(batch_1(conv_1(o_grid))))

    # 10 residual layers
    for _ in range(config.blocks):
        residual_layer = ResidualLayer()
        a_grid, o_grid = residual_layer(a_grid, o_grid)
    
    # 1x1 Kernel
    kernel_1 = keras.layers.Conv2D(config.kernels, (1, 1))
    a_grid = kernel_1(a_grid)
    o_grid = kernel_1(o_grid)

    flatten_1 = keras.layers.Flatten()
    a_grid = flatten_1(a_grid)
    o_grid = flatten_1(o_grid)

    batch_2 = keras.layers.BatchNormalization()
    relu_2 = keras.layers.Activation('relu')
    a_grid = relu_2(batch_2(a_grid))
    o_grid = relu_2(batch_2(o_grid))

    # Shrink opponent grid info
    o_grid = keras.layers.Dense(config.o_side_neurons)(o_grid)
    o_grid = keras.layers.BatchNormalization()(o_grid)
    o_grid = keras.layers.Activation('relu')(o_grid)

    # Combine with other features
    x = keras.layers.Concatenate()([a_grid, o_grid, *a_features, *o_features, *non_player_features])

    value_output = ValueHead(config)(x)
    policy_output = PolicyHead()(x)

    model = keras.Model(inputs=inputs, outputs=[value_output, policy_output])

    return model




def test_4(config) -> keras.Model: # Using bottleneck residual blocks
    # The network uses the same neural network to apply convolutions to both grids
    def ResidualLayer():
        def inside(in_1_1, in_1_2):
            # Bottleneck residual block
            # Lots of skip connections here, very confusing code
            batch_1 = keras.layers.BatchNormalization()
            dropout_1 = keras.layers.Dropout(config.dropout)
            relu_1 = keras.layers.Activation('relu')
            conv_1 = keras.layers.Conv2D(config.filters // 2, (1, 1), padding="same") # 1x1

            in_2_1 = conv_1(relu_1(dropout_1(batch_1(in_1_1))))
            in_2_2 = conv_1(relu_1(dropout_1(batch_1(in_1_2))))

            batch_2 = keras.layers.BatchNormalization()
            relu_2 = keras.layers.Activation('relu')
            conv_2 = keras.layers.Conv2D(config.filters // 2, (3, 3), padding="same") # 3x3
            batch_3 = keras.layers.BatchNormalization()
            dropout_2 = keras.layers.Dropout(config.dropout)
            relu_3 = keras.layers.Activation('relu')
            conv_3 = keras.layers.Conv2D(config.filters // 2, (3, 3), padding="same") # 3x3

            in_3_1 = conv_3(relu_3(dropout_2(batch_3(conv_2(relu_2(batch_2(in_2_1)))))))
            in_3_2 = conv_3(relu_3(dropout_2(batch_3(conv_2(relu_2(batch_2(in_2_2)))))))

            in_3_1 = keras.layers.Add()([in_2_1, in_3_1])
            in_3_2 = keras.layers.Add()([in_2_2, in_3_2])

            batch_4 = keras.layers.BatchNormalization()
            relu_4 = keras.layers.Activation('relu')
            conv_4 = keras.layers.Conv2D(config.filters // 2, (3, 3), padding="same") # 3x3
            batch_5 = keras.layers.BatchNormalization()
            dropout_3 = keras.layers.Dropout(config.dropout)
            relu_5 = keras.layers.Activation('relu')
            conv_5 = keras.layers.Conv2D(config.filters // 2, (3, 3), padding="same") # 3x3

            in_4_1 = conv_5(relu_5(dropout_3(batch_5(conv_4(relu_4(batch_4(in_3_1)))))))
            in_4_2 = conv_5(relu_5(dropout_3(batch_5(conv_4(relu_4(batch_4(in_3_2)))))))

            in_4_1 = keras.layers.Add()([in_3_1, in_4_1])
            in_4_2 = keras.layers.Add()([in_3_2, in_4_2])

            batch_6 = keras.layers.BatchNormalization()
            dropout_4 = keras.layers.Dropout(config.dropout)
            relu_6 = keras.layers.Activation('relu')
            conv_6 = keras.layers.Conv2D(config.filters, (1, 1), padding="same") # 1x1

            in_4_1 = conv_6(relu_6(dropout_4(batch_6(in_4_1))))
            in_4_2 = conv_6(relu_6(dropout_4(batch_6(in_4_2))))

            in_4_1 = keras.layers.Add()([in_1_1, in_4_1])
            in_4_2 = keras.layers.Add()([in_1_2, in_4_2])            

            return in_4_1, in_4_1

        return inside

    inputs, a_grid, a_features, o_grid, o_features, non_player_features = create_input_layers()

    # Start with a convolutional layer
    # Because each grid needs the same network, use the same layers for each side
    conv_1 = keras.layers.Conv2D(config.filters, (5, 5), padding="same")
    
    a_grid = conv_1(a_grid)
    o_grid = conv_1(o_grid)

    # n residual layers
    for _ in range(config.blocks):
        residual_layer = ResidualLayer()
        a_grid, o_grid = residual_layer(a_grid, o_grid)
    
    batch_1 = keras.layers.BatchNormalization()
    relu_1 = keras.layers.Activation('relu')

    a_grid = relu_1(batch_1(a_grid))
    o_grid = relu_1(batch_1(o_grid))
    
    # 1x1 Kernel
    kernel_1 = keras.layers.Conv2D(config.kernels, (1, 1))
    a_grid = kernel_1(a_grid)
    o_grid = kernel_1(o_grid)

    flatten_1 = keras.layers.Flatten()
    a_grid = flatten_1(a_grid)
    o_grid = flatten_1(o_grid)

    batch_2 = keras.layers.BatchNormalization()
    relu_2 = keras.layers.Activation('relu')
    a_grid = relu_2(batch_2(a_grid))
    o_grid = relu_2(batch_2(o_grid))

    # Shrink opponent grid info
    o_grid = keras.layers.Dense(config.o_side_neurons)(o_grid)
    o_grid = keras.layers.BatchNormalization()(o_grid)
    o_grid = keras.layers.Activation('relu')(o_grid)

    # Combine with other features
    x = keras.layers.Concatenate()([a_grid, o_grid, *a_features, *o_features, *non_player_features])

    value_output = ValueHead(config)(x)
    policy_output = PolicyHead()(x)

    model = keras.Model(inputs=inputs, outputs=[value_output, policy_output])

    return model

def test_5(config) -> keras.Model: # Same as test 4 but with dropout
    # The network uses the same neural network to apply convolutions to both grids
    def ResidualLayer():
        def inside(in_1_1, in_1_2):
            # Bottleneck residual block
            # Lots of skip connections here, very confusing code
            batch_1 = keras.layers.BatchNormalization()
            relu_1 = keras.layers.Activation('relu')
            conv_1 = keras.layers.Conv2D(config.filters // 2, (1, 1), padding="same") # 1x1

            in_2_1 = conv_1(relu_1(batch_1(in_1_1)))
            in_2_2 = conv_1(relu_1(batch_1(in_1_2)))

            batch_2 = keras.layers.BatchNormalization()
            relu_2 = keras.layers.Activation('relu')
            conv_2 = keras.layers.Conv2D(config.filters // 2, (3, 3), padding="same") # 3x3
            batch_3 = keras.layers.BatchNormalization()
            relu_3 = keras.layers.Activation('relu')
            conv_3 = keras.layers.Conv2D(config.filters // 2, (3, 3), padding="same") # 3x3

            in_3_1 = conv_3(relu_3(batch_3(conv_2(relu_2(batch_2(in_2_1))))))
            in_3_2 = conv_3(relu_3(batch_3(conv_2(relu_2(batch_2(in_2_2))))))

            in_3_1 = keras.layers.Add()([in_2_1, in_3_1])
            in_3_2 = keras.layers.Add()([in_2_2, in_3_2])

            batch_4 = keras.layers.BatchNormalization()
            relu_4 = keras.layers.Activation('relu')
            conv_4 = keras.layers.Conv2D(config.filters // 2, (3, 3), padding="same") # 3x3
            batch_5 = keras.layers.BatchNormalization()
            relu_5 = keras.layers.Activation('relu')
            conv_5 = keras.layers.Conv2D(config.filters // 2, (3, 3), padding="same") # 3x3

            in_4_1 = conv_5(relu_5(batch_5(conv_4(relu_4(batch_4(in_3_1))))))
            in_4_2 = conv_5(relu_5(batch_5(conv_4(relu_4(batch_4(in_3_2))))))

            in_4_1 = keras.layers.Add()([in_3_1, in_4_1])
            in_4_2 = keras.layers.Add()([in_3_2, in_4_2])

            batch_6 = keras.layers.BatchNormalization()
            relu_6 = keras.layers.Activation('relu')
            conv_6 = keras.layers.Conv2D(config.filters, (1, 1), padding="same") # 1x1

            in_4_1 = conv_6(relu_6(batch_6(in_4_1)))
            in_4_2 = conv_6(relu_6(batch_6(in_4_2)))

            in_4_1 = keras.layers.Add()([in_1_1, in_4_1])
            in_4_2 = keras.layers.Add()([in_1_2, in_4_2])            

            return in_4_1, in_4_1

        return inside

    inputs, a_grid, a_features, o_grid, o_features, non_player_features = create_input_layers()

    # Start with a convolutional layer
    # Because each grid needs the same network, use the same layers for each side
    conv_1 = keras.layers.Conv2D(config.filters, (5, 5), padding="same")
    
    a_grid = conv_1(a_grid)
    o_grid = conv_1(o_grid)

    # n residual layers
    for _ in range(config.blocks):
        residual_layer = ResidualLayer()
        a_grid, o_grid = residual_layer(a_grid, o_grid)
    
    batch_1 = keras.layers.BatchNormalization()
    relu_1 = keras.layers.Activation('relu')

    a_grid = relu_1(batch_1(a_grid))
    o_grid = relu_1(batch_1(o_grid))
    
    # 1x1 Kernel
    kernel_1 = keras.layers.Conv2D(config.kernels, (1, 1))
    a_grid = kernel_1(a_grid)
    o_grid = kernel_1(o_grid)

    flatten_1 = keras.layers.Flatten()
    a_grid = flatten_1(a_grid)
    o_grid = flatten_1(o_grid)

    batch_2 = keras.layers.BatchNormalization()
    relu_2 = keras.layers.Activation('relu')
    a_grid = relu_2(batch_2(a_grid))
    o_grid = relu_2(batch_2(o_grid))

    # Shrink opponent grid info
    o_grid = keras.layers.Dense(config.o_side_neurons)(o_grid)
    o_grid = keras.layers.BatchNormalization()(o_grid)
    o_grid = keras.layers.Activation('relu')(o_grid)

    # Combine with other features
    x = keras.layers.Concatenate()([a_grid, o_grid, *a_features, *o_features, *non_player_features])

    value_output = ValueHead(config)(x)
    policy_output = PolicyHead()(x)

    model = keras.Model(inputs=inputs, outputs=[value_output, policy_output])

    return model