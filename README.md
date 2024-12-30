**Bugs**
 - Fix fake t spins

**Todo:**
 - FIX LOSS
 - Fix reduction fpu if node is root and dirichlet noise is on
 - Clean up code
 - To main, add switch boards and visualize AI moves
 - Add chance nodes/address randomness in MCTS
 - Investigate combo not sending the correct number of lines (?)
 - Test parameters
 - Finish pytorch merger, clean up merge code
 - Encoding garbage into the neural network/MCTS (Open loop MCTS)
 - Keep implementing katago strategies (read appendix)
     - Variable board sizes
     - Use monte carlo graph search
 - Update model architecture
 - Change how/how much data is used for training
 - Stop tensorflow prints

**Areas of optimization:**
 - Generating move matrix
 - Switching to pytorch (?)
 - Load data faster

**Testing results**
l1_neurons
l2_neurons

Alphalike model
blocks
pooling_blocks
filters
cpool
dropout
    - 0 < 0.25 > 0.4
kernels=1,
o_side_neurons=16,
value_head_neurons=16,

augment_data
    - True > False
learning_rate
    - When training on a single set ??? (Don't know how to change in keras after initialization)
loss_weights
epochs
batch_size

save_all
    - False > True

use_playout_cap_randomization=
playout_cap_chance=
playout_cap_mult=

use_dirichlet_noise
DIRICHLET_ALPHA
DIRICHLET_S
    - 'DIRICHLET_ALPHA' == 0.01: 10 > 50
DIRICHLET_EXPLORATION
use_dirichlet_s

FpuStrategy
    'reduction' > absolute
FpuValue
    'reduction': 0.4 ~= 0.2 > 0.0

use_forced_playouts_and_policy_target_pruning
CForcedPlayout

use_root_softmax
RootSoftmaxTemp

CPUCT

Miscellaneous
 - With 10 layers, 16 filters is max number of filters before inference time scales
 - Bottleneck residuals were worse, with or without dropout than normal residuals (test_4 and test_5)
 - test_data_paramemeters does work



**Data and Model Versions**

Data:
1.5 - Doubled training data, removed augment data
1.6 - Increased data window length
1.7 - Switched to S2 Ruleset:
 - Added was_just_rotated to matrix
   - Policy size: (19, 25, 11) -> (19, 2, 25, 11)
 - Changed algorithm used
   - faster-but-loss -> brute-force
 - Added pieces % 7 to data

Models:
4.9 - Increased data window length
5.0 - Faulty: Switched to S2 Ruleset: 
 - Created base_nn, a copy of alpha_same but uses 1.7 policy size
 - Switched networks
   - alpha_same -> base_nn
5.1 - Removed policy from loss weights, turned off dirichlet alpha, playout cap randomization, and forced policy target pruning




 When you instantiate a game, set the ruleset