# This is parameter setting for all deep learning algorithms
import sys
# Import games
sys.path.append("DQN_GAMES/")

# Action Num
import pong 
import dot
import dot_test 
import tetris 
import wormy 
import breakout as game
import dodge 

Gamma = 0.99
Learning_rate = 0.00025
Epsilon = 1
Final_epsilon = 0.1

Num_action = game.Return_Num_Action()

Num_replay_memory = 100000
Num_start_training = 50000
Num_training = 500000
Num_update = Num_training/100
Num_batch = 32
Num_test = 250000
Num_skipFrame = 4
Num_stackFrame = 4
Num_colorChannel = 1

Num_plot_step = 5000

Train_mode = True 
Load_model = False
Load_path = './saved_networks/breakout/20200310-16-00-37_DQN/model'

img_size = 80