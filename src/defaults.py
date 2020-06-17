BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 0.001              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic 
WEIGHT_DECAY = 0        # L2 weight decay

PRIO_ALPHA = 0.6
PRIO_BETA = 0.4
PRIO_EPSILON = 1e-5