from dataclasses import dataclass, field
from typing import List

@dataclass
class Config:
    '''Class for keeping track of an agents config'''
    buffer_size:int = int(1e6)      # replay buffer size
    batch_size:int = 128            # minibatch size
    gamma:float = 0.99              # discount factor
    tau:float = 1e-3                # for soft update of target parameters
    lr_actor:float = 1e-3           # learning rate of the actor 
    lr_critic:float = 1e-3          # learning rate of the critic
    weight_decay:float = 0          # L2 weight decay

    random_seed:int = 50

    update_n_step:int = 4

    actor_hidden_units:List = field(default_factory=lambda: [128,128])
    critic_hidden_units:List = field(default_factory=lambda: [128,128])

    # ---- Ornstein–Uhlenbeck process parameters ---- #
    add_noise:bool = True
    noise_mu:float = 0. 
    noise_theta:float = 0.15 
    noise_sigma:float = 0.05

    # ---- prioritized experience replay parameters ---- #
    use_per:bool = False            # use prioritized experience replay
    per_alpha:float = 0.6
    per_beta:float = 0.4
    per_epsilon:float = 1e-5

    # ---- batchnorm ---- #
    use_bn:bool = False