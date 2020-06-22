import datetime
import os

import gym
import air_gym
import torch

import numpy as np
import airsim

from gym import error, spaces, utils
from gym.utils import seeding
from gym.spaces import Tuple, Box, Discrete, MultiDiscrete

from collections import OrderedDict

class MuZeroConfig:
    def __init__(self, args):
        self.seed = 0  # Seed for numpy, torch and the game

        ### Game
        self.observation_shape = (3, 192,
                                  192)  # Dimensions of the game observation, must be 3D (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
        self.action_space = [i for i in range(7)]  # Fixed list of all possible actions. You should only edit the length
        self.players = [i for i in range(1)]  # List of players. You should only edit the length
        self.stacked_observations = 32  # Number of previous observations and previous actions to add to the current observation

        # Evaluate
        self.muzero_player = 0  # Turn Muzero begins to play (0: MuZero plays first, 1: MuZero plays second)
        self.opponent = None  # Hard coded agent that MuZero faces to assess his progress in multiplayer games. It doesn't influence training. None, "random" or "expert" if implemented in the Game class

        ### Self-Play
        self.num_actors = 1  # Number of simultaneous threads self-playing to feed the replay buffer
        self.max_moves = 27000  # Maximum number of moves if game is not finished before
        self.num_simulations = 50  # Number of future moves self-simulated
        self.discount = 0.997  # Chronological discount of the reward
        self.temperature_threshold = None  # Number of moves before dropping temperature to 0 (ie playing according to the max)

        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.25
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        ### Network
        self.network = "resnet"  # "resnet" / "fullyconnected"
        self.support_size = 300  # Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of -support_size to support_size

        # Residual Network
        self.downsample = True  # Downsample observations before representation network (See paper appendix Network Architecture)
        self.blocks = 16  # Number of blocks in the ResNet
        self.channels = 256  # Number of channels in the ResNet
        self.reduced_channels_reward = 256  # Number of channels in reward head
        self.reduced_channels_value = 256  # Number of channels in value head
        self.reduced_channels_policy = 256  # Number of channels in policy head
        self.resnet_fc_reward_layers = [256, 256]  # Define the hidden layers in the reward head of the dynamic network
        self.resnet_fc_value_layers = [256, 256]  # Define the hidden layers in the value head of the prediction network
        self.resnet_fc_policy_layers = [256,
                                        256]  # Define the hidden layers in the policy head of the prediction network

        # Fully Connected Network
        self.encoding_size = 10
        self.fc_representation_layers = []  # Define the hidden layers in the representation network
        self.fc_dynamics_layers = [16]  # Define the hidden layers in the dynamics network
        self.fc_reward_layers = [16]  # Define the hidden layers in the reward network
        self.fc_value_layers = []  # Define the hidden layers in the value network
        self.fc_policy_layers = []  # Define the hidden layers in the policy network

        ### Training
        self.results_path = os.path.join(os.path.dirname(__file__), "../results", os.path.basename(__file__)[:-3],
                                         datetime.datetime.now().strftime(
                                             "%Y-%m-%d--%H-%M-%S"))  # Path to store the model weights and TensorBoard logs
        self.training_steps = int(1000e3)  # Total number of training steps (ie weights update according to a batch)
        self.batch_size = 1024  # Number of parts of games to train on at each training step
        self.checkpoint_interval = int(1e3)  # Number of training steps before using the model for self-playing
        self.value_loss_weight = 0.25  # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
        self.training_device = "cuda" if torch.cuda.is_available() else "cpu"  # Train on GPU if available

        self.optimizer = "SGD"  # "Adam" or "SGD". Paper uses SGD
        self.weight_decay = 1e-4  # L2 weights regularization
        self.momentum = 0.9  # Used only if optimizer is SGD

        # Exponential learning rate schedule
        self.lr_init = args.learning_rate  # Initial learning rate
        self.lr_decay_rate = args.decay_rate  # Set it to 1 to use a constant learning rate
        self.lr_decay_steps = 350e3

        ### Replay Buffer
        self.window_size = int(1e6)  # Number of self-play games to keep in the replay buffer
        self.num_unroll_steps = 5  # Number of game moves to keep for every batch element
        self.td_steps = 10  # Number of steps in the future to take into account for calculating the target value
        self.use_last_model_value = True  # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)

        # Prioritized Replay (See paper appendix Training)
        self.PER = True  # Select in priority the elements in the replay buffer which are unexpected for the network
        self.use_max_priority = False  # If False, use the n-step TD error as initial priority. Better for large replay buffer
        self.PER_alpha = 0.5  # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1
        self.PER_beta = 1.0

        ### Adjust the self play / training ratio to avoid over/underfitting
        self.self_play_delay = 0  # Number of seconds to wait after each played game
        self.training_delay = 0  # Number of seconds to wait after each training step
        self.ratio = None  # Desired self played games per training step ratio. Equivalent to a synchronous version, training can take much longer. Set it to None to disable it

    def visit_softmax_temperature_fn(self, trained_steps):
        """
        Parameter to alter the visit count distribution to ensure that the action selection becomes greedier as training progresses.
        The smaller it is, the more likely the best action (ie with the highest visit count) is chosen.

        Returns:
            Positive float.
        """
        if trained_steps < 500e3:
            return 1.0
        elif trained_steps < 750e3:
            return 0.5
        else:
            return 0.25


class AirSimEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, image_shape):
        self.env = gym.make('air_gym:airsim-drone-v0',ip_address='130.108.129.54', control_type='discrete',step_length=10, image_shape=(192,192,3), goal=[20,20,-20])
        self.observation_space = spaces.Box(low=0, high=255, shape=image_shape, dtype=np.uint8)
        self._seed()

        self.viewer = None
        self.steps = 0
        self.no_episode = 0
        self.reward_sum = 0

    def __del__(self):
        raise NotImplementedError()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _compute_reward(self):
        raise NotImplementedError()

    def step(self, action):
        raise NotImplementedError()

    def _get_obs(self):
        raise NotImplementedError()

    def close(self):
        raise NotImplementedError()

    def render(self, mode='human'):
        img = self._get_obs()
        if mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen
        elif mode == 'rgb_array':
            return img


class AirSimDroneEnv(AirSimEnv):

    def __init__(self, ip_address, control_type, step_length, image_shape, goal):
        super().__init__(image_shape)

        self.step_length = step_length
        self.control_type = control_type
        self.image_shape = image_shape
        self.goal = airsim.Vector3r(goal[0], goal[1], goal[2])

        if self.control_type is 'discrete':
            self.action_space = spaces.Discrete(7)
        if self.control_type is 'continuous':
            self.action_space = spaces.Box(low=-5, high=5, shape=(3,))
        else:
            print("Must choose a control type {'discrete','continuous'}. Defaulting to discrete.")
            self.action_space = spaces.Discrete(7)

        self.state = {"position": np.zeros(3), "collision": False}

        self.drone = airsim.MultirotorClient(ip=ip_address)
        self._setup_flight()

        self.image_request = airsim.ImageRequest('front_center', airsim.ImageType.Scene, False, False)

    def __del__(self):
        self.drone.reset()

    def _setup_flight(self):
        self.drone.__init__()
        self.drone.confirmConnection()
        self.drone.reset()
        self.drone.enableApiControl(True)
        self.drone.armDisarm(True)
        self.drone.moveToPositionAsync(0, 0, -2, 2).join()

    def _get_obs(self):
        response = self.drone.simGetImages([self.image_request])
        image = np.reshape(np.fromstring(response[0].image_data_uint8, dtype=np.uint8), self.image_shape)
        _drone_state = self.drone.getMultirotorState()
        position = _drone_state.kinematics_estimated.position.to_numpy_array()
        collision = self.drone.simGetCollisionInfo().has_collided

        self.state["position"] = position
        self.state["collision"] = collision

        return image

    def _compute_reward(self):
        pos = self.state["position"]
        current_pos = airsim.Vector3r(pos[0], pos[1], pos[2])
        if current_pos == self.goal:
            done = True
            reward = 10
            return reward, done
        elif self.state["collision"] == True:
            done = True
        else:
            done = False

        dist = current_pos.distance_to(self.goal)
        if dist > 30:
            reward = 0
        else:
            reward = (30 - dist) * 0.1

        return reward, done

    def _do_action(self, action):
        if self.control_type is 'discrete':
            new_position = self.actions_to_op(action)
            if new_position[2] > -1:
                new_position[2] = -1
            if new_position[2] < - 40:
                new_position[2] = -40
            self.drone.moveToPositionAsync(float(new_position[0]), float(new_position[1]), float(new_position[2]),
                                           8).join()
        else:
            self.drone.moveByVelocityAsync(float(action[0]), float(action[1]), float(action[2]),
                                           self.step_length).join()

    def noop(self):
        new_position = self.state["position"]
        return new_position

    def forward(self):
        new_position = self.state["position"]
        new_position[0] += self.step_length
        return new_position

    def backward(self):
        new_position = self.state["position"]
        new_position[0] -= self.step_length
        return new_position

    def right(self):
        new_position = self.state["position"]
        new_position[1] += self.step_length
        return new_position

    def left(self):
        new_position = self.state["position"]
        new_position[1] -= self.step_length
        return new_position

    def up(self):
        new_position = self.state["position"]
        new_position[2] += self.step_length
        return new_position

    def down(self):
        new_position = self.state["position"]
        new_position[2] -= self.step_length
        return new_position

    def step(self, action):
        self._do_action(action)
        obs = self._get_obs()
        reward, done = self._compute_reward()

        return obs, reward, done, self.state

    def reset(self):
        self._setup_flight()
        self._get_obs()

    def actions_to_op(self, action):
        switcher = {
            0: self.noop,
            1: self.forward,
            2: self.backward,
            3: self.right,
            4: self.left,
            5: self.up,
            6: self.down
        }

        func = switcher.get(action, lambda: "Invalid Action!")
        return func()

    def legal_actions(self):
        """
        Should return the legal actions at each turn, if it is not available, it can return
        the whole action space. At each turn, the game have to be able to handle one of returned actions.

        For complex game where calculating legal moves is too long, the idea is to define the legal actions
        equal to the action space but to return a negative reward if the action is illegal.

        Returns:
            An array of integers, subset of the action space.
        """
        return [i for i in range(7)]