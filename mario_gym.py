#!/usr/bin/env python3

"""
This script trains a simple DQN in the SuperMarioBros-v0 environment.
Each frame of the game is displayed in an OpenCV window
and simultaneously written to an MP4 video file.
If you interrupt the process with CTRL + C, the video file
will not be corrupted because we release the VideoWriter in a finally block.
"""

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 1) Multiprocessing fix (important for macOS, sometimes also for Linux/Windows)
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import multiprocessing
multiprocessing.set_start_method("spawn", force=True)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 2) Standard libraries and warnings
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import warnings
import os
import time
import random

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 3) External libraries (OpenCV, Gym, PyTorch, Numpy, etc.)
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import cv2
import gym
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from collections import deque
from torchvision import transforms as T

# Suppress Gym warnings (optional)
warnings.filterwarnings("ignore", message="Overwriting existing videos")
warnings.filterwarnings("ignore", message="The result returned by `env.reset()` was not a tuple")
warnings.filterwarnings("ignore", message="Disabling video recorder")
warnings.filterwarnings("ignore", message="No render modes was declared")
warnings.filterwarnings("ignore", message="Core environment is written in old step API")
warnings.filterwarnings("ignore", message="`np.bool8` is a deprecated alias for `np.bool_`")
warnings.filterwarnings("ignore", message="The environment creator metadata doesn't include `render_modes`")

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 4) Imports specific to gym_super_mario_bros
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

# Just a note in case MoviePy is not installed
try:
    import moviepy  # noqa
except ImportError:
    print("Warning: MoviePy is not installed! Please run 'pip install moviepy'")
    print("if you want to further process the generated .mp4 later.\n")

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 5) Automatically choose device (CPU / GPU)
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 6) Helper function: Frame preprocessing
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def preprocess_observation(obs):
    """
    Converts the input frame (RGB) to grayscale,
    rescales it to [84x84], and returns a numpy array
    with shape [1, 84, 84] (1=channel).
    Then we normalize pixel values from 0..255 to 0..1.
    """
    # obs should be shape (H, W, 3) = RGB
    if obs.ndim == 3:
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)  # to grayscale
    obs = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)  # resize
    obs = np.expand_dims(obs, axis=0)  # [1, 84, 84]
    return obs / 255.0

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 7) DQN network class
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class DQN(nn.Module):
    """
    A simple DQN with 3 convolutional layers and 2 dense layers.
    Expects input_dim channels (usually 1 for grayscale) and returns
    output_dim actions (e.g. 7 for SIMPLE_MOVEMENT).
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(input_dim, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        return self.network(x)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 8) Hyperparameters and global variables
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
GAMMA = 0.99         # Discount factor
LR = 0.00025         # Learning rate
MEMORY_SIZE = 10000  # Replay memory size
BATCH_SIZE = 32
EPSILON_DECAY = 0.9995
MIN_EPSILON = 0.01
UPDATE_TARGET = 1000  # synchronize target net every 1000 steps

# Counters for training steps + epsilon value
steps_done = 0
epsilon = 1.0

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 9) Environment without RecordVideo - we record frames ourselves!
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
env = gym_super_mario_bros.make(
    "SuperMarioBros-v0",
    apply_api_compatibility=True,  # new Gym API (5 return values)
    render_mode="rgb_array"        # important so that obs is an RGB array
)
env = JoypadSpace(env, SIMPLE_MOVEMENT)

# Actions
num_actions = env.action_space.n

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 10) Initialize DQN and replay buffer
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
policy_net = DQN(1, num_actions).to(device)
target_net = DQN(1, num_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()  # We only use the target net for Q-targets

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
memory = deque(maxlen=MEMORY_SIZE)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 11) Action selection: Epsilon-Greedy
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def select_action(state_tensor):
    """
    For the current state (batch=1, channels=1, 84, 84), either choose
    a random action (with probability epsilon) or
    the best action according to policy_net (with probability 1-epsilon).
    """
    global epsilon, steps_done
    if random.random() < epsilon:
        return torch.tensor([[random.randrange(num_actions)]], dtype=torch.long, device=device)
    else:
        with torch.no_grad():
            return policy_net(state_tensor).max(dim=1)[1].view(1, 1)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 12) Training: Replay sampling + backprop
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return  # not enough data yet, no training
    batch = random.sample(memory, BATCH_SIZE)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.tensor(np.array(states), dtype=torch.float32, device=device)
    next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=device)
    actions = torch.tensor(actions, dtype=torch.long, device=device)
    rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
    dones = torch.tensor(dones, dtype=torch.float32, device=device)

    # Q(s,a)
    q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()
    # max Q(s', a')
    next_q_values = target_net(next_states).max(dim=1)[0]
    # Bellman update
    expected_q_values = rewards + (GAMMA * next_q_values * (1 - dones))

    loss = nn.functional.mse_loss(q_values, expected_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 13) Custom video recording with OpenCV VideoWriter
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
video_folder = f"./videos_custom_{int(time.time())}_{random.randint(0,9999)}"
os.makedirs(video_folder, exist_ok=True)

video_filename = os.path.join(video_folder, "mario_run.mp4")
writer = None  # VideoWriter is initialized only when we know the frame size

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 14) Main training function
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def main():
    global steps_done, epsilon, writer

    num_episodes = 20
    for episode in range(num_episodes):
        # Gym reset returns (obs, info), obs is in "rgb_array" format
        obs, info = env.reset()

        # If the VideoWriter doesn't exist yet, create it using the frame size
        if writer is None:
            height, width, channels = obs.shape  # e.g. (240, 256, 3)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = 30.0
            writer = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))

        # Convert state to grayscale, [1,84,84], Torch format
        state_arr = preprocess_observation(obs)
        state = torch.tensor(state_arr, dtype=torch.float32, device=device)

        done = False
        total_reward = 0.0

        # Loop until episode is finished
        while not done:
            # 1) Live display in the OpenCV window ("Mario")
            bgr_frame = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)  # Gym provides RGB, OpenCV wants BGR
            cv2.imshow("Mario", bgr_frame)
            cv2.waitKey(1)

            # 2) Write the frame to the video
            writer.write(bgr_frame)

            # 3) Select action (Epsilon-Greedy)
            action = select_action(state.unsqueeze(0))  # -> shape [1,1,84,84]
            next_obs, reward, terminated, truncated, info = env.step(action.item())
            done = terminated or truncated

            # 4) Next state
            next_state_arr = preprocess_observation(next_obs)
            next_state = torch.tensor(next_state_arr, dtype=torch.float32, device=device)

            # 5) Fill the ReplayMemory
            memory.append((
                state.cpu().numpy(),
                action.item(),
                reward,
                next_state.cpu().numpy(),
                float(done)
            ))

            # 6) Prepare for the next step
            state = next_state
            obs = next_obs
            total_reward += reward

            # 7) Training / Backprop
            optimize_model()
            steps_done += 1
            epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY)

            # 8) Update the target net
            if steps_done % UPDATE_TARGET == 0:
                target_net.load_state_dict(policy_net.state_dict())

        print(f"Episode {episode + 1}, Reward: {total_reward}")

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 15) Cleanup function (always called at the end)
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def cleanup():
    """
    Closes the environment, destroys OpenCV windows, and releases the VideoWriter
    so the MP4 video can be properly finalized.
    """
    print("Clean up: Closing environment, finalizing video ...")
    try:
        env.close()
    except Exception as e:
        print("Warning on env.close():", e)
    cv2.destroyAllWindows()
    global writer
    if writer is not None:
        writer.release()
    print(f"Video written to: {video_filename}")

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 16) Main guard (entry point) + exception catching
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nManual interrupt (Ctrl + C).")
    finally:
        # The finally block is always executed, regardless of normal exit or exception
        cleanup()
