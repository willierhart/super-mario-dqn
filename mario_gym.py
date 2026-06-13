#!/usr/bin/env python3
"""
This script trains a Dueling Double DQN in the SuperMarioBros-v0 environment
with TensorBoard logging for training metrics.
It includes optimizations for frame skipping, memory usage, video recording,
and stable target network updates.
"""

# =============================================================================
# Global Settings and Hyperparameters
# =============================================================================
GAMMA = 0.99            # Discount factor for future rewards.
LR = 0.00025            # Learning rate for the optimizer.
MEMORY_SIZE = 10000     # Maximum capacity of the replay memory.
BATCH_SIZE = 32         # Mini-batch size for training the network.
EPSILON_DECAY = 0.99995 # Slower decay rate for epsilon per environment step (~90k steps to 0.01).
MIN_EPSILON = 0.01      # Minimum value for epsilon to ensure some exploration.
STACK_SIZE = 4          # Number of consecutive frames stacked together to form a state.
NUM_EPISODES = 10000    # Total number of training episodes.
OPTIMIZE_FREQ = 4       # Perform optimization step every 4 environment steps.
TAU = 0.005             # Soft target network update rate (Polyak averaging).
FRAME_SKIP = 4          # Number of frames to repeat action (Frame Skipping).

# Recording and Rendering config
RECORD_VIDEO_EVERY = 50 # Save a video of the episode every 50 episodes.
MAX_EPISODE_FRAMES = 5000 # Safety limit for in-memory frames to prevent OOM.
SHOW_GUI = True         # Show the OpenCV display window. Set to False for headless servers.

# Global variables to track training progress.
steps_done = 0          # Total number of environment steps taken so far.
epsilon = 1.0           # Initial exploration rate for epsilon-greedy policy.

# =============================================================================
# Multiprocessing Fix
# =============================================================================
# Set the multiprocessing start method to 'spawn' to avoid issues on macOS.
import multiprocessing
try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass

# =============================================================================
# Standard Library Imports
# =============================================================================
import warnings
import os
import time
import random
from datetime import datetime

# =============================================================================
# External Libraries Imports
# =============================================================================
import cv2
import gym
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from collections import deque
from torchvision import transforms as T

# Import TensorBoard's SummaryWriter for logging training metrics.
from torch.utils.tensorboard import SummaryWriter

# =============================================================================
# Suppress Specific Gym Warnings
# =============================================================================
warnings.filterwarnings("ignore", message="Overwriting existing videos")
warnings.filterwarnings("ignore", message="The result returned by `env.reset()` was not a tuple")
warnings.filterwarnings("ignore", message="Disabling video recorder")
warnings.filterwarnings("ignore", message="No render modes was declared")
warnings.filterwarnings("ignore", message="Core environment is written in old step API")
warnings.filterwarnings("ignore", message="`np.bool8` is a deprecated alias for `np.bool_`")
warnings.filterwarnings("ignore", message="The environment creator metadata doesn't include `render_modes`")
warnings.filterwarnings("ignore", message="The environment SuperMarioBros-v0 is out of date. You should consider upgrading to version `v3`.")
warnings.filterwarnings("ignore", message="The environment creator metadata doesn't include `render_modes`, contains: ['render.modes', 'video.frames_per_second']")

# =============================================================================
# gym_super_mario_bros and NES-Py Imports
# =============================================================================
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

# If MoviePy is not installed, warn the user.
try:
    import moviepy  # noqa
except ImportError:
    print("Warning: MoviePy is not installed! Please run 'pip install moviepy'")
    print("if you want to further process the generated .mp4 later.\n")

# =============================================================================
# Device Selection (CPU / CUDA / Apple Metal)
# =============================================================================
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple Metal (MPS) device.")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using NVIDIA CUDA device.")
else:
    device = torch.device("cpu")
    print("Using CPU device.")

# =============================================================================
# Environment Wrappers
# =============================================================================
class SkipFrame(gym.Wrapper):
    """
    Repeat action for `skip` frames.
    Returns the final frame and the sum of rewards.
    """
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info

# =============================================================================
# Helper Functions for Frame Preprocessing and Stacking
# =============================================================================
def preprocess_observation(obs):
    """
    Convert the input RGB frame to grayscale, resize it to 84x84,
    and normalize pixel values from 0-255 to 0-1.
    """
    if obs.ndim == 3:
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    obs = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
    obs = np.expand_dims(obs, axis=0)
    return obs / 255.0

def stack_frames(frame_buffer, new_frame, is_new_episode):
    """
    Stack consecutive frames to form the state representation.
    """
    processed = preprocess_observation(new_frame)
    if is_new_episode or frame_buffer is None:
        frame_buffer = [processed for _ in range(STACK_SIZE)]
    else:
        frame_buffer.append(processed)
        frame_buffer.pop(0)
    stacked_state = np.concatenate(frame_buffer, axis=0)
    return frame_buffer, stacked_state

# =============================================================================
# Dueling Double DQN Network Definition
# =============================================================================
class DQN(nn.Module):
    """
    Dueling DQN model architecture with shared convolutional layers
    and split Streams for State-Value V(s) and Advantage A(s,a).
    """
    def __init__(self, input_dim, num_actions):
        super(DQN, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(input_dim, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.value_stream = nn.Sequential(
            nn.Linear(3136, 512),  # 3136 is the flattened feature size (64 * 7 * 7)
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )

    def forward(self, x):
        features = self.feature(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        # Combine the streams: Q(s,a) = V(s) + (A(s,a) - mean(A(s)))
        q_vals = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_vals

# =============================================================================
# Prioritized Replay Buffer Implementation
# =============================================================================
class PrioritizedReplayBuffer:
    """
    Prioritized replay buffer with importance-sampling weight computation.
    """
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = []
        self.pos = 0

    def push(self, transition):
        max_priority = max(self.priorities, default=1.0)
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
            self.priorities.append(max_priority)
        else:
            self.buffer[self.pos] = transition
            self.priorities[self.pos] = max_priority
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            priorities = np.array(self.priorities)
        else:
            priorities = np.array(self.priorities[:len(self.buffer)])
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        samples = [self.buffer[idx] for idx in indices]
        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        return samples, indices, weights

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

    def __len__(self):
        return len(self.buffer)

# =============================================================================
# Environment Initialization
# =============================================================================
env = gym_super_mario_bros.make(
    "SuperMarioBros-v0",
    apply_api_compatibility=True,
    render_mode="rgb_array"
)
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = SkipFrame(env, skip=FRAME_SKIP)  # Apply frame skipping
num_actions = env.action_space.n

# =============================================================================
# Initialize Networks, Optimizer, Scheduler, and Replay Buffer
# =============================================================================
policy_net = DQN(STACK_SIZE, num_actions).to(device)
target_net = DQN(STACK_SIZE, num_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.99)

memory = PrioritizedReplayBuffer(MEMORY_SIZE, alpha=0.6)

# =============================================================================
# Epsilon-Greedy Action Selection Function
# =============================================================================
def select_action(state_tensor):
    global epsilon
    if random.random() < epsilon:
        return torch.tensor([[random.randrange(num_actions)]], dtype=torch.long, device=device)
    else:
        with torch.no_grad():
            return policy_net(state_tensor).max(dim=1)[1].view(1, 1)

# =============================================================================
# Model Optimization Function (Training Step)
# =============================================================================
def optimize_model(tb_writer=None):
    global steps_done
    if len(memory) < BATCH_SIZE:
        return

    # Anneal beta from 0.4 to 1.0 over the first 100,000 steps
    beta = min(1.0, 0.4 + (steps_done / 100000) * 0.6)
    batch, indices, weights = memory.sample(BATCH_SIZE, beta)
    
    states, actions, rewards, next_states, dones = zip(*batch)
    states = torch.from_numpy(np.stack(states)).float().to(device)
    next_states = torch.from_numpy(np.stack(next_states)).float().to(device)
    actions = torch.tensor(actions, dtype=torch.long, device=device)
    rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
    dones = torch.tensor(dones, dtype=torch.float32, device=device)
    weights_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    
    q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()
    
    # Double DQN Target Action selection & Evaluation
    with torch.no_grad():
        next_state_actions = policy_net(next_states).max(dim=1)[1].unsqueeze(1)
        next_q_values = target_net(next_states).gather(1, next_state_actions).squeeze(1)
    
    expected_q_values = rewards + (GAMMA * next_q_values * (1 - dones))
    
    td_errors = torch.abs(q_values - expected_q_values).detach()
    loss = torch.nn.functional.smooth_l1_loss(q_values, expected_q_values, reduction='none')
    loss = (loss * weights_tensor).mean()
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    
    new_priorities = td_errors.cpu().numpy() + 1e-6
    memory.update_priorities(indices, new_priorities)
    
    if tb_writer is not None:
        tb_writer.add_scalar("Loss/Optimize", loss.item(), steps_done)
        tb_writer.add_scalar("Q_Value/Mean", q_values.mean().item(), steps_done)
        tb_writer.add_scalar("Learning_Rate", scheduler.get_last_lr()[0], steps_done)

def soft_update(target_network, policy_network, tau):
    """
    Polyak update target network: theta_target = tau * theta_policy + (1 - tau) * theta_target
    """
    for target_param, policy_param in zip(target_network.parameters(), policy_network.parameters()):
        target_param.data.copy_(tau * policy_param.data + (1.0 - tau) * target_param.data)

# =============================================================================
# Helper function to save a list of raw frames to a video file
# =============================================================================
def save_video(frames, filepath, zoom_factor=2.0, fps=15.0):
    """
    Saves an array of raw RGB frames (usually 240x256) to an MP4 video file,
    applying zoom and BGR conversion on the fly.
    """
    if len(frames) == 0:
        return
    height, width, _ = frames[0].shape
    new_width = int(width * zoom_factor)
    new_height = int(height * zoom_factor)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    writer = cv2.VideoWriter(filepath, fourcc, fps, (new_width, new_height))
    
    for frame in frames:
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        display_frame = cv2.resize(
            bgr_frame,
            None,
            fx=zoom_factor,
            fy=zoom_factor,
            interpolation=cv2.INTER_NEAREST
        )
        writer.write(display_frame)
    writer.release()
    print(f"[INFO] Video saved to {filepath}")

# =============================================================================
# Checkpoint Saving and Loading Functions
# =============================================================================
def save_checkpoint(filename="checkpoint.pth"):
    checkpoint = {
        "policy_net": policy_net.state_dict(),
        "target_net": target_net.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epsilon": epsilon,
        "steps_done": steps_done,
        "memory": memory.buffer,
        "priorities": memory.priorities,
        "pos": memory.pos
    }
    torch.save(checkpoint, filename)
    print(f"[INFO] Checkpoint saved to {filename}")

def load_checkpoint(filename="checkpoint.pth"):
    global policy_net, target_net, optimizer, scheduler, epsilon, steps_done, memory
    if not os.path.exists(filename):
        print(f"[WARNING] Checkpoint file '{filename}' does not exist. Skipping load.")
        return
    checkpoint = torch.load(filename, map_location=device)
    policy_net.load_state_dict(checkpoint["policy_net"])
    target_net.load_state_dict(checkpoint["target_net"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    epsilon = checkpoint["epsilon"]
    steps_done = checkpoint["steps_done"]
    memory.buffer = checkpoint["memory"]
    memory.priorities = checkpoint["priorities"]
    memory.pos = checkpoint["pos"]
    print(f"[INFO] Checkpoint loaded from {filename}")

# =============================================================================
# Main Training Loop
# =============================================================================
# Create a folder for this training run
run_folder = f"./{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_run"
os.makedirs(run_folder, exist_ok=True)

def main():
    global steps_done, epsilon
    best_episode_reward = float("-inf")
    log_filename = os.path.join(run_folder, "log.txt")
    log_file = open(log_filename, "a")
    
    tb_writer = SummaryWriter(log_dir=run_folder)
    zoom_factor = 2.0
    
    for episode in range(NUM_EPISODES):
        obs, info = env.reset()
        
        # We store raw observations in memory (low RAM footprint)
        episode_frames = []
        
        frame_buffer, state_stack = stack_frames(None, obs, True)
        state = torch.from_numpy(state_stack).float().to(device)
        
        done = False
        total_reward = 0.0
        
        last_x_pos = info.get("x_pos", 0)
        last_lives = info.get("life", 2)
        
        while not done:
            # Optionally display game in a GUI window
            if SHOW_GUI:
                bgr_frame = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
                display_frame = cv2.resize(
                    bgr_frame,
                    None,
                    fx=zoom_factor,
                    fy=zoom_factor,
                    interpolation=cv2.INTER_NEAREST
                )
                cv2.imshow("Mario", display_frame)
                cv2.waitKey(1)
            
            # Store raw frame for potential video recording (limit size for safety)
            if len(episode_frames) < MAX_EPISODE_FRAMES:
                episode_frames.append(obs.copy())
            
            # Select action
            action = select_action(state.unsqueeze(0))
            next_obs, reward, terminated, truncated, info = env.step(action.item())
            
            # Decay epsilon on every environment interaction step
            steps_done += 1
            epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY)
            
            # Lose life ends episode early
            current_lives = info.get("life", 2)
            if current_lives < last_lives and current_lives > 0:
                terminated = True
            
            # Reward shaping
            current_x_pos = info.get("x_pos", 0)
            if current_x_pos > last_x_pos:
                reward += (current_x_pos - last_x_pos) * 0.1
            if current_lives < last_lives:
                reward -= 50
            if info.get("flag_get", False):
                reward += 2000
            
            reward = np.clip(reward, -100, 2000)
            last_x_pos = current_x_pos
            last_lives = current_lives
            
            frame_buffer, next_state_stack = stack_frames(frame_buffer, next_obs, False)
            next_state = torch.from_numpy(next_state_stack).float().to(device)
            
            done_float = float(terminated or truncated)
            memory.push((
                state.cpu().numpy(),
                action.item(),
                reward,
                next_state.cpu().numpy(),
                done_float
            ))
            
            state = next_state
            obs = next_obs
            total_reward += reward
            
            # Optimize and soft update the networks every OPTIMIZE_FREQ steps
            if steps_done % OPTIMIZE_FREQ == 0:
                optimize_model(tb_writer)
                soft_update(target_net, policy_net, TAU)
            
            done = terminated or truncated
        
        # Log episode metrics
        tb_writer.add_scalar("Episode/Reward", total_reward, episode)
        tb_writer.add_scalar("Episode/Epsilon", epsilon, episode)
        
        # Check if we should save this episode's video (scheduled)
        should_record = (episode + 1) % RECORD_VIDEO_EVERY == 0
        if should_record:
            rec_path = os.path.join(run_folder, f"mario_episode_{episode + 1}_reward_{total_reward:.2f}.mp4")
            save_video(episode_frames, rec_path, fps=15.0)
            
        # Check if this is the best run and save if so
        if total_reward > best_episode_reward:
            best_episode_reward = total_reward
            best_path = os.path.join(run_folder, f"mario_best_run_{total_reward:.2f}.mp4")
            print(f"[INFO] New best episode with reward {total_reward:.2f}. Saving video.")
            save_video(episode_frames, best_path, fps=15.0)
        
        # Log details to console and file
        timestamp = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        log_message = f"[{timestamp}] Episode {episode + 1}, Steps: {steps_done}, Epsilon: {epsilon:.4f}, Reward: {total_reward:.2f}"
        print(log_message)
        log_file.write(log_message + "\n")
        log_file.flush()
        
        # Save checkpoints periodically
        if (episode + 1) % 50 == 0:
            save_checkpoint("checkpoint.pth")
            
    log_file.close()
    tb_writer.close()

# =============================================================================
# Cleanup Function
# =============================================================================
def cleanup():
    print("Cleanup: Closing environment and destroying windows...")
    try:
        env.close()
    except Exception as e:
        print("Warning during env.close():", e)
    cv2.destroyAllWindows()

# =============================================================================
# Main Guard
# =============================================================================
if __name__ == "__main__":
    user_input = input("Start new training (n) or load from checkpoint (l)? [n/l]: ")
    if user_input.lower() == 'l':
        print("Loading checkpoint...")
        load_checkpoint("checkpoint.pth")
    else:
        print("Starting new training from scratch...")
    try:
        main()
    except KeyboardInterrupt:
        print("\nManual interrupt (Ctrl + C).")
    finally:
        save_checkpoint("checkpoint_final.pth")
        cleanup()
