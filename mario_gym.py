#!/usr/bin/env python3
"""
This script trains a Dueling Double DQN in the SuperMarioBros-v0 environment
with TensorBoard logging for training metrics.
It includes optimizations for frame skipping, memory usage, video recording,
stable target network updates, gradient clipping, and a Sum-Tree based
Prioritized Replay Buffer for performance scaling.
"""

# =============================================================================
# Global Settings and Hyperparameters
# =============================================================================
GAMMA = 0.99            # Discount factor for future rewards.
LR = 0.00025            # Learning rate for the optimizer.
MEMORY_SIZE = 50000     # Increased capacity of the replay memory (scaled thanks to Sum-Tree).
BATCH_SIZE = 32         # Mini-batch size for training the network.
EPSILON_DECAY = 0.99995 # Slower decay rate for epsilon per environment step (~90k steps to 0.01).
MIN_EPSILON = 0.01      # Minimum value for epsilon to ensure some exploration.
STACK_SIZE = 4          # Number of consecutive frames stacked together to form a state.
NUM_EPISODES = 10000    # Total number of training episodes.
OPTIMIZE_FREQ = 4       # Perform optimization step every 4 environment steps.
TAU = 0.005             # Soft target network update rate (Polyak averaging).
FRAME_SKIP = 4          # Number of frames to repeat action (Frame Skipping).

# Multi-level Mixed Training configuration
TRAINING_MODE = "mixed" # "mixed" for random selection, "curriculum" for step-by-step
LEVELS = ["SuperMarioBros-1-1-v0", "SuperMarioBros-1-2-v0", "SuperMarioBros-1-3-v0", "SuperMarioBros-1-4-v0"]

# Warm-up phase configuration
INITIAL_EXPLORATION_STEPS = 2000 # Number of steps to collect random experiences before learning starts.

# Recording and Rendering config
RECORD_VIDEO_EVERY = 50 # Save a video of the episode every 50 episodes.
MAX_EPISODE_FRAMES = 5000 # Safety limit for in-memory frames to prevent OOM.
SHOW_GUI = True         # Show the OpenCV display window. Set to False for headless servers.

# Checkpoint configuration
SAVE_CHECKPOINTS = True  # Set to True to enable saving checkpoints (network weights/state).
SAVE_REPLAY_BUFFER = False # Set to True to save replay buffer data (requires ~2.8GB of disk space when enabled).
SAVE_CHECKPOINT_EVERY = 500 # Save checkpoint every N episodes.

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
import threading
import copy
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
    Convert the input RGB frame to grayscale and resize it to 84x84.
    Returns a uint8 numpy array to save memory.
    """
    if obs.ndim == 3:
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    obs = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
    obs = np.expand_dims(obs, axis=0)
    return obs

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
# Sum-Tree Data Structure for Prioritized Replay Buffer
# =============================================================================
class SumTree:
    """
    A binary tree data structure where parent nodes are the sum of their children.
    Enables O(log N) sampling and updates.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])

# =============================================================================
# Prioritized Replay Buffer using Sum-Tree
# =============================================================================
class PrioritizedReplayBuffer:
    """
    Prioritized replay buffer with O(log N) operations using Sum-Tree.
    """
    def __init__(self, capacity, alpha=0.6):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.epsilon = 1e-5
        self.max_priority = 1.0

    def push(self, transition):
        self.tree.add(self.max_priority, transition)

    def sample(self, batch_size, beta=0.4):
        batch = []
        indices = []
        priorities = []
        segment = self.tree.total() / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            indices.append(idx)

        # Compute importance-sampling weights
        sampling_probabilities = np.array(priorities) / (self.tree.total() if self.tree.total() > 0 else 1.0)
        weights = (self.tree.n_entries * sampling_probabilities) ** (-beta)
        weights /= weights.max() if weights.max() > 0 else 1.0

        return batch, indices, weights

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            p = (priority + self.epsilon) ** self.alpha
            self.tree.update(idx, p)
            self.max_priority = max(self.max_priority, p)

    def __len__(self):
        return self.tree.n_entries

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
    if len(memory) < INITIAL_EXPLORATION_STEPS:
        return

    # Anneal beta from 0.4 to 1.0 over the first 100,000 steps
    beta = min(1.0, 0.4 + (steps_done / 100000) * 0.6)
    batch, indices, weights = memory.sample(BATCH_SIZE, beta)
    
    states, actions, rewards, next_states, dones = zip(*batch)
    states = torch.from_numpy(np.stack(states)).float().to(device) / 255.0
    next_states = torch.from_numpy(np.stack(next_states)).float().to(device) / 255.0
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
    
    # Clip gradients to prevent explosion due to large shaped rewards
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=10.0)
    
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
def save_checkpoint(filename="checkpoint.pth", is_async=True):
    if not SAVE_CHECKPOINTS:
        return
    
    # Clone state dicts and deepcopy other states on CPU to avoid race conditions during training
    checkpoint_cpu = {
        "policy_net": {k: v.cpu().clone() for k, v in policy_net.state_dict().items()},
        "target_net": {k: v.cpu().clone() for k, v in target_net.state_dict().items()},
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "scheduler": copy.deepcopy(scheduler.state_dict()),
        "epsilon": epsilon,
        "steps_done": steps_done,
    }
    if SAVE_REPLAY_BUFFER:
        checkpoint_cpu.update({
            "buffer_tree": memory.tree.tree.copy(),
            "buffer_data": copy.deepcopy(memory.tree.data),
            "buffer_write": memory.tree.write,
            "buffer_n_entries": memory.tree.n_entries,
            "buffer_max_priority": memory.max_priority
        })
        
    def save_job(cp_dict, path):
        try:
            torch.save(cp_dict, path)
            print(f"[INFO] Checkpoint saved to {path}")
        except Exception as e:
            print(f"[ERROR] Failed to save checkpoint to {path}: {e}")

    if is_async:
        print(f"[INFO] Starting asynchronous checkpoint save to {filename}...")
        threading.Thread(target=save_job, args=(checkpoint_cpu, filename), daemon=True).start()
    else:
        save_job(checkpoint_cpu, filename)

def load_checkpoint(filename="checkpoint.pth", reset_exploration=False):
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
    
    # Restore SumTree state only if it is present in the checkpoint
    if "buffer_tree" in checkpoint:
        memory.tree.tree = checkpoint["buffer_tree"]
        
        # Convert loaded float state arrays to uint8 if they are floats to free up memory immediately!
        loaded_data = checkpoint["buffer_data"]
        converted_count = 0
        for i in range(len(loaded_data)):
            if loaded_data[i] is not None:
                state, action, reward, next_state, done = loaded_data[i]
                if state.dtype != np.uint8:
                    if state.max() <= 1.0:
                        state = (state * 255.0).astype(np.uint8)
                        next_state = (next_state * 255.0).astype(np.uint8)
                    else:
                        state = state.astype(np.uint8)
                        next_state = next_state.astype(np.uint8)
                    loaded_data[i] = (state, action, reward, next_state, done)
                    converted_count += 1
                    
        if converted_count > 0:
            print(f"[INFO] Converted {converted_count} buffer entries from float to uint8 for memory optimization.")
            
        memory.tree.data = loaded_data
        memory.tree.write = checkpoint["buffer_write"]
        memory.tree.n_entries = checkpoint["buffer_n_entries"]
        memory.max_priority = checkpoint.get("buffer_max_priority", 1.0)
        print(f"[INFO] Checkpoint and Replay Buffer loaded from {filename}")
    else:
        print(f"[INFO] Checkpoint loaded (without Replay Buffer) from {filename}")

    if reset_exploration:
        # Reset exploration rate to allow finding new strategies in challenging levels
        epsilon = 0.10
        # Reset learning rate in the optimizer to a healthy learning speed
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0001
        # Reinitialize scheduler settings with reset learning rate and step counter
        scheduler.base_lrs = [0.0001]
        scheduler.last_epoch = 0
        print(f"[INFO] Exploration reset: Epsilon set to {epsilon:.2f}, Learning Rate reset to {optimizer.param_groups[0]['lr']}")

# =============================================================================
# Main Training Loop
# =============================================================================
# Create a folder for this training run
run_folder = f"./{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_run"
os.makedirs(run_folder, exist_ok=True)

def main():
    global steps_done, epsilon, env
    best_rewards = {level: float("-inf") for level in LEVELS}
    current_level_idx = 0
    curriculum_rewards = []
    
    log_filename = os.path.join(run_folder, "log.txt")
    log_file = open(log_filename, "a")
    
    tb_writer = SummaryWriter(log_dir=run_folder)
    zoom_factor = 2.0
    
    for episode in range(NUM_EPISODES):
        # Select level
        if TRAINING_MODE == "mixed":
            level_name = random.choice(LEVELS)
        else:  # curriculum
            level_name = LEVELS[current_level_idx]
            
        # Recreate environment for selected level to clear state
        try:
            env.close()
        except Exception:
            pass
        env = gym_super_mario_bros.make(
            level_name,
            apply_api_compatibility=True,
            render_mode="rgb_array"
        )
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        env = SkipFrame(env, skip=FRAME_SKIP)
        
        obs, info = env.reset()
        
        # We store raw observations in memory (low RAM footprint)
        episode_frames = []
        
        frame_buffer, state_stack = stack_frames(None, obs, True)
        state = torch.from_numpy(state_stack).float().to(device) / 255.0
        
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
            
            # Increment environment steps
            steps_done += 1
            
            # Decay epsilon only after warm-up phase (exploration steps completed)
            if steps_done > INITIAL_EXPLORATION_STEPS:
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
            next_state = torch.from_numpy(next_state_stack).float().to(device) / 255.0
            
            done_float = float(terminated or truncated)
            memory.push((
                state_stack,
                action.item(),
                reward,
                next_state_stack,
                done_float
            ))
            
            state = next_state
            state_stack = next_state_stack
            obs = next_obs
            total_reward += reward
            
            # Optimize and soft update the networks every OPTIMIZE_FREQ steps
            if steps_done % OPTIMIZE_FREQ == 0:
                optimize_model(tb_writer)
                soft_update(target_net, policy_net, TAU)
            
            done = terminated or truncated
        
        # Extract level suffix for file names and logging (e.g. 1-1, 1-2)
        level_suffix = level_name.replace("SuperMarioBros-", "").replace("-v0", "")
        
        # Log episode metrics
        tb_writer.add_scalar("Episode/Reward", total_reward, episode)
        tb_writer.add_scalar(f"Episode_Reward/{level_suffix}", total_reward, episode)
        tb_writer.add_scalar("Episode/Epsilon", epsilon, episode)
        
        # Check if we should save this episode's video (scheduled)
        should_record = (episode + 1) % RECORD_VIDEO_EVERY == 0
        if should_record:
            rec_path = os.path.join(run_folder, f"mario_episode_{episode + 1}_{level_suffix}_reward_{total_reward:.2f}.mp4")
            save_video(episode_frames, rec_path, fps=15.0)
            
        # Check if this is the best run for this specific level and save if so
        if total_reward > best_rewards[level_name]:
            best_rewards[level_name] = total_reward
            best_path = os.path.join(run_folder, f"mario_best_run_{level_suffix}_{total_reward:.2f}.mp4")
            print(f"[INFO] New best episode for level {level_suffix} with reward {total_reward:.2f}. Saving video.")
            save_video(episode_frames, best_path, fps=15.0)
        
        # Log details to console and file
        timestamp = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        log_message = f"[{timestamp}] Episode {episode + 1}, Level: {level_suffix}, Steps: {steps_done}, Epsilon: {epsilon:.4f}, Reward: {total_reward:.2f}"
        print(log_message)
        log_file.write(log_message + "\n")
        log_file.flush()
        
        # Update curriculum learning state if active
        if TRAINING_MODE == "curriculum":
            curriculum_rewards.append(total_reward)
            if len(curriculum_rewards) >= 10:
                avg_reward = np.mean(curriculum_rewards[-10:])
                if avg_reward > 1900:
                    if current_level_idx < len(LEVELS) - 1:
                        current_level_idx += 1
                        curriculum_rewards.clear()
                        print(f"[CURRICULUM] Level {LEVELS[current_level_idx-1]} solved (Avg: {avg_reward:.2f})! Advancing to {LEVELS[current_level_idx]}.")
        
        # Save checkpoints periodically
        if (episode + 1) % SAVE_CHECKPOINT_EVERY == 0:
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
    import sys
    mode = None
    if len(sys.argv) > 1:
        if sys.argv[1] in ['-l', '--load']:
            mode = 'l'
        elif sys.argv[1] in ['-r', '--resume-reset']:
            mode = 'r'
        elif sys.argv[1] in ['-n', '--new']:
            mode = 'n'
    
    if mode is None:
        user_input = input("Start new training (n), load checkpoint (l), or load and reset exploration (r)? [n/l/r]: ")
        mode = user_input.lower()
        
    if mode == 'l':
        print("Loading checkpoint...")
        load_checkpoint("checkpoint.pth", reset_exploration=False)
    elif mode == 'r':
        print("Loading checkpoint and resetting exploration...")
        load_checkpoint("checkpoint.pth", reset_exploration=True)
    else:
        print("Starting new training from scratch...")
    try:
        main()
    except KeyboardInterrupt:
        print("\nManual interrupt (Ctrl + C).")
    finally:
        save_checkpoint("checkpoint_final.pth", is_async=False)
        cleanup()
