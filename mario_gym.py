#!/usr/bin/env python3

"""
This script trains a simple DQN in the SuperMarioBros-v0 environment.
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
from datetime import datetime  # imported for timestamping log entries

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 3) External libraries (OpenCV, Gym, PyTorch, Numpy, etc.)
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import cv2
import gym
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from collections import deque  # Not used anymore, but kept for reference
from torchvision import transforms as T

# Suppress Gym warnings (optional)
warnings.filterwarnings("ignore", message="Overwriting existing videos")
warnings.filterwarnings("ignore", message="The result returned by `env.reset()` was not a tuple")
warnings.filterwarnings("ignore", message="Disabling video recorder")
warnings.filterwarnings("ignore", message="No render modes was declared")
warnings.filterwarnings("ignore", message="Core environment is written in old step API")
warnings.filterwarnings("ignore", message="`np.bool8` is a deprecated alias for `np.bool_`")
warnings.filterwarnings("ignore", message="The environment creator metadata doesn't include `render_modes`")
warnings.filterwarnings("ignore", message="The environment SuperMarioBros-v0 is out of date. You should consider upgrading to version `v3`.")
warnings.filterwarnings("ignore", message="The environment creator metadata doesn't include `render_modes`, contains: ['render.modes', 'video.frames_per_second']")

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 4) Imports specific to gym_super_mario_bros
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

# Note: If MoviePy is not installed, you can install it via pip.
try:
    import moviepy  # noqa
except ImportError:
    print("Warning: MoviePy is not installed! Please run 'pip install moviepy'")
    print("if you want to further process the generated .mp4 later.\n")

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 5) Device selection (CPU / CUDA / Apple Metal if available)
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple Metal (MPS) device.")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using NVIDIA CUDA device.")
else:
    device = torch.device("cpu")
    print("Using CPU device.")

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 6) Hyperparameters and global variables
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
GAMMA = 0.99            # Discount factor
LR = 0.00025            # Initial learning rate
MEMORY_SIZE = 10000     # Replay memory size
BATCH_SIZE = 32
EPSILON_DECAY = 0.995   # Epsilon decay rate
MIN_EPSILON = 0.01
UPDATE_TARGET = 500     # Frequency of target network update
STACK_SIZE = 4          # Number of frames to stack for state representation

# Counters for training steps and epsilon value
steps_done = 0
epsilon = 1.0

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 7) Helper function: Frame preprocessing and stacking
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def preprocess_observation(obs):
    """
    Converts the input frame (RGB) to grayscale,
    resizes it to [84x84], and returns a numpy array
    with shape [1, 84, 84] (1 channel).
    Then, normalizes pixel values from 0..255 to 0..1.
    """
    if obs.ndim == 3:
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    obs = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
    obs = np.expand_dims(obs, axis=0)
    return obs / 255.0

def stack_frames(frame_buffer, new_frame, is_new_episode):
    """
    Stacks frames to create a state representation.
    If is_new_episode is True, initializes the frame buffer with the same frame.
    Otherwise, appends the new frame to the buffer and removes the oldest frame.
    Returns the updated frame buffer and the stacked state.
    """
    processed = preprocess_observation(new_frame)  # shape (1, 84, 84)
    if is_new_episode or frame_buffer is None:
        frame_buffer = [processed for _ in range(STACK_SIZE)]
    else:
        frame_buffer.append(processed)
        frame_buffer.pop(0)
    # Concatenate along the first dimension to form a stacked state (STACK_SIZE, 84, 84)
    stacked_state = np.concatenate(frame_buffer, axis=0)
    return frame_buffer, stacked_state

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 8) DQN network class
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class DQN(nn.Module):
    """
    A simple DQN with 3 convolutional layers and 2 dense layers.
    Expects input_dim channels (STACK_SIZE for frame stacking) and returns
    output_dim actions.
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
# 9) Prioritized Replay Buffer Class
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class PrioritizedReplayBuffer:
    """
    A simple implementation of Prioritized Experience Replay.
    Stores transitions with associated priorities.
    """
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = []
        self.pos = 0

    def push(self, transition):
        """Add a new transition with maximum priority."""
        max_priority = max(self.priorities, default=1.0)
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
            self.priorities.append(max_priority)
        else:
            self.buffer[self.pos] = transition
            self.priorities[self.pos] = max_priority
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        """Sample a batch of transitions with importance-sampling weights."""
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
        """Update priorities of sampled transitions."""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

    def __len__(self):
        return len(self.buffer)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 10) Environment without RecordVideo - we record frames manually!
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
env = gym_super_mario_bros.make(
    "SuperMarioBros-v0",
    apply_api_compatibility=True,  # new Gym API (5 return values)
    render_mode="rgb_array"        # ensures obs is an RGB array
)
env = JoypadSpace(env, SIMPLE_MOVEMENT)
num_actions = env.action_space.n

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 11) Initialize DQN, Target Network, Optimizer, Scheduler and Replay Buffer
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
policy_net = DQN(STACK_SIZE, num_actions).to(device)
target_net = DQN(STACK_SIZE, num_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
# Learning Rate Scheduler: Decays the learning rate every 1000 steps by gamma=0.99
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.99)

# Use Prioritized Experience Replay Buffer instead of a simple deque
memory = PrioritizedReplayBuffer(MEMORY_SIZE, alpha=0.6)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 12) Action selection: Epsilon-Greedy
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def select_action(state_tensor):
    """
    Selects an action using an epsilon-greedy policy.
    With probability epsilon, a random action is chosen.
    Otherwise, the best action is chosen according to policy_net.
    """
    global epsilon, steps_done
    if random.random() < epsilon:
        return torch.tensor([[random.randrange(num_actions)]], dtype=torch.long, device=device)
    else:
        with torch.no_grad():
            return policy_net(state_tensor).max(dim=1)[1].view(1, 1)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 13) Training: Replay sampling, optimization, and prioritized updates
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def optimize_model():
    global steps_done, epsilon
    if len(memory) < BATCH_SIZE:
        return  # Not enough samples yet
    beta = 0.4  # Importance-sampling beta parameter
    batch, indices, weights = memory.sample(BATCH_SIZE, beta)
    
    # Unpack batch and convert to tensors using torch.from_numpy for optimization
    states, actions, rewards, next_states, dones = zip(*batch)
    states = torch.from_numpy(np.stack(states)).float().to(device)
    next_states = torch.from_numpy(np.stack(next_states)).float().to(device)
    actions = torch.tensor(actions, dtype=torch.long, device=device)
    rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
    dones = torch.tensor(dones, dtype=torch.float32, device=device)
    weights_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    
    # Compute Q(s, a) and Q(s', a')
    q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()
    next_q_values = target_net(next_states).max(dim=1)[0]
    
    # Compute expected Q values using the Bellman equation
    expected_q_values = rewards + (GAMMA * next_q_values * (1 - dones))
    
    # Compute TD error and use Huber loss (smooth L1 loss) with element-wise reduction
    td_errors = torch.abs(q_values - expected_q_values).detach()
    loss = torch.nn.functional.smooth_l1_loss(q_values, expected_q_values, reduction='none')
    
    # Apply importance-sampling weights to stabilize learning
    loss = (loss * weights_tensor).mean()
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()  # Update learning rate scheduler
    
    # Update priorities in the replay buffer
    new_priorities = td_errors.cpu().numpy() + 1e-6  # small constant to avoid zero priority
    memory.update_priorities(indices, new_priorities)
    
    steps_done += 1
    epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 14) Custom video recording with OpenCV VideoWriter
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
run_folder = f"./{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_run"
os.makedirs(run_folder, exist_ok=True)
video_filepath = os.path.join(run_folder, "mario_run.mp4")
writer = None  # VideoWriter will be initialized after determining frame size

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 15) Checkpoint saving and loading
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def save_checkpoint(filename="checkpoint.pth"):
    """
    Saves the current training state including network weights,
    optimizer, scheduler state, epsilon, steps_done, and replay memory.
    """
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
    """
    Loads a checkpoint and restores network weights, optimizer,
    scheduler, epsilon, steps_done, and replay memory.
    """
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

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 16) Main training function
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def main():
    global steps_done, epsilon, writer
    num_episodes = 500
    zoom_factor = 2.0  # Scaling factor for display and video recording
    
    best_episode_reward = float("-inf")
    best_video_filepath = None
    log_filename = os.path.join(run_folder, "log.txt")
    log_file = open(log_filename, "a")
    
    for episode in range(num_episodes):
        episode_frames = []
        obs, info = env.reset()
        
        # Initialize VideoWriter if not already done
        if writer is None:
            height, width, channels = obs.shape
            new_width = int(width * zoom_factor)
            new_height = int(height * zoom_factor)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = 30.0
            writer = cv2.VideoWriter(video_filepath, fourcc, fps, (new_width, new_height))
        
        # Initialize frame stack for the new episode
        frame_buffer, state_stack = stack_frames(None, obs, True)
        state = torch.from_numpy(state_stack).float().to(device)
        
        done = False
        total_reward = 0.0
        last_x_pos = 0
        
        while not done:
            # Convert RGB to BGR for OpenCV display
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
            writer.write(display_frame)
            episode_frames.append(display_frame)
            
            # Select an action using epsilon-greedy policy
            action = select_action(state.unsqueeze(0))
            try:
                next_obs, reward, terminated, truncated, info = env.step(action.item())
            except ValueError:
                # The environment is done, so break out of the loop gracefully
                break

            # Reward shaping: encourage moving right, penalize moving left or dying
            if info.get("x_pos", 0) > last_x_pos:
                reward += 1
            elif info.get("x_pos", 0) < last_x_pos:
                reward -= 1
            if terminated:
                reward -= 50
            if info.get("flag_get", False):
                reward += 1000
            reward = np.clip(reward / 10, -1, 1)
            last_x_pos = info.get("x_pos", 0)
            
            # Update frame stack with new observation
            frame_buffer, next_state_stack = stack_frames(frame_buffer, next_obs, False)
            next_state = torch.from_numpy(next_state_stack).float().to(device)
            
            # Store the transition in the replay buffer
            memory.push((
                state.cpu().numpy(),  # stacked state
                action.item(),
                reward,
                next_state.cpu().numpy(),  # stacked next state
                float(terminated or truncated)
            ))
            
            state = next_state
            obs = next_obs
            total_reward += reward
            
            optimize_model()
            
            if steps_done % UPDATE_TARGET == 0:
                target_net.load_state_dict(policy_net.state_dict())
            
            # Check if the episode has ended
            done = terminated or truncated
        
        # Save best video if the episode achieved a higher reward
        if total_reward > best_episode_reward:
            best_episode_reward = total_reward
            if best_video_filepath is not None and os.path.exists(best_video_filepath):
                os.remove(best_video_filepath)
            best_video_filepath = os.path.join(run_folder, f"mario_best_run_{total_reward:.2f}.mp4")
            # Print reward rounded to two decimals
            print(f"[INFO] New best episode with reward {total_reward:.2f}. Saving best run video as '{os.path.basename(best_video_filepath)}'.")
            if len(episode_frames) > 0:
                height, width, _ = episode_frames[0].shape
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                fps = 30.0
                best_writer = cv2.VideoWriter(best_video_filepath, fourcc, fps, (width, height))
                for frame in episode_frames:
                    best_writer.write(frame)
                best_writer.release()
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Log reward rounded to two decimals
        log_message = f"[{timestamp}] Episode {episode + 1}, Reward: {total_reward:.2f}\n"
        print(log_message.strip())
        log_file.write(log_message)
        log_file.flush()
        
        if (episode + 1) % 50 == 0:
            save_checkpoint("checkpoint.pth")
    
    log_file.close()

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 17) Cleanup function
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def cleanup():
    """
    Closes the environment, destroys OpenCV windows, and releases the VideoWriter.
    """
    print("Cleanup: Closing environment and finalizing video...")
    try:
        env.close()
    except Exception as e:
        print("Warning during env.close():", e)
    cv2.destroyAllWindows()
    global writer
    if writer is not None:
        writer.release()
    print(f"Video written to: {video_filepath}")

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 18) Main guard and exception handling
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
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
