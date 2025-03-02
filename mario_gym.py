#!/usr/bin/env python3
"""
This script trains a Dueling Double DQN in the SuperMarioBros-v0 environment
with TensorBoard logging for training metrics.
"""

# =============================================================================
# Global Settings and Hyperparameters
# =============================================================================
# These settings control the behavior of the training and the DQN.
# Adjust these values to change the learning dynamics and experiment with different parameters.

GAMMA = 0.99            # Discount factor for future rewards.
LR = 0.00025            # Learning rate for the optimizer.
MEMORY_SIZE = 10000     # Maximum capacity of the replay memory.
BATCH_SIZE = 32         # Mini-batch size for training the network.
EPSILON_DECAY = 0.999   # Decay rate for the epsilon in the epsilon-greedy action selection.
MIN_EPSILON = 0.01      # Minimum value for epsilon to ensure some exploration.
UPDATE_TARGET = 100     # Frequency (in steps) to update the target network.
STACK_SIZE = 4          # Number of consecutive frames stacked together to form a state.
NUM_EPISODES = 10000    # Total number of training episodes.

# Global variables to track training progress.
steps_done = 0          # Total number of steps taken so far.
epsilon = 1.0           # Initial exploration rate for epsilon-greedy policy.

# =============================================================================
# Multiprocessing Fix
# =============================================================================
# Set the multiprocessing start method to 'spawn' to avoid issues on macOS (and sometimes Linux/Windows).
import multiprocessing
multiprocessing.set_start_method("spawn", force=True)

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
# These warnings may occur due to environment or video recording settings.
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
# Import libraries specific to the Super Mario Bros environment.
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

# If MoviePy is not installed, warn the user so they can install it for further video processing.
try:
    import moviepy  # noqa
except ImportError:
    print("Warning: MoviePy is not installed! Please run 'pip install moviepy'")
    print("if you want to further process the generated .mp4 later.\n")

# =============================================================================
# Device Selection (CPU / CUDA / Apple Metal)
# =============================================================================
# This block checks for available hardware accelerators and selects the best device.
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
# Helper Functions for Frame Preprocessing and Stacking
# =============================================================================
def preprocess_observation(obs):
    """
    Convert the input RGB frame to grayscale, resize it to 84x84,
    and normalize pixel values from 0-255 to 0-1.
    
    Args:
        obs (np.array): Input frame in RGB format.
        
    Returns:
        np.array: Processed frame with shape [1, 84, 84].
    """
    if obs.ndim == 3:
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    obs = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
    obs = np.expand_dims(obs, axis=0)
    return obs / 255.0

def stack_frames(frame_buffer, new_frame, is_new_episode):
    """
    Stack consecutive frames to form the state representation.
    
    If it's a new episode or no previous frames exist, initialize the buffer
    with the same processed frame; otherwise, append the new frame and remove
    the oldest frame.
    
    Args:
        frame_buffer (list or None): List of previous frames.
        new_frame (np.array): The latest frame from the environment.
        is_new_episode (bool): Flag indicating whether it's the start of a new episode.
        
    Returns:
        tuple: (updated frame buffer, stacked state as a numpy array with shape [STACK_SIZE, 84, 84])
    """
    processed = preprocess_observation(new_frame)  # Process the new frame.
    if is_new_episode or frame_buffer is None:
        frame_buffer = [processed for _ in range(STACK_SIZE)]
    else:
        frame_buffer.append(processed)
        frame_buffer.pop(0)
    # Concatenate the frames along the channel dimension.
    stacked_state = np.concatenate(frame_buffer, axis=0)
    return frame_buffer, stacked_state

# =============================================================================
# Dueling Double DQN Network Definition
# =============================================================================
class DQN(nn.Module):
    """
    A Dueling Double DQN model with shared convolutional layers for feature extraction
    and two separate streams for state-value and advantage estimation.
    The deeper architecture is designed to capture complex features from the input.
    """
    def __init__(self, input_dim, num_actions):
        """
        Initialize the DQN model.
        
        Args:
            input_dim (int): Number of input channels (e.g., STACK_SIZE).
            num_actions (int): Number of possible actions in the environment.
        """
        super(DQN, self).__init__()
        # Convolutional layers for feature extraction.
        self.feature = nn.Sequential(
            nn.Conv2d(input_dim, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        # Value stream: estimates the state value V(s).
        self.value_stream = nn.Sequential(
            nn.Linear(3136, 512),  # 3136 is the flattened feature size (assumed).
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        # Advantage stream: estimates the advantage for each action A(s,a).
        self.advantage_stream = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )

    def forward(self, x):
        """
        Forward pass of the DQN.
        
        Args:
            x (torch.Tensor): Input state tensor.
            
        Returns:
            torch.Tensor: Q-values for each action.
        """
        features = self.feature(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        # Combine the value and advantage streams:
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s)))
        q_vals = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_vals

# =============================================================================
# Prioritized Replay Buffer Implementation
# =============================================================================
class PrioritizedReplayBuffer:
    """
    A simple prioritized replay buffer that stores transitions along with their priorities.
    This enables sampling transitions based on their TD error.
    """
    def __init__(self, capacity, alpha=0.6):
        """
        Initialize the replay buffer.
        
        Args:
            capacity (int): Maximum number of transitions to store.
            alpha (float): How much prioritization is used (0 - no prioritization, 1 - full prioritization).
        """
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = []
        self.pos = 0

    def push(self, transition):
        """
        Add a new transition to the buffer with the maximum current priority.
        
        Args:
            transition (tuple): A tuple of (state, action, reward, next_state, done).
        """
        max_priority = max(self.priorities, default=1.0)
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
            self.priorities.append(max_priority)
        else:
            self.buffer[self.pos] = transition
            self.priorities[self.pos] = max_priority
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        """
        Sample a batch of transitions, computing importance-sampling weights.
        
        Args:
            batch_size (int): Number of samples to return.
            beta (float): Importance-sampling exponent.
        
        Returns:
            tuple: (samples, indices, importance-sampling weights)
        """
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
        """
        Update the priorities for a batch of transitions.
        
        Args:
            indices (list): Indices of the sampled transitions.
            priorities (list): New priority values.
        """
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

    def __len__(self):
        return len(self.buffer)

# =============================================================================
# Environment Initialization
# =============================================================================
# Create the Super Mario Bros environment with API compatibility and RGB output.
env = gym_super_mario_bros.make(
    "SuperMarioBros-v0",
    apply_api_compatibility=True,
    render_mode="rgb_array"
)
# Wrap the environment to restrict the action space to simple movements.
env = JoypadSpace(env, SIMPLE_MOVEMENT)
num_actions = env.action_space.n  # Get the number of possible actions.

# =============================================================================
# Initialize Networks, Optimizer, Scheduler, and Replay Buffer
# =============================================================================
# Create the policy and target networks and move them to the selected device.
policy_net = DQN(STACK_SIZE, num_actions).to(device)
target_net = DQN(STACK_SIZE, num_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())  # Initialize target network.
target_net.eval()  # Set target network to evaluation mode.

# Create the optimizer and learning rate scheduler.
optimizer = optim.Adam(policy_net.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.99)

# Initialize the prioritized replay buffer.
memory = PrioritizedReplayBuffer(MEMORY_SIZE, alpha=0.6)

# =============================================================================
# Epsilon-Greedy Action Selection Function
# =============================================================================
def select_action(state_tensor):
    """
    Select an action based on an epsilon-greedy policy.
    
    With probability epsilon, a random action is chosen to encourage exploration.
    Otherwise, the best action according to the policy network is selected.
    
    Args:
        state_tensor (torch.Tensor): Current state as a tensor.
        
    Returns:
        torch.Tensor: Chosen action as a tensor.
    """
    global epsilon, steps_done
    if random.random() < epsilon:
        # Choose a random action.
        return torch.tensor([[random.randrange(num_actions)]], dtype=torch.long, device=device)
    else:
        # Choose the action with the highest Q-value.
        with torch.no_grad():
            return policy_net(state_tensor).max(dim=1)[1].view(1, 1)

# =============================================================================
# Model Optimization Function (Training Step)
# =============================================================================
def optimize_model(tb_writer=None):
    """
    Perform a single optimization step:
    - Sample a batch from the replay buffer.
    - Compute the target Q-values using Double DQN.
    - Compute the loss (Huber loss weighted by importance-sampling weights).
    - Update network parameters and the learning rate scheduler.
    - Update priorities in the replay buffer.
    - Log training metrics to TensorBoard if a writer is provided.
    
    Args:
        tb_writer (SummaryWriter): TensorBoard writer for logging.
    """
    global steps_done, epsilon
    if len(memory) < BATCH_SIZE:
        return  # Not enough samples in the replay buffer yet.

    beta = 0.4  # Importance-sampling parameter.
    batch, indices, weights = memory.sample(BATCH_SIZE, beta)
    
    # Unpack the batch and convert data to tensors.
    states, actions, rewards, next_states, dones = zip(*batch)
    states = torch.from_numpy(np.stack(states)).float().to(device)
    next_states = torch.from_numpy(np.stack(next_states)).float().to(device)
    actions = torch.tensor(actions, dtype=torch.long, device=device)
    rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
    dones = torch.tensor(dones, dtype=torch.float32, device=device)
    weights_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    
    # Compute current Q-values for the actions taken.
    q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()
    
    # Double DQN: use the policy network to select actions and the target network to evaluate them.
    with torch.no_grad():
        next_state_actions = policy_net(next_states).max(dim=1)[1].unsqueeze(1)
        next_q_values = target_net(next_states).gather(1, next_state_actions).squeeze(1)
    
    # Compute the expected Q-values using the Bellman equation.
    expected_q_values = rewards + (GAMMA * next_q_values * (1 - dones))
    
    # Calculate the Huber loss (smooth L1 loss) weighted by importance-sampling weights.
    td_errors = torch.abs(q_values - expected_q_values).detach()
    loss = torch.nn.functional.smooth_l1_loss(q_values, expected_q_values, reduction='none')
    loss = (loss * weights_tensor).mean()
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    
    # Update the priorities in the replay buffer based on TD errors.
    new_priorities = td_errors.cpu().numpy() + 1e-6
    memory.update_priorities(indices, new_priorities)
    
    steps_done += 1
    epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY)
    
    # Log metrics to TensorBoard.
    if tb_writer is not None:
        tb_writer.add_scalar("Loss/Optimize", loss.item(), steps_done)
        tb_writer.add_scalar("Q_Value/Mean", q_values.mean().item(), steps_done)
        tb_writer.add_scalar("Learning_Rate", scheduler.get_last_lr()[0], steps_done)

# =============================================================================
# Custom Video Recording Setup with OpenCV
# =============================================================================
# Create a folder for this training run using the current timestamp.
run_folder = f"./{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_run"
os.makedirs(run_folder, exist_ok=True)
video_filepath = os.path.join(run_folder, "mario_run.mp4")
writer = None  # This will be initialized once the first frame is available.

# =============================================================================
# Checkpoint Saving and Loading Functions
# =============================================================================
def save_checkpoint(filename="checkpoint.pth"):
    """
    Save the current training state to a checkpoint file, including:
    - Policy network and target network weights.
    - Optimizer and scheduler state.
    - Epsilon, steps_done, and replay memory.
    
    Args:
        filename (str): File path to save the checkpoint.
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
    Load a checkpoint and restore the training state.
    
    Args:
        filename (str): File path from which to load the checkpoint.
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

# =============================================================================
# Main Training Loop
# =============================================================================
def main():
    """
    The main training loop which:
    - Runs for a specified number of episodes.
    - Resets the environment and initializes video recording.
    - Executes the epsilon-greedy policy for action selection.
    - Applies reward shaping and updates the replay buffer.
    - Periodically optimizes the model and updates the target network.
    - Logs training metrics to TensorBoard and saves the best run video.
    """
    global steps_done, epsilon, writer
    best_episode_reward = float("-inf")
    best_video_filepath = None
    log_filename = os.path.join(run_folder, "log.txt")
    log_file = open(log_filename, "a")
    
    # Initialize TensorBoard writer for logging.
    tb_writer = SummaryWriter(log_dir=run_folder)
    
    for episode in range(NUM_EPISODES):
        # Reset the environment at the beginning of each episode.
        obs, info = env.reset()
        
        # Initialize the VideoWriter on the first frame.
        if writer is None:
            height, width, channels = obs.shape
            zoom_factor = 2.0  # Factor to scale the display size.
            new_width = int(width * zoom_factor)
            new_height = int(height * zoom_factor)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = 30.0
            writer = cv2.VideoWriter(video_filepath, fourcc, fps, (new_width, new_height))
        
        episode_frames = []  # List to store frames for the current episode.
        frame_buffer, state_stack = stack_frames(None, obs, True)
        state = torch.from_numpy(state_stack).float().to(device)
        
        done = False
        total_reward = 0.0
        
        # Track Mario's last x-position and lives for custom reward shaping.
        last_x_pos = info.get("x_pos", 0)
        last_lives = info.get("life", 2)
        
        while not done:
            # Convert frame from RGB to BGR (as required by OpenCV) and resize for display.
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
            
            # Select an action using the epsilon-greedy policy.
            action = select_action(state.unsqueeze(0))
            next_obs, reward, terminated, truncated, info = env.step(action.item())
            
            # If Mario loses a life, end the episode early.
            current_lives = info.get("life", 2)
            if current_lives < last_lives and current_lives > 0:
                terminated = True
            
            # Reward shaping: reward forward movement, penalize life loss, and reward level completion.
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
            
            # Update the state by stacking the new observation.
            frame_buffer, next_state_stack = stack_frames(frame_buffer, next_obs, False)
            next_state = torch.from_numpy(next_state_stack).float().to(device)
            
            # Store the transition in the replay buffer.
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
            
            # Optimize the model after every step.
            optimize_model(tb_writer)
            
            # Update the target network periodically.
            if steps_done % UPDATE_TARGET == 0:
                target_net.load_state_dict(policy_net.state_dict())
            
            done = terminated or truncated
        
        # Log episode metrics to TensorBoard.
        tb_writer.add_scalar("Episode/Reward", total_reward, episode)
        tb_writer.add_scalar("Episode/Epsilon", epsilon, episode)
        
        # If this episode has the highest reward so far, save the run video.
        if total_reward > best_episode_reward:
            best_episode_reward = total_reward
            if best_video_filepath is not None and os.path.exists(best_video_filepath):
                os.remove(best_video_filepath)
            best_video_filepath = os.path.join(run_folder, f"mario_best_run_{total_reward:.2f}.mp4")
            print(f"[INFO] New best episode with reward {total_reward:.2f}. Saving best run video as '{os.path.basename(best_video_filepath)}'.")
            if len(episode_frames) > 0:
                height, width, _ = episode_frames[0].shape
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                fps = 30.0
                best_writer = cv2.VideoWriter(best_video_filepath, fourcc, fps, (width, height))
                for frame in episode_frames:
                    best_writer.write(frame)
                best_writer.release()
        
        # Log episode details to the console and a log file.
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] Episode {episode + 1}, Reward: {total_reward:.2f}"
        print(log_message)
        log_file.write(log_message + "\n")
        log_file.flush()
        
        # Save a checkpoint periodically.
        if (episode + 1) % 50 == 0:
            save_checkpoint("checkpoint.pth")
    
    log_file.close()
    tb_writer.close()

# =============================================================================
# Cleanup Function
# =============================================================================
def cleanup():
    """
    Perform cleanup operations:
    - Close the environment.
    - Destroy all OpenCV windows.
    - Release the VideoWriter resources.
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

# =============================================================================
# Main Guard and Exception Handling
# =============================================================================
if __name__ == "__main__":
    # Ask the user whether to start training from scratch or load a checkpoint.
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
