# Super Mario Dueling Double DQN – A Live & Recorded Reinforcement Learning Demo

This repository demonstrates an enhanced **Dueling Double DQN** training process on the classic *Super Mario Bros* environment using [gym_super_mario_bros](https://github.com/Kautenja/gym-super-mario-bros). The training process includes live gameplay visualization via an OpenCV window, periodic recording of episode runs, and automatic best-run extraction into `.mp4` video files.

![image](https://github.com/user-attachments/assets/32573929-c44e-47cb-8f35-628ef13a2f15)

> **Note:** Whether you stop the training normally or interrupt it (e.g., using **Ctrl + C**), the script ensures that the latest video and checkpoints are finalized correctly and the environment is properly closed.

---

## 1. Overview

- **Dueling Double DQN with Prioritized Replay:**  
  The script trains a DQN using a deeper network architecture with separate value and advantage streams. In addition, it employs a prioritized experience replay buffer with importance sampling, Huber (smooth L1) loss, and Polyak soft target updates for enhanced training stability.

- **Multiprocessing Compatibility:**  
  The script applies a fix by setting the multiprocessing start method to `spawn`, ensuring compatibility (especially on macOS).

- **Live Visualization & Recording:**  
  - **Real-Time Display:** An OpenCV window named **"Mario"** shows the gameplay in real time (displayed with a default 2× zoom for enhanced visibility). This can be turned off via the `SHOW_GUI` flag in the script for headless server training.  
  - **Periodic Recording:** Instead of writing a single huge file, the script records gameplay videos every 50 episodes (configurable via `RECORD_VIDEO_EVERY`) to reduce disk space. It buffers raw frames safely in RAM to prevent Out-of-Memory (OOM) errors.

- **Multi-Level Mixed Training:**  
  The agent is trained across multiple levels of World 1 (World 1-1, 1-2, 1-3, 1-4) selected randomly at the start of each episode. This allows the agent to generalize across different color schemes and gameplay structures (overworld, underworld, athletic platforms, and castle).

- **Best Run Extraction per Level:**  
  The script tracks the best run separately for each level and saves the gameplay video as `mario_best_run_<level>_<reward>.mp4` inside the run folder, providing clear level-by-level evaluation.

- **Custom Reward Shaping:**  
  The reward function has been tuned to guide the training process:
  - **Forward Progress:** A bonus of **0.1 × (distance advanced)** is added if Mario’s current `x_pos` exceeds the previous value.
  - **Life Loss Penalty:** Losing a life (while still having remaining lives) immediately terminates the episode and applies a **–50 penalty**.
  - **Level Completion Bonus:** Reaching the flag/axe gives a bonus of **+2000**.
  - **Clipping:** Rewards are clipped to the range **[–100, 2000]**.

- **Checkpointing & Resuming with Exploration Reset:**  
  - **Periodic Saving:** A checkpoint is automatically saved every 500 episodes (configurable via `SAVE_CHECKPOINT_EVERY`) as `checkpoint.pth`.
  - **Asynchronous & Memory-Optimized:** Checkpoint files are saved asynchronously in a background thread to prevent pausing the training loop, and the replay memory uses compact `uint8` structures.
  - **Final Checkpoint:** When the training stops (even via **Ctrl + C**), a final checkpoint is saved as `checkpoint_final.pth`.
  - **Loading & Exploration Reset:** On startup, you can load an existing checkpoint to resume training normally, or resume with an **exploration reset** (overriding epsilon to `0.10` and resetting learning rate to `0.0001` to prevent learning plateaus).

- **Automatic Device Selection:**  
  The script automatically detects and uses the best available device:
  1. **Apple Metal (MPS)** – for macOS users with Apple Silicon.
  2. **NVIDIA CUDA** – if a CUDA-capable GPU is available.
  3. **CPU** – as a fallback.

- **TensorBoard Logging & File Logging:**  
  Training metrics (Episode Reward, Loss, Q-Values, Learning Rate, and Epsilon) are logged to TensorBoard, and a `log.txt` file in the run folder records detailed episode logs.

---

## 2. Enhanced DQN Training Features

This implementation includes several improvements over the standard DQN algorithm:

- **Frame Stacking:**  
  Uses 4 consecutive frames as the state representation, providing temporal context to the agent.

- **Frame Skipping:**  
  Repeats the selected action for 4 frames (configurable via `FRAME_SKIP`), reducing neural network computational load and providing a wider motion representation.

- **Polyak Soft Target Updates:**  
  Updates target network weights slowly at every training step ($\tau = 0.005$) to stabilize value updates and avoid target value oscillations.

- **Beta Annealing:**  
  Gradually increases the importance sampling correction weight $\beta$ from 0.4 to 1.0 over the course of training steps.

- **Sum-Tree Prioritized Experience Replay:**  
  Implements prioritized experience replay using a Sum-Tree data structure. This allows sampling and priority update operations in $O(\log N)$ time, enabling the replay buffer size to scale up to 50,000 transitions (configurable via `MEMORY_SIZE`) without performance degradation. State transitions are stored as compact `uint8` arrays and normalized to `float32` on the fly on the GPU to drastically reduce memory usage.

- **Gradient Clipping:**  
  Clips network gradient norms to a maximum value of 10.0 to prevent exploding gradients caused by high shaped rewards (like the +2000 level completion bonus).

- **Exploration Warm-up Phase:**  
  Delays model training and epsilon decay for the first 2,000 environment steps (configurable via `INITIAL_EXPLORATION_STEPS`) to populate the replay buffer with purely random transitions.

- **Huber Loss:**  
  Replaces the Mean Squared Error (MSE) loss with the Huber (smooth L1) loss for improved training stability.

- **Learning Rate Scheduler:**  
  Introduces a learning rate scheduler to decay the learning rate over training steps, promoting better convergence.

- **Optimized Replay Buffer Conversion:**  
  Utilizes `torch.from_numpy` for efficient conversion of replay buffer batches, reducing computational overhead.

---

## 3. Installation

### 3.1 Using `requirements.txt` (Recommended)

If you already have a Python environment (Python 3.8 or higher), install all dependencies by running:

```bash
pip install -r requirements.txt
```

The `requirements.txt` includes pinned versions such as:

```text
gym==0.26.2
gym-super-mario-bros==7.4.0
torch==2.4.1
torchvision==0.19.1
opencv-python==4.11.0.86
numpy==1.24.4
moviepy==1.0.3
tensorboard  
six
```

### 3.2 Installing via Anaconda

Follow these steps to set up a **conda environment** (e.g., `mario_env`):

1. **Install [Anaconda](https://www.anaconda.com/download) or Miniconda** if you haven’t already.
2. **Create a new environment:**
   ```bash
   conda create -n mario_env python=3.8
   ```
3. **Activate the environment:**
   ```bash
   conda activate mario_env
   ```
4. **Install dependencies:**

   **Option A: Using conda channels**
   ```bash
   conda install -c conda-forge numpy pandas matplotlib
   conda install -c pytorch pytorch cpuonly  # or the appropriate CUDA version
   conda install -c conda-forge opencv gym gym_super_mario_bros
   conda install -c conda-forge moviepy ffmpeg
   conda install -c conda-forge tensorboard
   ```

   **Option B: Using pip**
   ```bash
   pip install torch torchvision opencv-python gym gym_super_mario_bros moviepy tensorboard
   ```
   *Alternatively, install exactly the pinned versions from `requirements.txt`:*
   ```bash
   pip install -r requirements.txt
   ```

5. **(macOS only) Fork-Safety Fix:**  
   If you experience segmentation faults on macOS, set the following environment variable before running the script:
   ```bash
   export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
   ```

### 3.3 Additional Windows Steps

For **Windows**, you might need to install Visual C++ Build Tools. For example, using [winget](https://learn.microsoft.com/en-us/windows/package-manager/winget/):

- **Windows 10:**
  ```bash
  winget install Microsoft.VisualStudio.2022.BuildTools --force --override "--wait --passive --add Microsoft.VisualStudio.Component.VC.Tools.x86.x64 --add Microsoft.VisualStudio.Component.Windows10SDK"
  ```
- **Windows 11:**
  ```bash
  winget install Microsoft.VisualStudio.2022.BuildTools --force --override "--wait --passive --add Microsoft.VisualStudio.Component.VC.Tools.x86.x64 --add Microsoft.VisualStudio.Component.Windows11SDK.22621"
  ```

For more details, refer to [How to install Visual C++ Build Tools](https://stackoverflow.com/questions/40504552/how-to-install-visual-c-build-tools).

---

## 4. Usage

1. **Clone or Download the Repository:**
   ```bash
   git clone https://github.com/yourusername/super-mario-dqn.git
   cd super-mario-dqn
   ```

2. **Activate the Environment (if using conda):**
   ```bash
   conda activate mario_env
   ```

3. **Run the Script:**
   ```bash
   python mario_gym.py
   ```
   - At startup, you will be prompted to choose between starting new training, loading a checkpoint, or resuming with an exploration reset:
     ```text
     Start new training (n), load checkpoint (l), or load and reset exploration (r)? [n/l/r]:
     ```
     You can also use command-line flags to skip the prompt:
     - `-n` or `--new`: Start new training.
     - `-l` or `--load`: Resume training from the checkpoint normally.
     - `-r` or `--resume-reset`: Resume training from the checkpoint and reset exploration (epsilon to `0.10` and learning rate to `0.0001`).
   - **Real-Time Display:** An OpenCV window named **"Mario"** will open, showing live gameplay with a 2× zoom (unless `SHOW_GUI` is set to `False` in the script).
   - **Video Recording:**  
     - A video is recorded periodically (every 50 episodes) and saved as `mario_episode_<episode>_<level>_reward_<reward>.mp4` inside the run folder.
     - The best-performing episode of each level is saved as `mario_best_run_<level>_<reward>.mp4`.
   - **Checkpointing:**  
     - Checkpoints are saved periodically (every 500 episodes by default) as `checkpoint.pth`.
     - Checkpoint writing is performed asynchronously in a background thread so training is not blocked.
     - Upon termination (even via **Ctrl + C**), a final checkpoint is saved synchronously as `checkpoint_final.pth`.

4. **Configurable Settings in `mario_gym.py`:**
   At the top of the file, you can customize:
   * `SHOW_GUI = True`: Set to `False` to run headless (e.g. on virtual machines/servers without a monitor).
   * `RECORD_VIDEO_EVERY = 50`: Freq of recording episode videos.
   * `MAX_EPISODE_FRAMES = 5000`: Caps maximum buffered video frames in RAM to prevent OOM.
   * `FRAME_SKIP = 4`: Frame skipping frequency.
   * `MEMORY_SIZE = 50000`: Capacity of the replay buffer.
   * `INITIAL_EXPLORATION_STEPS = 2000`: Warm-up steps using random actions before training starts.
   * `SAVE_CHECKPOINT_EVERY = 500`: Frequency (in episodes) for saving checkpoints.

5. **Output Observation:**
   - **OpenCV Window:** Displays real-time gameplay (when enabled).
   - **Terminal:** Shows episode logs, rewards, and progress messages.
   - **Run Folder:** Contains the recorded video files, TensorBoard logs, and a `log.txt` file with detailed episode logs.

---

## 5. TensorBoard Logging

The script logs key training metrics (such as Episode Reward, Loss, Q-Values, Learning Rate, and Epsilon) to TensorBoard. These logs are stored in the run folder alongside the video recordings.

### Installing TensorBoard

If TensorBoard is not installed yet, run:
```bash
pip install tensorboard
```

### Launching TensorBoard

To monitor training metrics:
1. Open a terminal and run, replacing `<run_folder>` with the path to your run folder:
   ```bash
   tensorboard --logdir <run_folder>
   ```
   For example:
   ```bash
   tensorboard --logdir ./2025-03-01_12-34-56_run
   ```
2. Open your web browser and navigate to [http://localhost:6006](http://localhost:6006) to view the TensorBoard dashboard.

---

## 6. Project Structure

The repository includes:

- **`mario_gym.py`**  
  The main training script, containing:
  - Multiprocessing start method fix for compatibility.
  - Automatic device selection (Apple MPS, CUDA, or CPU).
  - Dueling Double DQN architecture with value and advantage streams.
  - Prioritized experience replay implementation with Polyak soft updates.
  - Custom reward shaping that rewards forward progress, penalizes life loss, and heavily rewards level completion.
  - Live gameplay visualization via OpenCV and periodic memory-safe video recording.
  - Logging via TensorBoard and a local `log.txt` file.
  - Checkpoint saving and loading mechanisms.

- **`requirements.txt`**  
  Contains all required dependencies with pinned versions.

- **`README.md`**  
  Documentation and usage instructions for the project.

---

## 7. Checkpointing & Resuming

The script supports saving and resuming training through checkpoints:
- **Saving:**  
  - A checkpoint is automatically saved every 500 episodes (configurable via `SAVE_CHECKPOINT_EVERY`) as `checkpoint.pth`.
  - Checkpoint files are saved asynchronously in the background so that writing to disk does not interrupt training.
  - When interrupted (e.g., via **Ctrl + C**), a final checkpoint is saved synchronously as `checkpoint_final.pth`.
- **Resuming:**  
  - **Normal Resume (`l` / `--load`):** Restores model weights, optimizer state, replay memory, and steps_done directly from the checkpoint. 
  - **Resume & Exploration Reset (`r` / `--resume-reset`):** Restores weights and replay memory, but overrides epsilon back to `0.10` and resets the learning rate back to `0.0001` (by resetting the scheduler step count). This is useful to break out of learning plateaus caused by learning rate decay and low exploration rate.
  - Ensure the network architecture and list of levels remain unchanged between saving and loading.

---

## 8. Automatic Device Selection

At runtime, the script checks for the best available computation device:
1. **Apple Metal (MPS):** For macOS users with Apple Silicon.
2. **NVIDIA CUDA:** If a CUDA-capable GPU is available.
3. **CPU:** As a fallback if no GPU support is detected.

This ensures that model operations run optimally on your hardware.

---

## 9. Contributing

Feel free to open an **issue** or submit a **pull request** if you find bugs or have suggestions for additional features (e.g., alternative action spaces, advanced reward shaping, etc.).

---

**Enjoy training Mario with your enhanced Dueling Double DQN!**
