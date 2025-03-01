# Super Mario DQN - A Live & Recorded Reinforcement Learning Demo

This repository demonstrates a **Deep Q-Network (DQN) training** process on the classic *Super Mario Bros* environment using [gym_super_mario_bros](https://github.com/Kautenja/gym-super-mario-bros). It **simultaneously** displays live gameplay in an OpenCV window **and** continuously records it into an `.mp4` video file. If you interrupt the script (e.g., with **Ctrl + C**), the video is properly finalized and not corrupted.

![image](https://github.com/user-attachments/assets/32573929-c44e-47cb-8f35-628ef13a2f15)

---

## 1. Overview

- **Train a DQN** in the [NES Super Mario Bros](https://github.com/Kautenja/gym-super-mario-bros) environment.
- **Live visualization** via an OpenCV window: watch Mario’s actions in real time (displayed with a default 2x zoom for enhanced visibility).
- **Continuous recording**: Each frame is directly encoded into an `.mp4` using OpenCV’s `VideoWriter`.  
  > **Note:** The video is saved as `mario_run.mp4` inside a run folder automatically named with the current date and time in the format `YYYY-MM-DD_HH-MM-SS_run`.
- **Best Run Recording**: In addition to the full run video, the script also tracks the episode with the highest total reward. The best episode is recorded separately and saved as `mario_best_run_<reward>.mp4` in the same run folder (where `<reward>` is the achieved reward, formatted to two decimal places). This allows you to quickly identify and replay the best performance.
- **Graceful shutdown**: Whether the training stops normally or via **Ctrl + C**, the environment and video file are closed correctly, leaving you with a playable recording.
- **Checkpointing & Resuming**: Save and load model checkpoints (network weights, optimizer state, training counters, replay memory, etc.) to resume training.
  - A checkpoint is automatically saved every 50 episodes as `checkpoint.pth`.
  - Upon termination (even via a KeyboardInterrupt), a final checkpoint is saved as `checkpoint_final.pth`.
  - **Important:** Due to the extensive replay memory and saved states, checkpoint files can grow to approximately **3.5GB** in size, each!
- **Automatic Device Selection**: The script automatically checks for Apple Metal (MPS) on macOS, NVIDIA GPU (CUDA), or falls back to CPU—whichever is available.
- **Logging**: Episode results (including total rewards and timestamps) are logged to a `log.txt` file in the same run folder where the video is saved.
- **Custom Reward Shaping**: The reward function has been tailored to guide the training process:
  - **Forward Progress**: If Mario's current `x_pos` is greater than in the previous step, a bonus of +1 is added.
  - **Regression Penalty**: Moving backward (a decrease in `x_pos`) incurs a penalty of -1.
  - **Death Penalty**: If Mario dies (i.e., the episode terminates), a significant penalty of -50 is applied.
  - **Goal Reward**: Reaching the flag (completing the level) grants a large bonus of +1000.
  - **Scaling**: Finally, rewards are scaled by dividing by 10 and clipped to the range `[-1, 1]` to stabilize training.

### Enhanced DQN Training Features

This implementation includes several improvements over the standard DQN algorithm:

- **Frame Stacking**: Uses 4 consecutive frames as the state representation, providing temporal context to the agent.
- **Prioritized Experience Replay**: Implements prioritized replay with importance sampling and dynamic priority updates to improve learning efficiency.
- **Huber Loss**: Replaces the Mean Squared Error (MSE) loss with the Huber (smooth L1) loss for improved training stability.
- **Learning Rate Scheduler**: Introduces a learning rate scheduler to decay the learning rate over training steps, promoting better convergence.
- **Optimized Replay Buffer Conversion**: Utilizes `torch.from_numpy` for efficient conversion of replay buffer batches, reducing computational overhead.

---

## 2. Installation

### 2.1 Using `requirements.txt` (recommended)

If you already have a Python 3.8 environment (e.g., via Anaconda or any other method), you can install all dependencies by simply running:

```bash
pip install -r requirements.txt
```

where the `requirements.txt` contains pinned versions:

```text
gym==0.26.2
gym-super-mario-bros==7.4.0
torch==2.4.1
torchvision==0.19.1
opencv-python==4.11.0.86
numpy==1.24.4
moviepy==1.0.3
tensorboard  # Ensure TensorBoard is installed
```

This ensures you get the exact versions used for this project, minimizing potential compatibility issues.

---

### 2.2 Installing via Anaconda (detailed steps)

Below are instructions on how to set up a **conda environment** (e.g., `mario_env`) with Python 3.8 and install the necessary packages:

1. **Install [Anaconda](https://www.anaconda.com/download) or Miniconda** if you haven't already.

2. **Create a new environment** (e.g., `mario_env`) with Python 3.8:
   ```bash
   conda create -n mario_env python=3.8
   ```
3. **Activate** the newly created environment:
   ```bash
   conda activate mario_env
   ```
4. **Install dependencies** using either `conda` or `pip`:

   **Option A: Using conda channels**
   ```bash
   # Basic packages
   conda install -c conda-forge numpy pandas matplotlib

   # PyTorch (CPU-only or GPU version)
   # Example for CPU version:
   conda install pytorch cpuonly -c pytorch

   # OpenCV
   conda install -c conda-forge opencv

   # Gym, gym_super_mario_bros, other RL packages
   conda install -c conda-forge gym gym_super_mario_bros

   # (Optional) For video processing with MoviePy
   conda install -c conda-forge moviepy ffmpeg

   # TensorBoard
   conda install -c conda-forge tensorboard
   ```

   **Option B: Using pip**
   ```bash
   pip install torch torchvision \
               opencv-python \
               gym \
               gym_super_mario_bros \
               moviepy \
               tensorboard
   ```

   *Alternatively, you can install exactly the pinned versions used in this repo via the `requirements.txt` method:*
   ```bash
   pip install -r requirements.txt
   ```
   
5. **Install PyTorch with CUDA (Optional, for GPU acceleration)**  
   If you have an NVIDIA GPU and want CUDA support, install PyTorch with the appropriate CUDA version. For example, for CUDA 11.8:
   ```bash
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```
   Make sure you have a compatible NVIDIA driver and CUDA toolkit for your GPU before installing.

6. **(macOS only) Potential fork-safety issues**:  
   If you run into segmentation faults on macOS, you may need:
   ```bash
   export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
   ```
   before running the script.

---

### 2.3 Additional Windows-Only Steps

If you are on **Windows**, you may need some extra setup:

1. **Install Visual C++ Build Tools**  
   This is required for compiling certain Python packages. Depending on your version of Windows, you can use one of the following commands (via [winget](https://learn.microsoft.com/en-us/windows/package-manager/winget/)):

   - **Windows 10**  
     ```bash
     winget install Microsoft.VisualStudio.2022.BuildTools --force --override "--wait --passive --add Microsoft.VisualStudio.Component.VC.Tools.x86.x64 --add Microsoft.VisualStudio.Component.Windows10SDK"
     ```

   - **Windows 11**  
     ```bash
     winget install Microsoft.VisualStudio.2022.BuildTools --force --override "--wait --passive --add Microsoft.VisualStudio.Component.VC.Tools.x86.x64 --add Microsoft.VisualStudio.Component.Windows11SDK.22621"
     ```

   For more details, see:  
   [How to install Visual C++ Build Tools (StackOverflow)](https://stackoverflow.com/questions/40504552/how-to-install-visual-c-build-tools)

---

## 3. Usage

1. **Clone or download** this repository:
   ```bash
   git clone https://github.com/willierhart/super-mario-dqn.git
   cd super-mario-dqn
   ```
2. **Activate** the `mario_env` environment again (if not already):
   ```bash
   conda activate mario_env
   ```
3. **Run the script** (e.g., `mario_gym.py`):
   ```bash
   python mario_gym.py
   ```
   - A window named **"Mario"** will pop up, showing the live environment (with a default 2x zoom in the display window).
   - The `.mp4` video is continuously recorded inside an automatically created folder named with the current date and time (e.g., `2025-02-17_14-35-22_run`) and saved as `mario_run.mp4`.
   - In addition, the best performing episode (i.e., the episode with the highest total reward) is recorded separately as `mario_best_run_<reward>.mp4` within the same folder.
   - Training stops once the specified number of episodes is completed **or** if you hit **Ctrl + C** (KeyboardInterrupt). In either case, the script cleans up gracefully, closes the environment, and finalizes the video.

### Observing the Output

- **OpenCV Window**: Real-time gameplay (displayed with a 2x zoom).
- **Terminal Output**: Logs about the current episode, total reward, etc.
- **Video Files**: 
  - `mario_run.mp4` for the full run.
  - `mario_best_run_<reward>.mp4` for the best episode.
- **Log File**: A `log.txt` is created in the run folder, logging episode results with timestamps.

---

## 4. TensorBoard Logging

This project includes TensorBoard logging to help you monitor key training metrics such as Episode Reward, Loss, Q-Values, and Epsilon. The logs are automatically saved into the run folder along with the video recordings.

### Installing TensorBoard

If you haven't installed TensorBoard yet, you can install it via pip:
  
```bash
pip install tensorboard
```

### Launching TensorBoard

After you start the training script, a new run folder (e.g., `YYYY-MM-DD_HH-MM-SS_run`) is created with all TensorBoard logs. To view these metrics:

1. Open a terminal and run the following command, replacing `<run_folder>` with the path to your run folder:
   ```bash
   tensorboard --logdir <run_folder>
   ```
   For example:
   ```bash
   tensorboard --logdir ./2025-03-01_12-34-56_run
   ```

2. Open your web browser and navigate to [http://localhost:6006](http://localhost:6006) to access the TensorBoard dashboard. Here you will see graphs of your training metrics in real time.

---

## 5. Project Structure

Typical files in this repository include:

- **`mario_gym.py`**  
  The main script containing:
  - Automatic device selection (Apple Metal MPS, CUDA, or CPU)
  - Multiprocessing compatibility fix (`spawn` start method)
  - Checkpoint saving/loading (resuming training)
  - DQN architecture and replay buffer logic
  - **Enhanced DQN Features**: Frame stacking, prioritized experience replay, Huber loss, learning rate scheduler, and optimized replay buffer batch conversion.
  - Custom Reward Shaping: Rewards are adjusted to encourage forward progress, penalize regression and death, and significantly reward level completion by reaching the flag.
  - Training loop with live OpenCV display and continuous `.mp4` recording
  - Logging to a `log.txt` file and TensorBoard logging for training metrics
- **`requirements.txt`**  
  All pinned dependencies for this project.  
  Install them via `pip install -r requirements.txt`.
- **`README.md`**  
  This documentation.

---

## 6. Checkpointing & Resuming

The script supports **saving and loading** checkpoints so you can stop training at any point and later resume without losing progress:

- **Checkpoint Saving**  
  - A checkpoint is automatically saved every 50 episodes as `checkpoint.pth`.
  - Upon termination (e.g., via Ctrl + C), a final checkpoint is saved as `checkpoint_final.pth`, ensuring that training progress is preserved.
  
- **Checkpoint Loading**  
  At the beginning of the script, you are prompted to choose whether to start new training or load an existing checkpoint:
  ```python
  user_input = input("Start new training (n) or load from checkpoint (l)? [n/l]: ")
  if user_input.lower() == 'l':
      load_checkpoint("checkpoint.pth")
  else:
      print("Starting new training from scratch...")
  ```
  This restores all relevant variables (network weights, training counters, replay memory, etc.), so **ensure the network architecture remains unchanged** between saves and loads.

---

## 7. Automatic Device Selection

Upon startup, the script automatically selects the best available device:
1. **Apple Metal (MPS)** — on macOS with Apple Silicon.
2. **NVIDIA CUDA** — if you have a CUDA-capable GPU and the necessary PyTorch support.
3. **CPU** — if neither of the above is available.

All model operations then run on the chosen device, improving performance where possible.

---

## 8. Contributing

Feel free to open an **issue** or make a **pull request** if you find any bugs or want to add features (e.g., alternative action spaces, advanced reward shaping, etc.).

---

**Enjoy training Mario with your own enhanced DQN!**
