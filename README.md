# Super Mario Dueling Double DQN – A Live & Recorded Reinforcement Learning Demo

This repository demonstrates an enhanced **Dueling Double DQN** training process on the classic *Super Mario Bros* environment using [gym_super_mario_bros](https://github.com/Kautenja/gym-super-mario-bros). The training process includes live gameplay visualization via an OpenCV window and continuous recording of the run into an `.mp4` video file.

> **Note:** Whether you stop the training normally or interrupt it (e.g., using **Ctrl + C**), the script ensures that the video is finalized correctly and the environment is properly closed.

---

## 1. Overview

- **Dueling Double DQN with Prioritized Replay:**  
  The script trains a DQN using a deeper network architecture with separate value and advantage streams. In addition, it employs a prioritized experience replay buffer with importance sampling and uses the Huber (smooth L1) loss for enhanced training stability.

- **Multiprocessing Compatibility:**  
  The script applies a fix by setting the multiprocessing start method to `spawn`, ensuring compatibility (especially on macOS).

- **Live Visualization & Recording:**  
  - **Real-Time Display:** An OpenCV window named **"Mario"** shows the gameplay in real time (displayed with a default 2× zoom for enhanced visibility).  
  - **Continuous Recording:** Every frame is encoded into a full run video (`mario_run.mp4`) saved inside a folder automatically named with the current date and time (`YYYY-MM-DD_HH-MM-SS_run`).

- **Best Run Extraction:**  
  The script tracks the episode with the highest total reward and saves it separately as `mario_best_run_<reward>.mp4` in the same run folder.

- **Custom Reward Shaping:**  
  The reward function has been tuned to guide the training process:
  - **Forward Progress:** A bonus of **0.1 × (distance advanced)** is added if Mario’s current `x_pos` exceeds the previous value.
  - **Life Loss Penalty:** Losing a life (while still having remaining lives) immediately terminates the episode and applies a **–50 penalty**.
  - **Level Completion Bonus:** Reaching the flag gives a bonus of **+2000**.
  - **Clipping:** Rewards are clipped to the range **[–100, 2000]**.

- **Checkpointing & Resuming:**  
  - **Periodic Saving:** A checkpoint is automatically saved every 50 episodes as `checkpoint.pth`.
  - **Final Checkpoint:** When the training stops (even via **Ctrl + C**), a final checkpoint is saved as `checkpoint_final.pth`.
  - **Loading Option:** On startup, you can choose to load an existing checkpoint to resume training.

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

- **Prioritized Experience Replay:**  
  Implements prioritized replay with importance sampling and dynamic priority updates to improve learning efficiency.

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
   - At startup, you will be prompted to choose between starting new training or loading from a checkpoint:
     ```python
     Start new training (n) or load from checkpoint (l)? [n/l]:
     ```
   - **Real-Time Display:** An OpenCV window named **"Mario"** will open, showing live gameplay with a 2× zoom.
   - **Video Recording:**  
     - A full run video is saved as `mario_run.mp4` inside a run folder named with the current date and time (e.g., `2025-03-01_12-34-56_run`).
     - The best-performing episode is also saved as `mario_best_run_<reward>.mp4` (with `<reward>` showing the episode’s total reward formatted to two decimal places).
   - **Checkpointing:**  
     - Checkpoints are saved every 50 episodes as `checkpoint.pth`.
     - Upon termination (even via **Ctrl + C**), a final checkpoint is saved as `checkpoint_final.pth`.

4. **Output Observation:**
   - **OpenCV Window:** Displays real-time gameplay.
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

The repository typically includes:

- **`mario_gym.py`**  
  The main training script, which contains:
  - Multiprocessing start method fix for compatibility.
  - Automatic device selection (Apple MPS, CUDA, or CPU).
  - Dueling Double DQN architecture with deeper value and advantage streams.
  - Prioritized experience replay implementation.
  - Custom reward shaping that rewards forward progress, penalizes life loss, and heavily rewards level completion.
  - Live gameplay visualization via OpenCV and continuous `.mp4` recording.
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
  - A checkpoint is automatically saved every 50 episodes as `checkpoint.pth`.
  - When interrupted (e.g., via **Ctrl + C**), a final checkpoint is saved as `checkpoint_final.pth`.
- **Resuming:**  
  - At startup, choose to load from a checkpoint to restore the model weights, optimizer state, replay memory, and training counters.
  - Ensure the network architecture remains unchanged between saving and loading.

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
