# Super Mario DQN - A Live & Recorded Reinforcement Learning Demo

This repository demonstrates a **Deep Q-Network (DQN) training** process on the classic *Super Mario Bros* environment using [gym_super_mario_bros](https://github.com/Kautenja/gym-super-mario-bros). It **simultaneously** displays the live gameplay in an OpenCV window **and** continuously records it into an `.mp4` video file. If you interrupt the script (e.g., with **Ctrl + C**), the video is properly finalized and not corrupted.

## 1. Overview

- **Train a DQN** in the [NES Super Mario Bros](https://github.com/Kautenja/gym-super-mario-bros) environment.
- **Live visualization** via an OpenCV window: watch Mario’s actions in real time (optionally zoomed 2x for better visibility).
- **Continuous recording**: each frame is directly encoded into an `.mp4` using OpenCV’s `VideoWriter`.
- **Graceful shutdown**: if you press **Ctrl + C**, the environment and the video file are closed correctly, leaving you with a playable recording.
- **Checkpointing & Resuming**: Save and load model checkpoints (network weights, replay buffer, etc.) to resume training.
- **Automatic Device Selection**: The script automatically checks for Apple Metal (MPS), NVIDIA GPU (CUDA), or falls back to CPU.

## 2. Installation (Anaconda)

Below are instructions on how to set up a **conda environment** and install all necessary dependencies.

1. **Install [Anaconda](https://www.anaconda.com/download) or Miniconda** if you haven't already.

2. **Create a new environment** (e.g., `mario_env`) with Python 3.8:
   ```bash
   conda create -n mario_env python=3.8
   ```

3. **Activate** the newly created environment:
   ```bash
   conda activate mario_env
   ```

4. **Install dependencies** inside this environment.  
   You can install everything via `conda` (some packages may only be available on `conda-forge`), or you can combine `conda` with `pip`:
   ```bash
   # 4.1) Basic packages
   conda install -c conda-forge numpy pandas matplotlib

   # 4.2) PyTorch (CPU-only or GPU version)
   # Example for CPU version:
   conda install pytorch cpuonly -c pytorch

   # 4.3) OpenCV
   conda install -c conda-forge opencv

   # 4.4) Gym, gym_super_mario_bros, other RL packages
   conda install -c conda-forge gym gym_super_mario_bros

   # 4.5) (Optional) If you want to handle video editing with MoviePy
   conda install -c conda-forge moviepy ffmpeg
   ```

   Alternatively, using `pip`:
   ```bash
   pip install torch torchvision \
               opencv-python \
               gym \
               gym_super_mario_bros \
               moviepy
   ```

5. **(macOS only) Potential fork-safety issues**:  
   If you run into segmentation faults on macOS, you may need:
   ```bash
   export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
   ```
   before running the script.

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
   - An `.mp4` video is continuously recorded in a folder named `./videos_custom_<timestamp>_<random>`.
   - Training stops once the specified number of episodes is done, **or** if you hit **Ctrl + C** (KeyboardInterrupt). In either case, the script cleans up gracefully, closes the environment, and finalizes the `.mp4`.

### Observing the Output

- **OpenCV Window**: real-time gameplay (2x zoom in the live window).  
- **Terminal Output**: logs about the current episode, total reward, etc.  
- **Video File**: `mario_run.mp4` (or a similar name) inside the `videos_custom_*` folder.

## 4. Project Structure

Typical files in this repository might include:

- `mario_gym.py`  
  The main script containing:
  - Device selection (Apple Metal MPS, CUDA, or CPU)  
  - Checkpoint saving/loading (resuming training)  
  - DQN architecture
  - Replay buffer logic
  - Training loop
  - OpenCV display and continuous `.mp4` recording logic

- `README.md`  
  This documentation.

## 5. Checkpointing & Resuming

The script supports **saving and loading** checkpoints so you can stop training at any point and later resume without losing progress:

- **Checkpoint Saving**  
  Occurs automatically every few episodes (configurable in the code) and upon exit (`finally` block). For instance:
  ```python
  if (episode + 1) % 5 == 0:
      save_checkpoint("checkpoint.pth")
  ```
  and in the `finally` block:
  ```python
  finally:
      save_checkpoint("checkpoint_final.pth")
      cleanup()
  ```
  This saves:
  - `policy_net` and `target_net` weights
  - `optimizer` state
  - `epsilon`, `steps_done`
  - **Optional**: replay memory (may be large)

- **Checkpoint Loading**  
  At the beginning of the script, you can choose:
  ```python
  user_input = input("Start new training (n) or load from checkpoint (l)? [n/l]: ")
  if user_input.lower() == 'l':
      load_checkpoint("checkpoint.pth")
  else:
      print("Starting new training from scratch...")
  ```
  This restores all relevant variables (network weights, training counters, replay memory, etc.).

---

**Important:** Make sure that you define **exactly the same network architecture** in your code before calling `load_state_dict()`; otherwise, the loaded weight arrays won't match your current model's layers, and an error will occur.

## 6. Automatic Device Selection

Upon startup, the script checks if it can run on:
1. **Apple Metal (MPS)** — if available on macOS with Apple Silicon.  
2. **NVIDIA GPU (CUDA)** — if you have a CUDA-capable GPU and PyTorch with CUDA support.  
3. **CPU** — default fallback if neither MPS nor CUDA is available.  

All model operations then run on the chosen `device`.

## 7. Contributing

Feel free to open an **issue** or make a **pull request** if you find any bugs or want to add features (e.g., alternative action spaces, advanced reward shaping, etc.).

---

**Enjoy training Mario with your own DQN!**
