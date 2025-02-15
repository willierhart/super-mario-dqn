#!/usr/bin/env python3

"""
Dieses Skript trainiert ein einfaches DQN in der SuperMarioBros-v0-Umgebung.
Dabei wird jeder Frame des Spiels per OpenCV in einem Fenster angezeigt,
und zugleich in ein MP4-Video geschrieben.
Ein Abbruch per STRG + C erzeugt keine beschädigte Videodatei,
da wir den VideoWriter in einem finally-Block freigeben.
"""

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 1) Multiprocessing-Fix (wichtig für MacOS, teils auch für Linux/Windows)
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import multiprocessing
multiprocessing.set_start_method("spawn", force=True)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 2) Standard-Bibliotheken und Warnungen
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import warnings
import os
import time
import random

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 3) Externe Bibliotheken (OpenCV, Gym, PyTorch, Numpy etc.)
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import cv2
import gym
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from collections import deque
from torchvision import transforms as T

# Gym-Warnungen unterdrücken (optional)
warnings.filterwarnings("ignore", message="Overwriting existing videos")
warnings.filterwarnings("ignore", message="The result returned by `env.reset()` was not a tuple")
warnings.filterwarnings("ignore", message="Disabling video recorder")
warnings.filterwarnings("ignore", message="No render modes was declared")
warnings.filterwarnings("ignore", message="Core environment is written in old step API")
warnings.filterwarnings("ignore", message="`np.bool8` is a deprecated alias for `np.bool_`")
warnings.filterwarnings("ignore", message="The environment creator metadata doesn't include `render_modes`")

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 4) gym_super_mario_bros-spezifische Importe
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

# Nur ein Hinweis, falls MoviePy nicht installiert ist
try:
    import moviepy  # noqa
except ImportError:
    print("Achtung: MoviePy ist nicht installiert! Bitte 'pip install moviepy' ausführen,")
    print("falls du das erzeugte .mp4 nachträglich weiterverarbeiten willst.\n")

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 5) Gerät (CPU / GPU) automatisch wählen
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 6) Hilfsfunktion: Frame-Vorverarbeitung (Preprocessing)
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def preprocess_observation(obs):
    """
    Wandelt das Eingabe-Frame (RGB) in Graustufen um,
    skaliert es auf [84x84], und gibt ein numpy-Array
    mit Form [1, 84, 84] zurück (1=Kanal).
    Anschließend normalisieren wir die Pixelwerte von 0..255 auf 0..1.
    """
    # obs sollte shape (H, W, 3) = RGB haben
    if obs.ndim == 3:
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)  # zu Graustufen
    obs = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)  # resize
    obs = np.expand_dims(obs, axis=0)  # [1, 84, 84]
    return obs / 255.0

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 7) DQN-Netzwerk-Klasse
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class DQN(nn.Module):
    """
    Einfaches DQN mit 3 Convolution-Layern und 2 dichten Schichten.
    Erwartet input_dim Kanäle (meist 1 für Graustufen) und gibt
    output_dim Aktionen zurück (z.B. 7 bei SIMPLE_MOVEMENT).
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
# 8) Hyperparameter und globale Variablen
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
GAMMA = 0.99         # Discount-Faktor
LR = 0.00025         # Lernrate
MEMORY_SIZE = 10000  # Replay-Memory-Größe
BATCH_SIZE = 32
EPSILON_DECAY = 0.9995
MIN_EPSILON = 0.01
UPDATE_TARGET = 1000  # alle 1000 Steps Target-Net synchronisieren

# Zähler für Trainingsschritte + Epsilon-Wert
steps_done = 0
epsilon = 1.0

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 9) Environment ohne RecordVideo - Wir nehmen frames selbst auf!
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
env = gym_super_mario_bros.make(
    "SuperMarioBros-v0",
    apply_api_compatibility=True,  # neue Gym-API (5 Rückgabewerte)
    render_mode="rgb_array"       # wichtig, damit obs Frames=RGB-Array liefert
)
env = JoypadSpace(env, SIMPLE_MOVEMENT)

# Aktionen
num_actions = env.action_space.n

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 10) DQN- und Replay-Speicher initialisieren
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
policy_net = DQN(1, num_actions).to(device)
target_net = DQN(1, num_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()  # Wir nutzen das Target-Net nur für Q-Targets

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
memory = deque(maxlen=MEMORY_SIZE)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 11) Aktionen wählen: Epsilon-Greedy
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def select_action(state_tensor):
    """
    Für den aktuellen Zustand (Batch=1, Channels=1, 84, 84) wähle
    entweder eine zufällige Aktion (mit Wk. epsilon) oder
    die beste Aktion laut policy_net (mit Wk. 1-epsilon).
    """
    global epsilon, steps_done
    if random.random() < epsilon:
        return torch.tensor([[random.randrange(num_actions)]], dtype=torch.long, device=device)
    else:
        with torch.no_grad():
            return policy_net(state_tensor).max(dim=1)[1].view(1, 1)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 12) Training: Replay-Sampling + Backprop
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return  # noch zu wenige Daten, kein Training
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
    # Bellman-Update
    expected_q_values = rewards + (GAMMA * next_q_values * (1 - dones))

    loss = nn.functional.mse_loss(q_values, expected_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 13) Custom-Videoaufzeichnung mit OpenCV VideoWriter
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
video_folder = f"./videos_custom_{int(time.time())}_{random.randint(0,9999)}"
os.makedirs(video_folder, exist_ok=True)

video_filename = os.path.join(video_folder, "mario_run.mp4")
writer = None  # VideoWriter wird erst initialisiert, wenn wir die Frame-Größe kennen

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 14) Haupttrainingsfunktion
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def main():
    global steps_done, epsilon, writer

    num_episodes = 20
    for episode in range(num_episodes):
        # Gym-Reset liefert (obs, info), obs ist im "rgb_array"-Format
        obs, info = env.reset()

        # Falls der VideoWriter noch nicht existiert, Größe anlegen
        if writer is None:
            height, width, channels = obs.shape  # z.B. (240, 256, 3)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = 30.0
            writer = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))

        # Zustand in Graustufen, [1,84,84], Torch-Format
        state_arr = preprocess_observation(obs)
        state = torch.tensor(state_arr, dtype=torch.float32, device=device)

        done = False
        total_reward = 0.0

        # Schleife bis Episode beendet
        while not done:
            # 1) Live-Anzeige im OpenCV-Fenster ("Mario")
            bgr_frame = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)  # Gym liefert RGB, OpenCV mag BGR
            cv2.imshow("Mario", bgr_frame)
            cv2.waitKey(1)

            # 2) Frame ins Video schreiben
            writer.write(bgr_frame)

            # 3) Aktion wählen (Epsilon-Greedy)
            action = select_action(state.unsqueeze(0))  # -> shape [1,1,84,84]
            next_obs, reward, terminated, truncated, info = env.step(action.item())
            done = terminated or truncated

            # 4) Nächsten Zustand
            next_state_arr = preprocess_observation(next_obs)
            next_state = torch.tensor(next_state_arr, dtype=torch.float32, device=device)

            # 5) ReplayMemory befüllen
            memory.append((
                state.cpu().numpy(),
                action.item(),
                reward,
                next_state.cpu().numpy(),
                float(done)
            ))

            # 6) Auf den nächsten Schritt vorbereiten
            state = next_state
            obs = next_obs
            total_reward += reward

            # 7) Training / Backprop
            optimize_model()
            steps_done += 1
            epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY)

            # 8) Target-Net aktualisieren
            if steps_done % UPDATE_TARGET == 0:
                target_net.load_state_dict(policy_net.state_dict())

        print(f"Episode {episode + 1}, Reward: {total_reward}")

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 15) Cleanup-Funktion (Wird IMMER am Ende aufgerufen)
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def cleanup():
    """
    Schließt Environment, zerstört OpenCV-Fenster und gibt den VideoWriter frei,
    damit das MP4-Video sauber finalisiert wird.
    """
    print("Clean up: Environment schließen, Video finalisieren ...")
    try:
        env.close()
    except Exception as e:
        print("Warnung beim env.close():", e)
    cv2.destroyAllWindows()
    global writer
    if writer is not None:
        writer.release()
    print(f"Video geschrieben nach: {video_filename}")

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 16) Main-Guard (Einstiegspunkt) + Exception-Abfangen
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nManueller Abbruch (Strg + C).")
    finally:
        # finally-Block wird ausgeführt, egal ob normaler Abbruch oder Exception
        cleanup()
