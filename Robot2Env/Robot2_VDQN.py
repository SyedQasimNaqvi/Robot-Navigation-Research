import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from gymnasium import spaces
from collections import deque
import random
import matplotlib.pyplot as plt
import os
import csv

# if torch.cuda.is_available():
#     print("PyTorch is using GPU (CUDA).")
#     print(f"GPU Name: {torch.cuda.get_device_name(0)}")
#     print(f"Number of CUDA devices: {torch.cuda.device_count()}")
# else:
#     print("PyTorch is using CPU.")

xml_path = 'Robot2.xml'
simend = 500
button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0

# Graph data
x1 = []
y1 = []

# CSV data
episode_data = []

def init_controller(model,data):
    pass

def controller(model, data):
    pass

def keyboard(window, key, scancode, act, mods):
    if act == glfw.PRESS:
        if key == glfw.KEY_BACKSPACE:
            mj.mj_resetData(model, data)
            mj.mj_forward(model, data)
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)

def mouse_button(window, button, act, mods):
    # update button state
    global button_left
    global button_middle
    global button_right

    button_left = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
    button_middle = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
    button_right = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)

    # update mouse position
    glfw.get_cursor_pos(window)

# def mouse_move(window, xpos, ypos):
#     # compute mouse displacement, save
#     global lastx
#     global lasty
#     global button_left
#     global button_middle
#     global button_right

#     dx = xpos - lastx
#     dy = ypos - lasty
#     lastx = xpos
#     lasty = ypos

#     # no buttons down: nothing to do
#     if (not button_left) and (not button_middle) and (not button_right):
#         return

#     # get current window size
#     width, height = glfw.get_window_size(window)

#     # get shift key state
#     PRESS_LEFT_SHIFT = glfw.get_key(
#         window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
#     PRESS_RIGHT_SHIFT = glfw.get_key(
#         window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
#     mod_shift = (PRESS_LEFT_SHIFT or PRESS_RIGHT_SHIFT)

#     # determine action based on mouse button
#     if button_right:
#         if mod_shift:
#             action = mj.mjtMouse.mjMOUSE_MOVE_H
#         else:
#             action = mj.mjtMouse.mjMOUSE_MOVE_V
#     elif button_left:
#         if mod_shift:
#             action = mj.mjtMouse.mjMOUSE_ROTATE_H
#         else:
#             action = mj.mjtMouse.mjMOUSE_ROTATE_V
#     else:
#         action = mj.mjtMouse.mjMOUSE_ZOOM

#     mj.mjv_moveCamera(model, action, dx/height,
#                       dy/height, scene, cam)

def scroll(window, xoffset, yoffset):
    action = mj.mjtMouse.mjMOUSE_ZOOM
    mj.mjv_moveCamera(model, action, 0.0, -0.05 *
                      yoffset, scene, cam)

dirname = os.path.dirname(__file__)
abspath = os.path.join(dirname + "/" + xml_path)
xml_path = abspath

model = mj.MjModel.from_xml_path(xml_path)
data = mj.MjData(model)
cam = mj.MjvCamera()
opt = mj.MjvOption()

glfw.init()
window = glfw.create_window(1200, 900, "Demo", None, None)
glfw.make_context_current(window)
glfw.swap_interval(1)

mj.mjv_defaultCamera(cam)
mj.mjv_defaultOption(opt)
scene = mj.MjvScene(model, maxgeom=10000)
context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

glfw.set_key_callback(window, keyboard)
# glfw.set_cursor_pos_callback(window, mouse_move)
glfw.set_mouse_button_callback(window, mouse_button)
glfw.set_scroll_callback(window, scroll)

init_controller(model,data)

mj.set_mjcb_control(controller)

class BoxEnv(gym.Env):
    def __init__(self, model, data, target=np.array([.5, .5])):
        super(BoxEnv, self).__init__()
        self.model = model
        self.data = data
        self.target = target
        self.max_steps = 200
        self.current_step = 0

        # Actions: [stay, +, -, +steer, -steer, reset]
        self.action_space = spaces.Discrete(9)

        # Observations: [x, steer, x_vel, steer_pos]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
        )

    def reset(self, *, seed=None, options=None):
        mj.mj_resetData(self.model, self.data)   # <-- self.model/self.data
        self.current_step = 0
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        self.current_step += 1

        # apply action
        if action == 1:
            self.data.ctrl[0:1] = 10
        elif action == 2:
            self.data.ctrl[0:1] = -10
        elif action == 3:
            self.data.ctrl[0:1] = 0
        elif action == 4:
            self.data.ctrl[0] = 10
        elif action == 5:
            self.data.ctrl[0] = -10
        elif action == 6:
            self.data.ctrl[0] = 0
        elif action == 7:
            self.data.ctrl[1] = 10
        elif action == 8:
            self.data.ctrl[1] = -10
        elif action == 9:
            self.data.ctrl[1] = 0

        for __ in range(10):
            mj.mj_step(self.model, self.data)        # <-- self.model/self.data

        obs = self._get_obs()
        reward = -(np.linalg.norm(obs[:2] - self.target))
        if np.linalg.norm(obs[:2] - self.target) < 0.05:
            reward += 10.0
        
        done = (
            self.current_step >= self.max_steps
            or np.linalg.norm(obs[:2] - self.target) < 0.05
        )

        return obs, reward, done, False, {}

    def _get_obs(self):
        return np.array([
            self.data.qpos[0], self.data.qpos[1],
            self.data.qvel[0], self.data.qvel[1]
        ], dtype=np.float32)

# -----------------------
# DQN Agent
# -----------------------
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.fc(x)

# Training hyperparameters
state_dim = 4
action_dim = 9
lr = 1e-4
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.0
epsilon_decay = 0.995
batch_size = 64
replay_buffer = deque(maxlen=10000)

policy_net = DQN(state_dim, action_dim)
target_net = DQN(state_dim, action_dim)
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.Adam(policy_net.parameters(), lr=lr)

# -----------------------
# Training Loop
# -----------------------
def train_step():
    if len(replay_buffer) < batch_size:
        return
    batch = random.sample(replay_buffer, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
    rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
    next_states = torch.tensor(next_states, dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

    q_values = policy_net(states).gather(1, actions)
    with torch.no_grad():
        next_q = target_net(next_states).max(1)[0].unsqueeze(1)
        target = rewards + gamma * next_q * (1 - dones)

    loss = nn.MSELoss()(q_values, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

env = BoxEnv(model, data)

rewards_per_episode = []

for episode in range(1000):
    state, _ = env.reset()
    total_reward = 0
    done = False

    while not done:
        if episode % 100 == 0:  # Render every 100 episodes
            # Rendering loop
            time_prev = data.time

            # while (data.time - time_prev < 1.0/60.0):
                # mj.mj_step(model, data)

            if (data.time>=simend):
                break;

            # get framebuffer viewport
            viewport_width, viewport_height = glfw.get_framebuffer_size(
                window)
            viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

            # Update scene and render
            mj.mjv_updateScene(model, data, opt, None, cam,
                            mj.mjtCatBit.mjCAT_ALL.value, scene)
            mj.mjr_render(viewport, scene, context)

            # swap OpenGL buffers (blocking call due to v-sync)
            glfw.swap_buffers(window)

            # process pending GUI events, call GLFW callbacks
            glfw.poll_events()

        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action = policy_net(torch.tensor(state, dtype=torch.float32)).argmax().item()

        next_state, reward, done, _, _ = env.step(action)
        replay_buffer.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward

        train_step()
    rewards_per_episode.append(total_reward)

    episode_data.append([
        episode,
        total_reward,
        epsilon
    ])

    # update target network
    if episode % 10 == 0:
        target_net.load_state_dict(policy_net.state_dict())

    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    print(f"Episode {episode}, Total reward: {total_reward}")

with open("training_results_RobotVDQN.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Episode", "TotalReward", "Epsilon"])
    writer.writerows(episode_data)

print("Saved training_results_RobotVDQN.csv")

plt.figure(figsize=(8, 5))
plt.plot(rewards_per_episode, label="Reward")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Reward per Episode (Vanilla DQN)")
plt.legend()
plt.grid(True)
plt.show()

env.close()

glfw.terminate()