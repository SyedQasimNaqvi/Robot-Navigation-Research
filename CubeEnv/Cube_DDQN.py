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

xml_path = 'box.xml' #xml file (assumes this is in the same folder as this file)
simend = 200 #simulation time

# Graph data
x1 = []
y1 = []

# CSV data
episode_data = []

def init_controller(model,data):
    #initialize the controller here. This function is called once, in the beginning
    pass

def controller(model, data):
    #put the controller here. This function is called inside the simulation.
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
    def __init__(self, model, data, target=np.array([.25, .25])):
        super(BoxEnv, self).__init__()
        self.model = model
        self.data = data
        self.target = target
        self.max_steps = 200
        self.current_step = 0

        # Actions: [stay, +x, -x, +y, -y]
        self.action_space = spaces.Discrete(5)

        # Observations: [x, y, x_vel, y_vel]
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
        self.data.qvel[0] = 0.0
        self.data.qvel[1] = 0.0

        # apply action
        if action == 1:
            self.data.qvel[0] = 1
        elif action == 2:
            self.data.qvel[0] = -1
        elif action == 3:
            self.data.qvel[1] = 1
        elif action == 4:
            self.data.qvel[1] = -1
        

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
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.layers(x)

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)
    
class DoubleDQN:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, tau=1000):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau  # how often to update target

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer()
        self.steps = 0

    def act(self, state, epsilon=0.1):
        if random.random() < epsilon:
            return random.randrange(self.action_dim)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.q_net(state).argmax().item()

    def update(self, batch_size=64):
        if len(self.replay_buffer) < batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Current Q-values
        q_values = self.q_net(states).gather(1, actions)

        # Double DQN target:
        next_actions = self.q_net(next_states).argmax(1).unsqueeze(1)

        next_q_values = self.target_net(next_states).gather(1, next_actions)

        target_q = rewards + self.gamma * next_q_values * (1 - dones)

        # Loss
        loss = nn.MSELoss()(q_values, target_q.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.steps += 1
        if self.steps % self.tau == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

def train_double_dqn(episodes=100):
    env = BoxEnv(model, data)
    agent = DoubleDQN(env.observation_space.shape[0], env.action_space.n)
    epsilon, epsilon_min, epsilon_decay = 1.0, 0.01, 0.995

    rewards_per_episode = []

    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
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
            action = agent.act(state, epsilon)
            next_state, reward, done, _ = env.step(action)   # reward comes from YOUR env
            agent.replay_buffer.push(state, action, reward, next_state, done)
            agent.update()
            state = next_state
            total_reward += reward

        print(f"Episode {ep+1}, Reward: {total_reward}, Epsilon: {epsilon:.3f}")

        rewards_per_episode.append(total_reward)

        episode_data.append([
            ep,
            total_reward,
            epsilon
        ])

        # Update epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    with open("training_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Episode", "TotalReward", "Epsilon"])
        writer.writerows(episode_data)

    print("Saved training_results_DDQN.csv")

    plt.figure(figsize=(8, 5))
    plt.plot(rewards_per_episode, label="Reward")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Reward per Episode (Double DQN)")
    plt.legend()
    plt.grid(True)
    plt.show()

    env.close()

if __name__ == "__main__":
    train_double_dqn()

glfw.terminate()