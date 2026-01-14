import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from gymnasium import spaces
from collections import deque
import matplotlib.pyplot as plt
import random, os, csv

if torch.cuda.is_available():
    print("PyTorch is using GPU (CUDA).")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"Number of CUDA devices: {torch.cuda.device_count()}")
else:
    print("PyTorch is using CPU.")

xml_path = 'cube.xml'
simend = 200

x1 = []
y1 = []

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
mj.mj_forward(model, data)

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

cam.lookat[:] = np.array([0.0, 0.0, 0.1])
cam.distance = 3.0
cam.azimuth = 90
cam.elevation = -30

glfw.set_key_callback(window, keyboard)
# glfw.set_cursor_pos_callback(window, mouse_move)
glfw.set_mouse_button_callback(window, mouse_button)
glfw.set_scroll_callback(window, scroll)

class CubeEnv(gym.Env):
    def __init__(self, model, data, target=np.array([.25, .25])):
        super(CubeEnv, self).__init__()
        self.model = model
        self.data = data
        self.target = target
        self.max_steps = 200
        self.current_step = 0

        # Actions: [stay, +x, -x, +y, -y]
        self.action_space = spaces.Box( 
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

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

        action = np.clip(action, -1.0, 1.0)

        # scale to reasonable force
        self.data.ctrl[:] = 5.0 * action

        for _ in range(10):
            mj.mj_step(self.model, self.data)

        obs = self._get_obs()
        dist = np.linalg.norm(obs[:2] - self.target)

        reward = -dist
        if dist < 0.05:
            reward += 50.0

        terminated = False
        truncated = self.current_step >= self.max_steps

        return obs, reward, terminated, truncated, {}
    
    def _get_obs(self):
        return np.array([
            self.data.qpos[0], self.data.qpos[1],
            self.data.qvel[0], self.data.qvel[1]
        ], dtype=np.float32)
    
# -----------------------
# PPO Agent
# -----------------------
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()

        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, act_dim)
        )

        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def act(self, obs):
        mean = self.actor(obs)
        std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(mean, std)

        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob

    def value(self, obs):
        return self.critic(obs).squeeze(-1)
    
def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    adv = []
    gae = 0
    values = values + [0]

    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t+1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        adv.insert(0, gae)

    return torch.tensor(adv, dtype=torch.float32).detach()

def ppo_update(policy, optimizer, obs, acts, logp_old, adv, returns,
               clip=0.2, value_coef=0.5, entropy_coef=0.01):

    for _ in range(10):
        mean = policy.actor(obs)
        std = torch.exp(policy.log_std)
        dist = torch.distributions.Normal(mean, std)

        logp = dist.log_prob(acts).sum(-1)
        ratio = torch.exp(logp - logp_old)

        clipped = torch.clamp(ratio, 1 - clip, 1 + clip) * adv
        policy_loss = -(torch.min(ratio * adv, clipped)).mean()

        value_loss = ((returns - policy.value(obs)) ** 2).mean()
        entropy = dist.entropy().sum(-1).mean()

        loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

env = CubeEnv(model, data)
obs_dim = 4
act_dim = 2

rewards_per_episode = []

policy = ActorCritic(obs_dim, act_dim)
optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)

num_of_ep = 500
episode = 0
epoch = 0

while episode < num_of_ep:

    obs_buf, act_buf, logp_buf, rew_buf, val_buf, done_buf = [], [], [], [], [], []

    obs, _ = env.reset()
    done = False

    episode_return = 0.0

    for step in range(1024):
        
        # ---------- RENDER (non-blocking) ----------
        viewport_width, viewport_height = glfw.get_framebuffer_size(window)
        viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

        # mj.mjv_updateScene(
        #     model,
        #     data,
        #     opt,
        #     None,
        #     cam,
        #     mj.mjtCatBit.mjCAT_ALL.value,
        #     scene
        # )

        # mj.mjr_render(viewport, scene, context)

        # glfw.swap_buffers(window)
        # glfw.poll_events()
        # -------------------------------------------

        obs_t = torch.tensor(obs, dtype=torch.float32)

        with torch.no_grad():
            action, logp = policy.act(obs_t)
            value = policy.value(obs_t)

        next_obs, reward, term, trunc, _ = env.step(action.numpy())
        done = term or trunc

        episode_return += reward

        obs_buf.append(obs_t)
        act_buf.append(action)
        logp_buf.append(logp)
        rew_buf.append(reward)
        val_buf.append(value.detach())
        done_buf.append(done)

        obs = next_obs

        if done:
            obs, _ = env.reset()
            done = False
            print(f"Episode return: {episode_return:.3f}")
            rewards_per_episode.append(episode_return)
            episode += 1
            episode_return = 0.0

            if episode >= num_of_ep:
                break
        

    # ---------- PPO UPDATE ----------
    adv = compute_gae(rew_buf, val_buf, done_buf)
    returns = adv + torch.stack(val_buf)

    obs = torch.stack(obs_buf)
    acts = torch.stack(act_buf)
    logp_old = torch.stack(logp_buf).detach()

    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    ppo_update(policy, optimizer, obs, acts, logp_old, adv, returns)

    epoch += 1

    print(f"Epoch {epoch} complete")

plt.figure(figsize=(8, 5))
plt.plot(rewards_per_episode, label="Reward")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Reward per Episode (PPO)")
plt.legend()
plt.grid(True)
plt.show()

env.close()

glfw.terminate()