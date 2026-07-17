import mujoco as mj
from mujoco import viewer
import os
from mujoco.glfw import glfw
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces

xml_path = 'CubeEnv\env1.xml'

model = mj.MjModel.from_xml_path(xml_path)
data = mj.MjData(model)

runtime = 100.0
ctrl_step = 5.0

joint_id_x = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, 'move_x')
joint_id_y = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, 'move_y')

act_id_x = model.actuator('motor_x').id
act_id_y = model.actuator('motor_y').id

def clear_console():
    os.system('cls')

# def key_callback(keycode):
#     global target_vel_x, target_vel_y
#     if keycode == 265:      # Move forward/up
#         target_vel_y += ctrl_step
#     elif keycode == 264:    # Move backward/down
#         target_vel_y -= ctrl_step
#     elif keycode == 263:    # Move left
#         target_vel_x -= ctrl_step
#     elif keycode == 262:    # Move right
#         target_vel_x += ctrl_step
#     elif keycode == 82:    # Reset positions if needed
#         mj.mj_resetData(model, data)
#     if keycode == None:
#         target_vel_x = 0.0

# lidar/rangefinder
num_beams = 16
max_range = 5.0

# observations: rangefinders + x/y vel
obs_dim = num_beams + 2

class RobotEnv(gym.Env):
    def __init__(self, model, data, target=np.array([-.5, .5])):
        super(RobotEnv, self).__init__()
        self.model = model
        self.data = data
        self.target = target

        self.max_steps = 2000
        self.current_step = 0

        self.kp = 40.0
        self.ki = 150.0
        self.max_integral = 50.0
        self.err_integral_x = 0.0
        self.err_integral_y = 0.0

        low_vel = 1.0
        med_vel = 3.0
        high_vel = 5.0

        self.action_space = spaces.Discrete(15)
        self.action_map = {
            0: ("x", 0.0),
            1: ("x", low_vel),
            2: ("x", med_vel),
            3: ("x", high_vel),
            4: ("x", -low_vel),
            5: ("x", -med_vel),
            6: ("x", -high_vel),
            7: ("y", 0.0),
            8: ("y", low_vel),
            9: ("y", med_vel),
            10: ("y", high_vel),
            11: ("y", -low_vel),
            12: ("y", -med_vel),
            13: ("y", high_vel),
            14: (None, 0.0)
        }
        self.target_vel_x = 0
        self.target_vel_y = 0

        self.observation_space = spaces.Box(
            low= -8.0,
            high= 8.0,
            shape=(obs_dim,),
            dtype=np.float32
        )
        self.viewer = None

    def render(self):
        if self.viewer is None:
            self.viewer.launch_passive(
                self.model, self.data, show_left_ui=False, show_right_ui=False
            )

            viewer.cam.lookat = [0.0, 0.0, 0.0]
            viewer.cam.distance = 7.5
            viewer.cam.elevation = -90
            viewer.cam.azimuth = 90
        
        self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def _get_obs(self):
        vel_x = self.data.qvel[joint_id_x]
        vel_y = self.data.qvel[joint_id_y]

        range_data = self.data.sensordata[:num_beams]

        # distance and heading to target
        current_pos = self.data.qpos[:2]
        target_pos = self.target

        distance = np.linalg.norm(target_pos - current_pos)
        
        target_vector = target_pos - current_pos
        heading = np.arctan2(target_vector[1], target_vector[0])

        obs = np.concatenate([distance, heading, vel_x, vel_y], range_data).astype(np.float32)

        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mj.mj_resetData(self.model, self.data)

        self.current_step = 0

        self.err_integral_x = 0.0
        self.err_integral_y = 0.0

        self.target_vel_x = 0.0
        self.target_vel_y = 0.0

        return self._get_obs(), {}
    
    def step(self, action):
        self.current_step += 1
        axis, speed = self.action_map.get(action, (None, 0.0))

        if axis == "x":
            target_vel_x = speed
        elif axis == "y":
            target_vel_y = speed
        else:
            target_vel_x, target_vel_y = 0.0
        
        ### PI controller ###
        vel_x = self.data.qvel[joint_id_x]
        vel_y = self.data.qvel[joint_id_y]

        err_x = target_vel_x - vel_x
        err_y = target_vel_y - vel_y
        dt = model.opt.timestep

        self.err_integral_x = np.clip(
            self.err_integral_x + err_x * dt,
            -self.max_integral,
            self.max_integral)
        self.err_integral_y = np.clip(
            self.err_integral_y + err_y * dt,
            -self.max_integral,
            self.max_integral)
        
        self.data.ctrl[act_id_x] = (self.kp * err_x) + (self.ki * self.err_integral_x)
        self.data.ctrl[act_id_y] = (self.kp * err_y) + (self.ki * self.err_integral_y)

        mj.mj_step(model, data)

        obs = self._get_obs()

        current_pos = self.data.qpos[:2]
        target_pos = self.target
        distance = np.linalg.norm(target_pos - current_pos)
        reward = -distance

        range_data = self.data.sensordata[:num_beams]
        collision = np.any(range_data < 0.05)

        reached_target = distance < 0.1

        terminated = bool(reached_target or collision)
        truncated = self.current_step >= self.max_steps

        return obs, reward, terminated, truncated, {}
    

env = RobotEnv(model, data)

total_episodes = 1000
render_every_n_ep = 50

for episode in range(total_episodes):
    obs, info = env.reset()

    should_render = episode % render_every_n_ep == 0

    terminated = False
    truncated = False

    while not (terminated or truncated):
        step_start = time.time

        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
    
        if should_render:
            env.render()

            time_spent = time.time() - step_start
            time_until_next_step = model.opt.timestep - time_spent
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    if should_render:
        env.close()