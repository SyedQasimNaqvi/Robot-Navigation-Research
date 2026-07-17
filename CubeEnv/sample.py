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

# PI controller
joint_id_x = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, 'move_x')
joint_id_y = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, 'move_y')

act_id_x = model.actuator('motor_x').id
act_id_y = model.actuator('motor_y').id

kp = 40.0
ki = 150.0

err_integral_x = 0.0
err_integral_y = 0.0

target_vel_x = 0
target_vel_y = 0

def clear_console():
    os.system('cls')

def key_callback(keycode):
    global target_vel_x, target_vel_y
    if keycode == 265:      # Move forward/up
        target_vel_y += ctrl_step
    elif keycode == 264:    # Move backward/down
        target_vel_y -= ctrl_step
    elif keycode == 263:    # Move left
        target_vel_x -= ctrl_step
    elif keycode == 262:    # Move right
        target_vel_x += ctrl_step
    elif keycode == 82:    # Reset positions if needed
        mj.mj_resetData(model, data)
    if keycode == None:
        target_vel_x = 0.0

with mj.viewer.launch_passive(model, data, key_callback=key_callback, show_left_ui=False, show_right_ui=False) as viewer:

    viewer.cam.lookat = [0.0, 0.0, 0.0]
    viewer.cam.distance = 7.5
    viewer.cam.elevation = -90
    viewer.cam.azimuth = 90

    start_time = time.time()
    
    while viewer.is_running() and (time.time() - start_time) < runtime:
        step_start = time.time()

        vel_x = data.qvel[joint_id_x]
        vel_y = data.qvel[joint_id_y]

        clear_console()
        print(vel_x)
        print(vel_y)
        print(target_vel_x)
        print(target_vel_y)

        #PI controller
        err_x = target_vel_x - vel_x
        err_y = target_vel_y - vel_y

        dt = model.opt.timestep
        err_integral_x += err_x * dt
        err_integral_y += err_y * dt

        max_integral = 50.0
        err_integral_x = np.clip(err_integral_x, -max_integral, max_integral)
        err_integral_y = np.clip(err_integral_y, -max_integral, max_integral)

        data.ctrl[act_id_x] = (kp * err_x) + (ki * err_integral_x)
        data.ctrl[act_id_y] = (kp * err_y) + (ki * err_integral_y)

        mj.mj_step(model, data)
        
        viewer.sync()

        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)