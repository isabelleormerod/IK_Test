# Imports
from myosuite.simhive.myo_sim.test_sims import TestSims as loader
from IPython.display import HTML
import matplotlib.pyplot as plt
from base64 import b64encode
import scipy.sparse as spa
from copy import deepcopy
from tqdm import tqdm
import pandas as pd
import numpy as np
import skvideo.io
import mujoco
import osqp
import os
import psutil

# Memory Monitoring
def print_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"RSS: {mem_info.rss / (1024 * 1024):.2f} MB, VMS: {mem_info.vms / (1024 * 1024):.2f} MB")

# Video Display
def show_video(video_path, video_width=400):
    video_file = open(video_path, "r+b").read()
    video_url = f"data:video/mp4;base64,{b64encode(video_file).decode()}"
    return HTML(f"""<video autoplay width={video_width} controls><source src="{video_url}"></video>""")

# Quadratic Program Solver
def solve_qp(P, q, lb, ub, x0):
    P = spa.csc_matrix(P)
    A = spa.csc_matrix(spa.eye(q.shape[0]))
    m = osqp.OSQP()
    m.setup(P=P, q=q, A=A, l=lb, u=ub, verbose=False)
    m.warm_start(x=x0)
    res = m.solve()
    return res.x

# Plot Functions with memory usage monitoring
def plot_qxxx(qxxx, joint_names, labels):
    print_memory_usage()  # Check memory usage
    fig, axs = plt.subplots(4, 6, figsize=(12, 8))
    axs = axs.flatten()
    line_objects = []
    linestyle = ['-'] * qxxx.shape[2]
    linestyle[-1] = '--'
    for j in range(1, len(joint_names)+1):
        ax = axs[j-1]
        for i in range(qxxx.shape[2]):
            line, = ax.plot(qxxx[:, 0, -1], qxxx[:, j, i], linestyle[i])
            if j == 1:
                line_objects.append(line)
        ax.set_xlim([qxxx[:, 0].min(), qxxx[:, 0].max()])
        ax.set_ylim([qxxx[:, 1:, :].min(), qxxx[:, 1:, :].max()])
        ax.set_title(joint_names[j-1])
    legend_ax = axs[len(joint_names)]
    legend_ax.axis('off')
    legend_ax.legend(line_objects, labels, loc='center')
    plt.tight_layout()
    plt.show()
    del fig, axs  # Free memory immediately after plot

def plot_uxxx(uxxx, muscle_names, labels):
    print_memory_usage()  # Check memory usage
    fig, axs = plt.subplots(5, 8, figsize=(12, 8))
    axs = axs.flatten()
    line_objects = []
    for j in range(1, len(muscle_names)+1):
        ax = axs[j-1]
        for i in range(uxxx.shape[2]):
            line, = ax.plot(uxxx[:, 0, -1], uxxx[:, j, i])
            if j == 1:
                line_objects.append(line)
        ax.set_xlim([uxxx[:, 0].min(), uxxx[:, 0].max()])
        ax.set_ylim([uxxx[:, 1:, :].min(), uxxx[:, 1:, :].max()])
        ax.set_title(muscle_names[j-1])
    legend_ax = axs[len(muscle_names)]
    legend_ax.axis('off')
    legend_ax.legend(line_objects, labels, loc='center')
    plt.tight_layout()
    plt.show()
    del fig, axs  # Free memory after plot

# Load trajectory data
traj = pd.read_csv('data/6_trajectory.csv', dtype='float32').values  # Use float32 to save memory

# Calculate Forces
def get_qfrc(model, data, target_qpos):
    data_copy = deepcopy(data)
    data_copy.qacc = (((target_qpos - data.qpos) / model.opt.timestep) - data.qvel) / model.opt.timestep
    model.opt.disableflags += mujoco.mjtDisableBit.mjDSBL_CONSTRAINT
    mujoco.mj_inverse(model, data_copy)
    model.opt.disableflags -= mujoco.mjtDisableBit.mjDSBL_CONSTRAINT
    return data_copy.qfrc_inverse

# Main simulation with memory monitoring
try:
    model0 = loader.get_sim(None, 'hand/myohand.xml')
    data0 = mujoco.MjData(model0)
    qpos_eval = np.zeros((traj.shape[0], traj.shape[1], 2), dtype='float32')  # Use float32 for memory efficiency
    qpos_eval[:,:,-1] = traj
    for idx in tqdm(range(traj.shape[0])):
        if idx % 50 == 0:
            target_qpos = traj[idx, 1:]
            qfrc = get_qfrc(model0, data0, target_qpos)
            data0.qfrc_applied = qfrc
            mujoco.mj_step(model0, data0)
            qpos_eval[idx,:,0] = np.hstack((data0.time, data0.qpos))
            print_memory_usage()
    
    error = ((qpos_eval[:,1:,0] - qpos_eval[:,1:,-1])**2).mean(axis=0)
    print(f'Error max (rad): {error.max()}')
    
    joint_names = [model0.joint(i).name for i in range(model0.nq)]
    plot_qxxx(qpos_eval, joint_names, ['Achieved qpos', 'Reference qpos'])
    print_memory_usage()
    
    # Clear large arrays
    del qpos_eval, traj, error
    print_memory_usage()  # Check memory after cleanup

except Exception as e:
    print(f"An error occurred: {e}")
    print_memory_usage()
