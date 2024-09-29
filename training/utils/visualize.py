import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

matplotlib.use('Agg')
SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]

def create_axes3d(fig, rect=[0, 0, 1, 1], radius=4, elev=25, azim=45):
    ax = Axes3D(fig, rect=rect, proj_type='ortho')
    ax.set_xlim([-radius / 2, radius / 2])
    ax.set_ylim([-radius / 2, radius / 2])
    ax.set_zlim([0, radius])
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0)) # Hide YZ Plane
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0)) # Hide XZ Plane
    ax.zaxis.line.set_visible(False)
    ax.set_zticks([0.0])
    ax.grid(False)
    ax.set_xticks([0.0])
    ax.set_yticks([0.0])
    ax.view_init(elev=elev, azim=azim)
    fig.add_axes(ax)
    return ax

def clear_ax(ax, update_lines=None):
    try:
        if update_lines:
            for ln in update_lines:
                ln.remove()
            update_lines.clear()
    except:
        pass
    try:
        for line in ax.lines:
            line.remove() # For higher mpl versions.
        for text in ax.texts:
            text.remove()
    except:
        ax.lines = [] # For lower mpl versions.
        ax.texts = []

def plot_single_pose(joint_pos, ax, color, linewidth=2.0):
    lines = []
    for i in range(1, len(SMPL_PARENTS)):
        line = joint_pos[(SMPL_PARENTS[i], i), :]
        lines.extend(ax.plot3D(line[:, 0], line[:, 1], line[:, 2], linewidth=linewidth, color=color))
    return lines

def plot_traj(x, y, filename):
    plt.scatter(x, y)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Trajectory')
    axis_range = [-0.04, 0.1]
    # scale = 3
    # for i in range(len(axis_range)):
    #     axis_range[i] = axis_range[i] * scale
    plt.xlim(*axis_range)
    plt.ylim(*axis_range)
    plt.savefig(filename, format='jpg')

def plot_smpl_motion(data, save_path, kinematic_list, fps, figsize=(15, 15), radius=6):
    fig = plt.figure(figsize=figsize)
    ax = create_axes3d(fig, radius=radius)

    frame_number = data.shape[0]
    trajec = data[:, 0, [0, 1]]

    def update(index):
        for line in ax.lines:
            line.remove()

        if index > 1:
            ax.plot3D(trajec[:index, 0], trajec[:index, 1],
                      np.zeros_like(trajec[:index, 0]), linewidth=1.0,
                      color='blue')

        for i in range(1, len(kinematic_list)):
            linewidth = 2.0
            line = data[index, (kinematic_list[i], i), :]
            ax.plot3D(line[:, 0], line[:, 1], line[:, 2], linewidth=linewidth, color='red')

    ani = FuncAnimation(fig, update, frames=frame_number, interval=1000 / fps, repeat=False)
    ani.save(save_path, fps=fps)
    plt.close()