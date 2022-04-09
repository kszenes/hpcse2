import os
from math import sqrt
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

def read_csv(filename):

    timepoint = np.genfromtxt(filename, delimiter=',')
    return timepoint

def collect_data(file_list):

    n_frames = len(file_list)
    u0 = np.fromfile(file_list[0], dtype=np.float64)
    points_per_dim = int(sqrt(u0.size))
    shape = (points_per_dim, points_per_dim)
    u0 = u0.reshape(shape)
    u = np.empty((*shape, n_frames))
    u[:,:,0] = u0

    if n_frames > 1:
        for i in range(1, n_frames):
            u[:,:,i] = np.fromfile(file_list[i], dtype=np.float64).reshape(shape)

    h = 1.0 / points_per_dim
    x = np.linspace(h, 1.0-h, points_per_dim)
    y = np.linspace(h, 1.0-h, points_per_dim)
    x, y = np.meshgrid(x, y)

    return x, y, u


def create_animation(x, y, u):

    n_frames = u.shape[2]

    fps = 10 # frames per sec

    def update_plot(frame_number, u, plot):
        plot[0].remove()
        plot[0] = ax.plot_surface(x, y, u[:, :,frame_number], cmap="coolwarm", linewidth=0)


    fig = Figure()
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111, projection='3d')

    plot = [ax.plot_surface(x, y, u[:,:,0], color='0.75', rstride=1, cstride=1)]
    ax.set_zlim(0,1.0)
    ani = animation.FuncAnimation(fig, update_plot, n_frames, fargs=(u, plot), interval=1000/fps)
    ani.save("wave.gif",writer="imagemagick",fps=fps)


def main():
    data_path = "./output"
    if not os.path.isdir(data_path):
        raise ValueError("{} directory does not exist. Did you create it before running the MPI program?".format(data_path))

    file_list = ["./output/"+f for f in os.listdir(data_path)]
    file_list.sort()
    # Remove .gitkeep !
    file_list.pop(0)
    n_frames = len(file_list)

    x, y, u = collect_data(file_list)

    create_animation(x, y, u)

    ## Generating plain '.png' files
    #for i in range(40):
    #    timepoint = u[:,:,i]
    #    plt.imshow(timepoint, interpolation='nearest', cmap="gist_rainbow")
    #    plt.savefig("wave_"+str(i)+".png")

if __name__ == '__main__':
    main()

