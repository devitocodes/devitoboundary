import matplotlib
import matplotlib.pyplot as plt

__all__ = ['plot_boundary']


def plot_boundary(x, y, xs, ys, Lx, Ly):
    fig = plt.figure(figsize=(14, 7))
    ax1 = fig.add_subplot(111)
    ax1.plot(x, y, '-b')
    ax1.plot(xs, ys, '.r')
    ax1.axis([0, Lx, 0, Ly])
    ax1.set_xlabel('$x$')
    ax1.set_ylabel('$y$')
    ax1.minorticks_on()
    ax1.set_xticks(x, minor=True)
    ax1.grid(which='major', linestyle='-', linewidth='0.5', color='green')
    ax1.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    plt.show()
