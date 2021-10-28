import numpy as np
import matplotlib.pyplot as plt

infile = '../data/seismic_topography_wavefield.npy'
dat = np.load(infile)

plot_extent = [0, 10800,
               -3900, 1500]
for i in range(dat.shape[0] - 1):
    fig = plt.figure()
    plt.imshow(dat[i, :, 108, :].T,
               origin='lower', extent=plot_extent,
               vmin=-6, vmax=6, cmap='seismic')
    plt.xlabel("x (m)")
    plt.ylabel("z (m)")
    plt.savefig("animation/frame_%s" % str(i))
    # plt.show()
    plt.close()
