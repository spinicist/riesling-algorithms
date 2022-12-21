####################################################
#
# Convert the CG SENSE Reproducibility challenge brain
# dataset to riesling h5 format.
#
# How to use:
# - Download brain challenge and reference dataset
#       download_rrsg_data.sh
#
# - Create folder for riesling data named riesling_data
# Run script
#   python3 convert_data.py
#
# Emil Ljungberg and Tobias Wood
# December 2020, King's Colleg London
####################################################

import os
import sys

import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import requests
import riesling as rl
import contextlib

def download_data(fname, data_dir):
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)

    if not os.path.isfile(f'{data_dir}/{fname}'):
        print(f'Downloading {fname}')
        res = requests.get(f'https://zenodo.org/record/3975887/files/{fname}')
        with open(f'{data_dir}/{fname}', 'wb') as h5f:
            h5f.write(res.content)
    else:
        print(f'Already downloaded {fname}')


def create_info(matrix, tr, voxel_size, origin, direction):
    D = np.dtype({'names': ['matrix', 'tr', 'voxel_size', 'origin', 'direction'],
                  'formats': [('<i8', (3,)), '<f4', ('<f4', (3,)), ('<f4', (3,)), ('<f4', (9,))]})
    info = np.array([(matrix, tr, voxel_size, origin, direction)], dtype=D)
    return info

def convert_rrsg(input_fname, output_fname, matrix, voxel_size):
    data_f = h5py.File(input_fname, 'r')
    rawdata = data_f['rawdata'][...]
    traj = data_f['trajectory'][...]
    data_f.close()

    # Reshape data
    rawdata = np.transpose(rawdata, [0, 2, 1, 3])[:, np.newaxis, :, :, :]

    # Scale trajectory
    traj = traj/np.max(abs(traj)) * 0.5
    traj = traj.transpose((2, 1, 0))

    # Strip 3rd dimension of radial dataset
    if traj.shape[2] == 3:
        traj = traj[:,:,0:2]

    rl.data.write_noncartesian(output_fname, rawdata, traj, matrix, voxel_size)
    print("H5 file saved to {}".format(output_fname))

def diffsAlgos(fnames, titles, dsets1, dsets2,
               axis='z', slice_pos=0.5, difflim=None, diffmap='cmr.ember', scale=1, rotates=0):

    ifr = 0
    iv = 0
    ic = 0

    comp='mag'
    clim=None
    cmap='gray'
    interp='none'

    slice_dim, img_dims = rl.plot._get_dims('z', -1)
    with contextlib.ExitStack() as stack:
        files = [stack.enter_context(h5py.File(fn, 'r')) for fn in fnames]
        d1 = [f[dset] for f, dset in zip(files, dsets1)]
        d2 = [f[dset] for f, dset in zip(files, dsets2)]
        slice_index = int(np.floor(d1[0].shape[slice_dim] * slice_pos))
        imgs1 = [np.squeeze(rl.plot._get_slices(D, slice_dim, slice(slice_index, slice_index+1), img_dims))*scale for D in d1]
        imgs2 = [np.squeeze(rl.plot._get_slices(D, slice_dim, slice(slice_index, slice_index+1), img_dims))*scale for D in d2]
        nI = len(imgs1)
        diffs = []
        for ii in range(nI):
            diffs.append(100 * (imgs1[ii] - imgs2[ii]) / np.max(np.abs(imgs2[ii])))
        clim, cmap = rl.plot._get_colors(clim, cmap, imgs1[0], 'mag')
        difflim, diffmap = rl.plot._get_colors(difflim, diffmap, diffs[0], 'real')
        fig, ax = plt.subplots(2, nI, figsize=(
            nI*rl.plot.rc['figsize'], 2*rl.plot.rc['figsize']), facecolor='black')
        for ii in range(nI):
            imi = rl.plot._draw(ax[0, ii], rl.plot._orient(imgs1[ii], rotates), 'mag', clim, cmap)
            imd = rl.plot._draw(ax[1, ii], rl.plot._orient(diffs[ii], rotates), 'real', difflim, diffmap)
            ax[0, ii].text(0.1, 0.9, titles[ii], color='white', transform=ax[0, ii].transAxes, ha='left',
                fontsize=rl.plot.rc['fontsize'], path_effects=rl.plot.rc['effects'])
        fig.subplots_adjust(wspace=0, hspace=0)
        rl.plot._add_colorbar(True, 'real', fig, imd, difflim, 'Diff (%)', ax[1, :])
        plt.close()

        plt.close()
        return fig

def resid_plot(files, title, titles, ddir, skips, cols):
    matplotlib.rc('font', **{'size':14})
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.autoscale(enable=True, axis='x', tight=True)
    for f, sk, col in zip(files, skips, cols):
        data = np.genfromtxt(f'{ddir}/{f}.txt', delimiter=' ', max_rows=32, skip_header=sk)
        ax.plot(np.log(data[:, col]))
    ax.legend(titles, ncol=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel("log|r| or log$|A^†r|$")
    fig.suptitle(title)
    fig.tight_layout()
    plt.close()
    return fig
