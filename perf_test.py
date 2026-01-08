#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 15 14:55:54 2025

@author: arozman
"""

import os
import numpy as np
import matplotlib.pyplot as plt

def read_perf(fname, NROTOR, NPSI, NX, MREV, var_ix):

    # data = np.loadtxt("SUImrevperf.dat",skiprows=5, usecols=(0,1,2,10))
    
    with open(fname) as f:
        lines=f.readlines()
        
    rotor_delims = []
    for j in range(len(lines)):
        if "ROTOR" in lines[j]:
            rotor_delims.append(j)
            
    var = np.zeros((NROTOR, MREV, NPSI, NX))
    
    delim_ix = 1 # Start this at 1 because first 'ROTOR' instance is part of header
    for m in range(MREV):
        for r in range(NROTOR):
            data = np.loadtxt(lines[rotor_delims[delim_ix]+3:rotor_delims[delim_ix]+3+NPSI*NX])
            var[r, m, :, :] = data[:,var_ix].reshape((NPSI,NX))
            delim_ix += 1
            
    psis = np.unique(data[:,0])
    xs = np.unique(data[:,1])
            
    var_mean = np.mean(var, axis=1)

    return var_mean, psis, xs

# if given fname, will save to png of that name
def plot_polar(psis, xs, polar_data, vmin, vmax, title=None, fname=None):
    # Convert psis to radians
    theta = np.radians(psis)
    r = xs

    # Prepare for cyclic closing (fixing the "Pac-Man" gap)
    theta_cyclic = np.append(theta, theta[0] + 2*np.pi)
    # Create Meshgrid (indexing='ij' matches shape (NPSI+1, NX))
    Theta, R = np.meshgrid(theta_cyclic, r, indexing='ij')

    # Create a 2x2 grid, specifying that all subplots are polar
    fig, axs = plt.subplots(2, 2, figsize=(12, 10), 
                            subplot_kw={'projection': 'polar'},
                            layout='constrained')
    axs_flat = axs.flatten(order='F')

    mesh_objects = [] # Store plots to define colorbar later
    for i in range(NROTOR):
        ax = axs_flat[i]
        
        # Extract single rotor data
        Z = polar_data[i, :, :]
        
        # Make data cyclic (append first row to end) to match Theta_cyclic
        Z_cyclic = np.vstack((Z, Z[0, :]))
        
        c = ax.pcolormesh(Theta, R, Z_cyclic, cmap='jet', shading='auto',vmin=vmin, 
                          vmax=vmax, edgecolor='none')
        
        if i == 1 or i ==2:
            ax.set_theta_direction(-1)
            # ax.set_theta_zero_location('E')
            
        else:
            ax.set_theta_direction(1)
            # ax.set_theta_zero_location('E')

        mesh_objects.append(c)

        # Formatting individual plots
        ax.set_title(f"Rotor {i+1}",fontsize=40,pad=20)
        ax.grid(False)
        
        ax.tick_params(axis='x',labelsize=20,pad=10)
        ax.set_yticks([])
        ax.set_yticklabels([]) 
        
    cb = fig.colorbar(mesh_objects[-1], ax=axs, location='right', aspect=20, 
                      pad=0.05)
    
    cb.ax.yaxis.get_offset_text().set_fontsize(24)

    cb.formatter.set_scientific(True)
    cb.formatter.set_useMathText(True)
    cb.formatter.set_powerlimits((-2,2))
    
    cb.update_ticks()
    
    cb.set_label(title, fontsize=40)
    cb.ax.tick_params(labelsize=24)

    # plt.suptitle("Mean AoA Polar Distribution across Rotors", y=1.02, fontsize=14)
    if fname:
        plt.savefig(fname)

# %%
if __name__ == "__main__":
    # Get perf file path from environment variable
    perf_file_path = os.environ.get('CHARM_PERF_PATH')
    
    if perf_file_path is None:
        print("ERROR: CHARM_PERF_PATH environment variable not set")
        print("Please set it in setup.sh or run: export CHARM_PERF_PATH=/path/to/perf.dat")
        exit(1)
    
    # Check if file exists
    if not os.path.exists(perf_file_path):
        print(f"ERROR: File not found: {perf_file_path}")
        exit(1)
    
    print(f"Processing perf file: {perf_file_path}")
    
    # these values are all in the header of *perf.dat. At some point, this script
    # should be updated to read these values from the header automatically.
    NROTOR = 4
    NPSI = 48
    NX = 66
    MREV = 10

    # columns order in the perf file:
    # psi x=r/R dx dCT/dx dCQI/dx dCQP/dx X-force Y-force -Z-force P-mom AOAG CL2D
    # CD2D CM2D AOA2D MACH2D U-radial V-aft W-down W-induced Circulation

    os.makedirs("outputs", exist_ok=True)

    # read perf file
    var_ix = 14
    polar_aoa, psis, xs = read_perf(perf_file_path, NROTOR, NPSI, NX, MREV, var_ix)
    # visualize 
    plot_polar(psis, xs, polar_aoa, -10, 10, title='Local Airfoil AoA (deg)', fname='outputs/polar_aoa.png')

    var_ix = 15
    polar_mach, psis, xs = read_perf(perf_file_path, NROTOR, NPSI, NX, MREV, var_ix)
    plot_polar(psis, xs, polar_mach, 0, 0.4, title='Local Airfoil Mach', fname='outputs/polar_mach.png')

    # Create output directory for npy files
    outdir = "npy_files"
    os.makedirs(outdir, exist_ok=True)

    # aoa and Mach arrays are [N_Prop, N_PSI, NX]
    np.save(os.path.join(outdir, "polar_aoa.npy"), polar_aoa)
    np.save(os.path.join(outdir, "polar_mach.npy"), polar_mach)
    print("Saved aoa and mach arrays to npy_files/")
    print("Saved plots: polar_aoa.png and polar_mach.png")

    np.savetxt(os.path.join(outdir, "radial_locs.txt"), xs)
