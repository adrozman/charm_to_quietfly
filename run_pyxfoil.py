# ################################################################
# I'm using a locally stored pyxfoil install now. I belive
# this best for now in case we have to modify the read results 
# code of pyxfoil to amend the error when the .res file has double
# negative value. See the point of error in the the output of the 
# run log.
# resfile line issue example
# 1.05006  0.00043  0.00440-10.01466  0.000536  0.000246  2.180225    2.1528    1.6279  0.02464 -0.00537 -0.40174
# 
# we can possibly change this in the future
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
pyxfoil_src_path = os.path.join(current_dir, "pyxfoil", "src")

# Add that 'src' folder to the front of the search path
sys.path.insert(0, pyxfoil_src_path)

from pyxfoil import Xfoil, set_workdir, set_xfoilexe

############################


import numpy as np
from concurrent.futures import ProcessPoolExecutor

# -------------------------
# GLOBAL INPUTS
# -------------------------
aoa_file = "npy_files/polar_aoa.npy"
mach_file = "npy_files/polar_mach.npy"
xs_file = "npy_files/radial_locs.txt"

airfoil_dir = "inputs/Airfoils"
bg_file = "inputs/SUIbg.inp"

# Constants
C_SOUND = 343.0  # Speed of sound in m/s at sea level
RHO_INF = 1.225  # Density at sea level in kg/m³
P_INF = 101325.0  # Pressure at sea level in Pa
MU_INF = 1.789e-5

radius = 7.5 / 12 # ft, the default charm unit

#Re = 1e5 # will calculate for each stations
xtr_top = 0.2
xtr_bot = 0.2

results_root = "pyxfoil_results"
# how many ways you want to parallelize xfoil runs
N_WORKERS = 16

# Store the original working directory
original_dir = os.getcwd()

# -------------------------
# Function that reads stations and chords
# -------------------------
def read_stations_chords(bg_file):

    with open(bg_file) as f:
        lines = f.readlines()

    root_cutout = 0
    for j, line in enumerate(lines):
        if "CUTOUT" in line:
            root_cutout = float(lines[j+1].split()[0])
            break

    print(root_cutout)

    # read station widths
    widths = []
    for j, line in enumerate(lines):
        if "SL" in line:
            i = 1
            # read lines with numbers until you get a line with letters
            while not any(char.isalpha() for char in lines[j+i].strip()):
                values = lines[j+i].strip().split()
                for value in values:
                    widths.append(float(value))
                i += 1
            break

    # add the widths to the root cutout to get the station locations (centers of strip)
    stations = [root_cutout + widths[0]/2]
    for width in widths[1:]:
        stations.append(stations[-1]+width/2)

    print(stations)

    # read chord values
    chords = []
    for j, line in enumerate(lines):
        if "CHORD" in line:
            i = 1
            while not any(char.isalpha() for char in lines[j+i].strip()):
                values = lines[j+i].strip().split()
                for value in values:
                    chords.append(float(value))
                i += 1
            break

    # report the chord at the center of each segment (average of ends)
    chords = (np.array(chords[:-1]) + np.array(chords[1:])) / 2

    return stations, chords

# -------------------------
# Helper function to clean airfoil file
# -------------------------
def clean_airfoil_file(input_file, output_file, new_header):
    """Copy airfoil file and replace the header with a clean name"""
    with open(input_file, 'r') as f:
        lines = f.readlines()

    # Replace first line (header) with clean name
    lines[0] = new_header + '\n'

    with open(output_file, 'w') as f:
        f.writelines(lines)

# -------------------------
# function to run an xfoil case
# -------------------------

def run_single_xfoil(inputs):
    # Unpack the task data
    rtr, psi, airfoil_rR, alpha, mach, chord, airfoil_file, span_dir, original_dir = inputs

    # Constants (ensure these are accessible or passed in)
    # RHO_INF, C_SOUND, MU_INF, xtr_top, xtr_bot

    try:
        os.makedirs(span_dir, exist_ok=True)
        # Use absolute paths instead of os.chdir if possible.
        # If your Xfoil library REQUIRES chdir, it's okay inside a Process,
        # but you must be careful.
        os.chdir(span_dir)
        set_workdir('.')

        local_airfoil = f"airfoil_{int(round(airfoil_rR*100))}.dat"
        clean_header = f"SUI_rR{int(round(airfoil_rR*100))}"

        # Assume clean_airfoil_file and Xfoil are available in the worker scope
        clean_airfoil_file(airfoil_file, local_airfoil, clean_header)

        xfoil = Xfoil(clean_header)
        xfoil.points_from_dat(local_airfoil)
        xfoil.set_ppar(180)

        Re = RHO_INF * mach * C_SOUND * chord / MU_INF

        # this redirects the output from the xfoil run to reduce output clutter
        set_xfoilexe("( > /dev/null 2>&1 xfoil )")

        xfoil.run_result(
            alpha,
            mach=mach,
            Re=Re,
            xtrtop=xtr_top,
            xtrbot=xtr_bot
        )
        return "SUCCESS"
    except Exception as e:
        return f"FAIL: rotor {rtr+1}, psi {psi}, r/R {airfoil_rR:.2f}: {e}"
    finally:
        os.chdir(original_dir)

if __name__ == "__main__":
    # -------------------------
    # set up inputs for cases to run
    # -------------------------

    # read bg file
    stations, chords = read_stations_chords(bg_file)

    # Define airfoil r/R values (without decimals in filename)
    airfoil_rR_values = [0.12, 0.16, 0.21, 0.25, 0.29, 0.33, 0.37, 0.41, 0.45, 0.49, 
                         0.54, 0.58, 0.62, 0.66, 0.70, 0.74, 0.78, 0.82, 0.87, 0.91, 0.95, 0.99]

    # I tried this but the airfoil file names round a bit differently at 0.16, 0.83, 0.99...
    # it's maybe not the best to have the airfoil filenames hardcoded to these number tuples
    # for now, I'm leaving the hardcoded array above.
    #airfoil_rR_values = np.array(stations) / radius

    chords = np.array(chords) * 0.3048 # convert ft to meters for further calcs

    # -------------------------
    # Create mapping from r/R to filename
    # -------------------------
    airfoil_files = {}
    for rR in airfoil_rR_values:
        # Convert to filename format (e.g., 0.12 -> "12", 0.99 -> "99")
        rR_str = str(int(round(rR * 100)))
        filename = f"SUI_{rR_str}.dat"
        filepath = os.path.join(original_dir, airfoil_dir, filename)
        
        # Check if file exists
        if os.path.exists(filepath):
            airfoil_files[rR] = filepath
            print(f"Found: {filename}")
        else:
            print(f"WARNING: File not found: {filepath}")

    if len(airfoil_files) == 0:
        print("\nERROR: No airfoil files found! Check the airfoil_dir and filenames.")
        exit(1)

    print(f"\nFound {len(airfoil_files)} airfoil files")

    # -------------------------
    # LOAD CHARM DATA
    # -------------------------
    polar_aoa = np.load(aoa_file)      # shape (NROTOR, NPSI, NX)
    polar_mach = np.load(mach_file)

    NROTOR, NPSI, NX = polar_aoa.shape

    # Calculate r/R values for the NPY data
    #npy_rR_values = np.linspace(0, 1, NX)
    npy_rR_values = np.loadtxt(xs_file)

    # Find closest NPY indices for each airfoil r/R
    airfoil_indices = []
    airfoil_rR_matched = []
    valid_airfoil_rR = []

    for airfoil_rR in airfoil_rR_values:
        # Only process if we have the airfoil file
        if airfoil_rR not in airfoil_files:
            continue
        
        # Find the closest index in NPY data
        idx = np.argmin(np.abs(npy_rR_values - airfoil_rR))
        airfoil_indices.append(idx)
        airfoil_rR_matched.append(npy_rR_values[idx])
        valid_airfoil_rR.append(airfoil_rR)
        print(f"Airfoil r/R={airfoil_rR:.2f} matched to NPY index {idx} (r/R={npy_rR_values[idx]:.4f})")

    print(f"\nLoaded data: {NROTOR} rotors, {NPSI} psi positions, {len(airfoil_indices)} airfoil locations")
    print(f"Total runs to attempt: {NROTOR * NPSI * len(airfoil_indices)}")

    results_root_abs = os.path.abspath(results_root)
    os.makedirs(results_root_abs, exist_ok=True)

    # -------------------------
    # set up inputs for cases to run
    # -------------------------

    inputs = []
    skip_count = 0

    for rtr in range(NROTOR):
        for psi in range(NPSI):
            for i, (ix, airfoil_rR) in enumerate(zip(airfoil_indices, valid_airfoil_rR)):
                alpha = polar_aoa[rtr, psi, ix]
                mach = polar_mach[rtr, psi, ix]

                # Filter out unsuited cases
                if not np.isfinite(alpha) or not np.isfinite(mach) or mach <= 0.0:
                    skip_count += 1
                    continue

                span_dir = os.path.abspath(os.path.join(results_root_abs, f"rotor_{rtr+1}", f"psi_{psi:03d}", f"r_{ix:03d}"))
                
                # parallel executor likes one input. pack into a tuple
                input_data = (rtr, psi, airfoil_rR, alpha, mach, chords[i], 
                             airfoil_files[airfoil_rR], span_dir, original_dir)
                inputs.append(input_data)

    # 2. EXECUTE IN PARALLEL
    print(f"Starting {len(inputs)} XFOIL runs...")

    success_count = 0
    fail_count = 0

    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        # map returns results in the same order as inputs
        results = list(executor.map(run_single_xfoil, inputs))

    # run in serial (debugging)
    #results = [run_single_xfoil(input_i) for input_i in inputs[:3]]

    # 3. ANALYZE SUCCESSES
    for res in results:
        if res == "SUCCESS":
            success_count += 1
        else:
            fail_count += 1
            print(res)

    print("\n" + "="*60)
    print("All XFOIL runs completed.")
    print(f"Success: {success_count}, Fail: {fail_count}, Skipped: {skip_count}")
    print("="*60)

