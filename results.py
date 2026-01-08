import numpy as np
import os
import glob

# ----------------------------------
# Settings
# ----------------------------------
RES_DIR = "pyxfoil_results"
XC_TARGET = 0.99
OUT_DIR = "outputs/bl_final_outputs.npy"
bg_file = "inputs/SUIbg.inp"

# Constants
C_SOUND = 343.0  # Speed of sound in m/s at sea level
RHO_INF = 1.225  # Density at sea level in kg/m³
P_INF = 101325.0  # Pressure at sea level in Pa

# Define airfoil r/R values (same as in the XFOIL run script)
airfoil_rR_values = [0.12, 0.16, 0.21, 0.25, 0.29, 0.33, 0.37, 0.41, 0.45, 0.49, 
                     0.54, 0.58, 0.62, 0.66, 0.70, 0.74, 0.78, 0.82, 0.87, 0.91, 0.95, 0.99]

# ----------------------------------
# Helper: read .res file
# ----------------------------------
def read_res_file(fname):
    """Read .res file and return structured data"""
    data = []

    with open(fname, "r") as f:
        for line in f:
            if line.strip().startswith("#"):
                continue
            if not line.strip():
                continue
            parts = line.split()
            # Need at least 11 columns (up to K column which is index 10)
            if len(parts) < 11:
                continue
            try:
                # Read only the first 11 columns we actually need
                data.append([float(p) for p in parts[:11]])
            except ValueError:
                continue

    if len(data) == 0:
        return None
    
    return np.array(data)


# ----------------------------------
# Helper: interpolate at x/c = 0.99
# ----------------------------------
def interp_at_xc(xc, var, xc_target):
    return np.interp(xc_target, xc, var)


# ----------------------------------
# Helper: calculate dCp/dx at x/c = 0.99
# ----------------------------------
def calculate_dpdx_at_99c(xc, cp, x_physical, xc_target, U_inf):
    """
    Calculate dCp/dx at x/c = 0.99 using central difference
    Then dimensionalize to dP/dx
    """
    if len(xc) < 3:
        return 0.0
    
    # Find indices around xc_target for central difference
    idx = np.argmin(np.abs(xc - xc_target))
    
    # Make sure we can do central difference
    if idx == 0:
        idx = 1
    if idx == len(xc) - 1:
        idx = len(xc) - 2
    
    # Central difference for dCp/dx
    dx = x_physical[idx+1] - x_physical[idx-1]
    dcp = cp[idx+1] - cp[idx-1]
    dcp_dx = dcp / dx if dx != 0 else 0.0
    
    # Dimensionalize: Cp = (P - P_inf) / (0.5 * rho * U^2)
    # So dP/dx = 0.5 * rho * U^2 * dCp/dx
    dp_dx = 0.5 * RHO_INF * U_inf**2 * dcp_dx
    
    return dp_dx


# ----------------------------------
# Helper: extract r/R from directory index
# ----------------------------------
def get_rR_from_index(r_num, total_nx):
    """
    Map directory index to actual r/R value from NPY data
    and find closest airfoil r/R
    """
    # Calculate NPY r/R values
    npy_rR = r_num / (total_nx - 1) if total_nx > 1 else 0.0
    
    # Find closest airfoil r/R
    closest_idx = np.argmin(np.abs(np.array(airfoil_rR_values) - npy_rR))
    return airfoil_rR_values[closest_idx]


# ----------------------------------
# Helper: extract values at 0.99 chord
# ----------------------------------
def extract_values_at_99c(resfile, chord_length=1.0):
    """Extract upper and lower surface values at x/c = 0.99 with derived quantities"""
    data = read_res_file(resfile)
    
    if data is None or len(data) == 0:
        raise ValueError("No valid data in .res file")
    
    if data.ndim != 2 or data.shape[1] < 11:
        raise ValueError(f"Invalid data shape: {data.shape}, expected at least 11 columns")

    # Columns from .res file (only reading first 11)
    # s x y Ue/Vinf Dstar Theta Cf H H* P m K
    # 0 1 2    3      4     5    6  7  8  9 10
    s = data[:, 0]
    x = data[:, 1]
    y = data[:, 2]
    ue_ratio = data[:, 3]  # Ue/Vinf
    ds = data[:, 4]  # Dstar (deltastar)
    th = data[:, 5]  # Theta
    cf = data[:, 6]
    H = data[:, 7]
    
    # Physical x coordinate (assuming chord_length normalization)
    x_physical = x * chord_length

    # Split upper / lower surface
    idx_le = np.argmin(x)  # leading edge
    
    if idx_le == 0 or idx_le == len(x) - 1:
        raise ValueError("Cannot identify leading edge properly")

    upper = slice(0, idx_le + 1)
    lower = slice(idx_le + 1, None)

    results = {}

    for label, sl in zip(["upper", "lower"], [upper, lower]):
        x_s = x[sl]
        x_phys_s = x_physical[sl]
        ue_s = ue_ratio[sl]
        ds_s = ds[sl]
        th_s = th[sl]
        cf_s = cf[sl]
        H_s = H[sl]
        cp_s = 1-ue_ratio[sl]**2
        
        if len(x_s) < 2:
            raise ValueError(f"Insufficient data points for {label} surface")

        # ensure monotonic x for interpolation
        order = np.argsort(x_s)
        x_s = x_s[order]
        x_phys_s = x_phys_s[order]
        ue_s = ue_s[order]
        ds_s = ds_s[order]
        th_s = th_s[order]
        cf_s = cf_s[order]
        H_s = H_s[order]
        cp_s = cp_s[order]
        
        # Check if XC_TARGET is within range
        if XC_TARGET < x_s.min() or XC_TARGET > x_s.max():
            raise ValueError(f"x/c = {XC_TARGET} outside range [{x_s.min():.3f}, {x_s.max():.3f}] for {label}")

        # Interpolate values at XC_TARGET
        ue_val = interp_at_xc(x_s, ue_s, XC_TARGET)
        ds_val = interp_at_xc(x_s, ds_s, XC_TARGET)
        th_val = interp_at_xc(x_s, th_s, XC_TARGET)
        cf_val = interp_at_xc(x_s, cf_s, XC_TARGET)
        H_val = interp_at_xc(x_s, H_s, XC_TARGET)
        
        # get m_inf from the filename...
        Me = float(resfile.split("/")[-1].split("_")[-2])
        
        # Hk = (H - 0.290*Me^2) / (1 + 0.113*Me^2)
        Hk = (H_val - 0.290 * Me**2) / (1 + 0.113 * Me**2) if (1 + 0.113 * Me**2) != 0 else H_val
        
        # delta = theta * (3.15 + 1.72/(Hk - 1)) + deltastar
        delta = th_val * (3.15 + 1.72 / (Hk - 1)) + ds_val if abs(Hk - 1) > 0.01 else th_val * 3.15 + ds_val
        
        # Calculate dP/dx at x/c = 0.99
        U_inf = Me*C_SOUND
        dp_dx = calculate_dpdx_at_99c(x_s, cp_s, x_phys_s, XC_TARGET, U_inf)

        results[label] = {
            "ue": ue_val,
            "delta": delta,
            "ds": ds_val,
            "th": th_val,
            "dp_dx": dp_dx,
            "cf": cf_val,
        }

    return results

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


# ----------------------------------
# Main processing
# ----------------------------------

# Find all rotor directories
rotor_dirs = sorted(glob.glob(os.path.join(RES_DIR, "rotor_*")))

total_processed = 0
total_failed = 0

# First pass: determine total NX from any rotor/psi combination
sample_psi = glob.glob(os.path.join(rotor_dirs[0], "psi_*"))[0]
sample_r_dirs = glob.glob(os.path.join(sample_psi, "r_*"))
TOTAL_NX = max([int(os.path.basename(d).split("_")[1]) for d in sample_r_dirs]) + 1

print(f"Detected total spanwise locations (NX): {TOTAL_NX}")

stations, chords = read_stations_chords(bg_file)

chords *= .3048

VAR_KEYS = ["ue", "ds", "th", "cf", "H", "Me", "Hk", "delta", "dp_dx"]
NVARS = len(VAR_KEYS)

# hard-coding this but needs to be updated!
NROTOR = 4
NPSI = 48
NR = 22

VAR_KEYS = ["ue", "delta", "ds", "th", "cf", "dp_dx"]
NVARS = len(VAR_KEYS) + 1 # we want r/R also

# [rotor, psi, r_R, [upper,lower], var]
bl_data = np.full((NROTOR, NPSI, NR, 2, NVARS), np.nan)

for rotor_idx, rotor_dir in enumerate(rotor_dirs):
    rotor_name = os.path.basename(rotor_dir)

    # Find all psi directories within this rotor
    psi_dirs = sorted(glob.glob(os.path.join(rotor_dir, "psi_*")))

    for psi_idx, psi_dir in enumerate(psi_dirs):
        psi_name = os.path.basename(psi_dir)
        psi_num = psi_name.split("_")[1]

        # Find all r directories within this psi
        r_dirs = sorted(glob.glob(os.path.join(psi_dir, "r_*")))

        for r_idx, r_dir in enumerate(r_dirs):
            r_name = os.path.basename(r_dir)
            r_num = int(r_name.split("_")[1])
            r_over_R = get_rR_from_index(r_num, TOTAL_NX)

            res_files = glob.glob(os.path.join(r_dir, "*.res"))
            if len(res_files) == 0:
                total_failed += 1
                continue

            resfile = res_files[0]

            try:
                results = extract_values_at_99c(resfile, chord_length=chords[r_idx])

                for s_idx, label in enumerate(["upper","lower"]):
                    bl_data[rotor_idx,psi_idx,r_idx,s_idx,0] = r_over_R
                    for v_idx, key in enumerate(VAR_KEYS):
                        bl_data[rotor_idx,psi_idx,r_idx,s_idx,v_idx+1] = results[label][key]

                total_processed += 1

            except Exception as e:
                total_failed += 1
                print(f"XFOIL convergence failure - skipping r/R={r_over_R:.2f} at {r_dir}: {e}")
                continue

np.save(OUT_DIR, bl_data)

print("\n" + "="*60)
print("Processing complete!")
print(f"Successfully processed: {total_processed} cases")
print(f"Failed/skipped: {total_failed} cases")
print("="*60)
