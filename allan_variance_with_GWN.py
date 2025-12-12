import numpy as np
import allantools
from pathlib import Path
import matplotlib.pyplot as plt

def find_files_in_root(root_folder, fname):
    """
    Find all files in nested folders using pathlib.
    """
    root = Path(root_folder)
    found_files = list(root.rglob(fname))
    n_files =  len(found_files)
    print(f"Total files found: {n_files}")
    return found_files, n_files

def compute_allan_var(tau, data, fsample):
    """
    Computes Allan Variance.
    (Note: oadev returns Deviation, so we square it to get Variance)
    """
    (taus, allan_dev, _, _) = allantools.oadev(data, rate=fsample, data_type="freq", taus=tau)
    # Square deviation to get Variance (nm^2)
    allan_var = allan_dev ** 2
    return taus, allan_var

def analyze_slope(taus, variances):
    """
    Calculates the local slope of the Allan Variance plot in Log-Log space.
    White Noise = Slope of -1.
    """
    log_tau = np.log10(taus)
    log_var = np.log10(variances)

    # Calculate gradient (slope) of log-log data
    slopes = np.gradient(log_var, log_tau)
    return slopes

def find_linear_region(taus, variances, target_slope=-1.0, tolerance=0.2):
    """
    Finds the longest contiguous region where slope is within tolerance of target.

    Returns:
    start_tau, end_tau, indices_of_region
    """
    slopes = analyze_slope(taus, variances)
    print(slopes)

    # Check where slope is within tolerance (e.g., between -0.75 and -1.25)
    is_linear = np.abs(slopes - target_slope) < tolerance

    if not np.any(is_linear):
        return None, None, []

    indices = np.where(is_linear)[0]    # Find longest contiguous region
    splits = np.split(indices, np.where(np.diff(indices) != 1)[0] + 1)    # Group consecutive indices
    longest_segment = max(splits, key=len)    # Find the longest segment

    if len(longest_segment) < 2:
        return None, None, []

    start_idx = longest_segment[0]
    end_idx = longest_segment[-1]

    return taus[start_idx], taus[end_idx], longest_segment

def plot_allan_var(taus, allan_var, n_files, savedir, showfig=True):
    # Create figure with custom styling
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))


    # Calculate Averages
    mean_tau_x = np.mean(taus[0,:,:], axis=0)
    mean_allan_x = np.mean(allan_var[0,:,:], axis=0)
    mean_tau_y = np.mean(taus[1,:,:], axis=0)
    mean_allan_y = np.mean(allan_var[1,:,:], axis=0)

    # Analyze Slopes (White Noise Detection)
    t_start_x, t_end_x, ids_x = find_linear_region(mean_tau_x, mean_allan_x)
    t_start_y, t_end_y, ids_y = find_linear_region(mean_tau_y, mean_allan_y)

    if len(ids_x) > 0:
        GWN_min_idx = np.where(mean_allan_x == mean_allan_x[ids_x].min())[0][0]
        opt_tau_x = mean_tau_x[GWN_min_idx]
    if len(ids_y) > 0:
        GWN_min_idy = np.where(mean_allan_y == mean_allan_y[ids_y].min())[0][0]
        opt_tau_y = mean_tau_y[GWN_min_idy]

    # take the lowest tau value in the GWN region, combining the minimum from both lateral directions
    opt_tau_GWN = np.min([opt_tau_x, opt_tau_x])
    print(opt_tau_GWN)

    # Statistics for Global Minimum
    min_x = mean_allan_x.min()
    idx = mean_allan_x.argmin()
    min_y = mean_allan_y.min()
    idy = mean_allan_y.argmin()

    print("\n--- ANALYSIS RESULTS ---")
    print(f"X-Direction Minimum: {min_x:.2e} nm² at tau = {mean_tau_x[idx]:.4f} s")
    if t_start_x is not None:
        print(f"X-Direction White Noise Region (Slope ~ -1): {t_start_x:.4f} s to {t_end_x:.4f} s")

    print(f"Y-Direction Minimum: {min_y:.2e} nm² at tau = {mean_tau_y[idy]:.4f} s")
    if t_start_y is not None:
        print(f"Y-Direction White Noise Region (Slope ~ -1): {t_start_y:.4f} s to {t_end_y:.4f} s")
    print("------------------------\n")

    # --- PLOTTING ---
    # LEFT: X direction
    ax1.loglog(taus[0,:,:].T, allan_var[0,:,:].T, c="tab:blue", alpha=0.15)    # Plot individual traces
    ax1.loglog(mean_tau_x, mean_allan_x, 'k-', linewidth=2, label=f'Average (n={n_files})')    # Plot Average
    ax1.plot(mean_tau_x[idx], mean_allan_x[idx], 'o', c='tab:red', label='Global Minimum')    # Plot Minimum
    ax1.axvline(opt_tau_GWN, ls='--', c='tab:red', label=f'optimal GWN τ = {opt_tau_GWN:.3f} s')

    # Plot White Noise Linear Fit
    if len(ids_x) > 0:
        # Fit a line y = mx + c in log space for visualization
        log_t = np.log10(mean_tau_x[ids_x])
        log_v = np.log10(mean_allan_x[ids_x])
        coeffs = np.polyfit(log_t, log_v, 1) # Linear fit
        fit_y = 10**(np.polyval(coeffs, log_t))
        # ax1.loglog(mean_tau_x[ids_x], fit_y, 'r--', linewidth=2.5, label=f'White Noise fit (slope={coeffs[0]:.2f})')
        ax1.axvspan(t_start_x, t_end_x, color='green', alpha=0.1, label='White Noise Region')        # Add vertical spans to visualize the domain

    ax1.set_xlim(1e-4, 1)
    ax1.set_xlabel('τ (s)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Allan Variance (nm²)', fontsize=13, fontweight='bold')
    ax1.set_title('X Direction', fontsize=14, fontweight='bold', loc='left')
    ax1.legend(fontsize=9, framealpha=0.9, loc='upper right')
    ax1.grid(True, alpha=0.3, which='both', linestyle='--')

    # RIGHT: Y direction
    ax2.loglog(taus[1,:,:].T, allan_var[1,:,:].T, c="tab:orange", alpha=0.15)
    ax2.loglog(mean_tau_y, mean_allan_y, 'k-', linewidth=2, label=f'Average (n={n_files})')
    ax2.plot(mean_tau_y[idy], mean_allan_y[idy], 'o', c='tab:red', label='Global Minimum')
    ax2.axvline(opt_tau_GWN, ls='--', c='tab:red', label=f'optimal GWN τ = {opt_tau_GWN:.3f} s')
    
    # Plot White Noise Linear Fit
    if len(ids_y) > 0:
        log_t = np.log10(mean_tau_y[ids_y])
        log_v = np.log10(mean_allan_y[ids_y])
        coeffs = np.polyfit(log_t, log_v, 1)
        fit_y = 10**(np.polyval(coeffs, log_t))
        # ax2.loglog(mean_tau_y[ids_y], fit_y, 'r--', linewidth=2.5, label=f'White Noise fit (slope={coeffs[0]:.2f})')
        ax2.axvspan(t_start_y, t_end_y, color='green', alpha=0.1, label='White Noise Region') # Add vertical spans to visualize the domain

    ax2.set_xlim(1e-4, 1)
    ax2.set_xlabel('τ (s)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Allan Variance (nm²)', fontsize=13, fontweight='bold')
    ax2.set_title('Y Direction', fontsize=14, fontweight='bold', loc='left')
    ax2.legend(fontsize=9, framealpha=0.9, loc='upper right')
    ax2.grid(True, alpha=0.3, which='both', linestyle='--')

    plt.tight_layout()
    plt.savefig(savedir/"GWN_AV_analysis.pdf")
    if showfig:
        plt.show()


# Main
if __name__ == "__main__":
    save_dir = Path(r"D:\USE FOR STEPFIT\V15P\20251028_TitinV15P bead with FtsH_TV15P_FT2512")
    cali_root = Path(r"D:\LocalSaves\20251028_TitinV15P bead with FtsH_TV15P_FT2512")
    print(cali_root.name[:8])
    filename = "cali_nm.dat"
    fsample = 1e4
    cali_paths, n_files = find_files_in_root(cali_root, filename)
    tau = np.logspace(-5, 0, 50)
    tau = 1/fsample*2 ** np.linspace(0, 14, 15)
    allan_var = np.zeros((2, n_files, len(tau)))
    taus = np.zeros((2, n_files, len(tau)))

    # Analyze each file
    for i, path in enumerate(cali_paths):
        try:
            nm_data = np.loadtxt(path)
            taus[0,i,:], allan_var[0,i,:] = compute_allan_var(tau, nm_data[:,0], fsample)
            taus[1,i,:], allan_var[1,i,:] = compute_allan_var(tau, nm_data[:,1], fsample)
        except:
            print(f'Something wrong with file at {path}')
            n_files -= 1

    plot_allan_var(taus, allan_var, n_files, cali_root, showfig=True)