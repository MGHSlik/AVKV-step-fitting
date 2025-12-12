import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import allantools
from AVKV_step_detection import * # Assuming this file exists in your directory

# ==========================================
# 1. PHYSICS & BENCHMARK GENERATION
# ==========================================
def generate_benchmark_trace(duration, fs, step_size_avg=2, k_rate=1.25):
    """
    Generates benchmark data according to Carter et al (2008). / Kalafut-Visscher (2008) methods.
    Workflow:
    1. Generates 10 MHz high-speed geometric trace with exponential dwells.
    2. Inserts 0.2 ms linear transitions between steps.
    3. Blocks averages (integrates) down to acquisition frequency (fs).
    4. Adds Thermophilic/OT noise components.
    """
    print("--- GENERATING KV BENCHMARK DATA (Aquifex 50°C) ---")
    # --- A. HIGH SPEED GENERATION (10 MHz) ---
    fs_high = 10_000_000 # 10 MHz
    n_high = int(duration * fs_high)
    dt_high = 1/fs_high

    # Transition parameters
    transition_time = 0.0002 # 0.2 ms transition
    n_trans_pts = int(transition_time * fs_high)

    # Generate Dwells and Steps
    high_res_trace = np.zeros(n_high)
    current_idx = 0
    current_pos = 0.0

    # Store truth for the low-res timepoints later
    # Downsample the high_res_trace to get the "Effective Truth"
    while current_idx < n_high:
        # 1. Random Dwell Time (Exponential distribution)
        # k_rate is events per second. 
        dwell_time = np.random.exponential(1.0/k_rate)
        n_dwell = int(dwell_time * fs_high)

        end_dwell = min(current_idx + n_dwell, n_high)
        high_res_trace[current_idx:end_dwell] = current_pos
        current_idx = end_dwell

        if current_idx >= n_high:
            break

        # 2. Determine Step Size
        # Paper implies random stepping, but for FtsH we usually look for 
        # specific step sizes. Therefore we direction randomly but keep size fixed
        direction = np.random.choice([1, -1]) 
        # Optional: Add variance to step size: np.random.normal(step_size_avg, 1.0)
        next_pos = current_pos + (direction * step_size_avg)

        # 3. Insert Finite Transition (0.2 ms ramp)
        end_trans = min(current_idx + n_trans_pts, n_high)

        # Linear ramp from current_pos to next_pos
        ramp_len = end_trans - current_idx
        if ramp_len > 0:
            ramp = np.linspace(current_pos, next_pos, ramp_len)
            high_res_trace[current_idx:end_trans] = ramp

        current_pos = next_pos
        current_idx = end_trans

    # --- B. CAMERA INTEGRATION (Downsampling) ---
    # We avail of the reshape/mean trick to simulate camera exposure
    downsample_factor = int(fs_high / fs)
    n_samples_low = len(high_res_trace) // downsample_factor

    # Crop to multiple of factor
    high_res_cropped = high_res_trace[:n_samples_low * downsample_factor]

    # The "Effective Ground Truth" is the averaged signal (smoothed steps)
    truth_low = np.mean(high_res_cropped.reshape(-1, downsample_factor), axis=1)

    # Time vector for low res
    t_low = np.linspace(0, duration, n_samples_low)
    N = len(truth_low)
    dt = 1/fs

    # --- C. ADDING THERMOPHILIC NOISE ---
    # (Matches your original noise logic)
    kb_T = 1.38e-2*(273+50) # 50C
    k_trap = 0.15           # pN/nm
    gamma = 0.000015        # Drag

    # 1. Brownian (Colored)
    tau_relax = gamma / k_trap
    print(f"Physics: Relaxation Time = {tau_relax*1000:.3f} ms vs Sampling {dt*1000:.3f} ms")

    # Langevin scalar
    exp_term = np.exp(-dt/tau_relax)
    noise_amp = np.sqrt((kb_T/k_trap) * (1 - exp_term**2))

    brownian = np.zeros(N)
    x = 0
    rand_kick = np.random.normal(0, 1, N)
    for i in range(1, N):
        x = x * exp_term + noise_amp * rand_kick[i]
        brownian[i] = x

    # 2. Pink Drift
    white = np.random.normal(0, 1, N)
    X = np.fft.rfft(white)
    frequencies = np.fft.rfftfreq(N)
    frequencies[0] = frequencies[1]
    X_pink = X / np.sqrt(frequencies)
    pink = np.fft.irfft(X_pink, n=N)
    pink = pink / np.std(pink) * 3.0 # Amplitude

    # 3. Hum & Shot noise
    hum = 0.5 * np.sin(2 * np.pi * 50 * t_low)
    shot = np.random.normal(0, 2.5, N)

    # Combine all noise
    trace_final = truth_low + brownian + pink + hum + shot

    return t_low, trace_final, truth_low

def AV_OT_data(n, DURATION, FS):
    '''
    Generate averaged traces to simulate multiple independent bead calibrations to characterise
    the AV behaviour.
    '''
    taus = np.logspace(np.log10(1/FS), 0, 20, base=10)
    AV_traces = np.zeros((n, 19))

    for i in range(n):
        _, trace, _ = generate_benchmark_trace(DURATION, FS, 0)
        taus, iav = compute_allan_var(taus, trace, FS)
        AV_traces[i,:] = iav

    return taus, np.mean(AV_traces, axis=0)

# ==========================================
# 2. ANALYSIS ALGORITHMS & METRICS
# ==========================================
def compute_allan_var(tau, data, fsample):
    (taus, allan_dev, _, _) = allantools.oadev(data, rate=fsample, data_type="freq", taus=tau)
    return taus, allan_dev

def analyze_slope(taus, variances):
    log_tau = np.log10(taus)
    log_var = np.log10(variances)
    if len(log_tau) > 1:
        dx = log_tau[1] - log_tau[0]
        slopes = np.gradient(log_var, dx)
    else:
        slopes = np.zeros(len(log_tau))
    return slopes

def find_linear_region(taus, variances, target_slope=-1.0, tolerance=0.8):
    slopes = analyze_slope(taus, variances)
    is_linear = np.abs(slopes - target_slope) < tolerance
    if not np.any(is_linear):
        return None, None, []
    indices = np.where(is_linear)[0]
    splits = np.split(indices, np.where(np.diff(indices) != 1)[0] + 1)
    longest_segment = max(splits, key=len)
    if len(longest_segment) < 2:
        return None, None, []
    start_idx = longest_segment[0]
    end_idx = longest_segment[-1]
    return taus[start_idx], taus[end_idx], longest_segment

# --- step fit statistics ---
def evaluate_fit_performance(truth, fit, tolerance=50):
    """
    Compares the Ground Truth signal with the Fitted signal.
    
    Parameters:
    - truth: The ground truth array.
    - fit: The step-fitting result array.
    - tolerance: (int) The number of indices (datapoints) allowed between a true step 
                 and a fitted step to consider it a "match".
                 
    Returns a dictionary of error metrics including detection stats.
    """
    # Ensure lengths match
    n = min(len(truth), len(fit))
    truth = truth[:n]
    fit = fit[:n]

    # --- A. SPATIAL ACCURACY METRICS ---
    residuals = fit - truth
    mse = np.mean(residuals**2)
    rmsd = np.sqrt(mse) # Root Mean Square Deviation
    mae = np.mean(np.abs(residuals)) # Mean Absolute Error

    # --- B. EVENT DETECTION METRICS (Step Counting) ---
    # Find indices where steps occur (derivative is non-zero)
    # data can be floats, so we define a minimal change threshold
    thresh = 1e-5
    
    true_diff = np.diff(truth)
    fit_diff = np.diff(fit)
    
    true_locs = np.where(np.abs(true_diff) > thresh)[0]
    fit_locs = np.where(np.abs(fit_diff) > thresh)[0]
    
    # Validation counters
    true_positives = 0
    matched_fit_indices = set()

    # Loop through every GROUND TRUTH step to see if we found it
    for t_idx in true_locs:
        # 1. Find all fit steps within the tolerance window
        candidates = fit_locs[(fit_locs >= t_idx - tolerance) & (fit_locs <= t_idx + tolerance)]
        
        # 2. Filter candidates: Must match DIRECTION (Up or Down)
        t_dir = np.sign(true_diff[t_idx])
        valid_candidates = [c for c in candidates if np.sign(fit_diff[c]) == t_dir]
        
        # 3. Filter candidates: Must not have been matched to a previous true step
        unmatched_candidates = [c for c in valid_candidates if c not in matched_fit_indices]
        
        if len(unmatched_candidates) > 0:
            # We found a match! Pick the spatially closest one
            best_match = min(unmatched_candidates, key=lambda x: abs(x - t_idx))
            matched_fit_indices.add(best_match)
            true_positives += 1

    # False Negatives: True steps we missed
    false_negatives = len(true_locs) - true_positives
    
    # False Positives: Fitted steps that didn't match any true step
    false_positives = len(fit_locs) - true_positives

    # --- C. PRINT REPORT ---
    print(f"--- Performance Metrics ---")
    print(f"RMSD (Spatial Error): {rmsd:.2f} nm")
    print(f"MAE (Tracking):       {mae:.2f} nm")
    print(f"--- Step Detection Accuracy (Tol={tolerance} pts) ---")
    print(f"Total True Steps:     {len(true_locs)}")
    print(f"Total Detected Steps: {len(fit_locs)}")
    print(f"Correct Matches (TP): {true_positives}")
    print(f"Missed Steps (FN):    {false_negatives}")
    print(f"False Steps (FP):     {false_positives}")
    
    # Calculate Precision and Recall (Safety check for div by zero)
    recall = true_positives / len(true_locs) if len(true_locs) > 0 else 0
    precision = true_positives / len(fit_locs) if len(fit_locs) > 0 else 0
    print(f"Sensitivity (Recall): {recall*100:.1f}%")
    print(f"Precision:            {precision*100:.1f}%")

    return {
        "mse": mse, 
        "rmsd": rmsd, 
        "mae": mae,
        "true_positives": true_positives,
        "false_negatives": false_negatives,
        "false_positives": false_positives,
        "total_true": len(true_locs),
        "total_detected": len(fit_locs)
    }

def kv_algorithm(data):
    """Wrapper for external KV function"""
    found_indices = kalafut_visscher_step_detect(data)
    fit_line = generate_fit_trace(data, found_indices)
    return fit_line

# ==========================================
# 3. MAIN SIMULATION PIPELINE
# ==========================================
if __name__ == "__main__":
    # Parameters
    FS = 1000       # 1 kHz recording
    FRE = 300 # Naive resampling method
    DURATION = 30.0  # seconds
    STEP_SIZE = 2 # nm
    tol = int(2 / 30 * FS) # similar window to KV validation

    # 1. Generate Data (KV Paper Method)
    # Note: STEPS list is removed, steps are now random exponential events
    t, trace_raw, truth = generate_benchmark_trace(DURATION, FS)
    t_naive, naive_trace = resampling_avg(trace_raw, FS, FS)
    # _, AV_calib_data, _ = generate_benchmark_trace(5, FS, 0)

    # 2. Allan Variance Analysis
    taus = 1/(FS)*2 ** np.linspace(0, 14, 15)
    taus, av = AV_OT_data(5, 5, FS)

    # Find optimal Window (Gaussian White Noise Region)
    t_start, t_end, ids = find_linear_region(taus, av)

    if len(ids) > 0:
        # We look for the minimum of the GWN region
        min_id = np.where(av == av[ids].min())[0][0]
        opt_tau = taus[min_id]
        # opt_window_size = int(opt_tau * FS)
    # Fallback if detection fails
    else:
        opt_tau = 1/FRE
        opt_window_size = int(opt_tau*FS)

    print(f"AV Optimal Tau: {opt_tau*1000:.2f} ms")

    print(">>> Ground Truth Stats:")
    extract_step_statistics(truth, FS)

    # 3. Method A: Naive KV (Applied to Raw Data)
    print("\nRunning Naive KV...")
    fit_naive = kv_algorithm(naive_trace)
    
    # Evaluate Naive KV
    print(">>> Naive Method Stats:")
    extract_step_statistics(fit_naive, FS)
    evaluate_fit_performance(truth, fit_naive, tol)

    # 4. Method B: AV-Guided Resampling
    print("\nRunning AV-Guided KV...")
    # Downsample
    t_resampled, trace_resampled = resampling_avg(trace_raw, FS, 1/opt_tau)
    # n_chunks = len(trace_raw) // opt_window_size
    # trace_resampled = np.mean(trace_raw[:n_chunks*opt_window_size].reshape(-1, opt_window_size), axis=1)
    # t_resampled = t[:n_chunks*opt_window_size:opt_window_size]

    # Apply KV to resampled data
    fit_av_small = kv_algorithm(trace_resampled)

    # Project back to original time axis
    fit_av_projected = np.repeat(fit_av_small, opt_tau*FS)
    # Handle length mismatch due to floor division
    fit_av_projected = np.pad(fit_av_projected, (0, len(t)-len(fit_av_projected)), 'edge')

    # Evaluate AVKV
    print(">>> AV-Guided Method Stats:")
    extract_step_statistics(fit_av_projected, FS)
    evaluate_fit_performance(truth, fit_av_projected, tol)

    # ==========================================
    # 4. PLOTTING
    # ==========================================
    AVKVsim, axs = plt.subplots(3, 1, figsize=(20, 12))

    # Row 1: The Raw Data
    axs[0].plot(t, trace_raw, color='lightgray', alpha=0.5, lw=0.5, label='Simulated OT Data')
    # Plot true steps on top
    axs[0].plot(t, truth, color='k', lw=1.5, linestyle='--', label='Ground Truth')
    axs[0].set_title(f"A: Noisy Benchmark Data", fontsize=13, fontweight='bold', loc='left')
    axs[0].legend(loc='upper left')
    axs[0].set_xlim(0, DURATION)
    axs[0].set_ylabel('Distance (nm)', fontsize=12, fontweight='bold')

    # Row 2 Left: Naive Method
    axs[1].plot(t, trace_raw, color='lightgray', alpha=0.5)
    axs[1].plot(t_naive, fit_naive, color='tab:red', lw=2, label='Naive KV Fit')
    axs[1].plot(t, truth, color='k', linestyle='--', lw=1, alpha=0.8)
    axs[1].set_title("B: Naive KV step fitting", fontsize=13, fontweight='bold', loc='left')
    axs[1].legend(loc='upper left')
    axs[1].set_xlim(0, DURATION)
    axs[1].set_ylabel('Distance (nm)', fontsize=12, fontweight='bold')

    # Row 3: The AVKV Method
    axs[2].plot(t, trace_raw, color='lightgray', alpha=0.5)
    axs[2].plot(t_resampled, trace_resampled, color='green', lw=1, alpha=0.5, label=f'AV-guided resampled data (τ = {opt_tau:.2e} s)')
    axs[2].plot(t, fit_av_projected, color='tab:blue', lw=2, label='AVKV fit')
    axs[2].plot(t, truth, color='k', linestyle='--', lw=1, alpha=0.8)

    axs[2].set_title("C: AVKV step fitting", fontsize=13, fontweight='bold', loc='left')
    axs[2].legend(loc='upper left')
    axs[2].set_ylabel('Distance (nm)', fontsize=12, fontweight='bold')
    axs[2].set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    axs[2].set_xlim(0, DURATION)

    AVKVsim.tight_layout()
    savedirs = [Path(r"D:\LaTeX doc\figures\Chapter 4"), 
                Path(r"M:\Maarten\Studie\NanoBio MSc\MEP\Analysed Results\3. OT exps")]
    for dir in savedirs:
        AVKVsim.savefig(dir/"AVKV_comparison.pdf")

    # Plot AV Curve
    AV, ax = plt.subplots(figsize=(6,5))
    ax.loglog(taus, av, 'b-o', markersize=4)
    ax.axvline(opt_tau, color='r', linestyle='--', label=f'Optimal Tau: {opt_tau*1000:.1f}ms')
    if len(ids) > 0:
        ax.axvspan(t_start, t_end, color='green', alpha=0.1, label='White Noise Region')
    # ax.set_title("Allan Variance", fontsize=13)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"Allan Variance [nm$^2$]")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    plt.show()
