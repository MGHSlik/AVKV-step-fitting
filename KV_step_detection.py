import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def calc_sic(n_points, total_variance, n_steps):
    """
    Calculates the Schwarz Information Criterion.
    Formula: SIC = n * ln(variance) + p * ln(n)
    where p is increasing with the number of steps.
    """
    if total_variance <= 0:
        return -np.inf # Safety for perfect fits
    
    # According to KV2008: Penalty parameter usually scales with steps.
    # We penalize based on number of segments (n_steps + 1) * 2 params (mean + loc) roughly
    k_params = (n_steps + 1) # This follows the paper's simplistic penalty scaling
    
    sic = n_points * np.log(total_variance) + k_params * np.log(n_points)
    return sic

def get_total_variance(data, step_indices):
    """
    Calculates the global weighted variance for the current set of steps.
    data: The full data array.
    step_indices: A sorted list of indices where steps occur (including 0 and N).
    """
    n = len(data)
    weighted_var_sum = 0.0
    
    # Iterate through each plateau defined by the indices
    for i in range(len(step_indices) - 1):
        start = step_indices[i]
        end = step_indices[i+1]
        
        # Extract the segment
        segment = data[start:end]
        
        # Calculate variance of this specific plateau
        # ddof=0 is standard population variance (matches C logic usually)
        seg_var = np.var(segment) 
        
        # Add weighted variance to sum
        weighted_var_sum += (len(segment) * seg_var)
        
    return weighted_var_sum / n

def find_best_split(data, start_index, end_index):
    """
    Scans a segment (data[start:end]) to find the index that minimizes
    the combined variance of the two resulting sub-segments.
    """
    segment = data[start_index:end_index]
    n_seg = len(segment)
    
    if n_seg < 3: # Too small to split
        return None, float('inf')

    # Optimization: Use cumulative sums (cumsum) for O(N) calculation
    # instead of looping O(N^2). This is faster than the C++ snippet provided for large N.
    Y = segment
    sum_Y = np.cumsum(Y)
    sum_Y_sq = np.cumsum(Y**2)
    
    indices = np.arange(1, n_seg) # Possible split points (relative to start)
    
    # Variance formula using sums: Var = E[X^2] - (E[X])^2
    # Left side stats
    mean_L = sum_Y[:-1] / indices
    mean_sq_L = sum_Y_sq[:-1] / indices
    var_L = mean_sq_L - mean_L**2
    
    # Right side stats
    # Total sum minus cumulative sum gives the right side sum
    count_R = n_seg - indices
    mean_R = (sum_Y[-1] - sum_Y[:-1]) / count_R
    mean_sq_R = (sum_Y_sq[-1] - sum_Y_sq[:-1]) / count_R
    var_R = mean_sq_R - mean_R**2
    
    # Combined weighted variance for every possible split index
    combined_vars = (indices * var_L + count_R * var_R)
    
    # Find minimum
    best_local_idx = np.argmin(combined_vars)
    min_combined_var_sum = combined_vars[best_local_idx]
    
    # Return absolute index in the original array and the weighted sum of variances
    return (start_index + indices[best_local_idx]), min_combined_var_sum

def kalafut_visscher_step_detect(data):
    """
    The Main Driver Function.
    Iteratively adds steps until SIC stops improving.
    """
    n = len(data)
    # Start with indices [0, n] -> One plateau
    current_indices = [0, n]
    
    # Initial state
    current_global_var = np.var(data)
    current_sic = calc_sic(n, current_global_var, 0)
    
    print(f"Initial SIC: {current_sic:.2f} (0 steps)")
    
    iteration = 0
    while True:
        iteration += 1
        best_new_sic = float('inf')
        best_split_idx = -1
        best_idx_insertion_pos = -1 # Where in current_indices to insert
        
        # Loop through existing plateaus and try to split each one
        # This matches the C++ loop: "while(currnode!=NULL)..."
        total_weighted_var_sum_base = 0
        
        # First, calculate the sum of variances of all CURRENT segments
        # We need this to substitute the modified segment later
        segment_vars = []
        for i in range(len(current_indices) - 1):
            seg = data[current_indices[i]:current_indices[i+1]]
            segment_vars.append(len(seg) * np.var(seg))
        total_current_var_sum = sum(segment_vars)
        
        # Try a candidate split in every existing segment
        for i in range(len(current_indices) - 1):
            start = current_indices[i]
            end = current_indices[i+1]
            
            # Find best split for THIS segment
            split_idx, split_var_sum = find_best_split(data, start, end)
            
            if split_idx is not None:
                # Calculate what the NEW global variance would be if we accepted this split
                # Remove old contribution of this segment, add new contribution
                new_global_var_sum = total_current_var_sum - segment_vars[i] + split_var_sum
                new_global_var = new_global_var_sum / n
                
                # Calculate candidate SIC (we are adding 1 step)
                candidate_sic = calc_sic(n, new_global_var, len(current_indices) - 2 + 1)
                
                # Is this the best candidate seen so far in this iteration?
                if candidate_sic < best_new_sic:
                    best_new_sic = candidate_sic
                    best_split_idx = split_idx
                    best_idx_insertion_pos = i + 1

        # End of iteration loop. Check if we improved.
        if best_new_sic < current_sic:
            # We found a valid step! Match C++: "insert"
            print(f"Iter {iteration}: Step found at idx {best_split_idx}. SIC improved to {best_new_sic:.2f}")
            current_indices.insert(best_idx_insertion_pos, best_split_idx)
            current_sic = best_new_sic
        else:
            # No improvement found. We are done.
            print("Convergence reached. No further steps improve SIC.")
            break
            
    return sorted(current_indices)

def generate_fit_trace(data, indices):
    """
    Reconstructs the stepped fit line from indices.
    """
    fit = np.zeros_like(data)
    for i in range(len(indices) - 1):
        start = indices[i]
        end = indices[i+1]
        fit[start:end] = np.mean(data[start:end])
    return fit


if __name__ == "__main__":
    # ==========================================
    # Example data
    # ==========================================
    # 1. Create dummy data (Bead position vs time)
    np.random.seed(42)
    t = np.linspace(0, 10, 1000)
    # A trace with 3 levels (2 steps)
    true_signal = np.concatenate([
        np.ones(300) * 0,      # Dwell 1
        np.ones(400) * 5,      # Step up to 5 nm
        np.ones(300) * 2       # Step down to 2 nm
    ])
    noise = np.random.normal(0, 1.0, 1000) # Noise sigma = 1.0
    sim_data = true_signal + noise

    # ==========================================
    # Loading Experimental Data
    # ==========================================
    # root = Path(r"CHANGE DIRECTORY NAME")
    # fname = "CHANGE FILE NAME"
    # data = np.loadtxt(root/fname)
    # print(f"Data loaded from {root}")
    # print(f"Data shape: {data.shape}")


    # 2. Run Algorithm
    found_indices = kalafut_visscher_step_detect(sim_data)

    # 3. Generate Fit
    fit_line = generate_fit_trace(sim_data, found_indices)

    # 4. Statistics
    print(fit_line.shape)
    step_id = [x for x in np.where(np.sqrt((fit_line[1:]-fit_line[:-1])**2)>0.005)]
    print(fit_line[step_id])

    # 4. Plot
    plt.figure(figsize=(10, 6))
    plt.plot(t, sim_data, color='lightgray', label='Raw Data (FtsH)')
    plt.plot(t, fit_line, color='red', linewidth=1, label='KV Step Fit')
    plt.title(f"Kalafut-Visscher Step Detection (Found {len(found_indices)-2} steps)")
    plt.xlabel("Time (s)")
    plt.ylabel("Position (nm)")
    plt.legend()
    plt.show()