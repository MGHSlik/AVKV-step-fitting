import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit

def stratified_sampling(x, samples_per_bin=10, bins=50):
    """
    Sample uniformly from x-bins for guaranteed visual balance.
    
    Parameters
    ----------
    x, y : ndarray
        Data arrays
    samples_per_bin : int
        Max samples per bin
    bins : int
        Number of x-bins
    
    Returns
    -------
    indices : ndarray
        Selected indices (sorted)
    """
    # Create bins
    bin_edges = np.linspace(x.min(), x.max(), bins + 1)
    bin_indices = np.digitize(x, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, bins - 1)

    # Sample from each bin
    selected_indices = []

    for bin_idx in range(bins):
        bin_mask = bin_indices == bin_idx
        bin_points = np.where(bin_mask)[0]

        if len(bin_points) > 0:
            # Sample up to samples_per_bin from this bin
            n_sample = min(samples_per_bin, len(bin_points))
            sampled = np.random.choice(bin_points, size=n_sample, replace=False)
            selected_indices.extend(sampled)

    return np.sort(np.array(selected_indices))



class StepDataManager:
    """
    Manager for saving and analyzing step detection results using NumPy.

    File structure:
        base_dir/
        ├── measurement_001/
        │   └── stepstats.npy        # Complete dictionary
        ├── measurement_002/
        │   └── stepstats.npy
        └── pooled_analysis/
            ├── pooled_histograms.png
            └── pooled_cdf.png
    """

    def __init__(self, base_dir: str):
        """
        Initialize manager.

        Parameters
        ----------
        base_dir : str
            Root directory for all measurements
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    # =========================== #
    # LOADING & POOLING FUNCTIONS #
    # =========================== #

    def load_single_measurement(self, meas_dir: str) -> Dict:
        """
        Load complete statistics dictionary for a single measurement.

        Parameters
        ----------
        meas_dir : str
            Name of data file

        Returns
        -------
        data : dict
            Complete dictionary with all statistics and arrays
        """
        if not meas_dir.exists():
            raise FileNotFoundError(f"Measurement not found: {meas_dir}")

        # Load complete dictionary
        data = np.load(meas_dir, allow_pickle=True).item()

        return data

    def pool_all_measurements(self) -> Dict:
        """
        Load and pool all measurements in base directory.

        Returns
        -------
        pooled : dict
            Dictionary with keys:
                - measurement_names : list of str
                - n_measurements : int
                - total_steps : int
                - dwell_times : ndarray (pooled)
                - step_sizes : ndarray (pooled)
                - individual_stats : list of dict
        """
        # Find all measurement directories with stepstats.npy
        measurement_dirs = list(self.base_dir.rglob("stepstats.npy"))

        if not measurement_dirs:
            raise ValueError(f"No measurements found in {self.base_dir}")

        # Initialize lists for pooling
        pooled_dwells = []
        pooled_steps = []
        individual_stats = []
        measurement_names = []

        # Load each measurement
        for meas_dir in sorted(measurement_dirs):
            data = self.load_single_measurement(meas_dir)

            pooled_dwells.append(data["dwell_times_raw"])
            pooled_steps.append(np.abs(data["step_sizes_raw"]))

            # Store scalar statistics (exclude raw arrays for memory efficiency)
            individual_stats.append({
                "measurement_name": data.get("measurement_name", meas_dir.name),
                "step_count": data["step_count"],
                "mean_step_size": data["mean_step_size"],
                "std_step_size": data["std_step_size"],
                "mean_dwell": data["mean_dwell"],
                "std_dwell": data["std_dwell"]
            })
            measurement_names.append(data.get("measurement_name", meas_dir.name))

        # Concatenate all arrays
        pooled = {
            "measurement_names": measurement_names,
            "n_measurements": len(measurement_dirs),
            "total_steps": sum(len(s) for s in pooled_steps),
            "dwell_times": np.concatenate(pooled_dwells),
            "step_sizes": np.concatenate(pooled_steps),
            "individual_stats": individual_stats
        }

        print(f"\nPooled {pooled['n_measurements']} measurements")
        print(f"   Total steps: {pooled['total_steps']}")
        print(f"   Dwell times: {len(pooled['dwell_times'])} events")
        print(f"   Step sizes: {len(pooled['step_sizes'])} events")

        return pooled

    # ==================== #
    # ANALYSIS: HISTOGRAMS #
    # ==================== #

    def plot_histograms(self, pooled: Dict = None, bins: int = 100, 
                       save: bool = True, show: bool = True):
        """
        Plot histograms of dwell times and step sizes with fits.

        Parameters
        ----------
        pooled : dict, optional
            Pooled data from pool_all_measurements(). If None, loads automatically.
        bins : int
            Number of histogram bins
        save : bool
            Save figure to disk
        show : bool
            Display figure
        """
        if pooled is None:
            pooled = self.pool_all_measurements()

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # --- Dwell Times ---
        ax = axes[0]
        dwells = pooled["dwell_times"]
        mean = np.mean(dwells)
        std = np.std(dwells)

        # Calculate histogram
        counts, bin_edges, patches = ax.hist(
            dwells, bins=bins, density=True, alpha=0.7, 
            color='tab:blue', edgecolor='black', linewidth=0.5
        )

        ax.set_xlabel('Dwell Time (s)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Probability Density', fontsize=12, fontweight='bold')
        # ax.set_title(f'Dwell Times Distribution\n(n = {len(dwells)} events, '
        #             f'{pooled["n_measurements"]} measurements)', fontsize=13)
        ax.axvline(mean, ls='--', c='r', alpha=0.8, lw=1.5, label=f'Average dwell ({mean:.3f} s)')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, which='both', linestyle='--')
        ax.set_xlim(left=0)

        # --- Step Sizes ---
        ax = axes[1]
        steps = np.abs(pooled["step_sizes"])  # Absolute values
        mean = np.mean(steps)
        std = np.std(steps)

        # Calculate histogram
        counts, bin_edges, patches = ax.hist(
            steps, bins=bins, density=True, alpha=0.7,
            color='tab:orange', edgecolor='black', linewidth=0.5
        )

        ax.set_xlabel('Step Size (nm)', fontsize=12, fontweight='bold')
        # ax.set_ylabel('Probability Density', fontsize=12, fontweight='bold')
        # ax.set_title(f'Step Sizes Distribution\n(n = {len(steps)} events, '
        #             f'{pooled["n_measurements"]} measurements)', fontsize=13)
        ax.axvline(mean, ls='--', c='r', alpha=0.8, lw=1.5, label=f'Average step size ({mean:.3f} nm)')

        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, which='both', linestyle='--')
        ax.set_xlim(left=0)

        plt.tight_layout()

        if save:
            save_path = self.base_dir / "pooled_analysis"
            save_path.mkdir(exist_ok=True)
            fig.savefig(save_path / "pooled_histograms.png", dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path / 'pooled_histograms.png'}")

        if show:
            plt.show()
        else:
            plt.close()

    # =========================================== #
    # ANALYSIS: CUMULATIVE DISTRIBUTION FUNCTIONS #
    # =========================================== #

    def plot_cdf(self, pooled: Dict = None, save: bool = True, show: bool = True):
        """
        Plot empirical cumulative distribution functions (ECDF).

        Parameters
        ----------
        pooled : dict, optional
            Pooled data. If None, loads automatically.
        save : bool
            Save figure to disk
        show : bool
            Display figure
        """
        if pooled is None:
            pooled = self.pool_all_measurements()

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # --- Dwell Times CDF ---
        ax = axes[0]
        dwells = np.sort(pooled["dwell_times"])
        dwell_mean = np.mean(dwells)
        n = len(dwells)
        cdf = np.arange(1, n + 1) / n
        indices = stratified_sampling(dwells)


        ax.plot(dwells[indices], cdf[indices], 'o', c='xkcd:orange', markersize=5, alpha=0.5, label='Empirical CDF')

        # Theoretical exponential CDF: F(x) = 1 - exp(-λx)
        try:
            rate = 1 / dwell_mean
            x_theory = np.linspace(0, np.max(dwells), 200)
            cdf_theory = 1 - np.exp(-rate * x_theory)

            ax.plot(x_theory, cdf_theory, 'k-', linewidth=1.5,
                   label=f'Theoretical CDF (λ = {rate:.3f} s⁻¹)')
        except Exception as e:
            print(f"Warning: CDF fit failed: {e}")

        ax.set_xlabel('Dwell Time (s)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Cumulative Probability', fontsize=12, fontweight='bold')
        # ax.set_title('Dwell Times CDF', fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, which='both', linestyle='--')
        ax.set_xlim([0, dwells.max()])
        ax.set_ylim([0, 1.05])

        # --- Step Sizes CDF ---
        ax = axes[1]
        steps = np.sort(np.abs(pooled["step_sizes"]))
        n = len(steps)
        cdf = np.arange(1, n + 1) / n
        indices = stratified_sampling(steps)

        ax.plot(steps[indices], cdf[indices], 'o', markersize=5, c='xkcd:orange', alpha=0.5, label='Empirical CDF')

        # Theoretical Gaussian CDF
        try:
            mu = np.mean(steps)
            sigma = np.std(steps)
            x_theory = np.linspace(0, np.max(steps), 200)
            # CDF of Gaussian using error function
            from scipy.special import erf
            cdf_theory = 0.5 * (1 + erf((x_theory - mu) / (sigma * np.sqrt(2))))

            ax.plot(x_theory, cdf_theory, 'k-', linewidth=1.5,
                   label=f'Theoretical CDF (μ={mu:.2f}, σ={sigma:.2f})')
        except Exception as e:
            print(f"Warning: CDF fit failed: {e}")

        ax.set_xlabel('Step Size (nm)', fontsize=12, fontweight='bold')
        # ax.set_ylabel('Cumulative Probability', fontsize=12, fontweight='bold')
        # ax.set_title('Step Sizes CDF', fontsize=13)
        ax.legend(fontsize=10, loc='lower right')
        ax.grid(True, alpha=0.3, which='both', linestyle='--')
        ax.set_xlim([0, steps.max()])
        ax.set_ylim([0, 1.05])

        plt.tight_layout()

        if save:
            save_path = self.base_dir / "pooled_analysis"
            save_path.mkdir(exist_ok=True)
            fig.savefig(save_path / "pooled_cdf.png", dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path / 'pooled_cdf.png'}")

        if show:
            plt.show()
        else:
            plt.close()

    # =========================== #
    # ADVANCED ANALYSIS FUNCTIONS #
    # =========================== #

    def plot_dwell_vs_step(self, pooled: Dict = None, show: bool = True):
        """
        Plot correlation between dwell times and step sizes.
        Handles mismatched array lengths by taking minimum length.
        
        Parameters
        ----------
        pooled : dict, optional
            Pooled measurement data. If None, will call pool_all_measurements()
        show : bool
            Whether to display plot immediately
        """
        if pooled is None:
            pooled = self.pool_all_measurements()
        
        dwells = pooled["dwell_times"]
        steps = np.abs(pooled["step_sizes"])
        
        # Handle length mismatch
        min_length = min(len(dwells), len(steps))
        
        if len(dwells) != len(steps):
            print(f"\nWarning: Array length mismatch!")
            print(f"    Dwell times: {len(dwells)} events")
            print(f"    Step sizes:  {len(steps)} events")
            print(f"    Using first {min_length} events for correlation")
            
            dwells = dwells[:min_length]
            steps = steps[:min_length]
        
        # Now arrays have same length - calculate correlation
        corr = np.corrcoef(dwells, steps)[0, 1]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Hexbin for dense data
        if len(dwells) > 500:
            hb = ax.hexbin(steps, dwells, gridsize=50, cmap='Oranges', mincnt=1)
            plt.colorbar(hb, ax=ax, label='Count')
        else:
            ax.plot(steps, dwells, 'o', alpha=0.3, markersize=2)
        
        # Add correlation info
        text_str = f'Pearson r = {corr:.3f}\nn = {len(dwells)} events'
        if len(pooled["dwell_times"]) != len(pooled["step_sizes"]):
            text_str += f'\n(trimmed from {max(len(pooled["dwell_times"]), len(pooled["step_sizes"]))})'
        
        # ax.text(0.05, 0.95, text_str,
        #     transform=ax.transAxes, fontsize=11, verticalalignment='top',
        #     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax.set_xlabel('Step Size (nm)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Dwell Time (s)', fontsize=12, fontweight='bold')
        # ax.set_title('Dwell Time vs Step Size Correlation', fontsize=13)
        ax.grid(True, alpha=0.3, which='both', linestyle='--')
        
        plt.tight_layout()
        
        # Save figure
        save_dir = self.base_dir / "pooled_analysis"
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / "dwell_vs_step_correlation.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   Saved: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()

    def plot_rate_vs_measurement(self, pooled: Dict = None, save: bool = True, show: bool = True):
        """
        Bar plot: stepping rate for each measurement (check consistency).

        Useful to identify outlier measurements or drift over time.
        """
        if pooled is None:
            pooled = self.pool_all_measurements()

        names = pooled["measurement_names"]
        rates = np.zeros(len(names))
        rate_errs = np.zeros(len(names))

        for i, stat in enumerate(pooled["individual_stats"]):
            if stat["step_count"] > 0 and stat["mean_dwell"] > 0:
                # Rate = 1/mean_dwell (steps per second)
                rates[i] = 1 / stat["mean_dwell"]
                # Error propagation: δ(1/x) = δx/x²
                rate_errs[i] = stat["std_dwell"] / stat["mean_dwell"]**2

        fig, ax = plt.subplots(figsize=(max(10, len(names) * 0.8), 6))

        x_pos = np.arange(len(names))
        ax.bar(x_pos, rates, yerr=rate_errs, alpha=0.7, capsize=5,
              color='tab:cyan', edgecolor='black', linewidth=1.5)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.set_ylabel('Stepping Rate (s⁻¹)', fontsize=12, fontweight='bold')
        # ax.set_title('Stepping Rate Across Measurements', fontsize=13)
        ax.grid(axis='y', alpha=0.3)

        mean_rate = np.mean(rates[rates > 0])
        ax.axhline(mean_rate, color='red', linestyle='--', linewidth=2,
                  label=f'Mean = {mean_rate:.3f} s⁻¹')
        ax.legend()

        plt.tight_layout()

        if save:
            save_path = self.base_dir / "pooled_analysis"
            save_path.mkdir(exist_ok=True)
            fig.savefig(save_path / "rate_consistency.png", dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path / 'rate_consistency.png'}")

        if show:
            plt.show()
        else:
            plt.close()

    def bootstrap_confidence_intervals(self, pooled: Dict = None, n_bootstrap: int = 1000,
                                      confidence: float = 0.95):
        """
        Calculate bootstrap confidence intervals for mean dwell time and step size.

        Parameters
        ----------
        pooled : dict, optional
            Pooled data
        n_bootstrap : int
            Number of bootstrap samples
        confidence : float
            Confidence level (e.g., 0.95 for 95% CI)

        Returns
        -------
        results : dict
            Dictionary with mean and CI for dwell times and step sizes
        """
        if pooled is None:
            pooled = self.pool_all_measurements()

        dwells = pooled["dwell_times"]
        steps = np.abs(pooled["step_sizes"])

        n_dwell = len(dwells)
        n_steps = len(steps)

        # Bootstrap dwell times
        dwell_means = np.zeros(n_bootstrap)
        for i in range(n_bootstrap):
            sample_indices = np.random.randint(0, n_dwell, size=n_dwell)
            dwell_means[i] = np.mean(dwells[sample_indices])

        # Bootstrap step sizes
        step_means = np.zeros(n_bootstrap)
        for i in range(n_bootstrap):
            sample_indices = np.random.randint(0, n_steps, size=n_steps)
            step_means[i] = np.mean(steps[sample_indices])

        # Calculate percentiles
        alpha = 1 - confidence
        dwell_ci = np.percentile(dwell_means, [100 * alpha/2, 100 * (1 - alpha/2)])
        step_ci = np.percentile(step_means, [100 * alpha/2, 100 * (1 - alpha/2)])

        results = {
            "dwell_time": {
                "mean": np.mean(dwells),
                "ci_lower": dwell_ci[0],
                "ci_upper": dwell_ci[1]
            },
            "step_size": {
                "mean": np.mean(steps),
                "ci_lower": step_ci[0],
                "ci_upper": step_ci[1]
            }
        }

        print(f"\n📊 Bootstrap Confidence Intervals ({confidence*100:.0f}%):")
        print(f"   Dwell Time: {results['dwell_time']['mean']:.3f} s "
              f"[{results['dwell_time']['ci_lower']:.3f}, {results['dwell_time']['ci_upper']:.3f}]")
        print(f"   Step Size:  {results['step_size']['mean']:.2f} nm "
              f"[{results['step_size']['ci_lower']:.2f}, {results['step_size']['ci_upper']:.2f}]")

        return results

    def export_summary_table(self, pooled: Dict = None, filename: str = "summary_table.txt"):
        """
        Export summary statistics to text file.

        Parameters
        ----------
        pooled : dict, optional
            Pooled data
        filename : str
            Output filename
        """
        if pooled is None:
            pooled = self.pool_all_measurements()

        save_path = self.base_dir / "pooled_analysis"
        save_path.mkdir(exist_ok=True)
        txt_path = save_path / filename

        with open(txt_path, 'w') as f:
            # Header
            f.write("# FtsH Step Detection Summary Table\n")
            f.write("# " + "="*70 + "\n\n")

            # Column headers
            f.write(f"{'Measurement':<30} {'Steps':>8} {'Mean Dwell (s)':>15} "
                   f"{'Std Dwell (s)':>15} {'Mean Step (nm)':>15} {'Std Step (nm)':>15}\n")
            f.write("-" * 110 + "\n")

            # Individual measurements
            for stat in pooled["individual_stats"]:
                f.write(f"{stat['measurement_name']:<30} "
                       f"{stat['step_count']:>8} "
                       f"{stat['mean_dwell']:>15.4f} "
                       f"{stat['std_dwell']:>15.4f} "
                       f"{stat['mean_step_size']:>15.3f} "
                       f"{stat['std_step_size']:>15.3f}\n")

            # Pooled statistics
            f.write("-" * 110 + "\n")
            f.write(f"{'POOLED':<30} "
                   f"{pooled['total_steps']:>8} "
                   f"{np.mean(pooled['dwell_times']):>15.4f} "
                   f"{np.std(pooled['dwell_times']):>15.4f} "
                   f"{np.mean(np.abs(pooled['step_sizes'])):>15.3f} "
                   f"{np.std(np.abs(pooled['step_sizes'])):>15.3f}\n")

        print(f"📄 Summary table saved: {txt_path}")

        # Also save as NumPy-loadable array
        summary_array = np.array([
            [stat['step_count'], stat['mean_dwell'], stat['std_dwell'],
             stat['mean_step_size'], stat['std_step_size']]
            for stat in pooled["individual_stats"]
        ])

        np.save(save_path / "summary_array.npy", summary_array)
        print(f"📄 Summary array saved: {save_path / 'summary_array.npy'}")


# ============= #
# EXAMPLE USAGE #
# ============= #

if __name__ == "__main__":

    # Initialize manager
    manager = StepDataManager(base_dir=r"D:\USE FOR STEPFIT\V13P")

    print("\n" + "="*50)
    print("POOLING AND ANALYZING ALL DATA")
    print("="*50)

    pooled = manager.pool_all_measurements()

    # Generate all plots
    manager.plot_histograms(pooled, show=False)
    manager.plot_cdf(pooled, show=False)
    manager.plot_dwell_vs_step(pooled, show=False)
    manager.plot_rate_vs_measurement(pooled, show=False)

    # Statistical analysis
    ci_results = manager.bootstrap_confidence_intervals(pooled)

    # Export summary
    manager.export_summary_table(pooled)

    print("\nAnalysis complete!")
    print(f"Results saved in: {manager.base_dir / 'pooled_analysis'}")
