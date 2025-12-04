import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar

# ==========================================
# CONFIGURATION
# ==========================================
DATA_FILE = 'IvanData.csv'
BATTERY_SOC_MIN = 0.20  # 20% minimum state of charge
BATTERY_SOC_MAX = 0.80  # 80% maximum state of charge
BATTERY_EFFICIENCY = 1.0  # 100% efficiency (can be modified)
RAMP_HOURS = 1  # Hours for ramp-up before sunrise and ramp-down after sunset

# ==========================================
# DATA LOADING AND PARSING
# ==========================================

def load_and_parse_data(filepath):
    """
    Load CSV data and parse timestamps.
    Returns a pandas DataFrame with parsed datetime index.
    """
    print(f"Loading data from {filepath}...")
    
    # Read CSV file
    df = pd.read_csv(filepath)
    
    # Parse the timestamp format YYYYMMDD:HHMM
    df['datetime'] = pd.to_datetime(df['time'], format='%Y%m%d:%H%M')
    
    # Set datetime as index
    df.set_index('datetime', inplace=True)
    
    # Extract date for grouping
    df['date'] = df.index.date
    
    print(f"Loaded {len(df)} hourly records from {df.index[0].date()} to {df.index[-1].date()}")
    
    return df


def group_by_day(df):
    """
    Group data by day and return a dictionary of daily DataFrames.
    """
    daily_data = {}
    
    for date, group in df.groupby('date'):
        daily_data[date] = group.copy()
    
    print(f"Organized data into {len(daily_data)} days")
    
    return daily_data


# ==========================================
# IDEAL CURVE GENERATION
# ==========================================

def find_generation_window(daily_power):
    """
    Find the window where power generation occurs (P > 0).
    Returns indices of first and last generation, and peak time.
    """
    generation_mask = daily_power > 0
    
    if not generation_mask.any():
        return None, None, None
    
    generation_indices = np.where(generation_mask)[0]
    first_gen = generation_indices[0]
    last_gen = generation_indices[-1]
    peak_idx = first_gen + np.argmax(daily_power[first_gen:last_gen+1])
    
    return first_gen, last_gen, peak_idx


def create_smooth_ideal_curve(daily_power, hours):
    """
    Create a smooth ideal power curve using a modified sinusoidal shape.
    
    The curve:
    - Starts ramping up 1 hour before first generation
    - Peaks around midday
    - Ramps down to 0 one hour after last generation
    - Redistributes the same total daily energy smoothly
    
    Returns: ideal power curve (numpy array)
    """
    n_hours = len(daily_power)
    ideal_curve = np.zeros(n_hours)
    
    # Calculate total daily energy (kWh)
    total_energy = np.sum(daily_power)
    
    if total_energy == 0:
        return ideal_curve
    
    # Find generation window
    first_gen, last_gen, peak_idx = find_generation_window(daily_power)
    
    if first_gen is None:
        return ideal_curve
    
    # Define extended window with ramp periods
    start_idx = max(0, first_gen - RAMP_HOURS)
    end_idx = min(n_hours - 1, last_gen + RAMP_HOURS)
    
    # Total generation window length
    window_length = end_idx - start_idx + 1
    
    if window_length <= 0:
        return ideal_curve
    
    # Create smooth curve using raised cosine (Hann-like) function
    # This provides smooth onset and decline
    t = np.linspace(0, np.pi, window_length)
    
    # Raised cosine: smooth curve from 0 to 1 and back to 0
    smooth_shape = (1 - np.cos(t)) / 2
    
    # Scale to match total energy
    # Energy = sum of power values (since time step is 1 hour)
    # We need: sum(ideal_curve) = total_energy
    curve_area = np.sum(smooth_shape)
    scaling_factor = total_energy / curve_area if curve_area > 0 else 0
    
    ideal_curve[start_idx:end_idx+1] = smooth_shape * scaling_factor
    
    return ideal_curve


# ==========================================
# BATTERY SIMULATION
# ==========================================

def simulate_battery_soc(actual_power, ideal_power, efficiency=BATTERY_EFFICIENCY):
    """
    Simulate battery state of charge throughout the day.
    
    Battery charges when actual > ideal (excess generation)
    Battery discharges when actual < ideal (deficit)
    
    Returns:
    - soc_profile: SOC throughout the day (in kWh, relative to starting point)
    - min_soc: Minimum SOC excursion (kWh)
    - max_soc: Maximum SOC excursion (kWh)
    """
    n_hours = len(actual_power)
    
    # Power difference: positive = charging, negative = discharging
    power_diff = actual_power - ideal_power
    
    # Apply efficiency
    power_diff_adjusted = power_diff * efficiency
    
    # Cumulative energy difference (SOC relative to start)
    # This represents the battery charge level relative to beginning of day
    soc_profile = np.cumsum(power_diff_adjusted)
    
    # Find min and max excursions
    min_soc = np.min(soc_profile)
    max_soc = np.max(soc_profile)
    
    # Verify energy balance (should return to starting point)
    final_soc = soc_profile[-1]
    
    return soc_profile, min_soc, max_soc, final_soc


def calculate_required_battery_capacity(min_soc, max_soc):
    """
    Calculate the minimum battery capacity needed to stay within SOC limits.
    
    Battery must operate between BATTERY_SOC_MIN (20%) and BATTERY_SOC_MAX (80%).
    Usable range: 60% of total capacity
    
    Returns: Required battery capacity in kWh
    """
    # Total swing in SOC (kWh)
    soc_swing = max_soc - min_soc
    
    # Usable capacity percentage
    usable_fraction = BATTERY_SOC_MAX - BATTERY_SOC_MIN
    
    # Required total capacity
    required_capacity = soc_swing / usable_fraction if usable_fraction > 0 else 0
    
    return required_capacity


# ==========================================
# ANALYSIS FOR ALL DAYS
# ==========================================

def analyze_all_days(daily_data):
    """
    Analyze all days in the dataset and calculate battery requirements.
    
    Returns:
    - results: Dictionary with analysis results for each day
    - max_capacity_day: Date requiring maximum battery capacity
    - max_capacity: Maximum battery capacity required (kWh)
    """
    results = {}
    max_capacity = 0
    max_capacity_day = None
    
    print("\nAnalyzing daily power profiles...")
    
    for date, day_df in daily_data.items():
        # Extract power data
        actual_power = day_df['P'].values
        hours = np.arange(len(actual_power))
        
        # Generate ideal curve
        ideal_power = create_smooth_ideal_curve(actual_power, hours)
        
        # Simulate battery
        soc_profile, min_soc, max_soc, final_soc = simulate_battery_soc(actual_power, ideal_power)
        
        # Calculate required capacity
        required_capacity = calculate_required_battery_capacity(min_soc, max_soc)
        
        # Store results
        results[date] = {
            'actual_power': actual_power,
            'ideal_power': ideal_power,
            'soc_profile': soc_profile,
            'min_soc': min_soc,
            'max_soc': max_soc,
            'final_soc': final_soc,
            'required_capacity': required_capacity,
            'total_energy': np.sum(actual_power)
        }
        
        # Track maximum
        if required_capacity > max_capacity:
            max_capacity = required_capacity
            max_capacity_day = date
    
    print(f"\nAnalysis complete!")
    print(f"Maximum battery capacity required: {max_capacity:.2f} kWh")
    print(f"Required on date: {max_capacity_day}")
    
    return results, max_capacity_day, max_capacity


# ==========================================
# VISUALIZATION
# ==========================================

def plot_sample_days(results, sample_dates=None, n_samples=5):
    """
    Plot actual vs ideal power curves for sample days.
    """
    if sample_dates is None:
        # Select days with highest energy production
        sorted_dates = sorted(results.keys(), 
                            key=lambda d: results[d]['total_energy'], 
                            reverse=True)
        sample_dates = sorted_dates[:n_samples]
    
    n_plots = len(sample_dates)
    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 3*n_plots))
    
    if n_plots == 1:
        axes = [axes]
    
    for idx, date in enumerate(sample_dates):
        data = results[date]
        hours = np.arange(len(data['actual_power']))
        
        axes[idx].plot(hours, data['actual_power'], 'b-', linewidth=2, 
                      label='Actual Power', alpha=0.7)
        axes[idx].plot(hours, data['ideal_power'], 'r--', linewidth=2, 
                      label='Ideal Smooth Power')
        axes[idx].fill_between(hours, data['actual_power'], alpha=0.3)
        axes[idx].fill_between(hours, data['ideal_power'], alpha=0.2, color='red')
        
        axes[idx].set_xlabel('Hour of Day')
        axes[idx].set_ylabel('Power (W)')
        axes[idx].set_title(f'Date: {date} | Total Energy: {data["total_energy"]:.2f} Wh | '
                          f'Required Battery: {data["required_capacity"]:.2f} kWh')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_xlim(0, 23)
    
    plt.tight_layout()
    plt.savefig('daily_power_curves.png', dpi=300, bbox_inches='tight')
    print("\nSaved: daily_power_curves.png")
    plt.show()


def plot_battery_soc(results, sample_dates=None, n_samples=5):
    """
    Plot battery SOC profiles for sample days.
    """
    if sample_dates is None:
        # Select days with highest battery requirements
        sorted_dates = sorted(results.keys(), 
                            key=lambda d: results[d]['required_capacity'], 
                            reverse=True)
        sample_dates = sorted_dates[:n_samples]
    
    n_plots = len(sample_dates)
    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 3*n_plots))
    
    if n_plots == 1:
        axes = [axes]
    
    for idx, date in enumerate(sample_dates):
        data = results[date]
        hours = np.arange(len(data['soc_profile']))
        
        # Calculate SOC percentage if we use the required capacity
        capacity = data['required_capacity']
        if capacity > 0:
            # Shift SOC so it operates between 20% and 80%
            # Start at midpoint of usable range
            soc_offset = -data['min_soc'] + capacity * BATTERY_SOC_MIN
            soc_percent = (data['soc_profile'] + soc_offset) / capacity * 100
        else:
            soc_percent = np.zeros_like(data['soc_profile'])
        
        axes[idx].plot(hours, soc_percent, 'g-', linewidth=2)
        axes[idx].axhline(y=BATTERY_SOC_MIN*100, color='r', linestyle='--', 
                         label=f'Min SOC ({BATTERY_SOC_MIN*100}%)')
        axes[idx].axhline(y=BATTERY_SOC_MAX*100, color='r', linestyle='--', 
                         label=f'Max SOC ({BATTERY_SOC_MAX*100}%)')
        axes[idx].fill_between(hours, BATTERY_SOC_MIN*100, BATTERY_SOC_MAX*100, 
                              alpha=0.2, color='green', label='Usable Range')
        
        axes[idx].set_xlabel('Hour of Day')
        axes[idx].set_ylabel('Battery SOC (%)')
        axes[idx].set_title(f'Date: {date} | Battery Capacity: {capacity:.2f} kWh')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_xlim(0, 23)
        axes[idx].set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig('battery_soc_profiles.png', dpi=300, bbox_inches='tight')
    print("Saved: battery_soc_profiles.png")
    plt.show()


def plot_capacity_distribution(results):
    """
    Plot histogram of required battery capacities across all days.
    """
    capacities = [data['required_capacity'] for data in results.values() 
                 if data['total_energy'] > 0]  # Only days with generation
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(capacities, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Required Battery Capacity (kWh)')
    plt.ylabel('Number of Days')
    plt.title('Distribution of Daily Battery Capacity Requirements')
    plt.axvline(x=np.max(capacities), color='r', linestyle='--', 
               label=f'Maximum: {np.max(capacities):.2f} kWh')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    sorted_capacities = np.sort(capacities)
    percentiles = np.arange(1, len(sorted_capacities) + 1) / len(sorted_capacities) * 100
    plt.plot(sorted_capacities, percentiles, linewidth=2)
    plt.xlabel('Required Battery Capacity (kWh)')
    plt.ylabel('Cumulative Percentage of Days (%)')
    plt.title('Cumulative Distribution of Battery Requirements')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=95, color='orange', linestyle='--', alpha=0.5, 
               label='95th percentile')
    plt.axhline(y=99, color='red', linestyle='--', alpha=0.5, 
               label='99th percentile')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('battery_capacity_distribution.png', dpi=300, bbox_inches='tight')
    print("Saved: battery_capacity_distribution.png")
    plt.show()
    
    # Print statistics
    print("\n" + "="*60)
    print("BATTERY CAPACITY STATISTICS")
    print("="*60)
    print(f"Minimum required capacity: {np.min(capacities):.2f} kWh")
    print(f"Maximum required capacity: {np.max(capacities):.2f} kWh")
    print(f"Mean capacity: {np.mean(capacities):.2f} kWh")
    print(f"Median capacity: {np.median(capacities):.2f} kWh")
    print(f"95th percentile: {np.percentile(capacities, 95):.2f} kWh")
    print(f"99th percentile: {np.percentile(capacities, 99):.2f} kWh")
    print(f"Days with generation: {len(capacities)}")
    print("="*60)


# ==========================================
# MAIN EXECUTION
# ==========================================

def main():
    """
    Main execution function.
    """
    print("="*60)
    print("PV IDEAL CURVE SOLVER")
    print("Daily Power Smoothing with Battery Storage")
    print("="*60)
    
    # Load and parse data
    df = load_and_parse_data(DATA_FILE)
    
    # Group by day
    daily_data = group_by_day(df)
    
    # Analyze all days
    results, max_capacity_day, max_capacity = analyze_all_days(daily_data)
    
    # Visualizations
    print("\nGenerating visualizations...")
    
    # Plot sample days with highest energy production
    plot_sample_days(results, n_samples=5)
    
    # Plot battery SOC for days with highest capacity requirements
    plot_battery_soc(results, n_samples=5)
    
    # Plot capacity distribution
    plot_capacity_distribution(results)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print(f"\nMinimum battery capacity needed: {max_capacity:.2f} kWh")
    print(f"This ensures operation within {BATTERY_SOC_MIN*100}%-{BATTERY_SOC_MAX*100}% SOC limits")
    print(f"Usable capacity: {max_capacity * (BATTERY_SOC_MAX - BATTERY_SOC_MIN):.2f} kWh")
    print(f"Battery efficiency assumed: {BATTERY_EFFICIENCY*100}%")
    print("="*60)


if __name__ == "__main__":
    main()