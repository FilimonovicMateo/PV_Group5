"""
Debug script to visualize the curve generation process in detail
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Read data
df = pd.read_csv('IvanData.csv')
df['datetime'] = pd.to_datetime(df['time'], format='%Y%m%d:%H%M')
df.set_index('datetime', inplace=True)
df['date'] = df.index.date

# Select a few interesting days to examine
# Day with high generation, medium generation, and low generation
dates_to_check = []

daily_totals = []
for date, group in df.groupby('date'):
    total_energy = group['P'].sum()
    if total_energy > 0:
        daily_totals.append((date, total_energy))

daily_totals.sort(key=lambda x: x[1], reverse=True)

# Get top 3 days with different characteristics
dates_to_check = [
    daily_totals[0][0],      # Highest energy day
    daily_totals[len(daily_totals)//2][0],  # Medium energy day
    daily_totals[-10][0],    # Low energy day
]

print("Analyzing days:")
for date in dates_to_check:
    total = [x[1] for x in daily_totals if x[0] == date][0]
    print(f"  {date}: {total:.2f} Wh")

# Create detailed plots
fig, axes = plt.subplots(len(dates_to_check), 3, figsize=(18, 4*len(dates_to_check)))

for idx, date in enumerate(dates_to_check):
    day_data = df[df['date'] == date]
    
    actual_power = day_data['P'].values
    hours = np.arange(len(actual_power))
    
    # Find generation window
    gen_mask = actual_power > 0
    if not gen_mask.any():
        continue
    
    gen_indices = np.where(gen_mask)[0]
    first_gen = gen_indices[0]
    last_gen = gen_indices[-1]
    
    # Extended window
    start_idx = max(0, first_gen - 1)
    end_idx = min(len(actual_power) - 1, last_gen + 1)
    window_length = end_idx - start_idx + 1
    
    # Create ideal curve
    total_energy = np.sum(actual_power)
    ideal_curve = np.zeros(len(actual_power))
    
    t = np.linspace(0, np.pi, window_length)
    smooth_shape = np.sin(t)
    
    curve_area = np.sum(smooth_shape)
    scaling_factor = total_energy / curve_area if curve_area > 0 else 0
    ideal_curve[start_idx:end_idx+1] = smooth_shape * scaling_factor
    
    # Plot 1: Actual power with all data points visible
    axes[idx, 0].plot(hours, actual_power, 'b-', marker='o', markersize=5, linewidth=1.5, label='Actual Power')
    axes[idx, 0].axvline(x=first_gen, color='green', linestyle=':', alpha=0.7, label='First Generation')
    axes[idx, 0].axvline(x=last_gen, color='orange', linestyle=':', alpha=0.7, label='Last Generation')
    axes[idx, 0].axvline(x=start_idx, color='red', linestyle='--', alpha=0.5, label='Ideal Start')
    axes[idx, 0].axvline(x=end_idx, color='red', linestyle='--', alpha=0.5, label='Ideal End')
    axes[idx, 0].set_xlabel('Hour')
    axes[idx, 0].set_ylabel('Power (W)')
    axes[idx, 0].set_title(f'{date} - Actual Power Profile')
    axes[idx, 0].legend(fontsize=8)
    axes[idx, 0].grid(True, alpha=0.3)
    axes[idx, 0].set_xlim(0, 23)
    
    # Plot 2: Comparison of actual vs ideal
    axes[idx, 1].plot(hours, actual_power, 'b-', marker='o', markersize=4, linewidth=1.5, label='Actual', alpha=0.7)
    axes[idx, 1].plot(hours, ideal_curve, 'r--', linewidth=2.5, label='Ideal (Sine)', alpha=0.8)
    axes[idx, 1].fill_between(hours, actual_power, alpha=0.2, color='blue')
    axes[idx, 1].fill_between(hours, ideal_curve, alpha=0.15, color='red')
    axes[idx, 1].set_xlabel('Hour')
    axes[idx, 1].set_ylabel('Power (W)')
    axes[idx, 1].set_title(f'{date} - Actual vs Ideal\nTotal: {total_energy:.0f} Wh')
    axes[idx, 1].legend()
    axes[idx, 1].grid(True, alpha=0.3)
    axes[idx, 1].set_xlim(0, 23)
    
    # Plot 3: Power difference and cumulative SOC
    power_diff = actual_power - ideal_curve
    soc_profile = np.cumsum(power_diff)
    
    ax3a = axes[idx, 2]
    ax3b = ax3a.twinx()
    
    ax3a.bar(hours, power_diff, alpha=0.6, color=['green' if x > 0 else 'red' for x in power_diff], 
             label='Power Diff (Actual - Ideal)')
    ax3b.plot(hours, soc_profile, 'purple', linewidth=2.5, label='Cumulative SOC', marker='s', markersize=3)
    
    ax3a.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3a.set_xlabel('Hour')
    ax3a.set_ylabel('Power Difference (W)', color='black')
    ax3b.set_ylabel('Cumulative Energy (Wh)', color='purple')
    ax3a.set_title(f'{date} - Battery Charge/Discharge\nSOC range: [{soc_profile.min():.0f}, {soc_profile.max():.0f}] Wh')
    ax3a.grid(True, alpha=0.3)
    ax3a.legend(loc='upper left', fontsize=8)
    ax3b.legend(loc='upper right', fontsize=8)
    ax3a.set_xlim(0, 23)

plt.tight_layout()
plt.savefig('debug_curve_generation.png', dpi=300, bbox_inches='tight')
print("\nSaved: debug_curve_generation.png")
plt.show()

# Print detailed statistics
print("\n" + "="*70)
print("DETAILED STATISTICS")
print("="*70)
for date in dates_to_check:
    day_data = df[df['date'] == date]
    actual_power = day_data['P'].values
    
    gen_hours = np.where(actual_power > 0)[0]
    if len(gen_hours) == 0:
        continue
    
    print(f"\nDate: {date}")
    print(f"  Total energy: {np.sum(actual_power):.2f} Wh")
    print(f"  Generation hours: {len(gen_hours)} ({gen_hours[0]}:00 to {gen_hours[-1]}:00)")
    print(f"  Peak power: {np.max(actual_power):.2f} W at hour {np.argmax(actual_power)}")
    print(f"  Average power (during generation): {np.mean(actual_power[gen_hours]):.2f} W")
    print(f"  Power values during generation:")
    for i in gen_hours:
        print(f"    Hour {i:2d}: {actual_power[i]:7.2f} W")

print("="*70)
