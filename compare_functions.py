"""
Compare sine vs raised cosine curves
"""
import numpy as np
import matplotlib.pyplot as plt

# Create comparison of different smoothing functions
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Generate sample window
window_length = 10
t = np.linspace(0, np.pi, window_length)

# Function 1: Pure sine
sine_curve = np.sin(t)

# Function 2: Raised cosine (Hann window)
raised_cosine = (1 - np.cos(t)) / 2

# Function 3: Squared sine (more peaked)
squared_sine = np.sin(t)**2

# Function 4: Square root sine (flatter)
sqrt_sine = np.sqrt(np.sin(t))

# Plot 1: All curves together
axes[0, 0].plot(t, sine_curve, 'b-', linewidth=2.5, marker='o', label='Sine: sin(t)', markersize=6)
axes[0, 0].plot(t, raised_cosine, 'r--', linewidth=2.5, marker='s', label='Raised Cosine: (1-cos(t))/2', markersize=6)
axes[0, 0].plot(t, squared_sine, 'g-.', linewidth=2.5, marker='^', label='Squared Sine: sin²(t)', markersize=6)
axes[0, 0].plot(t, sqrt_sine, 'm:', linewidth=2.5, marker='d', label='Sqrt Sine: √sin(t)', markersize=6)
axes[0, 0].set_xlabel('Time (0 to π)', fontsize=11)
axes[0, 0].set_ylabel('Normalized Power', fontsize=11)
axes[0, 0].set_title('Comparison of Smoothing Functions', fontsize=13, fontweight='bold')
axes[0, 0].legend(fontsize=10)
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_ylim(-0.1, 1.1)

# Plot 2: Characteristics comparison
x_labels = ['Sine', 'Raised\nCosine', 'Sine²', '√Sine']
peak_values = [np.max(sine_curve), np.max(raised_cosine), np.max(squared_sine), np.max(sqrt_sine)]
areas = [np.sum(sine_curve), np.sum(raised_cosine), np.sum(squared_sine), np.sum(sqrt_sine)]

x = np.arange(len(x_labels))
width = 0.35

axes[0, 1].bar(x - width/2, peak_values, width, label='Peak Value', alpha=0.8, color='orange')
axes[0, 1].bar(x + width/2, [a/10 for a in areas], width, label='Area (÷10)', alpha=0.8, color='blue')
axes[0, 1].set_ylabel('Value', fontsize=11)
axes[0, 1].set_title('Peak and Area Comparison', fontsize=13, fontweight='bold')
axes[0, 1].set_xticks(x)
axes[0, 1].set_xticklabels(x_labels)
axes[0, 1].legend(fontsize=10)
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Plot 3: Shape characteristics (rate of change)
axes[1, 0].plot(t[1:], np.diff(sine_curve), 'b-', linewidth=2, marker='o', label='Sine', markersize=5)
axes[1, 0].plot(t[1:], np.diff(raised_cosine), 'r--', linewidth=2, marker='s', label='Raised Cosine', markersize=5)
axes[1, 0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
axes[1, 0].set_xlabel('Time', fontsize=11)
axes[1, 0].set_ylabel('Rate of Change', fontsize=11)
axes[1, 0].set_title('Derivative (Smoothness)', fontsize=13, fontweight='bold')
axes[1, 0].legend(fontsize=10)
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Simulated actual power vs different ideal curves
# Simulate a "realistic" noisy solar profile
np.random.seed(42)
actual_profile = sine_curve + np.random.normal(0, 0.1, len(sine_curve))
actual_profile = np.maximum(0, actual_profile)  # No negative power

# Normalize all to same total energy
total_energy = np.sum(actual_profile)
sine_scaled = sine_curve * (total_energy / np.sum(sine_curve))
cosine_scaled = raised_cosine * (total_energy / np.sum(raised_cosine))

axes[1, 1].plot(t, actual_profile, 'k-', linewidth=2, marker='o', label='Actual (simulated)', markersize=6, alpha=0.7)
axes[1, 1].plot(t, sine_scaled, 'b--', linewidth=2.5, label='Sine Ideal', alpha=0.8)
axes[1, 1].plot(t, cosine_scaled, 'r:', linewidth=2.5, label='Cosine Ideal', alpha=0.8)
axes[1, 1].set_xlabel('Time', fontsize=11)
axes[1, 1].set_ylabel('Power', fontsize=11)
axes[1, 1].set_title('Actual vs Ideal Curves (Energy Matched)', fontsize=13, fontweight='bold')
axes[1, 1].legend(fontsize=10)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('curve_function_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: curve_function_comparison.png")
print("\nFunction Characteristics:")
print(f"  Sine:          Peak={np.max(sine_curve):.3f}, Area={np.sum(sine_curve):.3f}")
print(f"  Raised Cosine: Peak={np.max(raised_cosine):.3f}, Area={np.sum(raised_cosine):.3f}")
print(f"  Sine²:         Peak={np.max(squared_sine):.3f}, Area={np.sum(squared_sine):.3f}")
print(f"  √Sine:         Peak={np.max(sqrt_sine):.3f}, Area={np.sum(sqrt_sine):.3f}")
print("\nNote: Pure sine function (sin(t)) provides the most natural solar curve shape")
print("      Raised cosine (1-cos(t))/2 is broader and flatter at the peak")

plt.show()
