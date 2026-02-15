import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. DATA ENTRY
# ==========================================
p = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Best Results (Our Algorithm)
d_ours = np.array([
    0.49999999, 0.49999999, 0.49999999, 0.49999999, 0.49999998, 
    0.49999998, 0.49999998, 0.43379114, 0.39458504, 0.39217645
])

# Baseline Results (Scipy DE)
d_baseline = np.array([
    0.49999742, 0.49999183, 0.49997360, 0.49997731, 0.49996936, 
    0.49938900, 0.47443668, 0.27026219, 0.36077084, 0.22369906
])

# Calculate Relative Error (%)
# Avoid division by zero if any; here data is safe.
r_err = (d_ours - d_baseline) / d_ours * 100

# ==========================================
# 2. PLOT 1: DISTANCE COMPARISON
# ==========================================
plt.figure(figsize=(8, 6))
plt.plot(p, d_ours, 'g-o', linewidth=2, label='Our Algorithm')
plt.plot(p, d_baseline, 'r--x', linewidth=2, label='Baseline (Scipy DE)')

# Neutral Title
plt.title('Comparison of Solution Quality (H-Shaped Box)', fontsize=14)
plt.xlabel('Number of Points ($p$)', fontsize=12)
plt.ylabel('Minimum Distance ($D$)', fontsize=12)
plt.xticks(p)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)

plt.tight_layout()
plt.savefig('hbox_distance_comparison.png', dpi=300)
plt.show()

# ==========================================
# 3. PLOT 2: OPTIMALITY GAP
# ==========================================
plt.figure(figsize=(8, 6))
plt.plot(p, r_err, 'r-x', linewidth=2, label='Relative Gap (%)')

plt.yscale('log')
plt.title('Relative Optimality Gap vs. Baseline', fontsize=14)
plt.xlabel('Number of Points ($p$)', fontsize=12)
plt.ylabel('Gap relative to Our Result (%)', fontsize=12)
plt.xticks(p)
plt.grid(True, which="both", ls="-", alpha=0.3)
plt.legend(fontsize=11)

plt.tight_layout()
plt.savefig('hbox_optimality_gap.png', dpi=300)
plt.show()