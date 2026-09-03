import os

import numpy as np
import matplotlib.pyplot as plt

# Where figures are written. Point CPDP_FIGURES_DIR at the paper source tree to
# regenerate the manuscript figures in place.
FIGURES_DIR = os.environ.get("CPDP_FIGURES_DIR", "figures")

# ==========================================
# 1. DATA ENTRY
# ==========================================
p = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 56, 60])

# Best Results (Our Algorithm)
d_ours = np.array([0.5000000061145241,
 0.4999999922217822,
 0.5000000004015258,
 0.49999999877848805,
 0.4999999969498282,
 0.49999999644518683,
 0.49999999051177124,
 0.4109359165108985,
 0.3972584677609625,
 0.3936088949396298,
 0.30427141373441935,
 0.28046437528075496,
 0.2669621445474951,
 0.24999982566954054,
 0.24999982566954054,
 0.19986383019086304])

# Baseline Results (Scipy DE)
d_baseline = np.array([
    0.49999742, 0.49999183, 0.49997360, 0.49997731, 0.49996936, 
    0.49938900, 0.47443668, 0.27026219, 0.36077084, 0.22369906, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
])

# Calculate Relative Error (%)
# Avoid division by zero if any; here data is safe.
r_err = (d_ours - d_baseline) / d_ours * 100

# ==========================================
# 2. PLOT 1: DISTANCE COMPARISON
# ==========================================

FONTSIZE = 24
x_ticks = [1, 5] + p[9:].tolist()  # Show 1,5 and then 20,30,40,50,56,60
plt.figure(figsize=(10, 8))
plt.plot(p, d_ours, 'g-o', linewidth=2, label='Our Algorithm')
plt.plot(p, d_baseline, 'r--x', linewidth=2, label='Baseline (Scipy DE)')

# Neutral Title
plt.title('Comparison of Solution Quality (H-Shaped Box)', fontsize=FONTSIZE)
plt.xlabel('Number of Points ($p$)', fontsize=FONTSIZE-2)
plt.ylabel('Minimum Distance ($D$)', fontsize=FONTSIZE-2)
plt.xticks(x_ticks)
plt.tick_params(axis='both', labelsize=FONTSIZE-2)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=FONTSIZE-2)
plt.tight_layout()
os.makedirs(FIGURES_DIR, exist_ok=True)
plt.savefig(os.path.join(FIGURES_DIR, 'comparison_h_box.png'), dpi=300)
plt.show()