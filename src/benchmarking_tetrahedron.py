import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. DATA ENTRY
# ==========================================
p_values = np.arange(1, 11)

# Best Results from Your Algorithm (Treated as "Optimum" for this analysis)
our_results = np.array([
    0.20412413, 0.14494893, 0.14494895, 0.14494895, 0.12247446, 
    0.11387057, 0.11237205, 0.11237234, 0.11237239, 0.11237235,
])

# Best Known Results (Kazakov et al., 2018)
kazakov_results = np.array([
    0.20045, 0.14204, 0.14204, 0.14204, 0.12147, 
    0.11243, 0.11062, 0.10600, 0.10250, 0.10200,
])

# SciPy Baseline Results
scipy_results = np.array([
    0.20412365, 0.14494889, 0.14494852, 0.14493123, 0.12244044, 
    0.11235068, 0.11213665, 0.11177796, 0.09080905, 0.08239677
])

# Create a graph with three lines: our results, Kazakov's results, and SciPy's results
plt.figure(figsize=(12, 7))
plt.plot(p_values, kazakov_results, marker='s', markersize=6, linewidth=2, label='Kazakov et al. (2018)', color='blue')
plt.plot(p_values, scipy_results, marker='^', markersize=6, linewidth=2, label='SciPy Baseline', color='green')
plt.plot(p_values, our_results, marker='o', markersize=6, linewidth=2, label='Our Algorithm', color='red')
plt.title('Comparison of Minimum Distances for Tetrahedron Container')
plt.xlabel('Number of Points (p)')
plt.ylabel('Minimum Distance (D)')
plt.xticks(p_values)
plt.grid(alpha=0.3)
plt.legend(loc='best')
plt.tight_layout()
plt.show()