import numpy as np
import matplotlib.pyplot as plt

import glob
import pickle
import os

import pandas as pd

def get_elapsed_times(root_dir="output/cube"):
    """
    Reads pkl files from output/cube/p<i>/euclidean/<subdir>/*.pkl
    and returns a dict of {p_value: elapsed_time}.
    """
    results = {}
    
    # Pattern to match the nested structure: p<i>, then euclidean, then any subdir, then .pkl
    # Adjust the pattern if your 'something_here' is multiple levels deep
    pattern = os.path.join(root_dir, "p*", "euclidean", "*", "*.pkl")
    
    for filepath in glob.glob(pattern):
        try:
            # Extract the 'p' value from the path (e.g., from 'p56' extract 56)
            # Path looks like: output/cube/p56/euclidean/...
            parts = filepath.split(os.sep)
            p_folder = [p for p in parts if p.startswith('p')][0 if root_dir != 'output/pointy-star' else 1]  # Handle case where 'p' folder is at different levels
            p_value = int(p_folder[1:]) 
            
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                
            # Store the elapsed time using p as the key
            if 'elapsed_time' in data:
                results[p_value] = data['elapsed_time']
                
        except (IndexError, ValueError, pickle.UnpicklingError, KeyError) as e:
            print(f"Skipping {filepath}: {e}")
            
    return results

cube_runtimes = {int(k): v for k, v in sorted(get_elapsed_times(root_dir="output/cube").items(), key=lambda x: int(x[0]))}
tetrahedron_runtimes = {int(k): v for k, v in sorted(get_elapsed_times(root_dir="output/tetrahedron").items(), key=lambda x: int(x[0]))}
h_box_runtimes = {int(k): v for k, v in sorted(get_elapsed_times(root_dir="output/h-box").items(), key=lambda x: int(x[0]))}
pointy_star_runtimes = {int(k): v for k, v in sorted(get_elapsed_times(root_dir="output/pointy-star").items(), key=lambda x: int(x[0]))}

# make a dataframe for the runtimes by p values (the p values for each set may differ) -- fill with NaN if a p value is missing for a set
df = pd.DataFrame({
    "p": sorted(set(cube_runtimes.keys()) | set(tetrahedron_runtimes.keys()) | set(h_box_runtimes.keys()) | set(pointy_star_runtimes.keys())),
    "Cube": [cube_runtimes.get(p, np.nan) for p in sorted(set(cube_runtimes.keys()) | set(tetrahedron_runtimes.keys()) | set(h_box_runtimes.keys()) | set(pointy_star_runtimes.keys()))],
    "Tetrahedron": [tetrahedron_runtimes.get(p, np.nan) for p in sorted(set(cube_runtimes.keys()) | set(tetrahedron_runtimes.keys()) | set(h_box_runtimes.keys()) | set(pointy_star_runtimes.keys()))],
    "H-Box": [h_box_runtimes.get(p, np.nan) for p in sorted(set(cube_runtimes.keys()) | set(tetrahedron_runtimes.keys()) | set(h_box_runtimes.keys()) | set(pointy_star_runtimes.keys()))],
    "Star-Shaped Polyhedron": [pointy_star_runtimes.get(p, np.nan) for p in sorted(set(cube_runtimes.keys()) | set(tetrahedron_runtimes.keys()) | set(h_box_runtimes.keys()) | set(pointy_star_runtimes.keys()))]
})

print(df)
    

if __name__ == "__main__":
    plt.figure(figsize=(12, 8)) # Increased width slightly to accommodate labels

    runtimes_data = [
        (tetrahedron_runtimes, "Tetrahedron", 4, "orange"),
        (cube_runtimes, "Cube", 6, "blue"),
        (h_box_runtimes, "H-Box", 14, "green"),
        (pointy_star_runtimes, "Star-Shaped Polyhedron", 24, "red")
    ]

    for data, label, m_faces, color in runtimes_data:
        sorted_items = sorted(data.items())
        x = np.array([k for k, v in sorted_items])
        y = np.array([v for k, v in sorted_items])
        
        # Plot actual data points
        plt.plot(x, y, 'o', color=color, label=f'{label} (Actual)', alpha=0.6)
        
        # Fit and plot quadratic fit
        if len(x) > 2:
            coeffs = np.polyfit(x, y, 2)
            poly = np.poly1d(coeffs)
            x_smooth = np.linspace(x.min(), x.max(), 100)
            y_smooth = poly(x_smooth)
            
            plt.plot(x_smooth, y_smooth, '--', color=color, label=f'{label} (Fit)')
            
            # Place M text at the end of the fit line (at x=10)
            # va='center' ensures it aligns with the line endpoint
            plt.text(x.max() + 0.2, poly(x.max()), f'M={m_faces}', 
                     color=color, va='center', fontsize=16, fontweight='bold')

    plt.title("Runtime Analysis Comparison")
    plt.xlabel("Number of Points (p)")
    plt.ylabel("Runtime (s)")
    # plt.xlim(0.5, 11) # Expand x-axis slightly so labels aren't cut off
    plt.legend(loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.show()