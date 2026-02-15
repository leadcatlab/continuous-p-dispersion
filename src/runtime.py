import numpy as np
import matplotlib.pyplot as plt

# Data definitions
cube_runtimes = {
    1: 4.72, 2: 8.65, 3: 15.21, 4: 18.71, 5: 34.55, 6: 45.30, 7: 43.40, 8: 34.28, 9: 68.67, 10: 89.31,
}

tetrahedron_runtimes = {
    1: 2.66922903, 2: 5.11073685, 3: 8.03986907, 4: 10.27118492, 5: 16.21308899, 
    6: 19.78887391, 7: 23.43137407, 8: 27.57343102, 9: 38.35076404, 10: 32.95176697, 
    11: 278.23813105, 20: 777.27, 30: 2307.85, 56: 16955.65,
}

h_box_runtimes = {
    1: 2.60671282, 2: 3.57589769, 3: 6.52290893, 4: 7.73551011, 5: 17.16410089,
    6: 22.38812327, 7: 36.28490019, 8: 42.34856319, 9: 74.65780282, 10: 201.28235388,
}

pointy_star_runtimes = {
    1: 4.51439476, 2: 8.42482805, 3: 14.38198614, 4: 19.30770612, 5: 26.45643210,
    6: 38.64741278, 7: 50.72459102, 8: 60.88067007, 9: 134.33394694, 10: 180.30770111,
    20: 852.00, 30: 2540.28, 40: 837.85, 50: 1140.64, 60: 1235.70,
}

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
    plt.xlim(0.5, 11) # Expand x-axis slightly so labels aren't cut off
    plt.legend(loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.show()