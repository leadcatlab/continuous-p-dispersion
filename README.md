# 3D p-Dispersion of Spheres

Simulation code for distributing _p_ spheres in three-dimensional Euclidean space, maximizing a minimum–distance (\(p\)-dispersion) criterion and studying their motion under custom dynamics.

To run the p-dispersion entry-point:

## 1. Clone the package

`git clone <package_name>`


## 2. Install the module inside a virtual environment

`python -m venv <venv_name>`

`source <venv_name>/bin/activate` on linux OR `.\<venv_name>\Scripts\activate` on Windows

`pip install -e .`

## 3. Run the program

`run-tsgo --p <p> --container <container_name> --D_init <initial radius> --plot <True/False> --save <True/False> --num_workers <n> --iterations <N> --parallel_iterations <P>`
