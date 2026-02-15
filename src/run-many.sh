#!/bin/bash

# tetrahedron runtimes for p=56, 84, 120
# run-tsgo --container tetrahedron --D_init 0.066 --p 56 --iterations 1 --parallel_iterations 5 --eps 1e-8 --save True --Q 10 --beta_max 10 --num_workers 3
# run-tsgo --container tetrahedron --D_init 0.0545 --p 84 --iterations 1 --parallel_iterations 5 --eps 1e-8 --save True --Q 10 --beta_max 10 --num_workers 3
# run-tsgo --container tetrahedron --D_init 0.049 --p 120 --iterations 2 --parallel_iterations 5 --eps 1e-8 --save True --Q 20 --beta_max 10 --num_workers 3

# h-box runtimes for p=30, 40, 50, 60, 70, 80, 90, 100
run-tsgo --container h-box --D_init 0.379511141 --p 10 --iterations 2 --parallel_iterations 5 --eps 1e-8 --save True --Q 3 --beta_max 5 --num_workers 3
run-tsgo --container h-box --D_init 0.301218192 --p 20 --iterations 2 --parallel_iterations 5 --eps 1e-8 --save True --Q 3 --beta_max 5 --num_workers 3
run-tsgo --container h-box --D_init 0.263138328 --p 30 --iterations 2 --parallel_iterations 5 --eps 1e-8 --save True --Q 3 --beta_max 5 --num_workers 3
run-tsgo --container h-box --D_init 0.239077038 --p 40 --iterations 2 --parallel_iterations 5 --eps 1e-8 --save True --Q 3 --beta_max 10 --num_workers 3
run-tsgo --container h-box --D_init 0.221939462 --p 50 --iterations 2 --parallel_iterations 5 --eps 1e-8 --save True --Q 3 --beta_max 10 --num_workers 3
run-tsgo --container h-box --D_init 0.20885303 --p 60 --iterations 2 --parallel_iterations 5 --eps 1e-8 --save True --Q 3 --beta_max 15 --num_workers 3
# run-tsgo --container h-box --D_init 0.20 --p 70 --iterations 1 --parallel_iterations 5 --eps 1e-8 --save True --Q 10 --beta_max 10 --num_workers 3
# run-tsgo --container h-box --D_init 0.20 --p 80 --iterations 1 --parallel_iterations 5 --eps 1e-8 --save True --Q 10 --beta_max 10 --num_workers 3
# run-tsgo --container h-box --D_init 0.21 --p 90 --iterations 2 --parallel_iterations 5 --eps 1e-8 --save True --Q 15 --beta_max 10 --num_workers 3
# run-tsgo --container h-box --D_init 0.20 --p 100 --iterations 2 --parallel_iterations 5 --eps 1e-8 --save True --Q 20 --beta_max 10 --num_workers 3

# pointy-star runtimes for p=30, 40, 50, 60, 70, 80, 90, 100
# run-tsgo --container pointy-star --D_init 0.659332202 --p 10 --iterations 2 --parallel_iterations 5 --eps 1e-8 --save True --Q 3 --beta_max 5 --num_workers 3
# run-tsgo --container pointy-star --D_init 0.523312315 --p 20 --iterations 2 --parallel_iterations 5 --eps 1e-8 --save True --Q 3 --beta_max 5 --num_workers 3
# run-tsgo --container pointy-star --D_init 0.457155416 --p 30 --iterations 3 --parallel_iterations 5 --eps 1e-8 --save True --Q 3 --beta_max 5 --num_workers 3
# run-tsgo --container pointy-star --D_init 0.41535326 --p 40 --iterations 3 --parallel_iterations 5 --eps 1e-8 --save True --Q 3 --beta_max 15 --num_workers 3
# run-tsgo --container pointy-star --D_init 0.385579811 --p 50 --iterations 4 --parallel_iterations 5 --eps 1e-8 --save True --Q 5 --beta_max 15 --num_workers 3
# run-tsgo --container pointy-star --D_init 0.362844494 --p 60 --iterations 4 --parallel_iterations 5 --eps 1e-8 --save True --Q 5 --beta_max 15 --num_workers 3
# run-tsgo --container pointy-star --D_init 0.35 --p 70 --iterations 1 --parallel_iterations 5 --eps 1e-8 --save True --Q 10 --beta_max 10 --num_workers 3
# run-tsgo --container pointy-star --D_init 0.30 --p 80 --iterations 1 --parallel_iterations 5 --eps 1e-8 --save True --Q 10 --beta_max 10 --num_workers 3
# run-tsgo --container pointy-star --D_init 0.31 --p 90 --iterations 2 --parallel_iterations 5 --eps 1e-8 --save True --Q 10 --beta_max 10 --num_workers 3
# run-tsgo --container pointy-star --D_init 0.33 --p 100 --iterations 2 --parallel_iterations 5 --eps 1e-8 --save True --Q 20 --beta_max 10 --num_workers 3
