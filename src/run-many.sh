#!/bin/bash

# pointy-star (small redos)
# run-tsgo --p 7 --container pointy-star --D_init 0.5 --eps 1e-8 --processes 6
# run-tsgo --p 8 --container pointy-star --D_init 0.5 --eps 1e-8 --processes 6

# tetrahedron
run-tsgo --p 35 --container tetrahedron --D_init 0.062254603 --eps 1e-8 --processes 6 --iterations 5 --verbose True # rho_init = 0.44
run-tsgo --p 56 --container tetrahedron --D_init 0.060474917 --eps 1e-8 --processes 6 --iterations 5 --verbose True # rho_init = 0.44

# # h-box
# run-tsgo --p 56 --container h-box --D_init 0.25 --eps 1e-8 --processes 6 --iterations 4 --metric SUP_NORM --verbose True
# run-tsgo --p 60 --container h-box --D_init 0.202955061 --eps 1e-8 --processes 6 --iterations 4

# # pointy-star (large)
# run-tsgo --p 40 --container pointy-star --D_init 0.3 --eps 1e-8 --processes 6 --iterations 5
# run-tsgo --p 50 --container pointy-star --D_init 0.3 --eps 1e-8 --processes 6 --iterations 5
# run-tsgo --p 60 --container pointy-star --D_init 0.3 --eps 1e-8 --processes 6 --iterations 5
