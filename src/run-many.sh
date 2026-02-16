#!/bin/bash

# cube
run-tsgo --p 40 --container cube --D_init 0.14433014 --eps 1e-8 --processes 6
run-tsgo --p 50 --container cube --D_init 0.133984233 --eps 1e-8 --processes 6
run-tsgo --p 60 --container cube --D_init 0.128419409 --eps 1e-8 --processes 6

# tetrahedron
run-tsgo --p 1 --container tetrahedron --D_init 0.20045 --eps 1e-8 --processes 6
run-tsgo --p 2 --container tetrahedron --D_init 0.14204 --eps 1e-8 --processes 6
run-tsgo --p 3 --container tetrahedron --D_init 0.14204 --eps 1e-8 --processes 6
run-tsgo --p 4 --container tetrahedron --D_init 0.14204 --eps 1e-8 --processes 6
run-tsgo --p 5 --container tetrahedron --D_init 0.12147 --eps 1e-8 --processes 6
run-tsgo --p 6 --container tetrahedron --D_init 0.11243 --eps 1e-8 --processes 6
run-tsgo --p 7 --container tetrahedron --D_init 0.11062 --eps 1e-8 --processes 6
run-tsgo --p 8 --container tetrahedron --D_init 0.10600 --eps 1e-8 --processes 6
run-tsgo --p 9 --container tetrahedron --D_init 0.10250 --eps 1e-8 --processes 6
run-tsgo --p 10 --container tetrahedron --D_init 0.10200 --eps 1e-8 --processes 6
run-tsgo --p 20 --container tetrahedron --D_init 0.01 --eps 1e-8 --processes 6 --iterations 3
run-tsgo --p 30 --container tetrahedron --D_init 0.01 --eps 1e-8 --processes 6 --iterations 3
run-tsgo --p 40 --container tetrahedron --D_init 0.01 --eps 1e-8 --processes 6 --iterations 3
run-tsgo --p 50 --container tetrahedron --D_init 0.01 --eps 1e-8 --processes 6 --iterations 3
run-tsgo --p 60 --container tetrahedron --D_init 0.01 --eps 1e-8 --processes 6 --iterations 3

# h-box
run-tsgo --p 1 --container h-box --D_init 0.2 --eps 1e-8 --processes 6
run-tsgo --p 2 --container h-box --D_init 0.2 --eps 1e-8 --processes 6
run-tsgo --p 3 --container h-box --D_init 0.2 --eps 1e-8 --processes 6
run-tsgo --p 4 --container h-box --D_init 0.2 --eps 1e-8 --processes 6
run-tsgo --p 5 --container h-box --D_init 0.2 --eps 1e-8 --processes 6
run-tsgo --p 6 --container h-box --D_init 0.2 --eps 1e-8 --processes 6
run-tsgo --p 7 --container h-box --D_init 0.2 --eps 1e-8 --processes 6
run-tsgo --p 8 --container h-box --D_init 0.2 --eps 1e-8 --processes 6 --iterations 3
run-tsgo --p 9 --container h-box --D_init 0.2 --eps 1e-8 --processes 6 --iterations 3
run-tsgo --p 10 --container h-box --D_init 0.2 --eps 1e-8 --processes 6 --iterations 3
run-tsgo --p 20 --container h-box --D_init 0.01 --eps 1e-8 --processes 6 --iterations 5
run-tsgo --p 30 --container h-box --D_init 0.01 --eps 1e-8 --processes 6 --iterations 5
run-tsgo --p 40 --container h-box --D_init 0.01 --eps 1e-8 --processes 6 --iterations 5
run-tsgo --p 50 --container h-box --D_init 0.01 --eps 1e-8 --processes 6 --iterations 5
run-tsgo --p 60 --container h-box --D_init 0.01 --eps 1e-8 --processes 6 --iterations 5

# pointy-star
run-tsgo --p 1 --container pointy-star --D_init 1 --eps 1e-8 --processes 6
run-tsgo --p 2 --container pointy-star --D_init 0.5 --eps 1e-8 --processes 6
run-tsgo --p 3 --container pointy-star --D_init 0.5 --eps 1e-8 --processes 6
run-tsgo --p 4 --container pointy-star --D_init 0.5 --eps 1e-8 --processes 6
run-tsgo --p 5 --container pointy-star --D_init 0.5 --eps 1e-8 --processes 6
run-tsgo --p 6 --container pointy-star --D_init 0.5 --eps 1e-8 --processes 6
run-tsgo --p 7 --container pointy-star --D_init 0.5 --eps 1e-8 --processes 6
run-tsgo --p 8 --container pointy-star --D_init 0.5 --eps 1e-8 --processes 6
run-tsgo --p 9 --container pointy-star --D_init 0.5 --eps 1e-8 --processes 6
run-tsgo --p 10 --container pointy-star --D_init 0.5 --eps 1e-8 --processes 6
run-tsgo --p 20 --container pointy-star --D_init 0.3 --eps 1e-8 --processes 6 --iterations 3
run-tsgo --p 30 --container pointy-star --D_init 0.3 --eps 1e-8 --processes 6 --iterations 3
run-tsgo --p 40 --container pointy-star --D_init 0.3 --eps 1e-8 --processes 6 --iterations 3
run-tsgo --p 50 --container pointy-star --D_init 0.3 --eps 1e-8 --processes 6 --iterations 3
run-tsgo --p 60 --container pointy-star --D_init 0.3 --eps 1e-8 --processes 6 --iterations 3
