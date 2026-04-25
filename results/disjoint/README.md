# Disjoint Problem Set — Raw Results

**250 problem instances** (disjoint constraint variable supports).
Empty cells indicate runs that did not complete (transient cluster errors).

---

## Problem Definitions

| COP | $n_x$ | Families | Structural constraints | Penalized constraints | $n_\text{qubits}$ (H) | SP gates (H) | Layer gates (H) | $n_\text{qubits}$ (P) | SP gates (P) | Layer gates (P) |
|-----|-------|----------|------------------------|-----------------------|----------------------|--------------|----------------|----------------------|--------------|----------------|
| 0 | 6 | cardinality, quadratic_knapsack | $x_{0} + x_{1} \leq 0$ (cardinality)<br>$3 \cdot x_{2} \cdot x_{2} + 1 \cdot x_{2} \cdot x_{3} + 3 \cdot x_{2} \cdot x_{4} + 4 \cdot x_{2} \cdot x_{5} + 5 \cdot x_{3} \cdot x_{3} + 4 \cdot x_{3} \cdot x_{4} + 2 \cdot x_{3} \cdot x_{5} + 2 \cdot x_{4} \cdot x_{4} + 4 \cdot x_{4} \cdot x_{5} + 4 \cdot x_{5} \cdot x_{5} \leq 17$ (quadratic_knapsack) | — | 6 | 19 | 101 | 11 | 5 | 664 |
| 1 | 7 | cardinality, cardinality | $x_{0} + x_{1} + x_{2} + x_{3} \leq 2$ (cardinality)<br>$x_{4} + x_{5} + x_{6} \geq 0$ (cardinality) | — | 7 | 19 | 126 | 11 | 4 | 130 |
| 2 | 8 | knapsack, knapsack | $9 \cdot x_{0} + 2 \cdot x_{1} + 1 \cdot x_{2} + 9 \cdot x_{3} + 6 \cdot x_{4} \leq 13$ (knapsack)<br>$6 \cdot x_{5} + 2 \cdot x_{6} + 2 \cdot x_{7} \leq 3$ (knapsack) | — | 8 | 67 | 239 | 14 | 6 | 211 |
| 3 | 7 | cardinality, cardinality | $x_{0} + x_{1} + x_{2} = 2$ (cardinality)<br>$x_{3} + x_{4} + x_{5} + x_{6} \leq 1$ (cardinality) | — | 7 | 24 | 129 | 8 | 1 | 87 |
| 4 | 8 | knapsack, cardinality | $3 \cdot x_{0} + 1 \cdot x_{1} + 4 \cdot x_{2} + 8 \cdot x_{3} \leq 7$ (knapsack)<br>$x_{4} + x_{5} + x_{6} + x_{7} = 4$ (cardinality) | — | 8 | 61 | 229 | 11 | 3 | 147 |
| 5 | 6 | quadratic_knapsack, knapsack | $5 \cdot x_{0} \cdot x_{0} + 1 \cdot x_{0} \cdot x_{1} + 2 \cdot x_{0} \cdot x_{2} + 5 \cdot x_{1} \cdot x_{1} + 1 \cdot x_{1} \cdot x_{2} + 3 \cdot x_{2} \cdot x_{2} \leq 6$ (quadratic_knapsack)<br>$6 \cdot x_{3} + 2 \cdot x_{4} + 2 \cdot x_{5} \leq 3$ (knapsack) | — | 6 | 31 | 121 | 11 | 5 | 272 |
| 6 | 6 | quadratic_knapsack, knapsack | $5 \cdot x_{0} \cdot x_{0} + 3 \cdot x_{0} \cdot x_{1} + 5 \cdot x_{0} \cdot x_{2} + 4 \cdot x_{1} \cdot x_{1} + 5 \cdot x_{1} \cdot x_{2} + 3 \cdot x_{2} \cdot x_{2} \leq 14$ (quadratic_knapsack)<br>$2 \cdot x_{3} + 5 \cdot x_{4} + 2 \cdot x_{5} \leq 5$ (knapsack) | — | 6 | 75 | 219 | 13 | 7 | 336 |
| 7 | 7 | assignment, flow | $x_{0} + x_{1} + x_{2} = 1$ (assignment)<br>$x_{3} + x_{4} - x_{5} - x_{6} = 0$ (flow) | — | 7 | 17 | 119 | 7 | 0 | 70 |
| 8 | 5 | cardinality, quadratic_knapsack | $x_{0} + x_{1} \leq 2$ (cardinality)<br>$1 \cdot x_{2} \cdot x_{2} + 5 \cdot x_{2} \cdot x_{3} + 4 \cdot x_{2} \cdot x_{4} + 2 \cdot x_{3} \cdot x_{3} + 2 \cdot x_{3} \cdot x_{4} + 2 \cdot x_{4} \cdot x_{4} \leq 8$ (quadratic_knapsack) | — | 5 | 18 | 86 | 11 | 6 | 297 |
| 9 | 7 | quadratic_knapsack, cardinality | $1 \cdot x_{0} \cdot x_{0} + 5 \cdot x_{0} \cdot x_{1} + 3 \cdot x_{0} \cdot x_{2} + 4 \cdot x_{1} \cdot x_{1} + 2 \cdot x_{1} \cdot x_{2} + 4 \cdot x_{2} \cdot x_{2} \leq 7$ (quadratic_knapsack)<br>$x_{3} + x_{4} + x_{5} + x_{6} \geq 3$ (cardinality) | — | 7 | 8 | 104 | 11 | 4 | 282 |
| 10 | 7 | knapsack, quadratic_knapsack | $7 \cdot x_{0} + 7 \cdot x_{1} + 10 \cdot x_{2} + 2 \cdot x_{3} \leq 10$ (knapsack)<br>$5 \cdot x_{4} \cdot x_{4} + 3 \cdot x_{4} \cdot x_{5} + 5 \cdot x_{4} \cdot x_{6} + 4 \cdot x_{5} \cdot x_{5} + 5 \cdot x_{5} \cdot x_{6} + 3 \cdot x_{6} \cdot x_{6} \leq 14$ (quadratic_knapsack) | — | 7 | 281 | 643 | 15 | 8 | 384 |
| 11 | 8 | knapsack, cardinality | $3 \cdot x_{0} + 8 \cdot x_{1} + 3 \cdot x_{2} \leq 5$ (knapsack)<br>$x_{3} + x_{4} + x_{5} + x_{6} + x_{7} \geq 0$ (cardinality) | — | 8 | 103 | 318 | 14 | 6 | 197 |
| 12 | 8 | quadratic_knapsack, cardinality | $2 \cdot x_{0} \cdot x_{0} + 5 \cdot x_{0} \cdot x_{1} + 2 \cdot x_{0} \cdot x_{2} + 2 \cdot x_{1} \cdot x_{1} + 1 \cdot x_{1} \cdot x_{2} + 1 \cdot x_{2} \cdot x_{2} \leq 7$ (quadratic_knapsack)<br>$x_{3} + x_{4} + x_{5} + x_{6} + x_{7} = 5$ (cardinality) | — | 8 | 56 | 217 | 11 | 3 | 286 |
| 13 | 7 | quadratic_knapsack, cardinality | $2 \cdot x_{0} \cdot x_{0} + 5 \cdot x_{0} \cdot x_{1} + 4 \cdot x_{0} \cdot x_{2} + 2 \cdot x_{0} \cdot x_{3} + 4 \cdot x_{0} \cdot x_{4} + 5 \cdot x_{1} \cdot x_{1} + 5 \cdot x_{1} \cdot x_{2} + 2 \cdot x_{1} \cdot x_{3} + 1 \cdot x_{1} \cdot x_{4} + 4 \cdot x_{2} \cdot x_{2} + 1 \cdot x_{2} \cdot x_{3} + 2 \cdot x_{2} \cdot x_{4} + 5 \cdot x_{3} \cdot x_{3} + 4 \cdot x_{3} \cdot x_{4} + 2 \cdot x_{4} \cdot x_{4} \leq 29$ (quadratic_knapsack)<br>$x_{5} + x_{6} \leq 1$ (cardinality) | — | 7 | 37 | 155 | 13 | 6 | 1332 |
| 14 | 8 | quadratic_knapsack, knapsack | $1 \cdot x_{0} \cdot x_{0} + 4 \cdot x_{0} \cdot x_{1} + 1 \cdot x_{0} \cdot x_{2} + 5 \cdot x_{0} \cdot x_{3} + 2 \cdot x_{1} \cdot x_{1} + 3 \cdot x_{1} \cdot x_{2} + 3 \cdot x_{1} \cdot x_{3} + 5 \cdot x_{2} \cdot x_{2} + 4 \cdot x_{2} \cdot x_{3} + 5 \cdot x_{3} \cdot x_{3} \leq 22$ (quadratic_knapsack)<br>$3 \cdot x_{4} + 1 \cdot x_{5} + 4 \cdot x_{6} + 8 \cdot x_{7} \leq 7$ (knapsack) | — | 8 | 80 | 267 | 16 | 8 | 755 |
| 15 | 8 | cardinality, knapsack | $x_{0} + x_{1} + x_{2} \geq 2$ (cardinality)<br>$2 \cdot x_{3} + 2 \cdot x_{4} + 9 \cdot x_{5} + 10 \cdot x_{6} + 4 \cdot x_{7} \leq 18$ (knapsack) | — | 8 | 35 | 180 | 14 | 6 | 229 |
| 16 | 6 | quadratic_knapsack, quadratic_knapsack | $2 \cdot x_{0} \cdot x_{0} + 5 \cdot x_{0} \cdot x_{1} + 2 \cdot x_{0} \cdot x_{2} + 2 \cdot x_{1} \cdot x_{1} + 1 \cdot x_{1} \cdot x_{2} + 1 \cdot x_{2} \cdot x_{2} \leq 7$ (quadratic_knapsack)<br>$3 \cdot x_{3} \cdot x_{3} + 1 \cdot x_{3} \cdot x_{4} + 3 \cdot x_{3} \cdot x_{5} + 5 \cdot x_{4} \cdot x_{4} + 2 \cdot x_{4} \cdot x_{5} + 2 \cdot x_{5} \cdot x_{5} \leq 8$ (quadratic_knapsack) | — | 6 | 74 | 214 | 13 | 7 | 482 |
| 17 | 8 | cardinality, knapsack | $x_{0} + x_{1} + x_{2} + x_{3} + x_{4} = 1$ (cardinality)<br>$2 \cdot x_{5} + 7 \cdot x_{6} + 7 \cdot x_{7} \leq 7$ (knapsack) | — | 8 | 17 | 144 | 11 | 3 | 142 |
| 18 | 8 | knapsack, flow | $9 \cdot x_{0} + 9 \cdot x_{1} + 4 \cdot x_{2} + 1 \cdot x_{3} \leq 9$ (knapsack)<br>$x_{4} - x_{5} - x_{6} - x_{7} = 0$ (flow) | — | 8 | 75 | 263 | 12 | 4 | 169 |
| 19 | 8 | cardinality, cardinality | $x_{0} + x_{1} + x_{2} \geq 2$ (cardinality)<br>$x_{3} + x_{4} + x_{5} + x_{6} + x_{7} \leq 5$ (cardinality) | — | 8 | 37 | 184 | 12 | 4 | 172 |
| 20 | 6 | cardinality, quadratic_knapsack | $x_{0} + x_{1} + x_{2} = 1$ (cardinality)<br>$5 \cdot x_{3} \cdot x_{3} + 3 \cdot x_{3} \cdot x_{4} + 5 \cdot x_{3} \cdot x_{5} + 4 \cdot x_{4} \cdot x_{4} + 5 \cdot x_{4} \cdot x_{5} + 3 \cdot x_{5} \cdot x_{5} \leq 14$ (quadratic_knapsack) | — | 6 | 17 | 97 | 10 | 4 | 289 |
| 21 | 6 | independent_set, quadratic_knapsack | $x_{0} \cdot x_{1} = 0$ (independent_set)<br>$1 \cdot x_{2} \cdot x_{2} + 1 \cdot x_{2} \cdot x_{3} + 3 \cdot x_{2} \cdot x_{4} + 2 \cdot x_{2} \cdot x_{5} + 1 \cdot x_{3} \cdot x_{3} + 2 \cdot x_{3} \cdot x_{4} + 3 \cdot x_{3} \cdot x_{5} + 2 \cdot x_{4} \cdot x_{4} + 3 \cdot x_{4} \cdot x_{5} + 3 \cdot x_{5} \cdot x_{5} \leq 7$ (quadratic_knapsack) | — | 6 | 128 | 325 | 9 | 3 | 529 |
| 22 | 6 | cardinality, quadratic_knapsack | $x_{0} + x_{1} + x_{2} \leq 3$ (cardinality)<br>$1 \cdot x_{3} \cdot x_{3} + 5 \cdot x_{3} \cdot x_{4} + 4 \cdot x_{3} \cdot x_{5} + 2 \cdot x_{4} \cdot x_{4} + 2 \cdot x_{4} \cdot x_{5} + 2 \cdot x_{5} \cdot x_{5} \leq 8$ (quadratic_knapsack) | — | 6 | 25 | 113 | 12 | 6 | 316 |
| 23 | 8 | cardinality, knapsack | $x_{0} + x_{1} + x_{2} + x_{3} \geq 2$ (cardinality)<br>$8 \cdot x_{4} + 5 \cdot x_{5} + 7 \cdot x_{6} + 4 \cdot x_{7} \leq 11$ (knapsack) | — | 8 | 57 | 221 | 14 | 6 | 204 |
| 24 | 8 | quadratic_knapsack, cardinality | $3 \cdot x_{0} \cdot x_{0} + 1 \cdot x_{0} \cdot x_{1} + 3 \cdot x_{0} \cdot x_{2} + 1 \cdot x_{1} \cdot x_{1} + 4 \cdot x_{1} \cdot x_{2} + 3 \cdot x_{2} \cdot x_{2} \leq 5$ (quadratic_knapsack)<br>$x_{3} + x_{4} + x_{5} + x_{6} + x_{7} = 3$ (cardinality) | — | 8 | 114 | 338 | 11 | 3 | 289 |
| 25 | 7 | cardinality, cardinality | $x_{0} + x_{1} + x_{2} + x_{3} + x_{4} \leq 1$ (cardinality)<br>$x_{5} + x_{6} = 2$ (cardinality) | — | 7 | 17 | 122 | 8 | 1 | 99 |
| 26 | 6 | cardinality, quadratic_knapsack | $x_{0} + x_{1} + x_{2} = 0$ (cardinality)<br>$1 \cdot x_{3} \cdot x_{3} + 2 \cdot x_{3} \cdot x_{4} + 1 \cdot x_{3} \cdot x_{5} + 1 \cdot x_{4} \cdot x_{4} + 5 \cdot x_{4} \cdot x_{5} + 1 \cdot x_{5} \cdot x_{5} \leq 6$ (quadratic_knapsack) | — | 6 | 43 | 152 | 9 | 3 | 245 |
| 27 | 5 | cardinality, flow | $x_{0} + x_{1} \geq 2$ (cardinality)<br>$x_{2} - x_{3} - x_{4} = 0$ (flow) | — | 5 | 8 | 66 | 5 | 0 | 39 |
| 28 | 8 | quadratic_knapsack, flow | $3 \cdot x_{0} \cdot x_{0} + 1 \cdot x_{0} \cdot x_{1} + 3 \cdot x_{0} \cdot x_{2} + 5 \cdot x_{1} \cdot x_{1} + 2 \cdot x_{1} \cdot x_{2} + 2 \cdot x_{2} \cdot x_{2} \leq 8$ (quadratic_knapsack)<br>$x_{3} + x_{4} - x_{5} - x_{6} - x_{7} = 0$ (flow) | — | 8 | 41 | 189 | 12 | 4 | 325 |
| 29 | 7 | cardinality, cardinality | $x_{0} + x_{1} \geq 1$ (cardinality)<br>$x_{2} + x_{3} + x_{4} + x_{5} + x_{6} \geq 3$ (cardinality) | — | 7 | 0 | 81 | 10 | 3 | 126 |
| 30 | 8 | cardinality, cardinality | $x_{0} + x_{1} + x_{2} + x_{3} \geq 0$ (cardinality)<br>$x_{4} + x_{5} + x_{6} + x_{7} = 3$ (cardinality) | — | 8 | 23 | 162 | 11 | 3 | 143 |
| 31 | 7 | knapsack, quadratic_knapsack | $5 \cdot x_{0} + 4 \cdot x_{1} + 1 \cdot x_{2} \leq 6$ (knapsack)<br>$1 \cdot x_{3} \cdot x_{3} + 4 \cdot x_{3} \cdot x_{4} + 1 \cdot x_{3} \cdot x_{5} + 2 \cdot x_{3} \cdot x_{6} + 5 \cdot x_{4} \cdot x_{4} + 4 \cdot x_{4} \cdot x_{5} + 3 \cdot x_{4} \cdot x_{6} + 5 \cdot x_{5} \cdot x_{5} + 4 \cdot x_{5} \cdot x_{6} + 2 \cdot x_{6} \cdot x_{6} \leq 18$ (quadratic_knapsack) | — | 7 | 386 | 851 | 15 | 8 | 725 |
| 32 | 7 | cardinality, knapsack | $x_{0} + x_{1} \geq 0$ (cardinality)<br>$1 \cdot x_{2} + 6 \cdot x_{3} + 9 \cdot x_{4} + 8 \cdot x_{5} + 9 \cdot x_{6} \leq 18$ (knapsack) | — | 7 | 139 | 357 | 14 | 7 | 215 |
| 33 | 6 | assignment, cardinality | $x_{0} + x_{1} + x_{2} = 1$ (assignment)<br>$x_{3} + x_{4} + x_{5} = 2$ (cardinality) | — | 6 | 17 | 95 | 6 | 0 | 53 |
| 34 | 8 | cardinality, knapsack | $x_{0} + x_{1} + x_{2} + x_{3} + x_{4} \geq 4$ (cardinality)<br>$7 \cdot x_{5} + 10 \cdot x_{6} + 9 \cdot x_{7} \leq 8$ (knapsack) | — | 8 | 11 | 128 | 13 | 5 | 179 |
| 35 | 4 | assignment, cardinality | $x_{0} + x_{1} = 1$ (assignment)<br>$x_{2} + x_{3} \geq 1$ (cardinality) | — | 4 | 3 | 39 | 5 | 1 | 36 |
| 36 | 7 | cardinality, knapsack | $x_{0} + x_{1} \geq 0$ (cardinality)<br>$5 \cdot x_{2} + 10 \cdot x_{3} + 1 \cdot x_{4} + 9 \cdot x_{5} + 6 \cdot x_{6} \leq 19$ (knapsack) | — | 7 | 407 | 899 | 14 | 7 | 217 |
| 37 | 8 | cardinality, flow | $x_{0} + x_{1} \geq 1$ (cardinality)<br>$x_{2} + x_{3} + x_{4} - x_{5} - x_{6} - x_{7} = 0$ (flow) | — | 8 | 26 | 159 | 9 | 1 | 108 |
| 38 | 7 | knapsack, cardinality | $1 \cdot x_{0} + 9 \cdot x_{1} + 5 \cdot x_{2} + 2 \cdot x_{3} + 10 \cdot x_{4} \leq 10$ (knapsack)<br>$x_{5} + x_{6} \geq 2$ (cardinality) | — | 7 | 141 | 363 | 11 | 4 | 168 |
| 39 | 8 | knapsack, cardinality | $1 \cdot x_{0} + 4 \cdot x_{1} + 8 \cdot x_{2} + 10 \cdot x_{3} + 8 \cdot x_{4} \leq 16$ (knapsack)<br>$x_{5} + x_{6} + x_{7} = 1$ (cardinality) | — | 8 | 412 | 934 | 13 | 5 | 217 |
| 40 | 8 | quadratic_knapsack, knapsack | $1 \cdot x_{0} \cdot x_{0} + 5 \cdot x_{0} \cdot x_{1} + 3 \cdot x_{0} \cdot x_{2} + 4 \cdot x_{1} \cdot x_{1} + 2 \cdot x_{1} \cdot x_{2} + 4 \cdot x_{2} \cdot x_{2} \leq 7$ (quadratic_knapsack)<br>$6 \cdot x_{3} + 7 \cdot x_{4} + 8 \cdot x_{5} + 9 \cdot x_{6} + 5 \cdot x_{7} \leq 14$ (knapsack) | — | 8 | 511 | 1134 | 15 | 7 | 379 |
| 41 | 5 | cardinality, knapsack | $x_{0} + x_{1} = 1$ (cardinality)<br>$2 \cdot x_{2} + 5 \cdot x_{3} + 2 \cdot x_{4} \leq 5$ (knapsack) | — | 5 | 66 | 182 | 8 | 3 | 87 |
| 42 | 7 | cardinality, knapsack | $x_{0} + x_{1} + x_{2} = 0$ (cardinality)<br>$7 \cdot x_{3} + 7 \cdot x_{4} + 10 \cdot x_{5} + 2 \cdot x_{6} \leq 10$ (knapsack) | — | 7 | 269 | 619 | 11 | 4 | 147 |
| 43 | 8 | quadratic_knapsack, quadratic_knapsack | $1 \cdot x_{0} \cdot x_{0} + 5 \cdot x_{0} \cdot x_{1} + 3 \cdot x_{0} \cdot x_{2} + 4 \cdot x_{1} \cdot x_{1} + 2 \cdot x_{1} \cdot x_{2} + 4 \cdot x_{2} \cdot x_{2} \leq 7$ (quadratic_knapsack)<br>$3 \cdot x_{3} \cdot x_{3} + 1 \cdot x_{3} \cdot x_{4} + 5 \cdot x_{3} \cdot x_{5} + 2 \cdot x_{3} \cdot x_{6} + 5 \cdot x_{3} \cdot x_{7} + 2 \cdot x_{4} \cdot x_{4} + 5 \cdot x_{4} \cdot x_{5} + 5 \cdot x_{4} \cdot x_{6} + 5 \cdot x_{4} \cdot x_{7} + 2 \cdot x_{5} \cdot x_{5} + 3 \cdot x_{5} \cdot x_{6} + 3 \cdot x_{5} \cdot x_{7} + 4 \cdot x_{6} \cdot x_{6} + 1 \cdot x_{6} \cdot x_{7} + 5 \cdot x_{7} \cdot x_{7} \leq 18$ (quadratic_knapsack) | — | 8 | 415 | 937 | 16 | 8 | 1533 |
| 44 | 7 | cardinality, cardinality | $x_{0} + x_{1} + x_{2} + x_{3} \geq 2$ (cardinality)<br>$x_{4} + x_{5} + x_{6} \leq 1$ (cardinality) | — | 7 | 8 | 104 | 10 | 3 | 120 |
| 45 | 8 | quadratic_knapsack, cardinality | $3 \cdot x_{0} \cdot x_{0} + 1 \cdot x_{0} \cdot x_{1} + 3 \cdot x_{0} \cdot x_{2} + 1 \cdot x_{1} \cdot x_{1} + 4 \cdot x_{1} \cdot x_{2} + 3 \cdot x_{2} \cdot x_{2} \leq 5$ (quadratic_knapsack)<br>$x_{3} + x_{4} + x_{5} + x_{6} + x_{7} \leq 1$ (cardinality) | — | 8 | 98 | 312 | 12 | 4 | 309 |
| 46 | 8 | cardinality, knapsack | $x_{0} + x_{1} + x_{2} = 2$ (cardinality)<br>$1 \cdot x_{3} + 10 \cdot x_{4} + 3 \cdot x_{5} + 5 \cdot x_{6} + 1 \cdot x_{7} \leq 12$ (knapsack) | — | 8 | 83 | 276 | 12 | 4 | 187 |
| 47 | 7 | knapsack, independent_set | $1 \cdot x_{0} + 10 \cdot x_{1} + 3 \cdot x_{2} + 5 \cdot x_{3} + 1 \cdot x_{4} \leq 12$ (knapsack)<br>$x_{5} \cdot x_{6} = 0$ (independent_set) | — | 7 | 71 | 224 | 11 | 4 | 165 |
| 48 | 7 | quadratic_knapsack, knapsack | $3 \cdot x_{0} \cdot x_{0} + 1 \cdot x_{0} \cdot x_{1} + 3 \cdot x_{0} \cdot x_{2} + 4 \cdot x_{0} \cdot x_{3} + 5 \cdot x_{1} \cdot x_{1} + 4 \cdot x_{1} \cdot x_{2} + 2 \cdot x_{1} \cdot x_{3} + 2 \cdot x_{2} \cdot x_{2} + 4 \cdot x_{2} \cdot x_{3} + 4 \cdot x_{3} \cdot x_{3} \leq 17$ (quadratic_knapsack)<br>$5 \cdot x_{4} + 4 \cdot x_{5} + 1 \cdot x_{6} \leq 6$ (knapsack) | — | 7 | 30 | 139 | 15 | 8 | 725 |
| 49 | 6 | knapsack, cardinality | $1 \cdot x_{0} + 3 \cdot x_{1} + 9 \cdot x_{2} + 8 \cdot x_{3} \leq 10$ (knapsack)<br>$x_{4} + x_{5} \leq 1$ (cardinality) | — | 6 | 105 | 269 | 11 | 5 | 143 |
| 50 | 6 | knapsack, cardinality | $2 \cdot x_{0} + 7 \cdot x_{1} + 7 \cdot x_{2} \leq 7$ (knapsack)<br>$x_{3} + x_{4} + x_{5} \leq 3$ (cardinality) | — | 6 | 21 | 108 | 11 | 5 | 128 |
| 51 | 8 | flow, cardinality | $x_{0} - x_{1} - x_{2} = 0$ (flow)<br>$x_{3} + x_{4} + x_{5} + x_{6} + x_{7} \leq 3$ (cardinality) | — | 8 | 38 | 192 | 10 | 2 | 135 |
| 52 | 6 | quadratic_knapsack, cardinality | $1 \cdot x_{0} \cdot x_{0} + 5 \cdot x_{0} \cdot x_{1} + 3 \cdot x_{0} \cdot x_{2} + 4 \cdot x_{1} \cdot x_{1} + 2 \cdot x_{1} \cdot x_{2} + 4 \cdot x_{2} \cdot x_{2} \leq 7$ (quadratic_knapsack)<br>$x_{3} + x_{4} + x_{5} \leq 0$ (cardinality) | — | 6 | 8 | 75 | 9 | 3 | 242 |
| 53 | 8 | quadratic_knapsack, cardinality | $2 \cdot x_{0} \cdot x_{0} + 5 \cdot x_{0} \cdot x_{1} + 4 \cdot x_{0} \cdot x_{2} + 4 \cdot x_{0} \cdot x_{3} + 3 \cdot x_{0} \cdot x_{4} + 4 \cdot x_{1} \cdot x_{1} + 4 \cdot x_{1} \cdot x_{2} + 1 \cdot x_{1} \cdot x_{3} + 2 \cdot x_{1} \cdot x_{4} + 3 \cdot x_{2} \cdot x_{2} + 1 \cdot x_{2} \cdot x_{3} + 4 \cdot x_{2} \cdot x_{4} + 3 \cdot x_{3} \cdot x_{3} + 4 \cdot x_{3} \cdot x_{4} + 3 \cdot x_{4} \cdot x_{4} \leq 16$ (quadratic_knapsack)<br>$x_{5} + x_{6} + x_{7} \geq 3$ (cardinality) | — | 8 | 1080 | 2270 | 13 | 5 | 1342 |
| 54 | 7 | cardinality, cardinality | $x_{0} + x_{1} + x_{2} + x_{3} \geq 3$ (cardinality)<br>$x_{4} + x_{5} + x_{6} \geq 3$ (cardinality) | — | 7 | 3 | 94 | 8 | 1 | 90 |
| 55 | 6 | cardinality, cardinality | $x_{0} + x_{1} + x_{2} + x_{3} \leq 2$ (cardinality)<br>$x_{4} + x_{5} \leq 0$ (cardinality) | — | 6 | 19 | 99 | 8 | 2 | 87 |
| 56 | 6 | assignment, quadratic_knapsack | $x_{0} + x_{1} = 1$ (assignment)<br>$5 \cdot x_{2} \cdot x_{2} + 3 \cdot x_{2} \cdot x_{3} + 1 \cdot x_{2} \cdot x_{4} + 5 \cdot x_{2} \cdot x_{5} + 3 \cdot x_{3} \cdot x_{3} + 1 \cdot x_{3} \cdot x_{4} + 4 \cdot x_{3} \cdot x_{5} + 5 \cdot x_{4} \cdot x_{4} + 5 \cdot x_{4} \cdot x_{5} + 1 \cdot x_{5} \cdot x_{5} \leq 20$ (quadratic_knapsack) | — | 6 | 378 | 819 | 11 | 5 | 666 |
| 57 | 8 | cardinality, flow | $x_{0} + x_{1} + x_{2} + x_{3} + x_{4} \geq 1$ (cardinality)<br>$x_{5} - x_{6} - x_{7} = 0$ (flow) | — | 8 | 6 | 119 | 11 | 3 | 156 |
| 58 | 6 | cardinality, cardinality | $x_{0} + x_{1} + x_{2} + x_{3} \geq 1$ (cardinality)<br>$x_{4} + x_{5} \leq 2$ (cardinality) | — | 6 | 6 | 75 | 10 | 4 | 112 |
| 59 | 8 | knapsack, flow | $4 \cdot x_{0} + 6 \cdot x_{1} + 1 \cdot x_{2} + 1 \cdot x_{3} \leq 6$ (knapsack)<br>$x_{4} + x_{5} - x_{6} - x_{7} = 0$ (flow) | — | 8 | 281 | 669 | 11 | 3 | 143 |
| 60 | 8 | cardinality, knapsack | $x_{0} + x_{1} + x_{2} + x_{3} + x_{4} \leq 4$ (cardinality)<br>$2 \cdot x_{5} + 7 \cdot x_{6} + 7 \cdot x_{7} \leq 7$ (knapsack) | — | 8 | 44 | 201 | 14 | 6 | 206 |
| 61 | 6 | quadratic_knapsack, flow | $3 \cdot x_{0} \cdot x_{0} + 1 \cdot x_{0} \cdot x_{1} + 3 \cdot x_{0} \cdot x_{2} + 5 \cdot x_{1} \cdot x_{1} + 2 \cdot x_{1} \cdot x_{2} + 2 \cdot x_{2} \cdot x_{2} \leq 8$ (quadratic_knapsack)<br>$x_{3} - x_{4} - x_{5} = 0$ (flow) | — | 6 | 29 | 121 | 10 | 4 | 286 |
| 62 | 7 | knapsack, quadratic_knapsack | $2 \cdot x_{0} + 7 \cdot x_{1} + 7 \cdot x_{2} \leq 7$ (knapsack)<br>$5 \cdot x_{3} \cdot x_{3} + 1 \cdot x_{3} \cdot x_{4} + 3 \cdot x_{3} \cdot x_{5} + 1 \cdot x_{3} \cdot x_{6} + 4 \cdot x_{4} \cdot x_{4} + 1 \cdot x_{4} \cdot x_{5} + 5 \cdot x_{4} \cdot x_{6} + 4 \cdot x_{5} \cdot x_{5} + 2 \cdot x_{5} \cdot x_{6} + 3 \cdot x_{6} \cdot x_{6} \leq 19$ (quadratic_knapsack) | — | 7 | 31 | 150 | 15 | 8 | 728 |
| 63 | 7 | knapsack, knapsack | $4 \cdot x_{0} + 6 \cdot x_{1} + 2 \cdot x_{2} + 10 \cdot x_{3} \leq 7$ (knapsack)<br>$10 \cdot x_{4} + 8 \cdot x_{5} + 10 \cdot x_{6} \leq 9$ (knapsack) | — | 7 | 68 | 219 | 14 | 7 | 192 |
| 64 | 8 | quadratic_knapsack, cardinality | $1 \cdot x_{0} \cdot x_{0} + 5 \cdot x_{0} \cdot x_{1} + 1 \cdot x_{0} \cdot x_{2} + 3 \cdot x_{1} \cdot x_{1} + 3 \cdot x_{1} \cdot x_{2} + 5 \cdot x_{2} \cdot x_{2} \leq 10$ (quadratic_knapsack)<br>$x_{3} + x_{4} + x_{5} + x_{6} + x_{7} = 3$ (cardinality) | — | 8 | 42 | 200 | 12 | 4 | 333 |
| 65 | 5 | cardinality, cardinality | $x_{0} + x_{1} + x_{2} \geq 1$ (cardinality)<br>$x_{3} + x_{4} \leq 0$ (cardinality) | — | 5 | 0 | 50 | 7 | 2 | 67 |
| 66 | 5 | independent_set, knapsack | $x_{0} \cdot x_{1} = 0$ (independent_set)<br>$10 \cdot x_{2} + 8 \cdot x_{3} + 10 \cdot x_{4} \leq 9$ (knapsack) | — | 5 | 11 | 72 | 9 | 4 | 104 |
| 67 | 8 | knapsack, cardinality | $2 \cdot x_{0} + 5 \cdot x_{1} + 2 \cdot x_{2} \leq 5$ (knapsack)<br>$x_{3} + x_{4} + x_{5} + x_{6} + x_{7} \geq 5$ (cardinality) | — | 8 | 68 | 243 | 11 | 3 | 141 |
| 68 | 6 | quadratic_knapsack, quadratic_knapsack | $3 \cdot x_{0} \cdot x_{0} + 1 \cdot x_{0} \cdot x_{1} + 3 \cdot x_{0} \cdot x_{2} + 1 \cdot x_{1} \cdot x_{1} + 4 \cdot x_{1} \cdot x_{2} + 3 \cdot x_{2} \cdot x_{2} \leq 5$ (quadratic_knapsack)<br>$5 \cdot x_{3} \cdot x_{3} + 1 \cdot x_{3} \cdot x_{4} + 2 \cdot x_{3} \cdot x_{5} + 5 \cdot x_{4} \cdot x_{4} + 1 \cdot x_{4} \cdot x_{5} + 3 \cdot x_{5} \cdot x_{5} \leq 6$ (quadratic_knapsack) | — | 6 | 91 | 245 | 12 | 6 | 439 |
| 69 | 6 | flow, quadratic_knapsack | $x_{0} - x_{1} = 0$ (flow)<br>$4 \cdot x_{2} \cdot x_{2} + 5 \cdot x_{2} \cdot x_{3} + 2 \cdot x_{2} \cdot x_{4} + 5 \cdot x_{2} \cdot x_{5} + 4 \cdot x_{3} \cdot x_{3} + 3 \cdot x_{3} \cdot x_{4} + 5 \cdot x_{3} \cdot x_{5} + 3 \cdot x_{4} \cdot x_{4} + 2 \cdot x_{4} \cdot x_{5} + 3 \cdot x_{5} \cdot x_{5} \leq 22$ (quadratic_knapsack) | — | 6 | 27 | 120 | 11 | 5 | 665 |
| 70 | 6 | knapsack, cardinality | $2 \cdot x_{0} + 5 \cdot x_{1} + 2 \cdot x_{2} \leq 5$ (knapsack)<br>$x_{3} + x_{4} + x_{5} \geq 0$ (cardinality) | — | 6 | 63 | 189 | 11 | 5 | 122 |
| 71 | 7 | cardinality, cardinality | $x_{0} + x_{1} + x_{2} \geq 2$ (cardinality)<br>$x_{3} + x_{4} + x_{5} + x_{6} \leq 3$ (cardinality) | — | 7 | 23 | 131 | 10 | 3 | 119 |
| 72 | 7 | quadratic_knapsack, cardinality | $1 \cdot x_{0} \cdot x_{0} + 2 \cdot x_{0} \cdot x_{1} + 1 \cdot x_{0} \cdot x_{2} + 1 \cdot x_{1} \cdot x_{1} + 5 \cdot x_{1} \cdot x_{2} + 1 \cdot x_{2} \cdot x_{2} \leq 6$ (quadratic_knapsack)<br>$x_{3} + x_{4} + x_{5} + x_{6} \leq 1$ (cardinality) | — | 7 | 55 | 198 | 11 | 4 | 282 |
| 73 | 7 | knapsack, cardinality | $7 \cdot x_{0} + 10 \cdot x_{1} + 9 \cdot x_{2} \leq 8$ (knapsack)<br>$x_{3} + x_{4} + x_{5} + x_{6} \leq 2$ (cardinality) | — | 7 | 30 | 142 | 13 | 6 | 172 |
| 74 | 8 | flow, cardinality, independent_set | $x_{0} + x_{1} - x_{2} = 0$ (flow)<br>$x_{3} + x_{4} + x_{5} \leq 3$ (cardinality)<br>$x_{6} \cdot x_{7} = 0$ (independent_set) | — | 8 | 19 | 145 | 10 | 2 | 98 |
| 75 | 8 | cardinality, knapsack | $x_{0} + x_{1} + x_{2} = 1$ (cardinality)<br>$6 \cdot x_{3} + 7 \cdot x_{4} + 8 \cdot x_{5} + 9 \cdot x_{6} + 5 \cdot x_{7} \leq 14$ (knapsack) | — | 8 | 508 | 1126 | 12 | 4 | 187 |
| 76 | 8 | quadratic_knapsack, knapsack | $2 \cdot x_{0} \cdot x_{0} + 5 \cdot x_{0} \cdot x_{1} + 4 \cdot x_{0} \cdot x_{2} + 4 \cdot x_{0} \cdot x_{3} + 3 \cdot x_{0} \cdot x_{4} + 4 \cdot x_{1} \cdot x_{1} + 4 \cdot x_{1} \cdot x_{2} + 1 \cdot x_{1} \cdot x_{3} + 2 \cdot x_{1} \cdot x_{4} + 3 \cdot x_{2} \cdot x_{2} + 1 \cdot x_{2} \cdot x_{3} + 4 \cdot x_{2} \cdot x_{4} + 3 \cdot x_{3} \cdot x_{3} + 4 \cdot x_{3} \cdot x_{4} + 3 \cdot x_{4} \cdot x_{4} \leq 16$ (quadratic_knapsack)<br>$3 \cdot x_{5} + 8 \cdot x_{6} + 3 \cdot x_{7} \leq 5$ (knapsack) | — | 8 | 1180 | 2470 | 16 | 8 | 1387 |
| 77 | 7 | knapsack, cardinality | $1 \cdot x_{0} + 6 \cdot x_{1} + 9 \cdot x_{2} + 8 \cdot x_{3} + 9 \cdot x_{4} \leq 18$ (knapsack)<br>$x_{5} + x_{6} \geq 2$ (cardinality) | — | 7 | 141 | 366 | 12 | 5 | 199 |
| 78 | 7 | quadratic_knapsack, cardinality | $2 \cdot x_{0} \cdot x_{0} + 4 \cdot x_{0} \cdot x_{1} + 5 \cdot x_{0} \cdot x_{2} + 1 \cdot x_{0} \cdot x_{3} + 4 \cdot x_{0} \cdot x_{4} + 2 \cdot x_{1} \cdot x_{1} + 4 \cdot x_{1} \cdot x_{2} + 4 \cdot x_{1} \cdot x_{3} + 1 \cdot x_{1} \cdot x_{4} + 2 \cdot x_{2} \cdot x_{2} + 3 \cdot x_{2} \cdot x_{3} + 3 \cdot x_{2} \cdot x_{4} + 3 \cdot x_{3} \cdot x_{3} + 2 \cdot x_{3} \cdot x_{4} + 1 \cdot x_{4} \cdot x_{4} \leq 19$ (quadratic_knapsack)<br>$x_{5} + x_{6} = 2$ (cardinality) | — | 7 | 34 | 156 | 12 | 5 | 1326 |
| 79 | 7 | cardinality, cardinality | $x_{0} + x_{1} + x_{2} + x_{3} \leq 4$ (cardinality)<br>$x_{4} + x_{5} + x_{6} \leq 3$ (cardinality) | — | 7 | 37 | 159 | 12 | 5 | 155 |
| 80 | 6 | knapsack, quadratic_knapsack | $3 \cdot x_{0} + 8 \cdot x_{1} + 3 \cdot x_{2} \leq 5$ (knapsack)<br>$5 \cdot x_{3} \cdot x_{3} + 1 \cdot x_{3} \cdot x_{4} + 2 \cdot x_{3} \cdot x_{5} + 5 \cdot x_{4} \cdot x_{4} + 1 \cdot x_{4} \cdot x_{5} + 3 \cdot x_{5} \cdot x_{5} \leq 6$ (quadratic_knapsack) | — | 6 | 111 | 285 | 12 | 6 | 292 |
| 81 | 7 | assignment, cardinality | $x_{0} + x_{1} = 1$ (assignment)<br>$x_{2} + x_{3} + x_{4} + x_{5} + x_{6} = 3$ (cardinality) | — | 7 | 34 | 156 | 7 | 0 | 81 |
| 82 | 8 | cardinality, knapsack | $x_{0} + x_{1} + x_{2} = 0$ (cardinality)<br>$9 \cdot x_{3} + 2 \cdot x_{4} + 1 \cdot x_{5} + 9 \cdot x_{6} + 6 \cdot x_{7} \leq 13$ (knapsack) | — | 8 | 44 | 204 | 12 | 4 | 186 |
| 83 | 7 | quadratic_knapsack, quadratic_knapsack | $5 \cdot x_{0} \cdot x_{0} + 1 \cdot x_{0} \cdot x_{1} + 2 \cdot x_{0} \cdot x_{2} + 5 \cdot x_{1} \cdot x_{1} + 1 \cdot x_{1} \cdot x_{2} + 3 \cdot x_{2} \cdot x_{2} \leq 6$ (quadratic_knapsack)<br>$5 \cdot x_{3} \cdot x_{3} + 3 \cdot x_{3} \cdot x_{4} + 1 \cdot x_{3} \cdot x_{5} + 5 \cdot x_{3} \cdot x_{6} + 3 \cdot x_{4} \cdot x_{4} + 1 \cdot x_{4} \cdot x_{5} + 4 \cdot x_{4} \cdot x_{6} + 5 \cdot x_{5} \cdot x_{5} + 5 \cdot x_{5} \cdot x_{6} + 1 \cdot x_{6} \cdot x_{6} \leq 20$ (quadratic_knapsack) | — | 7 | 383 | 848 | 15 | 8 | 873 |
| 84 | 6 | quadratic_knapsack, independent_set | $4 \cdot x_{0} \cdot x_{0} + 4 \cdot x_{0} \cdot x_{1} + 1 \cdot x_{0} \cdot x_{2} + 5 \cdot x_{0} \cdot x_{3} + 5 \cdot x_{1} \cdot x_{1} + 5 \cdot x_{1} \cdot x_{2} + 5 \cdot x_{1} \cdot x_{3} + 1 \cdot x_{2} \cdot x_{2} + 4 \cdot x_{2} \cdot x_{3} + 1 \cdot x_{3} \cdot x_{3} \leq 11$ (quadratic_knapsack)<br>$x_{4} \cdot x_{5} = 0$ (independent_set) | — | 6 | 221 | 496 | 10 | 4 | 590 |
| 85 | 7 | quadratic_knapsack, knapsack | $5 \cdot x_{0} \cdot x_{0} + 3 \cdot x_{0} \cdot x_{1} + 5 \cdot x_{0} \cdot x_{2} + 4 \cdot x_{1} \cdot x_{1} + 5 \cdot x_{1} \cdot x_{2} + 3 \cdot x_{2} \cdot x_{2} \leq 14$ (quadratic_knapsack)<br>$4 \cdot x_{3} + 6 \cdot x_{4} + 1 \cdot x_{5} + 1 \cdot x_{6} \leq 6$ (knapsack) | — | 7 | 281 | 644 | 14 | 7 | 361 |
| 86 | 7 | cardinality, cardinality | $x_{0} + x_{1} + x_{2} = 1$ (cardinality)<br>$x_{3} + x_{4} + x_{5} + x_{6} \leq 4$ (cardinality) | — | 7 | 29 | 143 | 10 | 3 | 128 |
| 87 | 8 | cardinality, quadratic_knapsack | $x_{0} + x_{1} + x_{2} + x_{3} + x_{4} = 3$ (cardinality)<br>$1 \cdot x_{5} \cdot x_{5} + 5 \cdot x_{5} \cdot x_{6} + 4 \cdot x_{5} \cdot x_{7} + 2 \cdot x_{6} \cdot x_{6} + 2 \cdot x_{6} \cdot x_{7} + 2 \cdot x_{7} \cdot x_{7} \leq 8$ (quadratic_knapsack) | — | 8 | 43 | 193 | 12 | 4 | 330 |
| 88 | 7 | quadratic_knapsack, cardinality | $3 \cdot x_{0} \cdot x_{0} + 1 \cdot x_{0} \cdot x_{1} + 3 \cdot x_{0} \cdot x_{2} + 5 \cdot x_{1} \cdot x_{1} + 2 \cdot x_{1} \cdot x_{2} + 2 \cdot x_{2} \cdot x_{2} \leq 8$ (quadratic_knapsack)<br>$x_{3} + x_{4} + x_{5} + x_{6} = 3$ (cardinality) | — | 7 | 46 | 173 | 11 | 4 | 306 |
| 89 | 8 | quadratic_knapsack, knapsack | $1 \cdot x_{0} \cdot x_{0} + 4 \cdot x_{0} \cdot x_{1} + 1 \cdot x_{0} \cdot x_{2} + 5 \cdot x_{0} \cdot x_{3} + 2 \cdot x_{1} \cdot x_{1} + 3 \cdot x_{1} \cdot x_{2} + 3 \cdot x_{1} \cdot x_{3} + 5 \cdot x_{2} \cdot x_{2} + 4 \cdot x_{2} \cdot x_{3} + 5 \cdot x_{3} \cdot x_{3} \leq 22$ (quadratic_knapsack)<br>$7 \cdot x_{4} + 7 \cdot x_{5} + 10 \cdot x_{6} + 2 \cdot x_{7} \leq 10$ (knapsack) | — | 8 | 292 | 694 | 17 | 9 | 780 |
| 90 | 8 | cardinality, quadratic_knapsack | $x_{0} + x_{1} + x_{2} + x_{3} \leq 1$ (cardinality)<br>$3 \cdot x_{4} \cdot x_{4} + 1 \cdot x_{4} \cdot x_{5} + 3 \cdot x_{4} \cdot x_{6} + 4 \cdot x_{4} \cdot x_{7} + 5 \cdot x_{5} \cdot x_{5} + 4 \cdot x_{5} \cdot x_{6} + 2 \cdot x_{5} \cdot x_{7} + 2 \cdot x_{6} \cdot x_{6} + 4 \cdot x_{6} \cdot x_{7} + 4 \cdot x_{7} \cdot x_{7} \leq 17$ (quadratic_knapsack) | — | 8 | 31 | 168 | 14 | 6 | 715 |
| 91 | 6 | cardinality, cardinality | $x_{0} + x_{1} + x_{2} + x_{3} \geq 4$ (cardinality)<br>$x_{4} + x_{5} \leq 2$ (cardinality) | — | 6 | 10 | 86 | 8 | 2 | 80 |
| 92 | 8 | cardinality, assignment, cardinality | $x_{0} + x_{1} + x_{2} + x_{3} = 2$ (cardinality)<br>$x_{4} + x_{5} = 1$ (assignment)<br>$x_{6} + x_{7} \geq 0$ (cardinality) | — | 8 | 21 | 154 | 10 | 2 | 99 |
| 93 | 7 | cardinality, quadratic_knapsack | $x_{0} + x_{1} + x_{2} \geq 2$ (cardinality)<br>$1 \cdot x_{3} \cdot x_{3} + 4 \cdot x_{3} \cdot x_{4} + 1 \cdot x_{3} \cdot x_{5} + 2 \cdot x_{3} \cdot x_{6} + 5 \cdot x_{4} \cdot x_{4} + 4 \cdot x_{4} \cdot x_{5} + 3 \cdot x_{4} \cdot x_{6} + 5 \cdot x_{5} \cdot x_{5} + 4 \cdot x_{5} \cdot x_{6} + 2 \cdot x_{6} \cdot x_{6} \leq 18$ (quadratic_knapsack) | — | 7 | 375 | 838 | 13 | 6 | 695 |
| 94 | 8 | quadratic_knapsack, knapsack | $4 \cdot x_{0} \cdot x_{0} + 1 \cdot x_{0} \cdot x_{1} + 4 \cdot x_{0} \cdot x_{2} + 3 \cdot x_{0} \cdot x_{3} + 4 \cdot x_{1} \cdot x_{1} + 5 \cdot x_{1} \cdot x_{2} + 4 \cdot x_{1} \cdot x_{3} + 3 \cdot x_{2} \cdot x_{2} + 5 \cdot x_{2} \cdot x_{3} + 5 \cdot x_{3} \cdot x_{3} \leq 24$ (quadratic_knapsack)<br>$4 \cdot x_{4} + 6 \cdot x_{5} + 1 \cdot x_{6} + 1 \cdot x_{7} \leq 6$ (knapsack) | — | 8 | 441 | 992 | 16 | 8 | 756 |
| 95 | 8 | cardinality, knapsack | $x_{0} + x_{1} + x_{2} + x_{3} = 1$ (cardinality)<br>$8 \cdot x_{4} + 10 \cdot x_{5} + 1 \cdot x_{6} + 2 \cdot x_{7} \leq 12$ (knapsack) | — | 8 | 64 | 241 | 12 | 4 | 173 |
| 96 | 7 | cardinality, cardinality | $x_{0} + x_{1} = 0$ (cardinality)<br>$x_{2} + x_{3} + x_{4} + x_{5} + x_{6} \geq 4$ (cardinality) | — | 7 | 0 | 85 | 8 | 1 | 96 |
| 97 | 6 | knapsack, cardinality | $6 \cdot x_{0} + 2 \cdot x_{1} + 2 \cdot x_{2} \leq 3$ (knapsack)<br>$x_{3} + x_{4} + x_{5} \leq 1$ (cardinality) | — | 6 | 31 | 131 | 9 | 3 | 96 |
| 98 | 8 | knapsack, cardinality | $6 \cdot x_{0} + 7 \cdot x_{1} + 8 \cdot x_{2} + 9 \cdot x_{3} + 5 \cdot x_{4} \leq 14$ (knapsack)<br>$x_{5} + x_{6} + x_{7} \geq 2$ (cardinality) | — | 8 | 503 | 1118 | 13 | 5 | 199 |
| 99 | 5 | cardinality, cardinality | $x_{0} + x_{1} + x_{2} = 0$ (cardinality)<br>$x_{3} + x_{4} \geq 1$ (cardinality) | — | 5 | 0 | 47 | 6 | 1 | 47 |
| 100 | 7 | quadratic_knapsack, quadratic_knapsack | $1 \cdot x_{0} \cdot x_{0} + 1 \cdot x_{0} \cdot x_{1} + 3 \cdot x_{0} \cdot x_{2} + 2 \cdot x_{0} \cdot x_{3} + 1 \cdot x_{1} \cdot x_{1} + 2 \cdot x_{1} \cdot x_{2} + 3 \cdot x_{1} \cdot x_{3} + 2 \cdot x_{2} \cdot x_{2} + 3 \cdot x_{2} \cdot x_{3} + 3 \cdot x_{3} \cdot x_{3} \leq 7$ (quadratic_knapsack)<br>$1 \cdot x_{4} \cdot x_{4} + 5 \cdot x_{4} \cdot x_{5} + 1 \cdot x_{4} \cdot x_{6} + 3 \cdot x_{5} \cdot x_{5} + 3 \cdot x_{5} \cdot x_{6} + 5 \cdot x_{6} \cdot x_{6} \leq 10$ (quadratic_knapsack) | — | 7 | 139 | 360 | 14 | 7 | 780 |
| 101 | 8 | quadratic_knapsack, quadratic_knapsack | $1 \cdot x_{0} \cdot x_{0} + 1 \cdot x_{0} \cdot x_{1} + 3 \cdot x_{0} \cdot x_{2} + 2 \cdot x_{0} \cdot x_{3} + 1 \cdot x_{1} \cdot x_{1} + 2 \cdot x_{1} \cdot x_{2} + 3 \cdot x_{1} \cdot x_{3} + 2 \cdot x_{2} \cdot x_{2} + 3 \cdot x_{2} \cdot x_{3} + 3 \cdot x_{3} \cdot x_{3} \leq 7$ (quadratic_knapsack)<br>$4 \cdot x_{4} \cdot x_{4} + 5 \cdot x_{4} \cdot x_{5} + 2 \cdot x_{4} \cdot x_{6} + 5 \cdot x_{4} \cdot x_{7} + 4 \cdot x_{5} \cdot x_{5} + 3 \cdot x_{5} \cdot x_{6} + 5 \cdot x_{5} \cdot x_{7} + 3 \cdot x_{6} \cdot x_{6} + 2 \cdot x_{6} \cdot x_{7} + 3 \cdot x_{7} \cdot x_{7} \leq 22$ (quadratic_knapsack) | — | 8 | 153 | 416 | 16 | 8 | 1175 |
| 102 | 8 | quadratic_knapsack, cardinality | $4 \cdot x_{0} \cdot x_{0} + 3 \cdot x_{0} \cdot x_{1} + 3 \cdot x_{0} \cdot x_{2} + 2 \cdot x_{0} \cdot x_{3} + 2 \cdot x_{0} \cdot x_{4} + 2 \cdot x_{1} \cdot x_{1} + 3 \cdot x_{1} \cdot x_{2} + 2 \cdot x_{1} \cdot x_{3} + 3 \cdot x_{1} \cdot x_{4} + 3 \cdot x_{2} \cdot x_{2} + 5 \cdot x_{2} \cdot x_{3} + 4 \cdot x_{2} \cdot x_{4} + 3 \cdot x_{3} \cdot x_{3} + 5 \cdot x_{3} \cdot x_{4} + 1 \cdot x_{4} \cdot x_{4} \leq 22$ (quadratic_knapsack)<br>$x_{5} + x_{6} + x_{7} \geq 3$ (cardinality) | — | 8 | 35 | 180 | 13 | 5 | 1342 |
| 103 | 6 | knapsack, cardinality | $2 \cdot x_{0} + 5 \cdot x_{1} + 2 \cdot x_{2} \leq 5$ (knapsack)<br>$x_{3} + x_{4} + x_{5} = 2$ (cardinality) | — | 6 | 75 | 209 | 9 | 3 | 98 |
| 104 | 8 | quadratic_knapsack, quadratic_knapsack | $1 \cdot x_{0} \cdot x_{0} + 5 \cdot x_{0} \cdot x_{1} + 4 \cdot x_{0} \cdot x_{2} + 2 \cdot x_{1} \cdot x_{1} + 2 \cdot x_{1} \cdot x_{2} + 2 \cdot x_{2} \cdot x_{2} \leq 8$ (quadratic_knapsack)<br>$4 \cdot x_{3} \cdot x_{3} + 1 \cdot x_{3} \cdot x_{4} + 2 \cdot x_{3} \cdot x_{5} + 4 \cdot x_{3} \cdot x_{6} + 1 \cdot x_{3} \cdot x_{7} + 2 \cdot x_{4} \cdot x_{4} + 3 \cdot x_{4} \cdot x_{5} + 1 \cdot x_{4} \cdot x_{6} + 1 \cdot x_{4} \cdot x_{7} + 3 \cdot x_{5} \cdot x_{5} + 3 \cdot x_{5} \cdot x_{6} + 4 \cdot x_{5} \cdot x_{7} + 1 \cdot x_{6} \cdot x_{6} + 3 \cdot x_{6} \cdot x_{7} + 2 \cdot x_{7} \cdot x_{7} \leq 11$ (quadratic_knapsack) | — | 8 | 191 | 489 | 16 | 8 | 1475 |
| 105 | 8 | cardinality, cardinality, assignment | $x_{0} + x_{1} + x_{2} \leq 2$ (cardinality)<br>$x_{3} + x_{4} \leq 0$ (cardinality)<br>$x_{5} + x_{6} + x_{7} = 1$ (assignment) | — | 8 | 17 | 141 | 10 | 2 | 103 |
| 106 | 6 | quadratic_knapsack, cardinality | $1 \cdot x_{0} \cdot x_{0} + 2 \cdot x_{0} \cdot x_{1} + 1 \cdot x_{0} \cdot x_{2} + 1 \cdot x_{1} \cdot x_{1} + 5 \cdot x_{1} \cdot x_{2} + 1 \cdot x_{2} \cdot x_{2} \leq 6$ (quadratic_knapsack)<br>$x_{3} + x_{4} + x_{5} \geq 3$ (cardinality) | — | 6 | 46 | 155 | 9 | 3 | 247 |
| 107 | 8 | knapsack, flow | $3 \cdot x_{0} + 8 \cdot x_{1} + 3 \cdot x_{2} \leq 5$ (knapsack)<br>$x_{3} + x_{4} + x_{5} - x_{6} - x_{7} = 0$ (flow) | — | 8 | 121 | 355 | 11 | 3 | 138 |
| 108 | 8 | knapsack, flow | $2 \cdot x_{0} + 6 \cdot x_{1} + 1 \cdot x_{2} \leq 3$ (knapsack)<br>$x_{3} + x_{4} - x_{5} - x_{6} - x_{7} = 0$ (flow) | — | 8 | 25 | 162 | 10 | 2 | 119 |
| 109 | 7 | quadratic_knapsack, cardinality | $2 \cdot x_{0} \cdot x_{0} + 5 \cdot x_{0} \cdot x_{1} + 4 \cdot x_{0} \cdot x_{2} + 4 \cdot x_{0} \cdot x_{3} + 3 \cdot x_{0} \cdot x_{4} + 4 \cdot x_{1} \cdot x_{1} + 4 \cdot x_{1} \cdot x_{2} + 1 \cdot x_{1} \cdot x_{3} + 2 \cdot x_{1} \cdot x_{4} + 3 \cdot x_{2} \cdot x_{2} + 1 \cdot x_{2} \cdot x_{3} + 4 \cdot x_{2} \cdot x_{4} + 3 \cdot x_{3} \cdot x_{3} + 4 \cdot x_{3} \cdot x_{4} + 3 \cdot x_{4} \cdot x_{4} \leq 16$ (quadratic_knapsack)<br>$x_{5} + x_{6} \geq 2$ (cardinality) | — | 7 | 1079 | 2240 | 12 | 5 | 1324 |
| 110 | 5 | independent_set, quadratic_knapsack | $x_{0} \cdot x_{1} = 0$ (independent_set)<br>$5 \cdot x_{2} \cdot x_{2} + 1 \cdot x_{2} \cdot x_{3} + 2 \cdot x_{2} \cdot x_{4} + 5 \cdot x_{3} \cdot x_{3} + 1 \cdot x_{3} \cdot x_{4} + 3 \cdot x_{4} \cdot x_{4} \leq 6$ (quadratic_knapsack) | — | 5 | 8 | 66 | 8 | 3 | 230 |
| 111 | 5 | knapsack, assignment | $2 \cdot x_{0} + 6 \cdot x_{1} + 1 \cdot x_{2} \leq 3$ (knapsack)<br>$x_{3} + x_{4} = 1$ (assignment) | — | 5 | 10 | 70 | 7 | 2 | 69 |
| 112 | 6 | knapsack, assignment | $9 \cdot x_{0} + 9 \cdot x_{1} + 4 \cdot x_{2} + 1 \cdot x_{3} \leq 9$ (knapsack)<br>$x_{4} + x_{5} = 1$ (assignment) | — | 6 | 69 | 192 | 10 | 4 | 133 |
| 113 | 8 | cardinality, quadratic_knapsack, cardinality | $x_{0} + x_{1} = 1$ (cardinality)<br>$3 \cdot x_{2} \cdot x_{2} + 1 \cdot x_{2} \cdot x_{3} + 3 \cdot x_{2} \cdot x_{4} + 5 \cdot x_{3} \cdot x_{3} + 2 \cdot x_{3} \cdot x_{4} + 2 \cdot x_{4} \cdot x_{4} \leq 8$ (quadratic_knapsack)<br>$x_{5} + x_{6} + x_{7} \geq 1$ (cardinality) | — | 8 | 26 | 162 | 14 | 6 | 340 |
| 114 | 6 | cardinality, knapsack | $x_{0} + x_{1} + x_{2} = 0$ (cardinality)<br>$10 \cdot x_{3} + 8 \cdot x_{4} + 10 \cdot x_{5} \leq 9$ (knapsack) | — | 6 | 11 | 85 | 10 | 4 | 118 |
| 115 | 7 | knapsack, cardinality | $5 \cdot x_{0} + 4 \cdot x_{1} + 1 \cdot x_{2} \leq 6$ (knapsack)<br>$x_{3} + x_{4} + x_{5} + x_{6} = 0$ (cardinality) | — | 7 | 11 | 104 | 10 | 3 | 114 |
| 116 | 7 | cardinality, cardinality | $x_{0} + x_{1} \leq 2$ (cardinality)<br>$x_{2} + x_{3} + x_{4} + x_{5} + x_{6} = 5$ (cardinality) | — | 7 | 11 | 106 | 9 | 2 | 100 |
| 117 | 8 | cardinality, knapsack | $x_{0} + x_{1} + x_{2} + x_{3} \geq 0$ (cardinality)<br>$8 \cdot x_{4} + 5 \cdot x_{5} + 7 \cdot x_{6} + 4 \cdot x_{7} \leq 11$ (knapsack) | — | 8 | 57 | 224 | 15 | 7 | 219 |
| 118 | 8 | cardinality, cardinality | $x_{0} + x_{1} + x_{2} + x_{3} = 0$ (cardinality)<br>$x_{4} + x_{5} + x_{6} + x_{7} = 1$ (cardinality) | — | 8 | 7 | 127 | 8 | 0 | 91 |
| 119 | 8 | knapsack, knapsack | $3 \cdot x_{0} + 8 \cdot x_{1} + 3 \cdot x_{2} \leq 5$ (knapsack)<br>$1 \cdot x_{3} + 10 \cdot x_{4} + 3 \cdot x_{5} + 5 \cdot x_{6} + 1 \cdot x_{7} \leq 12$ (knapsack) | — | 8 | 174 | 460 | 15 | 7 | 232 |
| 120 | 6 | cardinality, knapsack | $x_{0} + x_{1} + x_{2} = 0$ (cardinality)<br>$5 \cdot x_{3} + 4 \cdot x_{4} + 1 \cdot x_{5} \leq 6$ (knapsack) | — | 6 | 11 | 81 | 9 | 3 | 95 |
| 121 | 8 | knapsack, cardinality | $2 \cdot x_{0} + 2 \cdot x_{1} + 9 \cdot x_{2} + 10 \cdot x_{3} + 4 \cdot x_{4} \leq 18$ (knapsack)<br>$x_{5} + x_{6} + x_{7} = 3$ (cardinality) | — | 8 | 38 | 181 | 13 | 5 | 214 |
| 122 | 7 | cardinality, cardinality | $x_{0} + x_{1} + x_{2} + x_{3} = 4$ (cardinality)<br>$x_{4} + x_{5} + x_{6} \leq 3$ (cardinality) | — | 7 | 17 | 113 | 9 | 2 | 99 |
| 123 | 7 | cardinality, quadratic_knapsack | $x_{0} + x_{1} + x_{2} \geq 1$ (cardinality)<br>$4 \cdot x_{3} \cdot x_{3} + 5 \cdot x_{3} \cdot x_{4} + 2 \cdot x_{3} \cdot x_{5} + 5 \cdot x_{3} \cdot x_{6} + 4 \cdot x_{4} \cdot x_{4} + 3 \cdot x_{4} \cdot x_{5} + 5 \cdot x_{4} \cdot x_{6} + 3 \cdot x_{5} \cdot x_{5} + 2 \cdot x_{5} \cdot x_{6} + 3 \cdot x_{6} \cdot x_{6} \leq 22$ (quadratic_knapsack) | — | 7 | 25 | 131 | 14 | 7 | 707 |
| 124 | 8 | cardinality, knapsack | $x_{0} + x_{1} + x_{2} + x_{3} \geq 2$ (cardinality)<br>$7 \cdot x_{4} + 7 \cdot x_{5} + 10 \cdot x_{6} + 2 \cdot x_{7} \leq 10$ (knapsack) | — | 8 | 269 | 651 | 14 | 6 | 206 |
| 125 | 8 | cardinality, cardinality, cardinality | $x_{0} + x_{1} + x_{2} \leq 1$ (cardinality)<br>$x_{3} + x_{4} + x_{5} \geq 0$ (cardinality)<br>$x_{6} + x_{7} \leq 2$ (cardinality) | — | 8 | 14 | 133 | 13 | 5 | 131 |
| 126 | 8 | cardinality, knapsack | $x_{0} + x_{1} + x_{2} + x_{3} + x_{4} \geq 1$ (cardinality)<br>$2 \cdot x_{5} + 5 \cdot x_{6} + 2 \cdot x_{7} \leq 5$ (knapsack) | — | 8 | 63 | 236 | 14 | 6 | 205 |
| 127 | 8 | quadratic_knapsack, knapsack | $5 \cdot x_{0} \cdot x_{0} + 2 \cdot x_{0} \cdot x_{1} + 2 \cdot x_{0} \cdot x_{2} + 3 \cdot x_{0} \cdot x_{3} + 1 \cdot x_{0} \cdot x_{4} + 1 \cdot x_{1} \cdot x_{1} + 4 \cdot x_{1} \cdot x_{2} + 4 \cdot x_{1} \cdot x_{3} + 3 \cdot x_{1} \cdot x_{4} + 1 \cdot x_{2} \cdot x_{2} + 3 \cdot x_{2} \cdot x_{3} + 1 \cdot x_{2} \cdot x_{4} + 5 \cdot x_{3} \cdot x_{3} + 1 \cdot x_{3} \cdot x_{4} + 5 \cdot x_{4} \cdot x_{4} \leq 23$ (quadratic_knapsack)<br>$7 \cdot x_{5} + 10 \cdot x_{6} + 9 \cdot x_{7} \leq 8$ (knapsack) | — | 8 | 43 | 193 | 17 | 9 | 1407 |
| 128 | 8 | quadratic_knapsack, cardinality | $4 \cdot x_{0} \cdot x_{0} + 4 \cdot x_{0} \cdot x_{1} + 2 \cdot x_{0} \cdot x_{2} + 5 \cdot x_{1} \cdot x_{1} + 1 \cdot x_{1} \cdot x_{2} + 2 \cdot x_{2} \cdot x_{2} \leq 9$ (quadratic_knapsack)<br>$x_{3} + x_{4} + x_{5} + x_{6} + x_{7} \leq 5$ (cardinality) | — | 8 | 56 | 228 | 15 | 7 | 396 |
| 129 | 8 | cardinality, independent_set, quadratic_knapsack | $x_{0} + x_{1} + x_{2} = 3$ (cardinality)<br>$x_{3} \cdot x_{4} = 0$ (independent_set)<br>$2 \cdot x_{5} \cdot x_{5} + 5 \cdot x_{5} \cdot x_{6} + 2 \cdot x_{5} \cdot x_{7} + 2 \cdot x_{6} \cdot x_{6} + 1 \cdot x_{6} \cdot x_{7} + 1 \cdot x_{7} \cdot x_{7} \leq 7$ (quadratic_knapsack) | — | 8 | 54 | 218 | 11 | 3 | 267 |
| 130 | 5 | assignment, cardinality | $x_{0} + x_{1} = 1$ (assignment)<br>$x_{2} + x_{3} + x_{4} \leq 2$ (cardinality) | — | 5 | 15 | 68 | 7 | 2 | 65 |
| 131 | 7 | quadratic_knapsack, cardinality | $1 \cdot x_{0} \cdot x_{0} + 5 \cdot x_{0} \cdot x_{1} + 4 \cdot x_{0} \cdot x_{2} + 2 \cdot x_{1} \cdot x_{1} + 2 \cdot x_{1} \cdot x_{2} + 2 \cdot x_{2} \cdot x_{2} \leq 8$ (quadratic_knapsack)<br>$x_{3} + x_{4} + x_{5} + x_{6} \leq 1$ (cardinality) | — | 7 | 24 | 136 | 12 | 5 | 324 |
| 132 | 6 | cardinality, quadratic_knapsack | $x_{0} + x_{1} \geq 0$ (cardinality)<br>$5 \cdot x_{2} \cdot x_{2} + 1 \cdot x_{2} \cdot x_{3} + 3 \cdot x_{2} \cdot x_{4} + 1 \cdot x_{2} \cdot x_{5} + 4 \cdot x_{3} \cdot x_{3} + 1 \cdot x_{3} \cdot x_{4} + 5 \cdot x_{3} \cdot x_{5} + 4 \cdot x_{4} \cdot x_{4} + 2 \cdot x_{4} \cdot x_{5} + 3 \cdot x_{5} \cdot x_{5} \leq 19$ (quadratic_knapsack) | — | 6 | 23 | 105 | 13 | 7 | 681 |
| 133 | 7 | cardinality, quadratic_knapsack | $x_{0} + x_{1} + x_{2} \leq 3$ (cardinality)<br>$5 \cdot x_{3} \cdot x_{3} + 1 \cdot x_{3} \cdot x_{4} + 3 \cdot x_{3} \cdot x_{5} + 1 \cdot x_{3} \cdot x_{6} + 4 \cdot x_{4} \cdot x_{4} + 1 \cdot x_{4} \cdot x_{5} + 5 \cdot x_{4} \cdot x_{6} + 4 \cdot x_{5} \cdot x_{5} + 2 \cdot x_{5} \cdot x_{6} + 3 \cdot x_{6} \cdot x_{6} \leq 19$ (quadratic_knapsack) | — | 7 | 36 | 160 | 14 | 7 | 710 |
| 134 | 7 | cardinality, quadratic_knapsack | $x_{0} + x_{1} \geq 2$ (cardinality)<br>$5 \cdot x_{2} \cdot x_{2} + 2 \cdot x_{2} \cdot x_{3} + 2 \cdot x_{2} \cdot x_{4} + 3 \cdot x_{2} \cdot x_{5} + 1 \cdot x_{2} \cdot x_{6} + 1 \cdot x_{3} \cdot x_{3} + 4 \cdot x_{3} \cdot x_{4} + 4 \cdot x_{3} \cdot x_{5} + 3 \cdot x_{3} \cdot x_{6} + 1 \cdot x_{4} \cdot x_{4} + 3 \cdot x_{4} \cdot x_{5} + 1 \cdot x_{4} \cdot x_{6} + 5 \cdot x_{5} \cdot x_{5} + 1 \cdot x_{5} \cdot x_{6} + 5 \cdot x_{6} \cdot x_{6} \leq 23$ (quadratic_knapsack) | — | 7 | 34 | 147 | 12 | 5 | 1323 |
| 135 | 7 | quadratic_knapsack, flow | $5 \cdot x_{0} \cdot x_{0} + 1 \cdot x_{0} \cdot x_{1} + 3 \cdot x_{0} \cdot x_{2} + 1 \cdot x_{0} \cdot x_{3} + 4 \cdot x_{1} \cdot x_{1} + 1 \cdot x_{1} \cdot x_{2} + 5 \cdot x_{1} \cdot x_{3} + 4 \cdot x_{2} \cdot x_{2} + 2 \cdot x_{2} \cdot x_{3} + 3 \cdot x_{3} \cdot x_{3} \leq 19$ (quadratic_knapsack)<br>$x_{4} - x_{5} - x_{6} = 0$ (flow) | — | 7 | 29 | 146 | 12 | 5 | 680 |
| 136 | 7 | cardinality, knapsack | $x_{0} + x_{1} + x_{2} \leq 3$ (cardinality)<br>$1 \cdot x_{3} + 3 \cdot x_{4} + 9 \cdot x_{5} + 8 \cdot x_{6} \leq 10$ (knapsack) | — | 7 | 113 | 308 | 13 | 6 | 178 |
| 137 | 6 | cardinality, quadratic_knapsack | $x_{0} + x_{1} + x_{2} \leq 1$ (cardinality)<br>$1 \cdot x_{3} \cdot x_{3} + 5 \cdot x_{3} \cdot x_{4} + 4 \cdot x_{3} \cdot x_{5} + 2 \cdot x_{4} \cdot x_{4} + 2 \cdot x_{4} \cdot x_{5} + 2 \cdot x_{5} \cdot x_{5} \leq 8$ (quadratic_knapsack) | — | 6 | 20 | 106 | 11 | 5 | 302 |
| 138 | 7 | cardinality, cardinality | $x_{0} + x_{1} = 1$ (cardinality)<br>$x_{2} + x_{3} + x_{4} + x_{5} + x_{6} \leq 3$ (cardinality) | — | 7 | 35 | 151 | 9 | 2 | 117 |
| 139 | 7 | cardinality, quadratic_knapsack | $x_{0} + x_{1} + x_{2} \leq 3$ (cardinality)<br>$3 \cdot x_{3} \cdot x_{3} + 1 \cdot x_{3} \cdot x_{4} + 3 \cdot x_{3} \cdot x_{5} + 4 \cdot x_{3} \cdot x_{6} + 5 \cdot x_{4} \cdot x_{4} + 4 \cdot x_{4} \cdot x_{5} + 2 \cdot x_{4} \cdot x_{6} + 2 \cdot x_{5} \cdot x_{5} + 4 \cdot x_{5} \cdot x_{6} + 4 \cdot x_{6} \cdot x_{6} \leq 17$ (quadratic_knapsack) | — | 7 | 32 | 149 | 14 | 7 | 709 |
| 140 | 6 | quadratic_knapsack, cardinality | $1 \cdot x_{0} \cdot x_{0} + 4 \cdot x_{0} \cdot x_{1} + 1 \cdot x_{0} \cdot x_{2} + 5 \cdot x_{0} \cdot x_{3} + 2 \cdot x_{1} \cdot x_{1} + 3 \cdot x_{1} \cdot x_{2} + 3 \cdot x_{1} \cdot x_{3} + 5 \cdot x_{2} \cdot x_{2} + 4 \cdot x_{2} \cdot x_{3} + 5 \cdot x_{3} \cdot x_{3} \leq 22$ (quadratic_knapsack)<br>$x_{4} + x_{5} \geq 2$ (cardinality) | — | 6 | 25 | 113 | 11 | 5 | 666 |
| 141 | 7 | knapsack, knapsack | $2 \cdot x_{0} + 10 \cdot x_{1} + 4 \cdot x_{2} \leq 6$ (knapsack)<br>$9 \cdot x_{3} + 9 \cdot x_{4} + 4 \cdot x_{5} + 1 \cdot x_{6} \leq 9$ (knapsack) | — | 7 | 73 | 227 | 14 | 7 | 195 |
| 142 | 7 | quadratic_knapsack, quadratic_knapsack | $5 \cdot x_{0} \cdot x_{0} + 3 \cdot x_{0} \cdot x_{1} + 5 \cdot x_{0} \cdot x_{2} + 4 \cdot x_{1} \cdot x_{1} + 5 \cdot x_{1} \cdot x_{2} + 3 \cdot x_{2} \cdot x_{2} \leq 14$ (quadratic_knapsack)<br>$5 \cdot x_{3} \cdot x_{3} + 1 \cdot x_{3} \cdot x_{4} + 3 \cdot x_{3} \cdot x_{5} + 1 \cdot x_{3} \cdot x_{6} + 4 \cdot x_{4} \cdot x_{4} + 1 \cdot x_{4} \cdot x_{5} + 5 \cdot x_{4} \cdot x_{6} + 4 \cdot x_{5} \cdot x_{5} + 2 \cdot x_{5} \cdot x_{6} + 3 \cdot x_{6} \cdot x_{6} \leq 19$ (quadratic_knapsack) | — | 7 | 35 | 149 | 16 | 9 | 914 |
| 143 | 5 | cardinality, knapsack | $x_{0} + x_{1} \geq 1$ (cardinality)<br>$10 \cdot x_{2} + 1 \cdot x_{3} + 9 \cdot x_{4} \leq 7$ (knapsack) | — | 5 | 11 | 71 | 9 | 4 | 95 |
| 144 | 6 | knapsack, knapsack | $2 \cdot x_{0} + 7 \cdot x_{1} + 7 \cdot x_{2} \leq 7$ (knapsack)<br>$2 \cdot x_{3} + 10 \cdot x_{4} + 4 \cdot x_{5} \leq 6$ (knapsack) | — | 6 | 15 | 91 | 12 | 6 | 143 |
| 145 | 6 | cardinality, knapsack | $x_{0} + x_{1} + x_{2} = 2$ (cardinality)<br>$2 \cdot x_{3} + 7 \cdot x_{4} + 7 \cdot x_{5} \leq 7$ (knapsack) | — | 6 | 20 | 106 | 9 | 3 | 101 |
| 146 | 7 | assignment, cardinality | $x_{0} + x_{1} = 1$ (assignment)<br>$x_{2} + x_{3} + x_{4} + x_{5} + x_{6} = 0$ (cardinality) | — | 7 | 3 | 94 | 7 | 0 | 76 |
| 147 | 6 | cardinality, quadratic_knapsack | $x_{0} + x_{1} + x_{2} \geq 2$ (cardinality)<br>$1 \cdot x_{3} \cdot x_{3} + 5 \cdot x_{3} \cdot x_{4} + 4 \cdot x_{3} \cdot x_{5} + 2 \cdot x_{4} \cdot x_{4} + 2 \cdot x_{4} \cdot x_{5} + 2 \cdot x_{5} \cdot x_{5} \leq 8$ (quadratic_knapsack) | — | 6 | 12 | 87 | 11 | 5 | 301 |
| 148 | 8 | quadratic_knapsack, cardinality | $2 \cdot x_{0} \cdot x_{0} + 4 \cdot x_{0} \cdot x_{1} + 4 \cdot x_{0} \cdot x_{2} + 4 \cdot x_{0} \cdot x_{3} + 5 \cdot x_{0} \cdot x_{4} + 1 \cdot x_{1} \cdot x_{1} + 2 \cdot x_{1} \cdot x_{2} + 2 \cdot x_{1} \cdot x_{3} + 5 \cdot x_{1} \cdot x_{4} + 1 \cdot x_{2} \cdot x_{2} + 3 \cdot x_{2} \cdot x_{3} + 1 \cdot x_{2} \cdot x_{4} + 5 \cdot x_{3} \cdot x_{3} + 4 \cdot x_{3} \cdot x_{4} + 1 \cdot x_{4} \cdot x_{4} \leq 28$ (quadratic_knapsack)<br>$x_{5} + x_{6} + x_{7} \leq 0$ (cardinality) | — | 8 | 809 | 1728 | 13 | 5 | 1339 |
| 149 | 7 | quadratic_knapsack, knapsack | $1 \cdot x_{0} \cdot x_{0} + 4 \cdot x_{0} \cdot x_{1} + 1 \cdot x_{0} \cdot x_{2} + 2 \cdot x_{0} \cdot x_{3} + 5 \cdot x_{1} \cdot x_{1} + 4 \cdot x_{1} \cdot x_{2} + 3 \cdot x_{1} \cdot x_{3} + 5 \cdot x_{2} \cdot x_{2} + 4 \cdot x_{2} \cdot x_{3} + 2 \cdot x_{3} \cdot x_{3} \leq 18$ (quadratic_knapsack)<br>$6 \cdot x_{4} + 2 \cdot x_{5} + 2 \cdot x_{6} \leq 3$ (knapsack) | — | 7 | 398 | 875 | 14 | 7 | 707 |
| 150 | 7 | knapsack, quadratic_knapsack | $4 \cdot x_{0} + 2 \cdot x_{1} + 10 \cdot x_{2} + 8 \cdot x_{3} \leq 13$ (knapsack)<br>$1 \cdot x_{4} \cdot x_{4} + 5 \cdot x_{4} \cdot x_{5} + 4 \cdot x_{4} \cdot x_{6} + 2 \cdot x_{5} \cdot x_{5} + 2 \cdot x_{5} \cdot x_{6} + 2 \cdot x_{6} \cdot x_{6} \leq 8$ (quadratic_knapsack) | — | 7 | 69 | 223 | 15 | 8 | 386 |
| 151 | 7 | cardinality, quadratic_knapsack | $x_{0} + x_{1} + x_{2} \geq 1$ (cardinality)<br>$3 \cdot x_{3} \cdot x_{3} + 1 \cdot x_{3} \cdot x_{4} + 3 \cdot x_{3} \cdot x_{5} + 4 \cdot x_{3} \cdot x_{6} + 5 \cdot x_{4} \cdot x_{4} + 4 \cdot x_{4} \cdot x_{5} + 2 \cdot x_{4} \cdot x_{6} + 2 \cdot x_{5} \cdot x_{5} + 4 \cdot x_{5} \cdot x_{6} + 4 \cdot x_{6} \cdot x_{6} \leq 17$ (quadratic_knapsack) | — | 7 | 19 | 122 | 14 | 7 | 708 |
| 152 | 8 | quadratic_knapsack, knapsack | $1 \cdot x_{0} \cdot x_{0} + 4 \cdot x_{0} \cdot x_{1} + 1 \cdot x_{0} \cdot x_{2} + 5 \cdot x_{0} \cdot x_{3} + 2 \cdot x_{1} \cdot x_{1} + 3 \cdot x_{1} \cdot x_{2} + 3 \cdot x_{1} \cdot x_{3} + 5 \cdot x_{2} \cdot x_{2} + 4 \cdot x_{2} \cdot x_{3} + 5 \cdot x_{3} \cdot x_{3} \leq 22$ (quadratic_knapsack)<br>$8 \cdot x_{4} + 5 \cdot x_{5} + 7 \cdot x_{6} + 4 \cdot x_{7} \leq 11$ (knapsack) | — | 8 | 80 | 276 | 17 | 9 | 782 |
| 153 | 7 | quadratic_knapsack, cardinality | $5 \cdot x_{0} \cdot x_{0} + 3 \cdot x_{0} \cdot x_{1} + 4 \cdot x_{0} \cdot x_{2} + 5 \cdot x_{0} \cdot x_{3} + 1 \cdot x_{0} \cdot x_{4} + 1 \cdot x_{1} \cdot x_{1} + 4 \cdot x_{1} \cdot x_{2} + 2 \cdot x_{1} \cdot x_{3} + 3 \cdot x_{1} \cdot x_{4} + 4 \cdot x_{2} \cdot x_{2} + 5 \cdot x_{2} \cdot x_{3} + 1 \cdot x_{2} \cdot x_{4} + 1 \cdot x_{3} \cdot x_{3} + 5 \cdot x_{3} \cdot x_{4} + 2 \cdot x_{4} \cdot x_{4} \leq 17$ (quadratic_knapsack)<br>$x_{5} + x_{6} \geq 0$ (cardinality) | — | 7 | 145 | 373 | 14 | 7 | 1340 |
| 154 | 6 | quadratic_knapsack, cardinality | $1 \cdot x_{0} \cdot x_{0} + 5 \cdot x_{0} \cdot x_{1} + 1 \cdot x_{0} \cdot x_{2} + 3 \cdot x_{1} \cdot x_{1} + 3 \cdot x_{1} \cdot x_{2} + 5 \cdot x_{2} \cdot x_{2} \leq 10$ (quadratic_knapsack)<br>$x_{3} + x_{4} + x_{5} = 1$ (cardinality) | — | 6 | 16 | 95 | 10 | 4 | 289 |
| 155 | 8 | cardinality, knapsack | $x_{0} + x_{1} + x_{2} + x_{3} + x_{4} \leq 2$ (cardinality)<br>$2 \cdot x_{5} + 7 \cdot x_{6} + 7 \cdot x_{7} \leq 7$ (knapsack) | — | 8 | 33 | 179 | 13 | 5 | 182 |
| 156 | 6 | cardinality, cardinality | $x_{0} + x_{1} = 0$ (cardinality)<br>$x_{2} + x_{3} + x_{4} + x_{5} \leq 0$ (cardinality) | — | 6 | 0 | 63 | 6 | 0 | 52 |
| 157 | 7 | quadratic_knapsack, cardinality | $5 \cdot x_{0} \cdot x_{0} + 3 \cdot x_{0} \cdot x_{1} + 1 \cdot x_{0} \cdot x_{2} + 5 \cdot x_{0} \cdot x_{3} + 3 \cdot x_{1} \cdot x_{1} + 1 \cdot x_{1} \cdot x_{2} + 4 \cdot x_{1} \cdot x_{3} + 5 \cdot x_{2} \cdot x_{2} + 5 \cdot x_{2} \cdot x_{3} + 1 \cdot x_{3} \cdot x_{3} \leq 20$ (quadratic_knapsack)<br>$x_{4} + x_{5} + x_{6} \geq 0$ (cardinality) | — | 7 | 375 | 832 | 14 | 7 | 703 |
| 158 | 8 | knapsack, knapsack | $9 \cdot x_{0} + 9 \cdot x_{1} + 4 \cdot x_{2} + 1 \cdot x_{3} \leq 9$ (knapsack)<br>$3 \cdot x_{4} + 1 \cdot x_{5} + 4 \cdot x_{6} + 8 \cdot x_{7} \leq 7$ (knapsack) | — | 8 | 123 | 356 | 15 | 7 | 226 |
| 159 | 5 | knapsack, cardinality | $2 \cdot x_{0} + 7 \cdot x_{1} + 7 \cdot x_{2} \leq 7$ (knapsack)<br>$x_{3} + x_{4} = 2$ (cardinality) | — | 5 | 10 | 58 | 8 | 3 | 83 |
| 160 | 7 | assignment, cardinality | $x_{0} + x_{1} = 1$ (assignment)<br>$x_{2} + x_{3} + x_{4} + x_{5} + x_{6} = 5$ (cardinality) | — | 7 | 8 | 99 | 7 | 0 | 78 |
| 161 | 7 | cardinality, knapsack | $x_{0} + x_{1} + x_{2} + x_{3} \geq 0$ (cardinality)<br>$2 \cdot x_{4} + 7 \cdot x_{5} + 7 \cdot x_{6} \leq 7$ (knapsack) | — | 7 | 8 | 104 | 13 | 6 | 167 |
| 162 | 8 | knapsack, knapsack | $8 \cdot x_{0} + 5 \cdot x_{1} + 7 \cdot x_{2} + 4 \cdot x_{3} \leq 11$ (knapsack)<br>$4 \cdot x_{4} + 6 \cdot x_{5} + 2 \cdot x_{6} + 10 \cdot x_{7} \leq 7$ (knapsack) | — | 8 | 114 | 344 | 15 | 7 | 228 |
| 163 | 7 | cardinality, cardinality | $x_{0} + x_{1} + x_{2} + x_{3} \leq 4$ (cardinality)<br>$x_{4} + x_{5} + x_{6} \leq 0$ (cardinality) | — | 7 | 24 | 131 | 10 | 3 | 123 |
| 164 | 8 | cardinality, knapsack, cardinality | $x_{0} + x_{1} + x_{2} \geq 2$ (cardinality)<br>$5 \cdot x_{3} + 4 \cdot x_{4} + 1 \cdot x_{5} \leq 6$ (knapsack)<br>$x_{6} + x_{7} \geq 2$ (cardinality) | — | 8 | 13 | 132 | 12 | 4 | 134 |
| 165 | 8 | cardinality, knapsack | $x_{0} + x_{1} + x_{2} + x_{3} + x_{4} = 3$ (cardinality)<br>$3 \cdot x_{5} + 8 \cdot x_{6} + 3 \cdot x_{7} \leq 5$ (knapsack) | — | 8 | 134 | 374 | 11 | 3 | 140 |
| 166 | 7 | cardinality, cardinality | $x_{0} + x_{1} \geq 1$ (cardinality)<br>$x_{2} + x_{3} + x_{4} + x_{5} + x_{6} \leq 5$ (cardinality) | — | 7 | 37 | 156 | 11 | 4 | 151 |
| 167 | 7 | knapsack, flow | $2 \cdot x_{0} + 10 \cdot x_{1} + 4 \cdot x_{2} \leq 6$ (knapsack)<br>$x_{3} - x_{4} - x_{5} - x_{6} = 0$ (flow) | — | 7 | 16 | 120 | 10 | 3 | 116 |
| 168 | 6 | knapsack, cardinality | $2 \cdot x_{0} + 10 \cdot x_{1} + 4 \cdot x_{2} \leq 6$ (knapsack)<br>$x_{3} + x_{4} + x_{5} \leq 0$ (cardinality) | — | 6 | 7 | 73 | 9 | 3 | 95 |
| 169 | 6 | cardinality, cardinality | $x_{0} + x_{1} \leq 2$ (cardinality)<br>$x_{2} + x_{3} + x_{4} + x_{5} \geq 0$ (cardinality) | — | 6 | 6 | 75 | 11 | 5 | 126 |
| 170 | 8 | cardinality, cardinality | $x_{0} + x_{1} + x_{2} + x_{3} + x_{4} \leq 0$ (cardinality)<br>$x_{5} + x_{6} + x_{7} = 1$ (cardinality) | — | 8 | 5 | 117 | 8 | 0 | 91 |
| 171 | 8 | flow, cardinality | $x_{0} + x_{1} - x_{2} = 0$ (flow)<br>$x_{3} + x_{4} + x_{5} + x_{6} + x_{7} \leq 2$ (cardinality) | — | 8 | 31 | 174 | 10 | 2 | 133 |
| 172 | 6 | quadratic_knapsack, quadratic_knapsack | $1 \cdot x_{0} \cdot x_{0} + 5 \cdot x_{0} \cdot x_{1} + 3 \cdot x_{0} \cdot x_{2} + 4 \cdot x_{1} \cdot x_{1} + 2 \cdot x_{1} \cdot x_{2} + 4 \cdot x_{2} \cdot x_{2} \leq 7$ (quadratic_knapsack)<br>$5 \cdot x_{3} \cdot x_{3} + 1 \cdot x_{3} \cdot x_{4} + 2 \cdot x_{3} \cdot x_{5} + 5 \cdot x_{4} \cdot x_{4} + 1 \cdot x_{4} \cdot x_{5} + 3 \cdot x_{5} \cdot x_{5} \leq 6$ (quadratic_knapsack) | — | 6 | 16 | 101 | 12 | 6 | 441 |
| 173 | 8 | knapsack, quadratic_knapsack | $6 \cdot x_{0} + 7 \cdot x_{1} + 8 \cdot x_{2} + 9 \cdot x_{3} + 5 \cdot x_{4} \leq 14$ (knapsack)<br>$1 \cdot x_{5} \cdot x_{5} + 5 \cdot x_{5} \cdot x_{6} + 3 \cdot x_{5} \cdot x_{7} + 4 \cdot x_{6} \cdot x_{6} + 2 \cdot x_{6} \cdot x_{7} + 4 \cdot x_{7} \cdot x_{7} \leq 7$ (quadratic_knapsack) | — | 8 | 511 | 1132 | 15 | 7 | 379 |
| 174 | 8 | cardinality, cardinality | $x_{0} + x_{1} + x_{2} = 0$ (cardinality)<br>$x_{3} + x_{4} + x_{5} + x_{6} + x_{7} = 1$ (cardinality) | — | 8 | 9 | 131 | 8 | 0 | 95 |
| 175 | 8 | quadratic_knapsack, cardinality | $1 \cdot x_{0} \cdot x_{0} + 5 \cdot x_{0} \cdot x_{1} + 4 \cdot x_{0} \cdot x_{2} + 2 \cdot x_{1} \cdot x_{1} + 2 \cdot x_{1} \cdot x_{2} + 2 \cdot x_{2} \cdot x_{2} \leq 8$ (quadratic_knapsack)<br>$x_{3} + x_{4} + x_{5} + x_{6} + x_{7} \leq 1$ (cardinality) | — | 8 | 27 | 161 | 13 | 5 | 348 |
| 176 | 7 | cardinality, cardinality | $x_{0} + x_{1} + x_{2} \geq 3$ (cardinality)<br>$x_{3} + x_{4} + x_{5} + x_{6} = 2$ (cardinality) | — | 7 | 21 | 130 | 7 | 0 | 75 |
| 177 | 6 | quadratic_knapsack, knapsack | $3 \cdot x_{0} \cdot x_{0} + 1 \cdot x_{0} \cdot x_{1} + 3 \cdot x_{0} \cdot x_{2} + 5 \cdot x_{1} \cdot x_{1} + 2 \cdot x_{1} \cdot x_{2} + 2 \cdot x_{2} \cdot x_{2} \leq 8$ (quadratic_knapsack)<br>$5 \cdot x_{3} + 4 \cdot x_{4} + 1 \cdot x_{5} \leq 6$ (knapsack) | — | 6 | 34 | 134 | 13 | 7 | 335 |
| 178 | 6 | assignment, knapsack | $x_{0} + x_{1} + x_{2} = 1$ (assignment)<br>$5 \cdot x_{3} + 4 \cdot x_{4} + 1 \cdot x_{5} \leq 6$ (knapsack) | — | 6 | 16 | 86 | 9 | 3 | 97 |
| 179 | 8 | knapsack, knapsack | $6 \cdot x_{0} + 7 \cdot x_{1} + 8 \cdot x_{2} + 9 \cdot x_{3} + 5 \cdot x_{4} \leq 14$ (knapsack)<br>$10 \cdot x_{5} + 1 \cdot x_{6} + 9 \cdot x_{7} \leq 7$ (knapsack) | — | 8 | 514 | 1138 | 15 | 7 | 232 |
| 180 | 8 | quadratic_knapsack, quadratic_knapsack | $3 \cdot x_{0} \cdot x_{0} + 1 \cdot x_{0} \cdot x_{1} + 3 \cdot x_{0} \cdot x_{2} + 1 \cdot x_{1} \cdot x_{1} + 4 \cdot x_{1} \cdot x_{2} + 3 \cdot x_{2} \cdot x_{2} \leq 5$ (quadratic_knapsack)<br>$2 \cdot x_{3} \cdot x_{3} + 4 \cdot x_{3} \cdot x_{4} + 4 \cdot x_{3} \cdot x_{5} + 4 \cdot x_{3} \cdot x_{6} + 5 \cdot x_{3} \cdot x_{7} + 1 \cdot x_{4} \cdot x_{4} + 2 \cdot x_{4} \cdot x_{5} + 2 \cdot x_{4} \cdot x_{6} + 5 \cdot x_{4} \cdot x_{7} + 1 \cdot x_{5} \cdot x_{5} + 3 \cdot x_{5} \cdot x_{6} + 1 \cdot x_{5} \cdot x_{7} + 5 \cdot x_{6} \cdot x_{6} + 4 \cdot x_{6} \cdot x_{7} + 1 \cdot x_{7} \cdot x_{7} \leq 28$ (quadratic_knapsack) | — | 8 | 892 | 1891 | 16 | 8 | 1533 |
| 181 | 8 | cardinality, cardinality | $x_{0} + x_{1} + x_{2} + x_{3} + x_{4} = 0$ (cardinality)<br>$x_{5} + x_{6} + x_{7} = 1$ (cardinality) | — | 8 | 5 | 120 | 8 | 0 | 92 |
| 182 | 8 | quadratic_knapsack, knapsack | $5 \cdot x_{0} \cdot x_{0} + 2 \cdot x_{0} \cdot x_{1} + 2 \cdot x_{0} \cdot x_{2} + 3 \cdot x_{0} \cdot x_{3} + 1 \cdot x_{0} \cdot x_{4} + 1 \cdot x_{1} \cdot x_{1} + 4 \cdot x_{1} \cdot x_{2} + 4 \cdot x_{1} \cdot x_{3} + 3 \cdot x_{1} \cdot x_{4} + 1 \cdot x_{2} \cdot x_{2} + 3 \cdot x_{2} \cdot x_{3} + 1 \cdot x_{2} \cdot x_{4} + 5 \cdot x_{3} \cdot x_{3} + 1 \cdot x_{3} \cdot x_{4} + 5 \cdot x_{4} \cdot x_{4} \leq 23$ (quadratic_knapsack)<br>$2 \cdot x_{5} + 7 \cdot x_{6} + 7 \cdot x_{7} \leq 7$ (knapsack) | — | 8 | 40 | 193 | 16 | 8 | 1388 |
| 183 | 8 | quadratic_knapsack, knapsack | $3 \cdot x_{0} \cdot x_{0} + 1 \cdot x_{0} \cdot x_{1} + 5 \cdot x_{0} \cdot x_{2} + 2 \cdot x_{0} \cdot x_{3} + 5 \cdot x_{0} \cdot x_{4} + 2 \cdot x_{1} \cdot x_{1} + 5 \cdot x_{1} \cdot x_{2} + 5 \cdot x_{1} \cdot x_{3} + 5 \cdot x_{1} \cdot x_{4} + 2 \cdot x_{2} \cdot x_{2} + 3 \cdot x_{2} \cdot x_{3} + 3 \cdot x_{2} \cdot x_{4} + 4 \cdot x_{3} \cdot x_{3} + 1 \cdot x_{3} \cdot x_{4} + 5 \cdot x_{4} \cdot x_{4} \leq 18$ (quadratic_knapsack)<br>$2 \cdot x_{5} + 10 \cdot x_{6} + 4 \cdot x_{7} \leq 6$ (knapsack) | — | 8 | 414 | 938 | 16 | 8 | 1387 |
| 184 | 7 | cardinality, cardinality | $x_{0} + x_{1} + x_{2} + x_{3} + x_{4} \geq 5$ (cardinality)<br>$x_{5} + x_{6} \geq 0$ (cardinality) | — | 7 | 5 | 95 | 9 | 2 | 97 |
| 185 | 8 | quadratic_knapsack, quadratic_knapsack | $2 \cdot x_{0} \cdot x_{0} + 4 \cdot x_{0} \cdot x_{1} + 5 \cdot x_{0} \cdot x_{2} + 1 \cdot x_{0} \cdot x_{3} + 4 \cdot x_{0} \cdot x_{4} + 2 \cdot x_{1} \cdot x_{1} + 4 \cdot x_{1} \cdot x_{2} + 4 \cdot x_{1} \cdot x_{3} + 1 \cdot x_{1} \cdot x_{4} + 2 \cdot x_{2} \cdot x_{2} + 3 \cdot x_{2} \cdot x_{3} + 3 \cdot x_{2} \cdot x_{4} + 3 \cdot x_{3} \cdot x_{3} + 2 \cdot x_{3} \cdot x_{4} + 1 \cdot x_{4} \cdot x_{4} \leq 19$ (quadratic_knapsack)<br>$3 \cdot x_{5} \cdot x_{5} + 1 \cdot x_{5} \cdot x_{6} + 3 \cdot x_{5} \cdot x_{7} + 1 \cdot x_{6} \cdot x_{6} + 4 \cdot x_{6} \cdot x_{7} + 3 \cdot x_{7} \cdot x_{7} \leq 5$ (quadratic_knapsack) | — | 8 | 115 | 340 | 16 | 8 | 1534 |
| 186 | 5 | cardinality, cardinality | $x_{0} + x_{1} + x_{2} \geq 0$ (cardinality)<br>$x_{3} + x_{4} \leq 1$ (cardinality) | — | 5 | 5 | 60 | 8 | 3 | 73 |
| 187 | 6 | cardinality, cardinality | $x_{0} + x_{1} = 2$ (cardinality)<br>$x_{2} + x_{3} + x_{4} + x_{5} \geq 1$ (cardinality) | — | 6 | 2 | 65 | 8 | 2 | 89 |
| 188 | 8 | knapsack, flow | $8 \cdot x_{0} + 10 \cdot x_{1} + 1 \cdot x_{2} + 2 \cdot x_{3} \leq 12$ (knapsack)<br>$x_{4} + x_{5} + x_{6} - x_{7} = 0$ (flow) | — | 8 | 66 | 239 | 12 | 4 | 167 |
| 189 | 8 | cardinality, cardinality | $x_{0} + x_{1} + x_{2} + x_{3} + x_{4} = 2$ (cardinality)<br>$x_{5} + x_{6} + x_{7} \leq 2$ (cardinality) | — | 8 | 36 | 177 | 10 | 2 | 121 |
| 190 | 6 | quadratic_knapsack, cardinality | $3 \cdot x_{0} \cdot x_{0} + 1 \cdot x_{0} \cdot x_{1} + 3 \cdot x_{0} \cdot x_{2} + 1 \cdot x_{1} \cdot x_{1} + 4 \cdot x_{1} \cdot x_{2} + 3 \cdot x_{2} \cdot x_{2} \leq 5$ (quadratic_knapsack)<br>$x_{3} + x_{4} + x_{5} \geq 1$ (cardinality) | — | 6 | 83 | 229 | 11 | 5 | 274 |
| 191 | 6 | quadratic_knapsack, cardinality | $1 \cdot x_{0} \cdot x_{0} + 5 \cdot x_{0} \cdot x_{1} + 3 \cdot x_{0} \cdot x_{2} + 4 \cdot x_{1} \cdot x_{1} + 2 \cdot x_{1} \cdot x_{2} + 4 \cdot x_{2} \cdot x_{2} \leq 7$ (quadratic_knapsack)<br>$x_{3} + x_{4} + x_{5} = 1$ (cardinality) | — | 6 | 13 | 95 | 9 | 3 | 249 |
| 192 | 6 | quadratic_knapsack, quadratic_knapsack | $5 \cdot x_{0} \cdot x_{0} + 3 \cdot x_{0} \cdot x_{1} + 5 \cdot x_{0} \cdot x_{2} + 4 \cdot x_{1} \cdot x_{1} + 5 \cdot x_{1} \cdot x_{2} + 3 \cdot x_{2} \cdot x_{2} \leq 14$ (quadratic_knapsack)<br>$1 \cdot x_{3} \cdot x_{3} + 5 \cdot x_{3} \cdot x_{4} + 4 \cdot x_{3} \cdot x_{5} + 2 \cdot x_{4} \cdot x_{4} + 2 \cdot x_{4} \cdot x_{5} + 2 \cdot x_{5} \cdot x_{5} \leq 8$ (quadratic_knapsack) | — | 6 | 24 | 114 | 14 | 8 | 524 |
| 193 | 8 | independent_set, quadratic_knapsack, knapsack | $x_{0} \cdot x_{1} = 0$ (independent_set)<br>$5 \cdot x_{2} \cdot x_{2} + 1 \cdot x_{2} \cdot x_{3} + 2 \cdot x_{2} \cdot x_{4} + 5 \cdot x_{3} \cdot x_{3} + 1 \cdot x_{3} \cdot x_{4} + 3 \cdot x_{4} \cdot x_{4} \leq 6$ (quadratic_knapsack)<br>$10 \cdot x_{5} + 1 \cdot x_{6} + 9 \cdot x_{7} \leq 7$ (knapsack) | — | 8 | 19 | 148 | 14 | 6 | 312 |
| 194 | 5 | cardinality, quadratic_knapsack | $x_{0} + x_{1} \geq 2$ (cardinality)<br>$3 \cdot x_{2} \cdot x_{2} + 1 \cdot x_{2} \cdot x_{3} + 3 \cdot x_{2} \cdot x_{4} + 1 \cdot x_{3} \cdot x_{3} + 4 \cdot x_{3} \cdot x_{4} + 3 \cdot x_{4} \cdot x_{4} \leq 5$ (quadratic_knapsack) | — | 5 | 85 | 208 | 8 | 3 | 230 |
| 195 | 8 | quadratic_knapsack, cardinality | $4 \cdot x_{0} \cdot x_{0} + 4 \cdot x_{0} \cdot x_{1} + 1 \cdot x_{0} \cdot x_{2} + 5 \cdot x_{0} \cdot x_{3} + 5 \cdot x_{1} \cdot x_{1} + 5 \cdot x_{1} \cdot x_{2} + 5 \cdot x_{1} \cdot x_{3} + 1 \cdot x_{2} \cdot x_{2} + 4 \cdot x_{2} \cdot x_{3} + 1 \cdot x_{3} \cdot x_{3} \leq 11$ (quadratic_knapsack)<br>$x_{4} + x_{5} + x_{6} + x_{7} = 4$ (cardinality) | — | 8 | 225 | 555 | 12 | 4 | 630 |
| 196 | 8 | knapsack, cardinality, quadratic_knapsack | $2 \cdot x_{0} + 10 \cdot x_{1} + 4 \cdot x_{2} \leq 6$ (knapsack)<br>$x_{3} + x_{4} \leq 2$ (cardinality)<br>$5 \cdot x_{5} \cdot x_{5} + 3 \cdot x_{5} \cdot x_{6} + 5 \cdot x_{5} \cdot x_{7} + 4 \cdot x_{6} \cdot x_{6} + 5 \cdot x_{6} \cdot x_{7} + 3 \cdot x_{7} \cdot x_{7} \leq 14$ (quadratic_knapsack) | — | 8 | 25 | 160 | 17 | 9 | 379 |
| 197 | 6 | cardinality, quadratic_knapsack | $x_{0} + x_{1} + x_{2} = 0$ (cardinality)<br>$1 \cdot x_{3} \cdot x_{3} + 5 \cdot x_{3} \cdot x_{4} + 3 \cdot x_{3} \cdot x_{5} + 4 \cdot x_{4} \cdot x_{4} + 2 \cdot x_{4} \cdot x_{5} + 4 \cdot x_{5} \cdot x_{5} \leq 7$ (quadratic_knapsack) | — | 6 | 8 | 79 | 9 | 3 | 244 |
| 198 | 7 | flow, cardinality | $x_{0} + x_{1} + x_{2} - x_{3} - x_{4} = 0$ (flow)<br>$x_{5} + x_{6} \leq 0$ (cardinality) | — | 7 | 18 | 119 | 7 | 0 | 71 |
| 199 | 7 | cardinality, cardinality, cardinality | $x_{0} + x_{1} \leq 1$ (cardinality)<br>$x_{2} + x_{3} + x_{4} \leq 3$ (cardinality)<br>$x_{5} + x_{6} \geq 2$ (cardinality) | — | 7 | 20 | 121 | 10 | 3 | 96 |
| 200 | 8 | knapsack, cardinality | $10 \cdot x_{0} + 8 \cdot x_{1} + 2 \cdot x_{2} + 1 \cdot x_{3} + 4 \cdot x_{4} \leq 10$ (knapsack)<br>$x_{5} + x_{6} + x_{7} \geq 2$ (cardinality) | — | 8 | 161 | 428 | 13 | 5 | 197 |
| 201 | 7 | knapsack, cardinality | $5 \cdot x_{0} + 1 \cdot x_{1} + 10 \cdot x_{2} + 4 \cdot x_{3} \leq 10$ (knapsack)<br>$x_{4} + x_{5} + x_{6} \geq 0$ (cardinality) | — | 7 | 57 | 193 | 13 | 6 | 172 |
| 202 | 8 | quadratic_knapsack, cardinality | $3 \cdot x_{0} \cdot x_{0} + 1 \cdot x_{0} \cdot x_{1} + 3 \cdot x_{0} \cdot x_{2} + 4 \cdot x_{0} \cdot x_{3} + 5 \cdot x_{1} \cdot x_{1} + 4 \cdot x_{1} \cdot x_{2} + 2 \cdot x_{1} \cdot x_{3} + 2 \cdot x_{2} \cdot x_{2} + 4 \cdot x_{2} \cdot x_{3} + 4 \cdot x_{3} \cdot x_{3} \leq 17$ (quadratic_knapsack)<br>$x_{4} + x_{5} + x_{6} + x_{7} = 2$ (cardinality) | — | 8 | 37 | 180 | 13 | 5 | 700 |
| 203 | 7 | knapsack, cardinality | $2 \cdot x_{0} + 10 \cdot x_{1} + 4 \cdot x_{2} \leq 6$ (knapsack)<br>$x_{3} + x_{4} + x_{5} + x_{6} = 1$ (cardinality) | — | 7 | 14 | 113 | 10 | 3 | 119 |
| 204 | 7 | quadratic_knapsack, quadratic_knapsack | $1 \cdot x_{0} \cdot x_{0} + 4 \cdot x_{0} \cdot x_{1} + 1 \cdot x_{0} \cdot x_{2} + 2 \cdot x_{0} \cdot x_{3} + 5 \cdot x_{1} \cdot x_{1} + 4 \cdot x_{1} \cdot x_{2} + 3 \cdot x_{1} \cdot x_{3} + 5 \cdot x_{2} \cdot x_{2} + 4 \cdot x_{2} \cdot x_{3} + 2 \cdot x_{3} \cdot x_{3} \leq 18$ (quadratic_knapsack)<br>$2 \cdot x_{4} \cdot x_{4} + 5 \cdot x_{4} \cdot x_{5} + 2 \cdot x_{4} \cdot x_{6} + 2 \cdot x_{5} \cdot x_{5} + 1 \cdot x_{5} \cdot x_{6} + 1 \cdot x_{6} \cdot x_{6} \leq 7$ (quadratic_knapsack) | — | 7 | 426 | 933 | 15 | 8 | 872 |
| 205 | 6 | quadratic_knapsack, quadratic_knapsack | $3 \cdot x_{0} \cdot x_{0} + 1 \cdot x_{0} \cdot x_{1} + 3 \cdot x_{0} \cdot x_{2} + 1 \cdot x_{1} \cdot x_{1} + 4 \cdot x_{1} \cdot x_{2} + 3 \cdot x_{2} \cdot x_{2} \leq 5$ (quadratic_knapsack)<br>$1 \cdot x_{3} \cdot x_{3} + 2 \cdot x_{3} \cdot x_{4} + 1 \cdot x_{3} \cdot x_{5} + 1 \cdot x_{4} \cdot x_{4} + 5 \cdot x_{4} \cdot x_{5} + 1 \cdot x_{5} \cdot x_{5} \leq 6$ (quadratic_knapsack) | — | 6 | 126 | 315 | 12 | 6 | 439 |
| 206 | 7 | quadratic_knapsack, cardinality | $4 \cdot x_{0} \cdot x_{0} + 4 \cdot x_{0} \cdot x_{1} + 1 \cdot x_{0} \cdot x_{2} + 5 \cdot x_{0} \cdot x_{3} + 5 \cdot x_{1} \cdot x_{1} + 5 \cdot x_{1} \cdot x_{2} + 5 \cdot x_{1} \cdot x_{3} + 1 \cdot x_{2} \cdot x_{2} + 4 \cdot x_{2} \cdot x_{3} + 1 \cdot x_{3} \cdot x_{3} \leq 11$ (quadratic_knapsack)<br>$x_{4} + x_{5} + x_{6} \geq 0$ (cardinality) | — | 7 | 221 | 523 | 13 | 6 | 633 |
| 207 | 7 | cardinality, cardinality | $x_{0} + x_{1} \geq 2$ (cardinality)<br>$x_{2} + x_{3} + x_{4} + x_{5} + x_{6} = 3$ (cardinality) | — | 7 | 33 | 151 | 7 | 0 | 80 |
| 208 | 8 | assignment, cardinality | $x_{0} + x_{1} + x_{2} = 1$ (assignment)<br>$x_{3} + x_{4} + x_{5} + x_{6} + x_{7} = 4$ (cardinality) | — | 8 | 43 | 199 | 8 | 0 | 98 |
| 209 | 6 | quadratic_knapsack, assignment | $5 \cdot x_{0} \cdot x_{0} + 3 \cdot x_{0} \cdot x_{1} + 1 \cdot x_{0} \cdot x_{2} + 5 \cdot x_{0} \cdot x_{3} + 3 \cdot x_{1} \cdot x_{1} + 1 \cdot x_{1} \cdot x_{2} + 4 \cdot x_{1} \cdot x_{3} + 5 \cdot x_{2} \cdot x_{2} + 5 \cdot x_{2} \cdot x_{3} + 1 \cdot x_{3} \cdot x_{3} \leq 20$ (quadratic_knapsack)<br>$x_{4} + x_{5} = 1$ (assignment) | — | 6 | 378 | 819 | 11 | 5 | 666 |
| 210 | 8 | quadratic_knapsack, cardinality | $5 \cdot x_{0} \cdot x_{0} + 2 \cdot x_{0} \cdot x_{1} + 2 \cdot x_{0} \cdot x_{2} + 3 \cdot x_{0} \cdot x_{3} + 1 \cdot x_{0} \cdot x_{4} + 1 \cdot x_{1} \cdot x_{1} + 4 \cdot x_{1} \cdot x_{2} + 4 \cdot x_{1} \cdot x_{3} + 3 \cdot x_{1} \cdot x_{4} + 1 \cdot x_{2} \cdot x_{2} + 3 \cdot x_{2} \cdot x_{3} + 1 \cdot x_{2} \cdot x_{4} + 5 \cdot x_{3} \cdot x_{3} + 1 \cdot x_{3} \cdot x_{4} + 5 \cdot x_{4} \cdot x_{4} \leq 23$ (quadratic_knapsack)<br>$x_{5} + x_{6} + x_{7} \leq 2$ (cardinality) | — | 8 | 44 | 195 | 15 | 7 | 1368 |
| 211 | 7 | cardinality, quadratic_knapsack | $x_{0} + x_{1} + x_{2} + x_{3} = 0$ (cardinality)<br>$1 \cdot x_{4} \cdot x_{4} + 5 \cdot x_{4} \cdot x_{5} + 4 \cdot x_{4} \cdot x_{6} + 2 \cdot x_{5} \cdot x_{5} + 2 \cdot x_{5} \cdot x_{6} + 2 \cdot x_{6} \cdot x_{6} \leq 8$ (quadratic_knapsack) | — | 7 | 12 | 112 | 11 | 4 | 305 |
| 212 | 5 | cardinality, cardinality | $x_{0} + x_{1} + x_{2} = 1$ (cardinality)<br>$x_{3} + x_{4} \leq 2$ (cardinality) | — | 5 | 11 | 72 | 7 | 2 | 63 |
| 213 | 6 | cardinality, knapsack | $x_{0} + x_{1} + x_{2} \geq 3$ (cardinality)<br>$5 \cdot x_{3} + 4 \cdot x_{4} + 1 \cdot x_{5} \leq 6$ (knapsack) | — | 6 | 14 | 91 | 9 | 3 | 100 |
| 214 | 6 | cardinality, cardinality | $x_{0} + x_{1} + x_{2} \geq 1$ (cardinality)<br>$x_{3} + x_{4} + x_{5} = 3$ (cardinality) | — | 6 | 3 | 72 | 8 | 2 | 83 |
| 215 | 8 | knapsack, cardinality, cardinality | $2 \cdot x_{0} + 6 \cdot x_{1} + 1 \cdot x_{2} \leq 3$ (knapsack)<br>$x_{3} + x_{4} = 2$ (cardinality)<br>$x_{5} + x_{6} + x_{7} = 2$ (cardinality) | — | 8 | 21 | 152 | 10 | 2 | 106 |
| 216 | 6 | knapsack, cardinality | $2 \cdot x_{0} + 7 \cdot x_{1} + 7 \cdot x_{2} \leq 7$ (knapsack)<br>$x_{3} + x_{4} + x_{5} \geq 0$ (cardinality) | — | 6 | 8 | 79 | 11 | 5 | 122 |
| 217 | 7 | cardinality, cardinality | $x_{0} + x_{1} + x_{2} + x_{3} + x_{4} \geq 3$ (cardinality)<br>$x_{5} + x_{6} \geq 0$ (cardinality) | — | 7 | 0 | 88 | 11 | 4 | 137 |
| 218 | 6 | cardinality, assignment | $x_{0} + x_{1} + x_{2} \leq 3$ (cardinality)<br>$x_{3} + x_{4} + x_{5} = 1$ (assignment) | — | 6 | 18 | 102 | 8 | 2 | 83 |
| 219 | 8 | knapsack, cardinality, cardinality | $9 \cdot x_{0} + 9 \cdot x_{1} + 4 \cdot x_{2} + 1 \cdot x_{3} \leq 9$ (knapsack)<br>$x_{4} + x_{5} \leq 2$ (cardinality)<br>$x_{6} + x_{7} \leq 1$ (cardinality) | — | 8 | 77 | 264 | 15 | 7 | 190 |
| 220 | 8 | quadratic_knapsack, cardinality, independent_set | $1 \cdot x_{0} \cdot x_{0} + 5 \cdot x_{0} \cdot x_{1} + 1 \cdot x_{0} \cdot x_{2} + 3 \cdot x_{1} \cdot x_{1} + 3 \cdot x_{1} \cdot x_{2} + 5 \cdot x_{2} \cdot x_{2} \leq 10$ (quadratic_knapsack)<br>$x_{3} + x_{4} + x_{5} \leq 1$ (cardinality)<br>$x_{6} \cdot x_{7} = 0$ (independent_set) | — | 8 | 19 | 148 | 13 | 5 | 321 |
| 221 | 8 | knapsack, knapsack | $1 \cdot x_{0} + 9 \cdot x_{1} + 5 \cdot x_{2} + 2 \cdot x_{3} + 10 \cdot x_{4} \leq 10$ (knapsack)<br>$10 \cdot x_{5} + 8 \cdot x_{6} + 10 \cdot x_{7} \leq 9$ (knapsack) | — | 8 | 150 | 412 | 16 | 8 | 253 |
| 222 | 5 | quadratic_knapsack, cardinality | $3 \cdot x_{0} \cdot x_{0} + 1 \cdot x_{0} \cdot x_{1} + 3 \cdot x_{0} \cdot x_{2} + 5 \cdot x_{1} \cdot x_{1} + 2 \cdot x_{1} \cdot x_{2} + 2 \cdot x_{2} \cdot x_{2} \leq 8$ (quadratic_knapsack)<br>$x_{3} + x_{4} \geq 0$ (cardinality) | — | 5 | 23 | 96 | 11 | 6 | 293 |
| 223 | 6 | cardinality, knapsack | $x_{0} + x_{1} \leq 1$ (cardinality)<br>$5 \cdot x_{2} + 1 \cdot x_{3} + 10 \cdot x_{4} + 4 \cdot x_{5} \leq 10$ (knapsack) | — | 6 | 62 | 190 | 11 | 5 | 146 |
| 224 | 5 | cardinality, knapsack | $x_{0} + x_{1} = 2$ (cardinality)<br>$5 \cdot x_{2} + 4 \cdot x_{3} + 1 \cdot x_{4} \leq 6$ (knapsack) | — | 5 | 13 | 72 | 8 | 3 | 85 |
| 225 | 6 | quadratic_knapsack, cardinality | $1 \cdot x_{0} \cdot x_{0} + 2 \cdot x_{0} \cdot x_{1} + 1 \cdot x_{0} \cdot x_{2} + 1 \cdot x_{1} \cdot x_{1} + 5 \cdot x_{1} \cdot x_{2} + 1 \cdot x_{2} \cdot x_{2} \leq 6$ (quadratic_knapsack)<br>$x_{3} + x_{4} + x_{5} \geq 0$ (cardinality) | — | 6 | 43 | 155 | 11 | 5 | 271 |
| 226 | 7 | knapsack, flow | $2 \cdot x_{0} + 5 \cdot x_{1} + 2 \cdot x_{2} \leq 5$ (knapsack)<br>$x_{3} - x_{4} - x_{5} - x_{6} = 0$ (flow) | — | 7 | 72 | 223 | 10 | 3 | 113 |
| 227 | 8 | cardinality, knapsack | $x_{0} + x_{1} + x_{2} + x_{3} \geq 4$ (cardinality)<br>$3 \cdot x_{4} + 1 \cdot x_{5} + 4 \cdot x_{6} + 8 \cdot x_{7} \leq 7$ (knapsack) | — | 8 | 61 | 234 | 11 | 3 | 148 |
| 228 | 7 | flow, assignment | $x_{0} + x_{1} + x_{2} - x_{3} = 0$ (flow)<br>$x_{4} + x_{5} + x_{6} = 1$ (assignment) | — | 7 | 14 | 110 | 7 | 0 | 69 |
| 229 | 7 | cardinality, cardinality | $x_{0} + x_{1} + x_{2} + x_{3} + x_{4} = 1$ (cardinality)<br>$x_{5} + x_{6} \leq 2$ (cardinality) | — | 7 | 15 | 112 | 9 | 2 | 100 |
| 230 | 6 | cardinality, cardinality | $x_{0} + x_{1} + x_{2} + x_{3} \geq 0$ (cardinality)<br>$x_{4} + x_{5} \geq 0$ (cardinality) | — | 6 | 0 | 69 | 11 | 5 | 124 |
| 231 | 8 | cardinality, cardinality | $x_{0} + x_{1} + x_{2} = 3$ (cardinality)<br>$x_{3} + x_{4} + x_{5} + x_{6} + x_{7} \geq 4$ (cardinality) | — | 8 | 3 | 116 | 9 | 1 | 115 |
| 232 | 4 | cardinality, assignment | $x_{0} + x_{1} = 0$ (cardinality)<br>$x_{2} + x_{3} = 1$ (assignment) | — | 4 | 3 | 40 | 4 | 0 | 26 |
| 233 | 8 | flow, quadratic_knapsack | $x_{0} - x_{1} - x_{2} - x_{3} = 0$ (flow)<br>$5 \cdot x_{4} \cdot x_{4} + 3 \cdot x_{4} \cdot x_{5} + 1 \cdot x_{4} \cdot x_{6} + 5 \cdot x_{4} \cdot x_{7} + 3 \cdot x_{5} \cdot x_{5} + 1 \cdot x_{5} \cdot x_{6} + 4 \cdot x_{5} \cdot x_{7} + 5 \cdot x_{6} \cdot x_{6} + 5 \cdot x_{6} \cdot x_{7} + 1 \cdot x_{7} \cdot x_{7} \leq 20$ (quadratic_knapsack) | — | 8 | 384 | 881 | 13 | 5 | 699 |
| 234 | 8 | knapsack, quadratic_knapsack | $7 \cdot x_{0} + 7 \cdot x_{1} + 10 \cdot x_{2} + 2 \cdot x_{3} \leq 10$ (knapsack)<br>$3 \cdot x_{4} \cdot x_{4} + 1 \cdot x_{4} \cdot x_{5} + 3 \cdot x_{4} \cdot x_{6} + 4 \cdot x_{4} \cdot x_{7} + 5 \cdot x_{5} \cdot x_{5} + 4 \cdot x_{5} \cdot x_{6} + 2 \cdot x_{5} \cdot x_{7} + 2 \cdot x_{6} \cdot x_{6} + 4 \cdot x_{6} \cdot x_{7} + 4 \cdot x_{7} \cdot x_{7} \leq 17$ (quadratic_knapsack) | — | 8 | 288 | 681 | 17 | 9 | 777 |
| 235 | 8 | flow, flow | $x_{0} + x_{1} - x_{2} - x_{3} - x_{4} = 0$ (flow)<br>$x_{5} + x_{6} - x_{7} = 0$ (flow) | — | 8 | 24 | 153 | 8 | 0 | 86 |
| 236 | 8 | cardinality, cardinality | $x_{0} + x_{1} + x_{2} + x_{3} + x_{4} \leq 0$ (cardinality)<br>$x_{5} + x_{6} + x_{7} \geq 3$ (cardinality) | — | 8 | 3 | 112 | 8 | 0 | 90 |
| 237 | 6 | cardinality, independent_set | $x_{0} + x_{1} + x_{2} + x_{3} = 2$ (cardinality)<br>$x_{4} \cdot x_{5} = 0$ (independent_set) | — | 6 | 18 | 99 | 6 | 0 | 54 |
| 238 | 7 | knapsack, quadratic_knapsack | $2 \cdot x_{0} + 10 \cdot x_{1} + 4 \cdot x_{2} \leq 6$ (knapsack)<br>$4 \cdot x_{3} \cdot x_{3} + 5 \cdot x_{3} \cdot x_{4} + 2 \cdot x_{3} \cdot x_{5} + 5 \cdot x_{3} \cdot x_{6} + 4 \cdot x_{4} \cdot x_{4} + 3 \cdot x_{4} \cdot x_{5} + 5 \cdot x_{4} \cdot x_{6} + 3 \cdot x_{5} \cdot x_{5} + 2 \cdot x_{5} \cdot x_{6} + 3 \cdot x_{6} \cdot x_{6} \leq 22$ (quadratic_knapsack) | — | 7 | 32 | 146 | 15 | 8 | 726 |
| 239 | 8 | knapsack, cardinality | $3 \cdot x_{0} + 8 \cdot x_{1} + 3 \cdot x_{2} \leq 5$ (knapsack)<br>$x_{3} + x_{4} + x_{5} + x_{6} + x_{7} = 3$ (cardinality) | — | 8 | 134 | 378 | 11 | 3 | 142 |
| 240 | 8 | quadratic_knapsack, cardinality | $1 \cdot x_{0} \cdot x_{0} + 1 \cdot x_{0} \cdot x_{1} + 3 \cdot x_{0} \cdot x_{2} + 2 \cdot x_{0} \cdot x_{3} + 1 \cdot x_{1} \cdot x_{1} + 2 \cdot x_{1} \cdot x_{2} + 3 \cdot x_{1} \cdot x_{3} + 2 \cdot x_{2} \cdot x_{2} + 3 \cdot x_{2} \cdot x_{3} + 3 \cdot x_{3} \cdot x_{3} \leq 7$ (quadratic_knapsack)<br>$x_{4} + x_{5} + x_{6} + x_{7} \geq 0$ (cardinality) | — | 8 | 128 | 363 | 14 | 6 | 613 |
| 241 | 7 | knapsack, knapsack | $8 \cdot x_{0} + 5 \cdot x_{1} + 7 \cdot x_{2} + 4 \cdot x_{3} \leq 11$ (knapsack)<br>$2 \cdot x_{4} + 10 \cdot x_{5} + 4 \cdot x_{6} \leq 6$ (knapsack) | — | 7 | 64 | 207 | 14 | 7 | 195 |
| 242 | 7 | cardinality, quadratic_knapsack | $x_{0} + x_{1} \geq 1$ (cardinality)<br>$5 \cdot x_{2} \cdot x_{2} + 2 \cdot x_{2} \cdot x_{3} + 2 \cdot x_{2} \cdot x_{4} + 3 \cdot x_{2} \cdot x_{5} + 1 \cdot x_{2} \cdot x_{6} + 1 \cdot x_{3} \cdot x_{3} + 4 \cdot x_{3} \cdot x_{4} + 4 \cdot x_{3} \cdot x_{5} + 3 \cdot x_{3} \cdot x_{6} + 1 \cdot x_{4} \cdot x_{4} + 3 \cdot x_{4} \cdot x_{5} + 1 \cdot x_{4} \cdot x_{6} + 5 \cdot x_{5} \cdot x_{5} + 1 \cdot x_{5} \cdot x_{6} + 5 \cdot x_{6} \cdot x_{6} \leq 23$ (quadratic_knapsack) | — | 7 | 32 | 143 | 13 | 6 | 1332 |
| 243 | 8 | knapsack, knapsack | $5 \cdot x_{0} + 1 \cdot x_{1} + 10 \cdot x_{2} + 4 \cdot x_{3} \leq 10$ (knapsack)<br>$7 \cdot x_{4} + 7 \cdot x_{5} + 10 \cdot x_{6} + 2 \cdot x_{7} \leq 10$ (knapsack) | — | 8 | 326 | 762 | 16 | 8 | 250 |
| 244 | 8 | quadratic_knapsack, cardinality | $1 \cdot x_{0} \cdot x_{0} + 4 \cdot x_{0} \cdot x_{1} + 1 \cdot x_{0} \cdot x_{2} + 5 \cdot x_{0} \cdot x_{3} + 2 \cdot x_{1} \cdot x_{1} + 3 \cdot x_{1} \cdot x_{2} + 3 \cdot x_{1} \cdot x_{3} + 5 \cdot x_{2} \cdot x_{2} + 4 \cdot x_{2} \cdot x_{3} + 5 \cdot x_{3} \cdot x_{3} \leq 22$ (quadratic_knapsack)<br>$x_{4} + x_{5} + x_{6} + x_{7} \leq 1$ (cardinality) | — | 8 | 35 | 177 | 14 | 6 | 716 |
| 245 | 5 | cardinality, knapsack | $x_{0} + x_{1} \leq 2$ (cardinality)<br>$6 \cdot x_{2} + 2 \cdot x_{3} + 2 \cdot x_{4} \leq 3$ (knapsack) | — | 5 | 29 | 104 | 9 | 4 | 88 |
| 246 | 6 | knapsack, cardinality | $1 \cdot x_{0} + 3 \cdot x_{1} + 9 \cdot x_{2} + 8 \cdot x_{3} \leq 10$ (knapsack)<br>$x_{4} + x_{5} \geq 1$ (cardinality) | — | 6 | 100 | 266 | 11 | 5 | 146 |
| 247 | 7 | quadratic_knapsack, cardinality | $5 \cdot x_{0} \cdot x_{0} + 2 \cdot x_{0} \cdot x_{1} + 2 \cdot x_{0} \cdot x_{2} + 3 \cdot x_{0} \cdot x_{3} + 1 \cdot x_{0} \cdot x_{4} + 1 \cdot x_{1} \cdot x_{1} + 4 \cdot x_{1} \cdot x_{2} + 4 \cdot x_{1} \cdot x_{3} + 3 \cdot x_{1} \cdot x_{4} + 1 \cdot x_{2} \cdot x_{2} + 3 \cdot x_{2} \cdot x_{3} + 1 \cdot x_{2} \cdot x_{4} + 5 \cdot x_{3} \cdot x_{3} + 1 \cdot x_{3} \cdot x_{4} + 5 \cdot x_{4} \cdot x_{4} \leq 23$ (quadratic_knapsack)<br>$x_{5} + x_{6} = 0$ (cardinality) | — | 7 | 32 | 148 | 12 | 5 | 1322 |
| 248 | 7 | cardinality, knapsack | $x_{0} + x_{1} + x_{2} + x_{3} \geq 0$ (cardinality)<br>$2 \cdot x_{4} + 5 \cdot x_{5} + 2 \cdot x_{6} \leq 5$ (knapsack) | — | 7 | 63 | 207 | 13 | 6 | 164 |
| 249 | 8 | cardinality, cardinality | $x_{0} + x_{1} + x_{2} + x_{3} \geq 4$ (cardinality)<br>$x_{4} + x_{5} + x_{6} + x_{7} = 3$ (cardinality) | — | 8 | 27 | 167 | 8 | 0 | 95 |

---

## Results

`H` = HybridQAOA, `P` = PenaltyQAOA.

| COP | $n_x$ | Method | $p=1$ AR$_f$ | $p=1$ $P_f$ | $p=1$ $P_o$ | $p=2$ AR$_f$ | $p=2$ $P_f$ | $p=2$ $P_o$ | $p=3$ AR$_f$ | $p=3$ $P_f$ | $p=3$ $P_o$ | $p=4$ AR$_f$ | $p=4$ $P_f$ | $p=4$ $P_o$ | $p=5$ AR$_f$ | $p=5$ $P_f$ | $p=5$ $P_o$ |
|-----|-------|--------|--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---|
| 0 | 6 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 0 | 6 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 1 | 7 | `H` | 0.464 | 1.000 | 0.025 | 0.558 | 1.000 | 0.044 | 0.671 | 1.000 | 0.115 | 0.770 | 1.000 | 0.201 | 0.839 | 1.000 | 0.276 |
| 1 | 7 | `P` | 0.315 | 0.818 | 0.003 | 0.387 | 0.919 | 0.006 | 0.485 | 0.971 | 0.077 | 0.125 | 0.910 | 0.000 | 0.669 | 0.982 | 0.002 |
| 2 | 8 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 2 | 8 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 3 | 7 | `H` | 0.647 | 1.000 | 0.047 | 0.825 | 1.000 | 0.220 | 0.974 | 1.000 | 0.520 | 0.989 | 1.000 | 0.755 | 0.997 | 1.000 | 0.948 |
| 3 | 7 | `P` | 0.428 | 0.620 | 0.002 | 0.631 | 0.921 | 0.000 | 0.663 | 0.965 | 0.000 | 0.777 | 0.966 | 0.434 | 0.827 | 0.983 | 0.486 |
| 4 | 8 | `H` | 0.867 | 1.000 | 0.299 | 0.952 | 1.000 | 0.636 | 0.995 | 1.000 | 0.969 | 1.000 | 1.000 | 0.999 | 1.000 | 1.000 | 1.000 |
| 4 | 8 | `P` | 0.028 | 0.062 | 0.009 | 0.021 | 0.038 | 0.007 | 0.031 | 0.055 | 0.004 | 0.016 | 0.028 | 0.007 | 0.014 | 0.030 | 0.004 |
| 5 | 6 | `H` | 0.539 | 1.000 | 0.232 | 0.723 | 1.000 | 0.614 | 0.950 | 1.000 | 0.919 | 0.996 | 1.000 | 0.995 | 1.000 | 1.000 | 1.000 |
| 5 | 6 | `P` | 0.067 | 0.191 | 0.031 | 0.069 | 0.177 | 0.037 | 0.057 | 0.168 | 0.025 | 0.063 | 0.209 | 0.013 | 0.067 | 0.225 | 0.016 |
| 6 | 6 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 6 | 6 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 7 | 7 | `H` | 0.749 | 1.000 | 0.283 | 0.932 | 1.000 | 0.543 | 0.986 | 1.000 | 0.765 | 0.998 | 1.000 | 0.971 | 1.000 | 1.000 | 1.000 |
| 7 | 7 | `P` | 0.233 | 0.579 | 0.131 | 0.796 | 0.880 | 0.058 | 0.563 | 0.936 | 0.006 | 0.945 | 0.958 | 0.911 | 0.598 | 0.958 | 0.012 |
| 8 | 5 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 8 | 5 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 9 | 7 | `H` | 0.795 | 1.000 | 0.167 | 0.910 | 1.000 | 0.225 | 0.944 | 1.000 | 0.317 | 0.963 | 1.000 | 0.466 | 0.980 | 1.000 | 0.701 |
| 9 | 7 | `P` | 0.141 | 0.346 | 0.004 | 0.076 | 0.143 | 0.004 | 0.081 | 0.157 | 0.008 | 0.094 | 0.162 | 0.014 | 0.099 | 0.181 | 0.007 |
| 10 | 7 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 10 | 7 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 11 | 8 | `H` | 0.415 | 1.000 | 0.026 | 0.516 | 1.000 | 0.059 | 0.629 | 1.000 | 0.087 | 0.732 | 1.000 | 0.131 | 0.795 | 1.000 | 0.169 |
| 11 | 8 | `P` | 0.133 | 0.370 | 0.004 | 0.127 | 0.360 | 0.007 | 0.136 | 0.381 | 0.003 | 0.136 | 0.379 | 0.002 | 0.130 | 0.374 | 0.003 |
| 12 | 8 | `H` | 0.944 | 1.000 | 0.932 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 12 | 8 | `P` | 0.067 | 0.167 | 0.052 | 0.022 | 0.080 | 0.015 | 0.025 | 0.067 | 0.021 | 0.038 | 0.055 | 0.035 | 0.023 | 0.052 | 0.019 |
| 13 | 7 | `H` | 0.575 | 1.000 | 0.015 | 0.701 | 1.000 | 0.066 | 0.821 | 1.000 | 0.148 | 0.866 | 1.000 | 0.177 | 0.895 | 1.000 | 0.221 |
| 13 | 7 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 14 | 8 | `H` | 0.433 | 1.000 | 0.017 | 0.517 | 1.000 | 0.020 | 0.615 | 1.000 | 0.035 | 0.690 | 1.000 | 0.061 | 0.752 | 1.000 | 0.095 |
| 14 | 8 | `P` | 0.157 | 0.407 | 0.003 | 0.159 | 0.421 | 0.003 | 0.161 | 0.417 | 0.004 | 0.158 | 0.410 | 0.005 | 0.156 | 0.412 | 0.004 |
| 15 | 8 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 15 | 8 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 16 | 6 | `H` | 0.568 | 1.000 | 0.115 | 0.708 | 1.000 | 0.248 | 0.865 | 1.000 | 0.424 | 0.958 | 1.000 | 0.530 | 0.985 | 1.000 | 0.603 |
| 16 | 6 | `P` | 0.172 | 0.449 | 0.015 | 0.177 | 0.465 | 0.017 | 0.185 | 0.478 | 0.018 | 0.175 | 0.466 | 0.012 | 0.170 | 0.470 | 0.017 |
| 17 | 8 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 17 | 8 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 18 | 8 | `H` | 0.618 | 1.000 | 0.068 | 0.772 | 1.000 | 0.145 | 0.835 | 1.000 | 0.235 | 0.875 | 1.000 | 0.346 | 0.909 | 1.000 | 0.548 |
| 18 | 8 | `P` | 0.058 | 0.155 | 0.003 | 0.051 | 0.103 | 0.008 | 0.053 | 0.109 | 0.004 | 0.047 | 0.095 | 0.005 | 0.039 | 0.086 | 0.002 |
| 19 | 8 | `H` | 0.448 | 1.000 | 0.009 | 0.523 | 1.000 | 0.032 | 0.617 | 1.000 | 0.044 | 0.679 | 1.000 | 0.061 | 0.740 | 1.000 | 0.095 |
| 19 | 8 | `P` | 0.197 | 0.528 | 0.001 | 0.254 | 0.670 | 0.000 | 0.266 | 0.623 | 0.000 | 0.208 | 0.512 | 0.003 | 0.192 | 0.494 | 0.003 |
| 20 | 6 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 20 | 6 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 21 | 6 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 21 | 6 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 22 | 6 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 22 | 6 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 23 | 8 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 23 | 8 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 24 | 8 | `H` | 0.677 | 1.000 | 0.131 | 0.849 | 1.000 | 0.148 | 0.920 | 1.000 | 0.334 | 0.960 | 1.000 | 0.511 | 0.978 | 1.000 | 0.700 |
| 24 | 8 | `P` | 0.213 | 0.309 | 0.097 | 0.132 | 0.260 | 0.014 | 0.122 | 0.232 | 0.036 | 0.121 | 0.224 | 0.024 | 0.108 | 0.211 | 0.019 |
| 25 | 7 | `H` | 0.978 | 1.000 | 0.962 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 25 | 7 | `P` | 0.365 | 0.650 | 0.355 | 0.735 | 0.893 | 0.014 | 0.808 | 0.941 | 0.003 | 0.750 | 0.949 | 0.576 | 0.842 | 0.978 | 0.016 |
| 26 | 6 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 26 | 6 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 27 | 5 | `H` | 0.997 | 1.000 | 0.988 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 27 | 5 | `P` | 0.301 | 0.717 | 0.000 | 0.720 | 0.994 | 0.000 | 0.795 | 0.987 | 0.000 | 0.985 | 0.989 | 0.970 | 0.979 | 0.980 | 0.977 |
| 28 | 8 | `H` | 0.499 | 1.000 | 0.053 | 0.675 | 1.000 | 0.119 | 0.778 | 1.000 | 0.180 | 0.853 | 1.000 | 0.278 | 0.904 | 1.000 | 0.358 |
| 28 | 8 | `P` | 0.214 | 0.417 | 0.018 | 0.079 | 0.196 | 0.003 | 0.077 | 0.194 | 0.003 | 0.077 | 0.195 | 0.005 | 0.080 | 0.200 | 0.004 |
| 29 | 7 | `H` | 0.556 | 1.000 | 0.042 | 0.651 | 1.000 | 0.084 | 0.757 | 1.000 | 0.142 | 0.838 | 1.000 | 0.207 | 0.902 | 1.000 | 0.322 |
| 29 | 7 | `P` | 0.419 | 0.792 | 0.000 | 0.542 | 0.930 | 0.002 | 0.530 | 0.974 | 0.003 | 0.557 | 0.992 | 0.000 | 0.631 | 0.984 | 0.001 |
| 30 | 8 | `H` | 0.576 | 1.000 | 0.024 | 0.642 | 1.000 | 0.037 | 0.758 | 1.000 | 0.083 | 0.842 | 1.000 | 0.160 | 0.897 | 1.000 | 0.238 |
| 30 | 8 | `P` | 0.176 | 0.396 | 0.005 | 0.352 | 0.556 | 0.004 | 0.406 | 0.755 | 0.007 | 0.391 | 0.743 | 0.013 | 0.747 | 0.923 | 0.009 |
| 31 | 7 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 31 | 7 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 32 | 7 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 32 | 7 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 33 | 6 | `H` | 0.938 | 1.000 | 0.722 | 0.998 | 1.000 | 0.988 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 33 | 6 | `P` | 0.760 | 0.881 | 0.444 | 0.859 | 0.997 | 0.500 | 0.861 | 0.999 | 0.501 | 0.925 | 0.981 | 0.784 | 0.985 | 0.994 | 0.967 |
| 34 | 8 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 34 | 8 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 35 | 4 | `H` | 0.760 | 1.000 | 0.720 | 0.993 | 1.000 | 0.992 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 35 | 4 | `P` | 0.384 | 0.984 | 0.247 | 0.969 | 0.982 | 0.965 | 0.991 | 0.999 | 0.989 | 0.993 | 0.998 | 0.992 | 0.999 | 1.000 | 0.999 |
| 36 | 7 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 36 | 7 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 37 | 8 | `H` | 0.462 | 1.000 | 0.037 | 0.595 | 1.000 | 0.083 | 0.713 | 1.000 | 0.145 | 0.807 | 1.000 | 0.230 | 0.864 | 1.000 | 0.295 |
| 37 | 8 | `P` | 0.368 | 0.852 | 0.049 | 0.348 | 0.952 | 0.000 | 0.346 | 0.975 | 0.000 | 0.382 | 0.967 | 0.002 | 0.399 | 0.923 | 0.058 |
| 38 | 7 | `H` | 0.796 | 1.000 | 0.371 | 0.935 | 1.000 | 0.509 | 0.985 | 1.000 | 0.829 | 0.999 | 1.000 | 0.996 | 1.000 | 1.000 | 1.000 |
| 38 | 7 | `P` | 0.083 | 0.195 | 0.014 | 0.018 | 0.047 | 0.003 | 0.064 | 0.147 | 0.012 | 0.020 | 0.050 | 0.003 | 0.044 | 0.100 | 0.010 |
| 39 | 8 | `H` | 0.648 | 1.000 | 0.214 | 0.833 | 1.000 | 0.428 | 0.941 | 1.000 | 0.585 | 0.972 | 1.000 | 0.725 | 0.989 | 1.000 | 0.897 |
| 39 | 8 | `P` | 0.116 | 0.236 | 0.007 | 0.100 | 0.224 | 0.010 | 0.088 | 0.204 | 0.007 | 0.086 | 0.193 | 0.008 | 0.087 | 0.195 | 0.007 |
| 40 | 8 | `H` | 0.506 | 1.000 | 0.018 | 0.621 | 1.000 | 0.044 | 0.722 | 1.000 | 0.081 | 0.797 | 1.000 | 0.113 | 0.860 | 1.000 | 0.149 |
| 40 | 8 | `P` | 0.086 | 0.183 | 0.004 | 0.090 | 0.190 | 0.005 | 0.085 | 0.185 | 0.004 | 0.086 | 0.185 | 0.004 | 0.088 | 0.189 | 0.003 |
| 41 | 5 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 41 | 5 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 42 | 7 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 42 | 7 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 43 | 8 | `H` | 0.413 | 1.000 | 0.022 | 0.513 | 1.000 | 0.037 | 0.573 | 1.000 | 0.061 | 0.655 | 1.000 | 0.102 | 0.736 | 1.000 | 0.160 |
| 43 | 8 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 44 | 7 | `H` | 0.484 | 1.000 | 0.036 | 0.659 | 1.000 | 0.106 | 0.758 | 1.000 | 0.171 | 0.856 | 1.000 | 0.218 | 0.903 | 1.000 | 0.301 |
| 44 | 7 | `P` | 0.414 | 0.764 | 0.003 | 0.564 | 0.974 | 0.000 | 0.676 | 0.987 | 0.002 | 0.696 | 0.994 | 0.000 | 0.695 | 0.995 | 0.003 |
| 45 | 8 | `H` | 0.716 | 1.000 | 0.173 | 0.818 | 1.000 | 0.356 | 0.915 | 1.000 | 0.627 | 0.975 | 1.000 | 0.901 | 0.997 | 1.000 | 0.988 |
| 45 | 8 | `P` | 0.229 | 0.388 | 0.010 | 0.196 | 0.391 | 0.005 | 0.097 | 0.168 | 0.004 | 0.065 | 0.111 | 0.009 | 0.073 | 0.112 | 0.011 |
| 46 | 8 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 46 | 8 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 47 | 7 | `H` | 0.579 | 1.000 | 0.035 | 0.668 | 1.000 | 0.065 | 0.753 | 1.000 | 0.091 | 0.827 | 1.000 | 0.121 | 0.880 | 1.000 | 0.182 |
| 47 | 7 | `P` | 0.276 | 0.541 | 0.012 | 0.248 | 0.502 | 0.005 | 0.259 | 0.495 | 0.008 | 0.240 | 0.486 | 0.006 | 0.292 | 0.509 | 0.015 |
| 48 | 7 | `H` | 0.569 | 1.000 | 0.007 | 0.667 | 1.000 | 0.050 | 0.744 | 1.000 | 0.082 | 0.804 | 1.000 | 0.126 | 0.853 | 1.000 | 0.191 |
| 48 | 7 | `P` | 0.261 | 0.521 | 0.007 | 0.261 | 0.524 | 0.012 | 0.258 | 0.527 | 0.005 | 0.261 | 0.526 | 0.008 | 0.256 | 0.510 | 0.005 |
| 49 | 6 | `H` | 0.614 | 1.000 | 0.034 | 0.694 | 1.000 | 0.055 | 0.778 | 1.000 | 0.079 | 0.840 | 1.000 | 0.096 | 0.861 | 1.000 | 0.185 |
| 49 | 6 | `P` | 0.215 | 0.471 | 0.029 | 0.175 | 0.496 | 0.005 | 0.200 | 0.468 | 0.012 | 0.175 | 0.494 | 0.007 | 0.295 | 0.560 | 0.034 |
| 50 | 6 | `H` | 0.682 | 1.000 | 0.152 | 0.769 | 1.000 | 0.235 | 0.875 | 1.000 | 0.366 | 0.939 | 1.000 | 0.551 | 0.967 | 1.000 | 0.671 |
| 50 | 6 | `P` | 0.292 | 0.504 | 0.034 | 0.273 | 0.494 | 0.025 | 0.288 | 0.526 | 0.031 | 0.272 | 0.480 | 0.030 | 0.290 | 0.497 | 0.035 |
| 51 | 8 | `H` | 0.436 | 1.000 | 0.029 | 0.545 | 1.000 | 0.052 | 0.651 | 1.000 | 0.091 | 0.715 | 1.000 | 0.133 | 0.782 | 1.000 | 0.196 |
| 51 | 8 | `P` | 0.256 | 0.594 | 0.005 | 0.317 | 0.723 | 0.010 | 0.391 | 0.826 | 0.045 | 0.339 | 0.813 | 0.007 | 0.303 | 0.734 | 0.013 |
| 52 | 6 | `H` | 0.963 | 1.000 | 0.953 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 52 | 6 | `P` | 0.046 | 0.124 | 0.033 | 0.129 | 0.236 | 0.095 | 0.115 | 0.234 | 0.085 | 0.022 | 0.064 | 0.012 | 0.035 | 0.075 | 0.029 |
| 53 | 8 | `H` | 0.798 | 1.000 | 0.001 | 0.884 | 1.000 | 0.002 | 0.942 | 1.000 | 0.009 | 0.964 | 1.000 | 0.032 | 0.967 | 1.000 | 0.038 |
| 53 | 8 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 54 | 7 | `H` | 0.891 | 1.000 | 0.776 | 0.999 | 1.000 | 0.999 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 54 | 7 | `P` | 0.126 | 0.287 | 0.037 | 0.383 | 0.769 | 0.009 | 0.455 | 0.902 | 0.007 | 0.468 | 0.930 | 0.003 | 0.490 | 0.966 | 0.001 |
| 55 | 6 | `H` | 0.606 | 1.000 | 0.349 | 0.783 | 1.000 | 0.659 | 0.960 | 1.000 | 0.935 | 0.999 | 1.000 | 0.999 | 1.000 | 1.000 | 1.000 |
| 55 | 6 | `P` | 0.230 | 0.606 | 0.110 | 0.708 | 0.960 | 0.504 | 0.824 | 0.984 | 0.692 | 0.836 | 0.932 | 0.767 | 0.438 | 0.987 | 0.030 |
| 56 | 6 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 56 | 6 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 57 | 8 | `H` | 0.478 | 1.000 | 0.003 | 0.599 | 1.000 | 0.031 | 0.702 | 1.000 | 0.073 | 0.793 | 1.000 | 0.101 | 0.852 | 1.000 | 0.135 |
| 57 | 8 | `P` | 0.335 | 0.567 | 0.004 | 0.398 | 0.772 | 0.006 | 0.566 | 0.947 | 0.142 | 0.677 | 0.990 | 0.186 | 0.569 | 0.997 | 0.183 |
| 58 | 6 | `H` | 0.605 | 1.000 | 0.085 | 0.693 | 1.000 | 0.128 | 0.756 | 1.000 | 0.166 | 0.831 | 1.000 | 0.256 | 0.902 | 1.000 | 0.368 |
| 58 | 6 | `P` | 0.590 | 0.980 | 0.080 | 0.649 | 0.999 | 0.037 | 0.607 | 0.997 | 0.009 | 0.578 | 0.999 | 0.025 | 0.615 | 0.999 | 0.001 |
| 59 | 8 | `H` | 0.453 | 1.000 | 0.022 | 0.584 | 1.000 | 0.056 | 0.685 | 1.000 | 0.088 | 0.776 | 1.000 | 0.127 | 0.842 | 1.000 | 0.198 |
| 59 | 8 | `P` | 0.118 | 0.329 | 0.002 | 0.086 | 0.234 | 0.005 | 0.072 | 0.211 | 0.001 | 0.070 | 0.193 | 0.002 | 0.078 | 0.206 | 0.008 |
| 60 | 8 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 60 | 8 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 61 | 6 | `H` | 0.746 | 1.000 | 0.218 | 0.897 | 1.000 | 0.398 | 0.957 | 1.000 | 0.674 | 0.996 | 1.000 | 0.978 | 1.000 | 1.000 | 1.000 |
| 61 | 6 | `P` | 0.237 | 0.320 | 0.033 | 0.242 | 0.365 | 0.018 | 0.168 | 0.280 | 0.019 | 0.198 | 0.305 | 0.019 | 0.131 | 0.265 | 0.014 |
| 62 | 7 | `H` | 0.441 | 1.000 | 0.043 | 0.536 | 1.000 | 0.085 | 0.645 | 1.000 | 0.143 | 0.735 | 1.000 | 0.218 | 0.789 | 1.000 | 0.287 |
| 62 | 7 | `P` | 0.158 | 0.450 | 0.009 | 0.159 | 0.461 | 0.009 | 0.171 | 0.481 | 0.010 | 0.161 | 0.481 | 0.007 | 0.162 | 0.471 | 0.009 |
| 63 | 7 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 63 | 7 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 64 | 8 | `H` | 0.543 | 1.000 | 0.044 | 0.662 | 1.000 | 0.103 | 0.767 | 1.000 | 0.155 | 0.833 | 1.000 | 0.240 | 0.894 | 1.000 | 0.335 |
| 64 | 8 | `P` | 0.203 | 0.393 | 0.011 | 0.099 | 0.236 | 0.003 | 0.089 | 0.206 | 0.002 | 0.099 | 0.233 | 0.003 | 0.094 | 0.226 | 0.003 |
| 65 | 5 | `H` | 0.772 | 1.000 | 0.624 | 0.989 | 1.000 | 0.986 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 65 | 5 | `P` | 0.185 | 0.523 | 0.003 | 0.397 | 0.855 | 0.004 | 0.684 | 0.923 | 0.442 | 0.548 | 0.993 | 0.490 | 0.271 | 0.997 | 0.000 |
| 66 | 5 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 66 | 5 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 67 | 8 | `H` | 0.971 | 1.000 | 0.844 | 0.989 | 1.000 | 0.928 | 0.998 | 1.000 | 0.989 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 67 | 8 | `P` | 0.021 | 0.068 | 0.013 | 0.044 | 0.083 | 0.010 | 0.056 | 0.110 | 0.024 | 0.038 | 0.079 | 0.020 | 0.016 | 0.031 | 0.007 |
| 68 | 6 | `H` | 0.606 | 1.000 | 0.045 | 0.778 | 1.000 | 0.081 | 0.866 | 1.000 | 0.148 | 0.885 | 1.000 | 0.184 | 0.904 | 1.000 | 0.284 |
| 68 | 6 | `P` | 0.132 | 0.309 | 0.009 | 0.149 | 0.325 | 0.017 | 0.151 | 0.319 | 0.016 | 0.150 | 0.322 | 0.018 | 0.150 | 0.322 | 0.014 |
| 69 | 6 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 69 | 6 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 70 | 6 | `H` | 0.577 | 1.000 | 0.068 | 0.725 | 1.000 | 0.142 | 0.820 | 1.000 | 0.261 | 0.896 | 1.000 | 0.414 | 0.951 | 1.000 | 0.578 |
| 70 | 6 | `P` | 0.320 | 0.658 | 0.028 | 0.309 | 0.655 | 0.027 | 0.285 | 0.619 | 0.027 | 0.314 | 0.662 | 0.038 | 0.307 | 0.649 | 0.038 |
| 71 | 7 | `H` | 0.582 | 1.000 | 0.043 | 0.666 | 1.000 | 0.075 | 0.740 | 1.000 | 0.110 | 0.818 | 1.000 | 0.166 | 0.879 | 1.000 | 0.230 |
| 71 | 7 | `P` | 0.358 | 0.922 | 0.000 | 0.488 | 0.972 | 0.000 | 0.504 | 0.949 | 0.003 | 0.320 | 0.996 | 0.000 | 0.491 | 0.995 | 0.000 |
| 72 | 7 | `H` | 0.541 | 1.000 | 0.074 | 0.675 | 1.000 | 0.137 | 0.775 | 1.000 | 0.237 | 0.841 | 1.000 | 0.344 | 0.893 | 1.000 | 0.537 |
| 72 | 7 | `P` | 0.205 | 0.518 | 0.009 | 0.221 | 0.575 | 0.002 | 0.107 | 0.354 | 0.003 | 0.125 | 0.355 | 0.002 | 0.100 | 0.250 | 0.012 |
| 73 | 7 | `H` | 0.781 | 1.000 | 0.164 | 0.908 | 1.000 | 0.214 | 0.934 | 1.000 | 0.212 | 0.947 | 1.000 | 0.311 | 0.966 | 1.000 | 0.506 |
| 73 | 7 | `P` | 0.123 | 0.231 | 0.008 | 0.098 | 0.178 | 0.006 | 0.093 | 0.172 | 0.008 | 0.093 | 0.168 | 0.009 | 0.090 | 0.168 | 0.006 |
| 74 | 8 | `H` | 0.546 | 1.000 | 0.038 | 0.695 | 1.000 | 0.079 | 0.769 | 1.000 | 0.104 | 0.832 | 1.000 | 0.161 | 0.888 | 1.000 | 0.227 |
| 74 | 8 | `P` | 0.237 | 0.506 | 0.003 | 0.408 | 0.848 | 0.029 | 0.454 | 0.969 | 0.006 | 0.480 | 0.984 | 0.029 | 0.515 | 0.995 | 0.032 |
| 75 | 8 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 75 | 8 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 76 | 8 | `H` | 0.645 | 1.000 | 0.010 | 0.759 | 1.000 | 0.024 | 0.842 | 1.000 | 0.020 | 0.877 | 1.000 | 0.051 | 0.896 | 1.000 | 0.066 |
| 76 | 8 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 77 | 7 | `H` | 0.662 | 1.000 | 0.340 | 0.832 | 1.000 | 0.579 | 0.967 | 1.000 | 0.895 | 0.995 | 1.000 | 0.989 | 0.998 | 1.000 | 0.996 |
| 77 | 7 | `P` | 0.080 | 0.187 | 0.008 | 0.063 | 0.142 | 0.006 | 0.057 | 0.138 | 0.005 | 0.072 | 0.173 | 0.010 | 0.066 | 0.152 | 0.007 |
| 78 | 7 | `H` | 0.574 | 1.000 | 0.073 | 0.704 | 1.000 | 0.152 | 0.806 | 1.000 | 0.293 | 0.874 | 1.000 | 0.461 | 0.939 | 1.000 | 0.657 |
| 78 | 7 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 79 | 7 | `H` | 0.434 | 1.000 | 0.013 | 0.522 | 1.000 | 0.027 | 0.583 | 1.000 | 0.029 | 0.642 | 1.000 | 0.058 | 0.707 | 1.000 | 0.090 |
| 79 | 7 | `P` | 0.445 | 1.000 | 0.030 | 0.406 | 1.000 | 0.004 | 0.441 | 1.000 | 0.007 | 0.509 | 1.000 | 0.030 | 0.418 | 1.000 | 0.001 |
| 80 | 6 | `H` | 0.747 | 1.000 | 0.278 | 0.844 | 1.000 | 0.464 | 0.933 | 1.000 | 0.740 | 0.995 | 1.000 | 0.983 | 1.000 | 1.000 | 1.000 |
| 80 | 6 | `P` | 0.131 | 0.271 | 0.014 | 0.101 | 0.187 | 0.015 | 0.080 | 0.160 | 0.022 | 0.140 | 0.270 | 0.014 | 0.151 | 0.246 | 0.045 |
| 81 | 7 | `H` | 0.806 | 1.000 | 0.259 | 0.944 | 1.000 | 0.595 | 0.987 | 1.000 | 0.890 | 1.000 | 1.000 | 0.994 | 1.000 | 1.000 | 1.000 |
| 81 | 7 | `P` | 0.447 | 0.730 | 0.065 | 0.612 | 0.986 | 0.119 | 0.624 | 0.997 | 0.121 | 0.842 | 0.955 | 0.232 | 0.794 | 0.897 | 0.238 |
| 82 | 8 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 82 | 8 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 83 | 7 | `H` | 0.599 | 1.000 | 0.036 | 0.699 | 1.000 | 0.074 | 0.792 | 1.000 | 0.117 | 0.847 | 1.000 | 0.174 | 0.882 | 1.000 | 0.253 |
| 83 | 7 | `P` | 0.223 | 0.409 | 0.007 | 0.224 | 0.413 | 0.009 | 0.210 | 0.394 | 0.008 | 0.223 | 0.414 | 0.007 | 0.219 | 0.408 | 0.008 |
| 84 | 6 | `H` | 0.535 | 1.000 | 0.017 | 0.675 | 1.000 | 0.033 | 0.756 | 1.000 | 0.057 | 0.809 | 1.000 | 0.071 | 0.846 | 1.000 | 0.117 |
| 84 | 6 | `P` | 0.208 | 0.548 | 0.009 | 0.222 | 0.494 | 0.016 | 0.229 | 0.515 | 0.018 | 0.242 | 0.569 | 0.017 | 0.210 | 0.502 | 0.013 |
| 85 | 7 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 85 | 7 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 86 | 7 | `H` | 0.574 | 1.000 | 0.047 | 0.691 | 1.000 | 0.086 | 0.781 | 1.000 | 0.145 | 0.866 | 1.000 | 0.250 | 0.918 | 1.000 | 0.347 |
| 86 | 7 | `P` | 0.334 | 0.749 | 0.012 | 0.391 | 0.829 | 0.013 | 0.418 | 0.871 | 0.012 | 0.387 | 0.916 | 0.015 | 0.435 | 0.939 | 0.005 |
| 87 | 8 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 87 | 8 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 88 | 7 | `H` | 0.785 | 1.000 | 0.137 | 0.876 | 1.000 | 0.253 | 0.933 | 1.000 | 0.392 | 0.979 | 1.000 | 0.688 | 0.994 | 1.000 | 0.900 |
| 88 | 7 | `P` | 0.121 | 0.234 | 0.014 | 0.100 | 0.188 | 0.015 | 0.081 | 0.145 | 0.007 | 0.079 | 0.144 | 0.008 | 0.084 | 0.151 | 0.009 |
| 89 | 8 | `H` | 0.511 | 1.000 | 0.061 | 0.610 | 1.000 | 0.122 | 0.691 | 1.000 | 0.156 | 0.781 | 1.000 | 0.289 | 0.850 | 1.000 | 0.395 |
| 89 | 8 | `P` | 0.173 | 0.411 | 0.008 | 0.175 | 0.413 | 0.010 | 0.171 | 0.409 | 0.008 | 0.176 | 0.415 | 0.008 | 0.173 | 0.414 | 0.010 |
| 90 | 8 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 90 | 8 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 91 | 6 | `H` | 0.865 | 1.000 | 0.754 | 0.987 | 1.000 | 0.972 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 91 | 6 | `P` | 0.025 | 0.733 | 0.015 | 0.468 | 0.957 | 0.465 | 0.511 | 0.982 | 0.509 | 0.620 | 0.967 | 0.148 | 0.720 | 0.973 | 0.527 |
| 92 | 8 | `H` | 0.531 | 1.000 | 0.046 | 0.650 | 1.000 | 0.124 | 0.754 | 1.000 | 0.209 | 0.831 | 1.000 | 0.324 | 0.870 | 1.000 | 0.396 |
| 92 | 8 | `P` | 0.249 | 0.497 | 0.000 | 0.357 | 0.906 | 0.000 | 0.428 | 0.993 | 0.000 | 0.424 | 0.984 | 0.000 | 0.428 | 0.994 | 0.000 |
| 93 | 7 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 93 | 7 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 94 | 8 | `H` | 0.600 | 1.000 | 0.001 | 0.675 | 1.000 | 0.007 | 0.765 | 1.000 | 0.020 | 0.821 | 1.000 | 0.033 | 0.863 | 1.000 | 0.044 |
| 94 | 8 | `P` | 0.255 | 0.479 | 0.006 | 0.271 | 0.496 | 0.008 | 0.266 | 0.493 | 0.007 | 0.265 | 0.498 | 0.007 | 0.266 | 0.491 | 0.008 |
| 95 | 8 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 95 | 8 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 96 | 7 | `H` | 0.903 | 1.000 | 0.767 | 0.992 | 1.000 | 0.983 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 96 | 7 | `P` | 0.229 | 0.771 | 0.001 | 0.613 | 0.919 | 0.002 | 0.465 | 0.928 | 0.020 | 0.645 | 0.977 | 0.003 | 0.706 | 0.993 | 0.001 |
| 97 | 6 | `H` | 0.693 | 1.000 | 0.298 | 0.933 | 1.000 | 0.525 | 0.970 | 1.000 | 0.663 | 0.996 | 1.000 | 0.961 | 1.000 | 1.000 | 1.000 |
| 97 | 6 | `P` | 0.135 | 0.304 | 0.015 | 0.277 | 0.415 | 0.007 | 0.066 | 0.198 | 0.006 | 0.129 | 0.186 | 0.092 | 0.124 | 0.263 | 0.059 |
| 98 | 8 | `H` | 0.527 | 1.000 | 0.008 | 0.606 | 1.000 | 0.020 | 0.658 | 1.000 | 0.031 | 0.707 | 1.000 | 0.050 | 0.753 | 1.000 | 0.096 |
| 98 | 8 | `P` | 0.117 | 0.241 | 0.006 | 0.093 | 0.209 | 0.005 | 0.082 | 0.187 | 0.003 | 0.099 | 0.207 | 0.008 | 0.081 | 0.180 | 0.005 |
| 99 | 5 | `H` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 99 | 5 | `P` | 0.219 | 0.506 | 0.021 | 0.378 | 0.986 | 0.003 | 0.370 | 0.984 | 0.001 | 0.743 | 0.991 | 0.000 | 0.686 | 0.988 | 0.001 |
| 100 | 7 | `H` | 0.506 | 1.000 | 0.012 | 0.597 | 1.000 | 0.024 | 0.654 | 1.000 | 0.041 | 0.698 | 1.000 | 0.059 | 0.744 | 1.000 | 0.081 |
| 100 | 7 | `P` | 0.203 | 0.459 | 0.005 | 0.224 | 0.487 | 0.011 | 0.218 | 0.467 | 0.007 | 0.225 | 0.477 | 0.010 | 0.217 | 0.477 | 0.008 |
| 101 | 8 | `H` | 0.541 | 1.000 | 0.015 | 0.609 | 1.000 | 0.032 | 0.682 | 1.000 | 0.055 | 0.743 | 1.000 | 0.087 | 0.790 | 1.000 | 0.117 |
| 101 | 8 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 102 | 8 | `H` | 0.558 | 1.000 | 0.100 | 0.666 | 1.000 | 0.178 | 0.759 | 1.000 | 0.329 | 0.830 | 1.000 | 0.529 | 0.909 | 1.000 | 0.772 |
| 102 | 8 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 103 | 6 | `H` | 0.692 | 1.000 | 0.282 | 0.862 | 1.000 | 0.687 | 0.974 | 1.000 | 0.928 | 0.995 | 1.000 | 0.986 | 1.000 | 1.000 | 0.999 |
| 103 | 6 | `P` | 0.277 | 0.570 | 0.000 | 0.333 | 0.734 | 0.004 | 0.278 | 0.727 | 0.076 | 0.441 | 0.868 | 0.069 | 0.413 | 0.906 | 0.001 |
| 104 | 8 | `H` | 0.444 | 1.000 | 0.030 | 0.567 | 1.000 | 0.067 | 0.633 | 1.000 | 0.092 | 0.702 | 1.000 | 0.141 | 0.763 | 1.000 | 0.190 |
| 104 | 8 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 105 | 8 | `H` | 0.705 | 1.000 | 0.130 | 0.841 | 1.000 | 0.265 | 0.913 | 1.000 | 0.450 | 0.957 | 1.000 | 0.688 | 0.990 | 1.000 | 0.949 |
| 105 | 8 | `P` | 0.141 | 0.217 | 0.032 | 0.491 | 0.757 | 0.010 | 0.603 | 0.918 | 0.089 | 0.661 | 0.990 | 0.128 | 0.666 | 0.993 | 0.158 |
| 106 | 6 | `H` | 0.673 | 1.000 | 0.520 | 0.876 | 1.000 | 0.810 | 0.999 | 1.000 | 0.998 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 106 | 6 | `P` | 0.111 | 0.311 | 0.053 | 0.205 | 0.563 | 0.101 | 0.240 | 0.670 | 0.115 | 0.239 | 0.715 | 0.109 | 0.249 | 0.729 | 0.125 |
| 107 | 8 | `H` | 0.605 | 1.000 | 0.093 | 0.759 | 1.000 | 0.192 | 0.900 | 1.000 | 0.326 | 0.944 | 1.000 | 0.453 | 0.970 | 1.000 | 0.610 |
| 107 | 8 | `P` | 0.065 | 0.206 | 0.010 | 0.048 | 0.142 | 0.003 | 0.035 | 0.099 | 0.003 | 0.037 | 0.112 | 0.004 | 0.044 | 0.120 | 0.005 |
| 108 | 8 | `H` | 0.485 | 1.000 | 0.065 | 0.581 | 1.000 | 0.116 | 0.685 | 1.000 | 0.182 | 0.834 | 1.000 | 0.367 | 0.926 | 1.000 | 0.468 |
| 108 | 8 | `P` | 0.087 | 0.247 | 0.006 | 0.111 | 0.359 | 0.002 | 0.137 | 0.417 | 0.000 | 0.125 | 0.422 | 0.000 | 0.112 | 0.416 | 0.000 |
| 109 | 7 | `H` | 0.782 | 1.000 | 0.092 | 0.858 | 1.000 | 0.173 | 0.886 | 1.000 | 0.290 | 0.914 | 1.000 | 0.546 | 0.977 | 1.000 | 0.880 |
| 109 | 7 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 110 | 5 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 110 | 5 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 111 | 5 | `H` | 0.906 | 1.000 | 0.497 | 0.956 | 1.000 | 0.692 | 0.992 | 1.000 | 0.948 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 111 | 5 | `P` | 0.107 | 0.314 | 0.007 | 0.174 | 0.391 | 0.028 | 0.080 | 0.295 | 0.004 | 0.130 | 0.255 | 0.025 | 0.086 | 0.184 | 0.034 |
| 112 | 6 | `H` | 0.916 | 1.000 | 0.743 | 0.979 | 1.000 | 0.930 | 0.998 | 1.000 | 0.995 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 112 | 6 | `P` | 0.120 | 0.231 | 0.036 | 0.152 | 0.255 | 0.051 | 0.128 | 0.221 | 0.041 | 0.093 | 0.172 | 0.023 | 0.118 | 0.202 | 0.042 |
| 113 | 8 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 113 | 8 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 114 | 6 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 114 | 6 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 115 | 7 | `H` | 0.844 | 1.000 | 0.237 | 0.924 | 1.000 | 0.593 | 0.986 | 1.000 | 0.926 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 115 | 7 | `P` | 0.072 | 0.129 | 0.015 | 0.104 | 0.168 | 0.049 | 0.145 | 0.235 | 0.021 | 0.119 | 0.304 | 0.037 | 0.082 | 0.142 | 0.003 |
| 116 | 7 | `H` | 0.922 | 1.000 | 0.780 | 0.991 | 1.000 | 0.986 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 116 | 7 | `P` | 0.275 | 0.359 | 0.167 | 0.283 | 0.913 | 0.002 | 0.751 | 0.933 | 0.444 | 0.277 | 0.783 | 0.004 | 0.510 | 0.905 | 0.196 |
| 117 | 8 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 117 | 8 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 118 | 8 | `H` | 0.978 | 1.000 | 0.922 | 1.000 | 1.000 | 0.999 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 118 | 8 | `P` | 0.179 | 0.268 | 0.113 | 0.342 | 0.797 | 0.000 | 0.441 | 0.932 | 0.020 | 0.561 | 0.932 | 0.000 | 0.405 | 0.920 | 0.001 |
| 119 | 8 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 119 | 8 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 120 | 6 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 120 | 6 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 121 | 8 | `H` | 0.667 | 1.000 | 0.002 | 0.747 | 1.000 | 0.011 | 0.836 | 1.000 | 0.077 | 0.883 | 1.000 | 0.229 | 0.930 | 1.000 | 0.383 |
| 121 | 8 | `P` | 0.046 | 0.106 | 0.005 | 0.037 | 0.088 | 0.003 | 0.037 | 0.082 | 0.003 | 0.039 | 0.094 | 0.003 | 0.038 | 0.092 | 0.003 |
| 122 | 7 | `H` | 0.935 | 1.000 | 0.429 | 0.972 | 1.000 | 0.576 | 0.982 | 1.000 | 0.695 | 0.999 | 1.000 | 0.977 | 1.000 | 1.000 | 1.000 |
| 122 | 7 | `P` | 0.225 | 0.554 | 0.145 | 0.632 | 0.837 | 0.392 | 0.287 | 0.873 | 0.031 | 0.467 | 0.917 | 0.430 | 0.445 | 0.929 | 0.358 |
| 123 | 7 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 123 | 7 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 124 | 8 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 124 | 8 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 125 | 8 | `H` | 0.559 | 1.000 | 0.005 | 0.623 | 1.000 | 0.013 | 0.689 | 1.000 | 0.032 | 0.743 | 1.000 | 0.045 | 0.798 | 1.000 | 0.092 |
| 125 | 8 | `P` | 0.456 | 0.773 | 0.009 | 0.442 | 0.832 | 0.002 | 0.596 | 0.963 | 0.000 | 0.615 | 0.982 | 0.000 | 0.565 | 0.994 | 0.055 |
| 126 | 8 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 126 | 8 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 127 | 8 | `H` | 0.570 | 1.000 | 0.054 | 0.696 | 1.000 | 0.105 | 0.773 | 1.000 | 0.147 | 0.827 | 1.000 | 0.211 | 0.878 | 1.000 | 0.299 |
| 127 | 8 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 128 | 8 | `H` | 0.482 | 1.000 | 0.005 | 0.526 | 1.000 | 0.008 | 0.563 | 1.000 | 0.016 | 0.610 | 1.000 | 0.025 | 0.665 | 1.000 | 0.044 |
| 128 | 8 | `P` | 0.341 | 0.764 | 0.007 | 0.335 | 0.742 | 0.003 | 0.335 | 0.753 | 0.004 | 0.336 | 0.748 | 0.003 | 0.337 | 0.748 | 0.004 |
| 129 | 8 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 129 | 8 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 130 | 5 | `H` | 0.586 | 1.000 | 0.094 | 0.805 | 1.000 | 0.265 | 0.868 | 1.000 | 0.361 | 0.920 | 1.000 | 0.613 | 0.988 | 1.000 | 0.938 |
| 130 | 5 | `P` | 0.516 | 0.980 | 0.141 | 0.751 | 0.978 | 0.437 | 0.915 | 0.999 | 0.784 | 0.827 | 1.000 | 0.539 | 0.958 | 0.977 | 0.931 |
| 131 | 7 | `H` | 0.654 | 1.000 | 0.157 | 0.790 | 1.000 | 0.289 | 0.867 | 1.000 | 0.401 | 0.932 | 1.000 | 0.618 | 0.973 | 1.000 | 0.753 |
| 131 | 7 | `P` | 0.170 | 0.340 | 0.004 | 0.147 | 0.280 | 0.019 | 0.141 | 0.284 | 0.011 | 0.151 | 0.290 | 0.014 | 0.135 | 0.272 | 0.011 |
| 132 | 6 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 132 | 6 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 133 | 7 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 133 | 7 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 134 | 7 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 134 | 7 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 135 | 7 | `H` | 0.550 | 1.000 | 0.053 | 0.640 | 1.000 | 0.051 | 0.762 | 1.000 | 0.119 | 0.862 | 1.000 | 0.211 | 0.917 | 1.000 | 0.302 |
| 135 | 7 | `P` | 0.194 | 0.457 | 0.016 | 0.179 | 0.448 | 0.010 | 0.160 | 0.386 | 0.007 | 0.145 | 0.353 | 0.007 | 0.143 | 0.350 | 0.009 |
| 136 | 7 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 136 | 7 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 137 | 6 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 137 | 6 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 138 | 7 | `H` | 0.477 | 1.000 | 0.031 | 0.618 | 1.000 | 0.073 | 0.738 | 1.000 | 0.108 | 0.859 | 1.000 | 0.171 | 0.909 | 1.000 | 0.227 |
| 138 | 7 | `P` | 0.461 | 0.848 | 0.012 | 0.439 | 0.982 | 0.000 | 0.536 | 0.993 | 0.123 | 0.545 | 0.997 | 0.180 | 0.447 | 0.992 | 0.000 |
| 139 | 7 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 139 | 7 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 140 | 6 | `H` | 0.617 | 1.000 | 0.217 | 0.731 | 1.000 | 0.331 | 0.846 | 1.000 | 0.528 | 0.914 | 1.000 | 0.725 | 0.987 | 1.000 | 0.963 |
| 140 | 6 | `P` | 0.100 | 0.271 | 0.018 | 0.107 | 0.288 | 0.012 | 0.101 | 0.238 | 0.029 | 0.086 | 0.223 | 0.011 | 0.074 | 0.191 | 0.011 |
| 141 | 7 | `H` | 0.603 | 1.000 | 0.129 | 0.722 | 1.000 | 0.219 | 0.814 | 1.000 | 0.298 | 0.865 | 1.000 | 0.466 | 0.914 | 1.000 | 0.685 |
| 141 | 7 | `P` | 0.086 | 0.191 | 0.005 | 0.085 | 0.189 | 0.006 | 0.085 | 0.187 | 0.007 | 0.091 | 0.196 | 0.009 | 0.086 | 0.188 | 0.008 |
| 142 | 7 | `H` | 0.509 | 1.000 | 0.014 | 0.588 | 1.000 | 0.033 | 0.682 | 1.000 | 0.060 | 0.736 | 1.000 | 0.094 | 0.776 | 1.000 | 0.116 |
| 142 | 7 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 143 | 5 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 143 | 5 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 144 | 6 | `H` | 0.569 | 1.000 | 0.119 | 0.727 | 1.000 | 0.283 | 0.816 | 1.000 | 0.426 | 0.908 | 1.000 | 0.684 | 0.982 | 1.000 | 0.958 |
| 144 | 6 | `P` | 0.106 | 0.285 | 0.009 | 0.098 | 0.250 | 0.014 | 0.088 | 0.248 | 0.008 | 0.097 | 0.234 | 0.017 | 0.102 | 0.254 | 0.013 |
| 145 | 6 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 145 | 6 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 146 | 7 | `H` | 0.999 | 1.000 | 0.999 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 146 | 7 | `P` | 0.305 | 0.598 | 0.305 | 0.508 | 0.908 | 0.508 | 0.776 | 0.984 | 0.776 | 0.771 | 0.986 | 0.771 | 0.984 | 0.990 | 0.984 |
| 147 | 6 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 147 | 6 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 148 | 8 | `H` | 0.588 | 1.000 | 0.026 | 0.711 | 1.000 | 0.050 | 0.765 | 1.000 | 0.080 | 0.794 | 1.000 | 0.104 | 0.806 | 1.000 | 0.152 |
| 148 | 8 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 149 | 7 | `H` | 0.698 | 1.000 | 0.139 | 0.793 | 1.000 | 0.230 | 0.865 | 1.000 | 0.311 | 0.915 | 1.000 | 0.441 | 0.942 | 1.000 | 0.594 |
| 149 | 7 | `P` | 0.183 | 0.356 | 0.017 | 0.155 | 0.308 | 0.004 | 0.174 | 0.331 | 0.009 | 0.175 | 0.319 | 0.012 | 0.158 | 0.289 | 0.008 |
| 150 | 7 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 150 | 7 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 151 | 7 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 151 | 7 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 152 | 8 | `H` | 0.490 | 1.000 | 0.000 | 0.532 | 1.000 | 0.000 | 0.579 | 1.000 | 0.000 | 0.618 | 1.000 | 0.000 | 0.649 | 1.000 | 0.000 |
| 152 | 8 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 153 | 7 | `H` | 0.488 | 1.000 | 0.023 | 0.583 | 1.000 | 0.039 | 0.647 | 1.000 | 0.055 | 0.717 | 1.000 | 0.116 | 0.757 | 1.000 | 0.151 |
| 153 | 7 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 154 | 6 | `H` | 0.625 | 1.000 | 0.177 | 0.789 | 1.000 | 0.327 | 0.912 | 1.000 | 0.528 | 0.967 | 1.000 | 0.774 | 0.992 | 1.000 | 0.960 |
| 154 | 6 | `P` | 0.166 | 0.384 | 0.003 | 0.140 | 0.366 | 0.019 | 0.126 | 0.316 | 0.013 | 0.125 | 0.328 | 0.012 | 0.130 | 0.306 | 0.015 |
| 155 | 8 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 155 | 8 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 156 | 6 | `H` |  | 1.000 | 1.000 |  | 1.000 | 1.000 |  | 1.000 | 1.000 |  | 1.000 | 1.000 |  | 1.000 | 1.000 |
| 156 | 6 | `P` |  | 0.810 | 0.810 |  | 0.974 | 0.974 |  | 0.998 | 0.998 |  | 0.999 | 0.999 |  | 1.000 | 1.000 |
| 157 | 7 | `H` | 0.566 | 1.000 | 0.034 | 0.630 | 1.000 | 0.064 | 0.704 | 1.000 | 0.087 | 0.779 | 1.000 | 0.126 | 0.826 | 1.000 | 0.177 |
| 157 | 7 | `P` | 0.400 | 0.826 | 0.004 | 0.403 | 0.813 | 0.006 | 0.408 | 0.830 | 0.006 | 0.394 | 0.810 | 0.007 | 0.397 | 0.813 | 0.006 |
| 158 | 8 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 158 | 8 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 159 | 5 | `H` | 0.976 | 1.000 | 0.957 | 0.999 | 1.000 | 0.998 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 159 | 5 | `P` | 0.175 | 0.322 | 0.050 | 0.160 | 0.297 | 0.085 | 0.116 | 0.229 | 0.064 | 0.064 | 0.127 | 0.033 | 0.093 | 0.168 | 0.065 |
| 160 | 7 | `H` | 0.996 | 1.000 | 0.996 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 160 | 7 | `P` | 0.208 | 0.397 | 0.208 | 0.479 | 0.948 | 0.479 | 0.529 | 0.983 | 0.529 | 0.515 | 0.994 | 0.515 | 0.558 | 0.996 | 0.558 |
| 161 | 7 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 161 | 7 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 162 | 8 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 162 | 8 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 163 | 7 | `H` | 0.557 | 1.000 | 0.153 | 0.724 | 1.000 | 0.337 | 0.858 | 1.000 | 0.662 | 0.971 | 1.000 | 0.933 | 1.000 | 1.000 | 0.999 |
| 163 | 7 | `P` | 0.086 | 0.175 | 0.006 | 0.259 | 0.365 | 0.129 | 0.361 | 0.724 | 0.001 | 0.220 | 0.882 | 0.000 | 0.383 | 0.932 | 0.004 |
| 164 | 8 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 164 | 8 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 165 | 8 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 165 | 8 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 166 | 7 | `H` | 0.580 | 1.000 | 0.021 | 0.683 | 1.000 | 0.035 | 0.734 | 1.000 | 0.061 | 0.795 | 1.000 | 0.099 | 0.841 | 1.000 | 0.143 |
| 166 | 7 | `P` | 0.467 | 0.748 | 0.018 | 0.445 | 0.849 | 0.001 | 0.603 | 0.940 | 0.014 | 0.559 | 0.983 | 0.000 | 0.550 | 0.989 | 0.001 |
| 167 | 7 | `H` | 0.811 | 1.000 | 0.171 | 0.916 | 1.000 | 0.379 | 0.977 | 1.000 | 0.525 | 0.986 | 1.000 | 0.686 | 0.997 | 1.000 | 0.942 |
| 167 | 7 | `P` | 0.197 | 0.337 | 0.040 | 0.099 | 0.183 | 0.004 | 0.113 | 0.171 | 0.005 | 0.100 | 0.159 | 0.008 | 0.097 | 0.147 | 0.009 |
| 168 | 6 | `H` | 0.936 | 1.000 | 0.813 | 0.987 | 1.000 | 0.956 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 168 | 6 | `P` | 0.124 | 0.253 | 0.064 | 0.102 | 0.203 | 0.051 | 0.087 | 0.211 | 0.045 | 0.123 | 0.305 | 0.065 | 0.070 | 0.168 | 0.033 |
| 169 | 6 | `H` | 0.496 | 1.000 | 0.028 | 0.581 | 1.000 | 0.046 | 0.714 | 1.000 | 0.090 | 0.780 | 1.000 | 0.148 | 0.828 | 1.000 | 0.222 |
| 169 | 6 | `P` | 0.444 | 1.000 | 0.006 | 0.712 | 1.000 | 0.173 | 0.683 | 1.000 | 0.215 | 0.601 | 1.000 | 0.019 | 0.568 | 1.000 | 0.002 |
| 170 | 8 | `H` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 170 | 8 | `P` | 0.337 | 0.385 | 0.200 | 0.677 | 0.776 | 0.381 | 0.853 | 0.916 | 0.664 | 0.710 | 0.885 | 0.189 | 0.798 | 0.943 | 0.365 |
| 171 | 8 | `H` | 0.520 | 1.000 | 0.046 | 0.669 | 1.000 | 0.085 | 0.783 | 1.000 | 0.164 | 0.869 | 1.000 | 0.246 | 0.914 | 1.000 | 0.345 |
| 171 | 8 | `P` | 0.303 | 0.583 | 0.027 | 0.435 | 0.878 | 0.003 | 0.545 | 0.920 | 0.194 | 0.473 | 0.953 | 0.000 | 0.471 | 0.942 | 0.234 |
| 172 | 6 | `H` | 0.730 | 1.000 | 0.245 | 0.856 | 1.000 | 0.360 | 0.963 | 1.000 | 0.633 | 0.989 | 1.000 | 0.842 | 0.998 | 1.000 | 0.976 |
| 172 | 6 | `P` | 0.117 | 0.253 | 0.010 | 0.130 | 0.276 | 0.019 | 0.111 | 0.239 | 0.015 | 0.128 | 0.265 | 0.015 | 0.135 | 0.281 | 0.022 |
| 173 | 8 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 173 | 8 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 174 | 8 | `H` | 0.946 | 1.000 | 0.839 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 174 | 8 | `P` | 0.181 | 0.288 | 0.081 | 0.737 | 0.846 | 0.424 | 0.680 | 0.898 | 0.048 | 0.883 | 0.946 | 0.730 | 0.727 | 0.977 | 0.009 |
| 175 | 8 | `H` | 0.638 | 1.000 | 0.057 | 0.767 | 1.000 | 0.146 | 0.870 | 1.000 | 0.237 | 0.910 | 1.000 | 0.300 | 0.936 | 1.000 | 0.420 |
| 175 | 8 | `P` | 0.075 | 0.134 | 0.001 | 0.122 | 0.228 | 0.006 | 0.089 | 0.174 | 0.004 | 0.094 | 0.174 | 0.005 | 0.090 | 0.175 | 0.003 |
| 176 | 7 | `H` | 0.848 | 1.000 | 0.631 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 176 | 7 | `P` | 0.215 | 0.432 | 0.001 | 0.413 | 0.827 | 0.001 | 0.454 | 0.924 | 0.003 | 0.474 | 0.953 | 0.000 | 0.722 | 0.884 | 0.424 |
| 177 | 6 | `H` | 0.466 | 1.000 | 0.080 | 0.620 | 1.000 | 0.162 | 0.725 | 1.000 | 0.257 | 0.833 | 1.000 | 0.436 | 0.927 | 1.000 | 0.719 |
| 177 | 6 | `P` | 0.189 | 0.470 | 0.009 | 0.183 | 0.498 | 0.018 | 0.164 | 0.464 | 0.014 | 0.175 | 0.469 | 0.014 | 0.180 | 0.485 | 0.020 |
| 178 | 6 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 178 | 6 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 179 | 8 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 179 | 8 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 180 | 8 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 180 | 8 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 181 | 8 | `H` | 0.983 | 1.000 | 0.964 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 181 | 8 | `P` | 0.192 | 0.227 | 0.116 | 0.745 | 0.867 | 0.439 | 0.844 | 0.984 | 0.494 | 0.822 | 0.955 | 0.491 | 0.499 | 0.963 | 0.499 |
| 182 | 8 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 182 | 8 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 183 | 8 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 183 | 8 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 184 | 7 | `H` | 0.911 | 1.000 | 0.854 | 0.999 | 1.000 | 0.999 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 184 | 7 | `P` | 0.138 | 0.327 | 0.004 | 0.289 | 0.536 | 0.282 | 0.304 | 0.716 | 0.015 | 0.605 | 0.882 | 0.502 | 0.310 | 0.664 | 0.176 |
| 185 | 8 | `H` | 0.450 | 1.000 | 0.005 | 0.498 | 1.000 | 0.007 | 0.575 | 1.000 | 0.013 | 0.631 | 1.000 | 0.017 | 0.671 | 1.000 | 0.028 |
| 185 | 8 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 186 | 5 | `H` | 0.509 | 1.000 | 0.120 | 0.621 | 1.000 | 0.232 | 0.746 | 1.000 | 0.373 | 0.848 | 1.000 | 0.592 | 0.934 | 1.000 | 0.838 |
| 186 | 5 | `P` | 0.397 | 0.933 | 0.005 | 0.715 | 0.998 | 0.469 | 0.772 | 1.000 | 0.506 | 0.518 | 1.000 | 0.235 | 0.625 | 0.995 | 0.136 |
| 187 | 6 | `H` | 0.840 | 1.000 | 0.225 | 0.884 | 1.000 | 0.297 | 0.927 | 1.000 | 0.460 | 0.955 | 1.000 | 0.644 | 0.980 | 1.000 | 0.825 |
| 187 | 6 | `P` | 0.323 | 0.484 | 0.033 | 0.894 | 0.936 | 0.852 | 0.924 | 0.980 | 0.507 | 0.901 | 0.965 | 0.464 | 0.774 | 0.980 | 0.309 |
| 188 | 8 | `H` | 0.649 | 1.000 | 0.119 | 0.756 | 1.000 | 0.187 | 0.852 | 1.000 | 0.313 | 0.913 | 1.000 | 0.516 | 0.953 | 1.000 | 0.719 |
| 188 | 8 | `P` | 0.096 | 0.234 | 0.003 | 0.077 | 0.193 | 0.005 | 0.068 | 0.173 | 0.004 | 0.077 | 0.183 | 0.004 | 0.074 | 0.183 | 0.005 |
| 189 | 8 | `H` | 0.664 | 1.000 | 0.037 | 0.753 | 1.000 | 0.090 | 0.829 | 1.000 | 0.183 | 0.878 | 1.000 | 0.245 | 0.917 | 1.000 | 0.291 |
| 189 | 8 | `P` | 0.456 | 0.670 | 0.048 | 0.750 | 0.923 | 0.213 | 0.528 | 0.800 | 0.108 | 0.583 | 0.874 | 0.070 | 0.709 | 0.962 | 0.078 |
| 190 | 6 | `H` | 0.544 | 1.000 | 0.024 | 0.655 | 1.000 | 0.053 | 0.739 | 1.000 | 0.100 | 0.797 | 1.000 | 0.144 | 0.834 | 1.000 | 0.189 |
| 190 | 6 | `P` | 0.324 | 0.580 | 0.026 | 0.297 | 0.566 | 0.035 | 0.299 | 0.591 | 0.013 | 0.296 | 0.566 | 0.016 | 0.306 | 0.600 | 0.023 |
| 191 | 6 | `H` | 0.810 | 1.000 | 0.332 | 0.942 | 1.000 | 0.566 | 0.989 | 1.000 | 0.846 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 191 | 6 | `P` | 0.169 | 0.310 | 0.044 | 0.100 | 0.208 | 0.014 | 0.097 | 0.186 | 0.021 | 0.101 | 0.189 | 0.020 | 0.103 | 0.198 | 0.019 |
| 192 | 6 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 192 | 6 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 193 | 8 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 193 | 8 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 194 | 5 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 194 | 5 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 195 | 8 | `H` | 0.590 | 1.000 | 0.048 | 0.784 | 1.000 | 0.155 | 0.898 | 1.000 | 0.301 | 0.922 | 1.000 | 0.472 | 0.957 | 1.000 | 0.746 |
| 195 | 8 | `P` | 0.038 | 0.113 | 0.012 | 0.071 | 0.180 | 0.032 | 0.016 | 0.048 | 0.002 | 0.011 | 0.029 | 0.002 | 0.025 | 0.069 | 0.005 |
| 196 | 8 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 196 | 8 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 197 | 6 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 197 | 6 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 198 | 7 | `H` | 0.849 | 1.000 | 0.207 | 0.938 | 1.000 | 0.525 | 0.980 | 1.000 | 0.825 | 1.000 | 1.000 | 0.999 | 1.000 | 1.000 | 1.000 |
| 198 | 7 | `P` | 0.263 | 0.503 | 0.116 | 0.822 | 0.935 | 0.627 | 0.983 | 0.995 | 0.977 | 0.995 | 0.999 | 0.993 | 0.999 | 0.999 | 0.998 |
| 199 | 7 | `H` | 0.603 | 1.000 | 0.100 | 0.753 | 1.000 | 0.226 | 0.854 | 1.000 | 0.352 | 0.926 | 1.000 | 0.467 | 0.958 | 1.000 | 0.631 |
| 199 | 7 | `P` | 0.250 | 0.474 | 0.016 | 0.417 | 0.730 | 0.059 | 0.441 | 0.915 | 0.012 | 0.572 | 0.951 | 0.311 | 0.556 | 0.981 | 0.224 |
| 200 | 8 | `H` | 0.623 | 1.000 | 0.006 | 0.714 | 1.000 | 0.044 | 0.786 | 1.000 | 0.065 | 0.850 | 1.000 | 0.118 | 0.911 | 1.000 | 0.197 |
| 200 | 8 | `P` | 0.143 | 0.261 | 0.010 | 0.097 | 0.178 | 0.002 | 0.102 | 0.187 | 0.004 | 0.105 | 0.191 | 0.005 | 0.105 | 0.193 | 0.005 |
| 201 | 7 | `H` | 0.525 | 1.000 | 0.005 | 0.636 | 1.000 | 0.028 | 0.713 | 1.000 | 0.052 | 0.788 | 1.000 | 0.082 | 0.834 | 1.000 | 0.107 |
| 201 | 7 | `P` | 0.275 | 0.571 | 0.003 | 0.246 | 0.554 | 0.006 | 0.260 | 0.566 | 0.007 | 0.251 | 0.563 | 0.009 | 0.267 | 0.583 | 0.006 |
| 202 | 8 | `H` | 0.568 | 1.000 | 0.025 | 0.662 | 1.000 | 0.071 | 0.726 | 1.000 | 0.093 | 0.782 | 1.000 | 0.145 | 0.824 | 1.000 | 0.203 |
| 202 | 8 | `P` | 0.150 | 0.303 | 0.004 | 0.136 | 0.282 | 0.003 | 0.121 | 0.249 | 0.003 | 0.120 | 0.245 | 0.003 | 0.129 | 0.264 | 0.004 |
| 203 | 7 | `H` | 0.747 | 1.000 | 0.185 | 0.865 | 1.000 | 0.342 | 0.936 | 1.000 | 0.527 | 0.961 | 1.000 | 0.693 | 0.990 | 1.000 | 0.929 |
| 203 | 7 | `P` | 0.073 | 0.261 | 0.011 | 0.089 | 0.309 | 0.021 | 0.100 | 0.335 | 0.022 | 0.074 | 0.158 | 0.013 | 0.054 | 0.110 | 0.006 |
| 204 | 7 | `H` | 0.566 | 1.000 | 0.024 | 0.657 | 1.000 | 0.038 | 0.730 | 1.000 | 0.067 | 0.803 | 1.000 | 0.082 | 0.864 | 1.000 | 0.102 |
| 204 | 7 | `P` | 0.292 | 0.620 | 0.009 | 0.290 | 0.608 | 0.008 | 0.296 | 0.619 | 0.007 | 0.296 | 0.614 | 0.008 | 0.296 | 0.621 | 0.008 |
| 205 | 6 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 205 | 6 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 206 | 7 | `H` | 0.519 | 1.000 | 0.007 | 0.575 | 1.000 | 0.016 | 0.622 | 1.000 | 0.019 | 0.689 | 1.000 | 0.025 | 0.748 | 1.000 | 0.059 |
| 206 | 7 | `P` | 0.301 | 0.624 | 0.007 | 0.299 | 0.629 | 0.007 | 0.295 | 0.623 | 0.007 | 0.299 | 0.634 | 0.010 | 0.303 | 0.631 | 0.009 |
| 207 | 7 | `H` | 0.791 | 1.000 | 0.244 | 0.894 | 1.000 | 0.667 | 0.992 | 1.000 | 0.974 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 207 | 7 | `P` | 0.320 | 0.609 | 0.148 | 0.469 | 0.878 | 0.000 | 0.346 | 0.926 | 0.000 | 0.562 | 0.965 | 0.001 | 0.506 | 0.966 | 0.238 |
| 208 | 8 | `H` | 0.788 | 1.000 | 0.210 | 0.939 | 1.000 | 0.483 | 0.981 | 1.000 | 0.830 | 1.000 | 1.000 | 0.999 | 1.000 | 1.000 | 1.000 |
| 208 | 8 | `P` | 0.298 | 0.473 | 0.000 | 0.502 | 0.817 | 0.002 | 0.441 | 0.774 | 0.014 | 0.581 | 0.945 | 0.002 | 0.540 | 0.928 | 0.002 |
| 209 | 6 | `H` | 0.553 | 1.000 | 0.069 | 0.685 | 1.000 | 0.136 | 0.796 | 1.000 | 0.243 | 0.872 | 1.000 | 0.296 | 0.916 | 1.000 | 0.444 |
| 209 | 6 | `P` | 0.198 | 0.438 | 0.039 | 0.200 | 0.478 | 0.011 | 0.181 | 0.428 | 0.015 | 0.190 | 0.409 | 0.017 | 0.186 | 0.414 | 0.017 |
| 210 | 8 | `H` | 0.378 | 1.000 | 0.002 | 0.421 | 1.000 | 0.008 | 0.481 | 1.000 | 0.020 | 0.552 | 1.000 | 0.040 | 0.614 | 1.000 | 0.058 |
| 210 | 8 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 211 | 7 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 211 | 7 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 212 | 5 | `H` | 0.637 | 1.000 | 0.206 | 0.852 | 1.000 | 0.388 | 0.917 | 1.000 | 0.604 | 0.990 | 1.000 | 0.966 | 1.000 | 1.000 | 1.000 |
| 212 | 5 | `P` | 0.200 | 0.800 | 0.000 | 0.270 | 0.951 | 0.004 | 0.226 | 0.973 | 0.003 | 0.154 | 0.962 | 0.000 | 0.208 | 0.977 | 0.001 |
| 213 | 6 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 213 | 6 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 214 | 6 | `H` | 0.703 | 1.000 | 0.158 | 0.922 | 1.000 | 0.784 | 0.997 | 1.000 | 0.994 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 214 | 6 | `P` | 0.176 | 0.406 | 0.073 | 0.417 | 0.914 | 0.230 | 0.627 | 0.975 | 0.406 | 0.738 | 0.976 | 0.489 | 0.814 | 0.990 | 0.697 |
| 215 | 8 | `H` | 0.797 | 1.000 | 0.246 | 0.881 | 1.000 | 0.459 | 0.957 | 1.000 | 0.778 | 0.994 | 1.000 | 0.969 | 1.000 | 1.000 | 1.000 |
| 215 | 8 | `P` | 0.082 | 0.110 | 0.038 | 0.182 | 0.266 | 0.001 | 0.254 | 0.343 | 0.099 | 0.265 | 0.412 | 0.001 | 0.326 | 0.435 | 0.000 |
| 216 | 6 | `H` | 0.674 | 1.000 | 0.133 | 0.806 | 1.000 | 0.216 | 0.890 | 1.000 | 0.416 | 0.945 | 1.000 | 0.576 | 0.977 | 1.000 | 0.733 |
| 216 | 6 | `P` | 0.334 | 0.612 | 0.007 | 0.252 | 0.408 | 0.014 | 0.346 | 0.646 | 0.022 | 0.324 | 0.557 | 0.040 | 0.289 | 0.478 | 0.038 |
| 217 | 7 | `H` | 0.531 | 1.000 | 0.028 | 0.644 | 1.000 | 0.049 | 0.735 | 1.000 | 0.112 | 0.795 | 1.000 | 0.176 | 0.859 | 1.000 | 0.265 |
| 217 | 7 | `P` | 0.451 | 0.848 | 0.009 | 0.641 | 0.915 | 0.237 | 0.680 | 0.971 | 0.206 | 0.559 | 0.999 | 0.000 | 0.589 | 0.978 | 0.000 |
| 218 | 6 | `H` | 0.647 | 1.000 | 0.098 | 0.790 | 1.000 | 0.200 | 0.894 | 1.000 | 0.305 | 0.941 | 1.000 | 0.468 | 0.970 | 1.000 | 0.716 |
| 218 | 6 | `P` | 0.412 | 0.842 | 0.054 | 0.457 | 0.894 | 0.186 | 0.479 | 0.968 | 0.178 | 0.464 | 0.961 | 0.120 | 0.528 | 0.990 | 0.198 |
| 219 | 8 | `H` | 0.529 | 1.000 | 0.051 | 0.633 | 1.000 | 0.097 | 0.720 | 1.000 | 0.156 | 0.793 | 1.000 | 0.198 | 0.854 | 1.000 | 0.279 |
| 219 | 8 | `P` | 0.115 | 0.296 | 0.003 | 0.109 | 0.284 | 0.004 | 0.107 | 0.278 | 0.003 | 0.111 | 0.281 | 0.005 | 0.111 | 0.284 | 0.004 |
| 220 | 8 | `H` | 0.536 | 1.000 | 0.035 | 0.641 | 1.000 | 0.065 | 0.733 | 1.000 | 0.139 | 0.814 | 1.000 | 0.189 | 0.873 | 1.000 | 0.223 |
| 220 | 8 | `P` | 0.197 | 0.429 | 0.001 | 0.188 | 0.404 | 0.001 | 0.137 | 0.305 | 0.005 | 0.144 | 0.316 | 0.004 | 0.130 | 0.291 | 0.005 |
| 221 | 8 | `H` | 0.624 | 1.000 | 0.138 | 0.809 | 1.000 | 0.303 | 0.906 | 1.000 | 0.451 | 0.950 | 1.000 | 0.585 | 0.970 | 1.000 | 0.741 |
| 221 | 8 | `P` | 0.040 | 0.088 | 0.005 | 0.037 | 0.082 | 0.003 | 0.039 | 0.086 | 0.004 | 0.041 | 0.086 | 0.005 | 0.039 | 0.086 | 0.004 |
| 222 | 5 | `H` | 0.621 | 1.000 | 0.190 | 0.777 | 1.000 | 0.362 | 0.863 | 1.000 | 0.487 | 0.906 | 1.000 | 0.624 | 0.963 | 1.000 | 0.862 |
| 222 | 5 | `P` | 0.222 | 0.645 | 0.034 | 0.219 | 0.624 | 0.029 | 0.241 | 0.637 | 0.034 | 0.243 | 0.637 | 0.032 | 0.229 | 0.637 | 0.034 |
| 223 | 6 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 223 | 6 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 224 | 5 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 224 | 5 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 225 | 6 | `H` | 0.534 | 1.000 | 0.062 | 0.667 | 1.000 | 0.075 | 0.771 | 1.000 | 0.133 | 0.848 | 1.000 | 0.149 | 0.902 | 1.000 | 0.220 |
| 225 | 6 | `P` | 0.345 | 0.749 | 0.032 | 0.340 | 0.752 | 0.026 | 0.362 | 0.743 | 0.065 | 0.325 | 0.747 | 0.031 | 0.354 | 0.796 | 0.020 |
| 226 | 7 | `H` | 0.619 | 1.000 | 0.141 | 0.755 | 1.000 | 0.259 | 0.864 | 1.000 | 0.419 | 0.935 | 1.000 | 0.683 | 0.982 | 1.000 | 0.904 |
| 226 | 7 | `P` | 0.120 | 0.343 | 0.011 | 0.182 | 0.355 | 0.029 | 0.125 | 0.251 | 0.007 | 0.107 | 0.182 | 0.017 | 0.140 | 0.304 | 0.017 |
| 227 | 8 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 227 | 8 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 228 | 7 | `H` | 0.829 | 1.000 | 0.258 | 0.955 | 1.000 | 0.434 | 0.978 | 1.000 | 0.658 | 0.990 | 1.000 | 0.835 | 1.000 | 1.000 | 0.998 |
| 228 | 7 | `P` | 0.401 | 0.496 | 0.171 | 0.802 | 0.956 | 0.451 | 0.847 | 0.996 | 0.491 | 0.851 | 0.999 | 0.497 | 0.854 | 0.999 | 0.504 |
| 229 | 7 | `H` | 0.715 | 1.000 | 0.138 | 0.842 | 1.000 | 0.240 | 0.934 | 1.000 | 0.349 | 0.969 | 1.000 | 0.613 | 0.991 | 1.000 | 0.904 |
| 229 | 7 | `P` | 0.604 | 0.693 | 0.067 | 0.898 | 0.960 | 0.462 | 0.917 | 0.986 | 0.298 | 0.952 | 0.992 | 0.495 | 0.958 | 0.994 | 0.489 |
| 230 | 6 | `H` | 0.475 | 1.000 | 0.025 | 0.597 | 1.000 | 0.051 | 0.701 | 1.000 | 0.111 | 0.789 | 1.000 | 0.166 | 0.855 | 1.000 | 0.220 |
| 230 | 6 | `P` | 0.371 | 1.000 | 0.005 | 0.489 | 1.000 | 0.011 | 0.607 | 1.000 | 0.172 | 0.639 | 1.000 | 0.003 | 0.660 | 1.000 | 0.025 |
| 231 | 8 | `H` | 0.929 | 1.000 | 0.700 | 0.998 | 1.000 | 0.995 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 231 | 8 | `P` | 0.251 | 0.345 | 0.011 | 0.355 | 0.575 | 0.000 | 0.522 | 0.698 | 0.001 | 0.523 | 0.826 | 0.002 | 0.663 | 0.881 | 0.011 |
| 232 | 4 | `H` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 232 | 4 | `P` | 0.428 | 0.818 | 0.428 | 0.925 | 0.926 | 0.925 | 0.999 | 0.999 | 0.999 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 233 | 8 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 233 | 8 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 234 | 8 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 234 | 8 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 235 | 8 | `H` | 0.685 | 1.000 | 0.129 | 0.833 | 1.000 | 0.168 | 0.895 | 1.000 | 0.268 | 0.914 | 1.000 | 0.345 | 0.943 | 1.000 | 0.549 |
| 235 | 8 | `P` | 0.282 | 0.489 | 0.000 | 0.546 | 0.929 | 0.000 | 0.585 | 0.969 | 0.000 | 0.562 | 0.987 | 0.000 | 0.584 | 0.992 | 0.000 |
| 236 | 8 | `H` |  | 1.000 | 1.000 |  | 1.000 | 1.000 |  | 1.000 | 1.000 |  | 1.000 | 1.000 |  | 1.000 | 1.000 |
| 236 | 8 | `P` |  | 0.226 | 0.226 |  | 0.719 | 0.719 |  | 0.954 | 0.954 |  | 0.985 | 0.985 |  | 0.994 | 0.994 |
| 237 | 6 | `H` | 0.626 | 1.000 | 0.174 | 0.791 | 1.000 | 0.325 | 0.888 | 1.000 | 0.522 | 0.959 | 1.000 | 0.777 | 0.995 | 1.000 | 0.978 |
| 237 | 6 | `P` | 0.461 | 0.838 | 0.198 | 0.572 | 0.990 | 0.229 | 0.766 | 0.950 | 0.170 | 0.906 | 0.988 | 0.462 | 0.924 | 0.996 | 0.498 |
| 238 | 7 | `H` | 0.698 | 1.000 | 0.100 | 0.801 | 1.000 | 0.201 | 0.869 | 1.000 | 0.282 | 0.918 | 1.000 | 0.406 | 0.962 | 1.000 | 0.525 |
| 238 | 7 | `P` | 0.282 | 0.439 | 0.023 | 0.292 | 0.461 | 0.021 | 0.275 | 0.438 | 0.021 | 0.277 | 0.437 | 0.025 | 0.283 | 0.445 | 0.028 |
| 239 | 8 | `H` | 0.644 | 1.000 | 0.145 | 0.791 | 1.000 | 0.296 | 0.903 | 1.000 | 0.453 | 0.954 | 1.000 | 0.553 | 0.977 | 1.000 | 0.665 |
| 239 | 8 | `P` | 0.073 | 0.177 | 0.001 | 0.069 | 0.197 | 0.001 | 0.099 | 0.226 | 0.000 | 0.045 | 0.109 | 0.003 | 0.054 | 0.128 | 0.002 |
| 240 | 8 | `H` | 0.502 | 1.000 | 0.018 | 0.578 | 1.000 | 0.031 | 0.643 | 1.000 | 0.057 | 0.698 | 1.000 | 0.080 | 0.747 | 1.000 | 0.111 |
| 240 | 8 | `P` | 0.289 | 0.620 | 0.005 | 0.298 | 0.629 | 0.007 | 0.305 | 0.635 | 0.010 | 0.310 | 0.624 | 0.014 | 0.316 | 0.663 | 0.007 |
| 241 | 7 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 241 | 7 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 242 | 7 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 242 | 7 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 243 | 8 | `H` | 0.435 | 1.000 | 0.014 | 0.549 | 1.000 | 0.036 | 0.640 | 1.000 | 0.046 | 0.699 | 1.000 | 0.077 | 0.734 | 1.000 | 0.108 |
| 243 | 8 | `P` | 0.087 | 0.243 | 0.004 | 0.091 | 0.259 | 0.004 | 0.089 | 0.246 | 0.005 | 0.088 | 0.247 | 0.003 | 0.087 | 0.249 | 0.003 |
| 244 | 8 | `H` | 0.531 | 1.000 | 0.020 | 0.624 | 1.000 | 0.046 | 0.716 | 1.000 | 0.074 | 0.796 | 1.000 | 0.138 | 0.828 | 1.000 | 0.184 |
| 244 | 8 | `P` | 0.139 | 0.275 | 0.005 | 0.159 | 0.339 | 0.004 | 0.140 | 0.295 | 0.003 | 0.133 | 0.281 | 0.004 | 0.134 | 0.285 | 0.002 |
| 245 | 5 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 245 | 5 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 246 | 6 | `H` | 0.670 | 1.000 | 0.053 | 0.812 | 1.000 | 0.120 | 0.895 | 1.000 | 0.240 | 0.946 | 1.000 | 0.289 | 0.971 | 1.000 | 0.366 |
| 246 | 6 | `P` | 0.221 | 0.431 | 0.009 | 0.208 | 0.377 | 0.026 | 0.213 | 0.394 | 0.019 | 0.197 | 0.388 | 0.012 | 0.209 | 0.381 | 0.018 |
| 247 | 7 | `H` | 0.676 | 1.000 | 0.078 | 0.820 | 1.000 | 0.152 | 0.895 | 1.000 | 0.275 | 0.946 | 1.000 | 0.377 | 0.967 | 1.000 | 0.468 |
| 247 | 7 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 248 | 7 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 248 | 7 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 249 | 8 | `H` | 0.979 | 1.000 | 0.837 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 249 | 8 | `P` | 0.088 | 0.098 | 0.031 | 0.381 | 0.541 | 0.006 | 0.472 | 0.545 | 0.003 | 0.646 | 0.830 | 0.000 | 0.846 | 0.846 | 0.845 |

---

*Generated by `analyze_results/generate_results_markdown.py`.*
