# Overlapping Problem Set — Raw Results

**250 problem instances** (overlapping constraint variable supports).
Empty cells indicate runs that did not complete (transient cluster errors).

---

## Problem Definitions

| COP | $n_x$ | Families | Structural constraints | Penalized constraints | Qubits (H/P) | SP gates (H/P) | Layer gates (H/P) |
|-----|-------|----------|------------------------|-----------------------|-------------|---------------|------------------|
| 0 | 5 | cardinality, quadratic_knapsack | $x_{1}+ x_{4} \leq 0$ *(cardinality)* | $3x_{1}^2+ x_{1}x_{0}+ 3x_{1}x_{2}$<br>$+ 4x_{1}x_{3} + 5x_{0}^2$<br>$+ 4x_{0}x_{2} + 2x_{0}x_{3}$<br>$+ 2x_{2}^2 + 4x_{2}x_{3}+ 4x_{3}^2 \leq 17$ *(quadratic_knapsack)* | / | / | / |
| 1 | 4 | cardinality, cardinality, cardinality | $x_{1}+ x_{2} \leq 1$ *(cardinality)* | $x_{0}+ x_{2} \leq 0$ *(cardinality)*<br>$x_{1}+ x_{2}+ x_{3} = 2$ *(cardinality)* | / | / | / |
| 2 | 7 | quadratic_knapsack, cardinality, quadratic_knapsack | $x_{4}+ x_{2} \leq 2$ *(cardinality)* | $x_{4}^2+ 4x_{4}x_{0}+ x_{4}x_{3}$<br>$+ 2x_{4}x_{5} + 5x_{0}^2$<br>$+ 4x_{0}x_{3} + 3x_{0}x_{5}$<br>$+ 5x_{3}^2 + 4x_{3}x_{5}+ 2x_{5}^2 \leq 18$ *(quadratic_knapsack)*<br>$5x_{6}^2+ 3x_{6}x_{5}+ 5x_{6}x_{4}$<br>$+ 4x_{5}^2 + 5x_{5}x_{4}+ 3x_{4}^2 \leq 14$ *(quadratic_knapsack)* | / | / | / |
| 3 | 3 | quadratic_knapsack, cardinality | $x_{2}^2+ 2x_{2}x_{0}+ x_{2}x_{1}$<br>$+ x_{0}^2 + 5x_{0}x_{1}+ x_{1}^2 \leq 6$ *(quadratic_knapsack)* | $x_{0}+ x_{2} \geq 2$ *(cardinality)* | / | / | / |
| 4 | 6 | cardinality, cardinality, flow | $x_{1}+ x_{4}- x_{2}- x_{0} = 0$ *(flow)* | $x_{5}+ x_{2}+ x_{0}+ x_{3}+ x_{4} \geq 0$ *(cardinality)*<br>$x_{2}+ x_{3}+ x_{5}+ x_{0} \geq 3$ *(cardinality)* | / | / | / |
| 5 | 8 | quadratic_knapsack, flow, knapsack | $x_{6}+ x_{4}- x_{3} = 0$ *(flow)* | $5x_{2}^2+ 3x_{2}x_{7}+ 4x_{2}x_{1}$<br>$+ 5x_{2}x_{5} + x_{2}x_{6}+ x_{7}^2$<br>$+ 4x_{7}x_{1} + 2x_{7}x_{5}$<br>$+ 3x_{7}x_{6} + 4x_{1}^2$<br>$+ 5x_{1}x_{5} + x_{1}x_{6}+ x_{5}^2$<br>$+ 5x_{5}x_{6} + 2x_{6}^2 \leq 17$ *(quadratic_knapsack)*<br>$6x_{5}+ 7x_{1}+ 8x_{6}+ 9x_{4}$<br>$+ 5x_{3} \leq 14$ *(knapsack)* | / | / | / |
| 6 | 5 | quadratic_knapsack, cardinality | $x_{3}+ x_{4}+ x_{2} \leq 1$ *(cardinality)* | $2x_{1}^2+ 5x_{1}x_{3}+ 4x_{1}x_{2}$<br>$+ 4x_{1}x_{0} + 3x_{1}x_{4}$<br>$+ 4x_{3}^2 + 4x_{3}x_{2}$<br>$+ x_{3}x_{0} + 2x_{3}x_{4}$<br>$+ 3x_{2}^2 + x_{2}x_{0}$<br>$+ 4x_{2}x_{4} + 3x_{0}^2$<br>$+ 4x_{0}x_{4} + 3x_{4}^2 \leq 16$ *(quadratic_knapsack)* | / | / | / |
| 7 | 3 | quadratic_knapsack, quadratic_knapsack | $2x_{2}^2+ 5x_{2}x_{1}+ 2x_{2}x_{0}$<br>$+ 2x_{1}^2 + x_{1}x_{0}+ x_{0}^2 \leq 7$ *(quadratic_knapsack)* | $3x_{0}^2+ x_{0}x_{1}+ 3x_{0}x_{2}$<br>$+ 5x_{1}^2 + 2x_{1}x_{2}+ 2x_{2}^2 \leq 8$ *(quadratic_knapsack)* | / | / | / |
| 8 | 8 | cardinality, flow, quadratic_knapsack | $x_{4}+ x_{7} = 0$ *(cardinality)*<br>$3x_{5}^2+ x_{5}x_{6}+ 3x_{5}x_{1}$<br>$+ 5x_{6}^2 + 2x_{6}x_{1}+ 2x_{1}^2 \leq 8$ *(quadratic_knapsack)* | $x_{2}+ x_{4}+ x_{0}- x_{7}- x_{1}$<br>$- x_{5} = 0$ *(flow)* | / | / | / |
| 9 | 6 | quadratic_knapsack, cardinality, independent_set | $3x_{1}^2+ x_{1}x_{3}+ 5x_{1}x_{5}$<br>$+ 2x_{1}x_{4} + 5x_{1}x_{2}$<br>$+ 2x_{3}^2 + 5x_{3}x_{5}$<br>$+ 5x_{3}x_{4} + 5x_{3}x_{2}$<br>$+ 2x_{5}^2 + 3x_{5}x_{4}$<br>$+ 3x_{5}x_{2} + 4x_{4}^2$<br>$+ x_{4}x_{2} + 5x_{2}^2 \leq 18$ *(quadratic_knapsack)* | $x_{1}+ x_{4} \geq 2$ *(cardinality)*<br>$x_{5}x_{1} = 0$ *(independent_set)* | / | / | / |
| 10 | 8 | quadratic_knapsack, knapsack, cardinality | $x_{3}+ x_{0}+ x_{6}+ x_{5}+ x_{1} \leq 5$ *(cardinality)* | $x_{7}^2+ 5x_{7}x_{0}+ 3x_{7}x_{1}$<br>$+ 4x_{0}^2 + 2x_{0}x_{1}+ 4x_{1}^2 \leq 7$ *(quadratic_knapsack)*<br>$x_{3}+ 6x_{0}+ 9x_{2}+ 8x_{6}$<br>$+ 9x_{1} \leq 18$ *(knapsack)* | / | / | / |
| 11 | 5 | quadratic_knapsack, cardinality | $x_{4}+ x_{1}+ x_{2}+ x_{3}+ x_{0} \leq 0$ *(cardinality)* | $2x_{4}^2+ 5x_{4}x_{1}+ 4x_{4}x_{3}$<br>$+ 4x_{4}x_{2} + 3x_{4}x_{0}$<br>$+ 4x_{1}^2 + 4x_{1}x_{3}$<br>$+ x_{1}x_{2} + 2x_{1}x_{0}$<br>$+ 3x_{3}^2 + x_{3}x_{2}$<br>$+ 4x_{3}x_{0} + 3x_{2}^2$<br>$+ 4x_{2}x_{0} + 3x_{0}^2 \leq 16$ *(quadratic_knapsack)* | / | / | / |
| 12 | 6 | cardinality, quadratic_knapsack | $x_{0}+ x_{1}+ x_{4}+ x_{5}+ x_{3} \geq 3$ *(cardinality)* | $5x_{1}^2+ 3x_{1}x_{3}+ 5x_{1}x_{5}$<br>$+ 4x_{3}^2 + 5x_{3}x_{5}+ 3x_{5}^2 \leq 14$ *(quadratic_knapsack)* | / | / | / |
| 13 | 5 | cardinality, quadratic_knapsack, flow | $x_{3}+ x_{1}- x_{4}- x_{0} = 0$ *(flow)* | $x_{3}+ x_{0}+ x_{4}+ x_{2} \geq 0$ *(cardinality)*<br>$4x_{0}^2+ x_{0}x_{1}+ 2x_{0}x_{4}$<br>$+ 4x_{0}x_{3} + x_{0}x_{2}$<br>$+ 2x_{1}^2 + 3x_{1}x_{4}$<br>$+ x_{1}x_{3} + x_{1}x_{2}+ 3x_{4}^2$<br>$+ 3x_{4}x_{3} + 4x_{4}x_{2}$<br>$+ x_{3}^2 + 3x_{3}x_{2}+ 2x_{2}^2 \leq 11$ *(quadratic_knapsack)* | / | / | / |
| 14 | 8 | cardinality, cardinality, cardinality | $x_{4}+ x_{3} = 0$ *(cardinality)*<br>$x_{7}+ x_{1}+ x_{6}+ x_{2}+ x_{5} \geq 1$ *(cardinality)* | $x_{0}+ x_{4}+ x_{5}+ x_{6} \geq 2$ *(cardinality)* | / | / | / |
| 15 | 8 | cardinality, cardinality, knapsack | $x_{2}+ x_{0} \geq 1$ *(cardinality)*<br>$x_{1}+ x_{6} \geq 0$ *(cardinality)* | $5x_{2}+ 10x_{0}+ x_{4}+ 9x_{6}$<br>$+ 6x_{1} \leq 19$ *(knapsack)* | / | / | / |
| 16 | 6 | flow, quadratic_knapsack | $x_{4}+ x_{5}+ x_{0}- x_{2} = 0$ *(flow)* | $5x_{0}^2+ 3x_{0}x_{3}+ 4x_{0}x_{2}$<br>$+ 5x_{0}x_{1} + x_{0}x_{5}+ x_{3}^2$<br>$+ 4x_{3}x_{2} + 2x_{3}x_{1}$<br>$+ 3x_{3}x_{5} + 4x_{2}^2$<br>$+ 5x_{2}x_{1} + x_{2}x_{5}+ x_{1}^2$<br>$+ 5x_{1}x_{5} + 2x_{5}^2 \leq 17$ *(quadratic_knapsack)* | / | / | / |
| 17 | 6 | cardinality, cardinality | $x_{5}+ x_{2}+ x_{3}+ x_{1} \geq 2$ *(cardinality)* | $x_{0}+ x_{5}+ x_{3}+ x_{2}+ x_{4} \geq 2$ *(cardinality)* | / | / | / |
| 18 | 5 | knapsack, cardinality | $9x_{2}+ 9x_{0}+ 4x_{3}+ x_{1} \leq 9$ *(knapsack)* | $x_{2}+ x_{1}+ x_{3}+ x_{4} \geq 2$ *(cardinality)* | / | / | / |
| 19 | 7 | quadratic_knapsack, quadratic_knapsack | $x_{2}^2+ 5x_{2}x_{5}+ 3x_{2}x_{0}$<br>$+ 4x_{5}^2 + 2x_{5}x_{0}+ 4x_{0}^2 \leq 7$ *(quadratic_knapsack)* | $3x_{1}^2+ x_{1}x_{2}+ 5x_{1}x_{0}$<br>$+ 2x_{1}x_{4} + 5x_{1}x_{5}$<br>$+ 2x_{2}^2 + 5x_{2}x_{0}$<br>$+ 5x_{2}x_{4} + 5x_{2}x_{5}$<br>$+ 2x_{0}^2 + 3x_{0}x_{4}$<br>$+ 3x_{0}x_{5} + 4x_{4}^2$<br>$+ x_{4}x_{5} + 5x_{5}^2 \leq 18$ *(quadratic_knapsack)* | / | / | / |
| 20 | 6 | knapsack, knapsack | $x_{0}+ 10x_{5}+ 3x_{2}+ 5x_{4}$<br>$+ x_{1} \leq 12$ *(knapsack)* | $6x_{1}+ 2x_{3}+ 2x_{4} \leq 3$ *(knapsack)* | / | / | / |
| 21 | 5 | cardinality, cardinality | $x_{1}+ x_{0}+ x_{4} \leq 1$ *(cardinality)* | $x_{4}+ x_{1}+ x_{3} = 0$ *(cardinality)* | / | / | / |
| 22 | 5 | cardinality, cardinality | $x_{2}+ x_{1}+ x_{4}+ x_{0}+ x_{3} = 3$ *(cardinality)* | $x_{2}+ x_{3}+ x_{4}+ x_{0}+ x_{1} \leq 4$ *(cardinality)* | / | / | / |
| 23 | 7 | quadratic_knapsack, cardinality, assignment | $x_{6}+ x_{4} \geq 2$ *(cardinality)*<br>$x_{2}+ x_{0} = 1$ *(assignment)* | $2x_{2}^2+ 5x_{2}x_{5}+ 4x_{2}x_{4}$<br>$+ 4x_{2}x_{3} + 3x_{2}x_{6}$<br>$+ 4x_{5}^2 + 4x_{5}x_{4}$<br>$+ x_{5}x_{3} + 2x_{5}x_{6}$<br>$+ 3x_{4}^2 + x_{4}x_{3}$<br>$+ 4x_{4}x_{6} + 3x_{3}^2$<br>$+ 4x_{3}x_{6} + 3x_{6}^2 \leq 16$ *(quadratic_knapsack)* | / | / | / |
| 24 | 7 | cardinality, knapsack, assignment | $x_{0}+ x_{6}+ x_{4}+ x_{3} = 1$ *(cardinality)* | $5x_{2}+ 10x_{5}+ x_{6}+ 9x_{3}$<br>$+ 6x_{4} \leq 19$ *(knapsack)*<br>$x_{4}+ x_{0} = 1$ *(assignment)* | / | / | / |
| 25 | 5 | cardinality, cardinality | $x_{0}+ x_{2} \leq 2$ *(cardinality)* | $x_{0}+ x_{1}+ x_{3}+ x_{2} \geq 1$ *(cardinality)* | / | / | / |
| 26 | 7 | quadratic_knapsack, quadratic_knapsack, cardinality | $x_{3}+ x_{5}+ x_{6} \leq 2$ *(cardinality)* | $x_{4}^2+ 5x_{4}x_{2}+ 4x_{4}x_{3}$<br>$+ 2x_{2}^2 + 2x_{2}x_{3}+ 2x_{3}^2 \leq 8$ *(quadratic_knapsack)*<br>$2x_{2}^2+ 4x_{2}x_{3}+ 4x_{2}x_{6}$<br>$+ 4x_{2}x_{1} + 5x_{2}x_{0}$<br>$+ x_{3}^2 + 2x_{3}x_{6}$<br>$+ 2x_{3}x_{1} + 5x_{3}x_{0}$<br>$+ x_{6}^2 + 3x_{6}x_{1}+ x_{6}x_{0}$<br>$+ 5x_{1}^2 + 4x_{1}x_{0}+ x_{0}^2 \leq 28$ *(quadratic_knapsack)* | / | / | / |
| 27 | 5 | cardinality, knapsack, quadratic_knapsack | $x_{2}+ x_{4}+ x_{0}+ x_{1}+ x_{3} \geq 3$ *(cardinality)* | $8x_{3}+ 10x_{4}+ x_{1}+ 2x_{0} \leq 12$ *(knapsack)*<br>$5x_{4}^2+ 2x_{4}x_{3}+ 2x_{4}x_{2}$<br>$+ 3x_{4}x_{0} + x_{4}x_{1}+ x_{3}^2$<br>$+ 4x_{3}x_{2} + 4x_{3}x_{0}$<br>$+ 3x_{3}x_{1} + x_{2}^2$<br>$+ 3x_{2}x_{0} + x_{2}x_{1}$<br>$+ 5x_{0}^2 + x_{0}x_{1}+ 5x_{1}^2 \leq 23$ *(quadratic_knapsack)* | / | / | / |
| 28 | 6 | cardinality, quadratic_knapsack, flow | $x_{5}+ x_{2}+ x_{1} = 2$ *(cardinality)* | $3x_{1}^2+ x_{1}x_{5}+ 5x_{1}x_{0}$<br>$+ 2x_{1}x_{3} + 5x_{1}x_{2}$<br>$+ 2x_{5}^2 + 5x_{5}x_{0}$<br>$+ 5x_{5}x_{3} + 5x_{5}x_{2}$<br>$+ 2x_{0}^2 + 3x_{0}x_{3}$<br>$+ 3x_{0}x_{2} + 4x_{3}^2$<br>$+ x_{3}x_{2} + 5x_{2}^2 \leq 18$ *(quadratic_knapsack)*<br>$x_{3}+ x_{4}+ x_{0}- x_{1}- x_{2} = 0$ *(flow)* | / | / | / |
| 29 | 4 | quadratic_knapsack, quadratic_knapsack | $3x_{3}^2+ x_{3}x_{1}+ 3x_{3}x_{0}$<br>$+ x_{1}^2 + 4x_{1}x_{0}+ 3x_{0}^2 \leq 5$ *(quadratic_knapsack)* | $5x_{1}^2+ x_{1}x_{2}+ 2x_{1}x_{0}$<br>$+ 5x_{2}^2 + x_{2}x_{0}+ 3x_{0}^2 \leq 6$ *(quadratic_knapsack)* | / | / | / |
| 30 | 8 | cardinality, cardinality, quadratic_knapsack | $x_{7}+ x_{0}+ x_{4}+ x_{1} \leq 2$ *(cardinality)* | $x_{1}+ x_{3}+ x_{7}+ x_{6} = 1$ *(cardinality)*<br>$x_{5}^2+ 5x_{5}x_{6}+ 2x_{5}x_{3}$<br>$+ 2x_{5}x_{4} + 3x_{6}^2$<br>$+ 2x_{6}x_{3} + 3x_{6}x_{4}$<br>$+ 2x_{3}^2 + 4x_{3}x_{4}+ 5x_{4}^2 \leq 13$ *(quadratic_knapsack)* | / | / | / |
| 31 | 8 | quadratic_knapsack, flow, knapsack | $x_{3}- x_{6}- x_{5}- x_{2} = 0$ *(flow)* | $5x_{2}^2+ 2x_{2}x_{5}+ 2x_{2}x_{3}$<br>$+ 3x_{2}x_{6} + x_{2}x_{7}+ x_{5}^2$<br>$+ 4x_{5}x_{3} + 4x_{5}x_{6}$<br>$+ 3x_{5}x_{7} + x_{3}^2$<br>$+ 3x_{3}x_{6} + x_{3}x_{7}$<br>$+ 5x_{6}^2 + x_{6}x_{7}+ 5x_{7}^2 \leq 23$ *(quadratic_knapsack)*<br>$8x_{7}+ 10x_{5}+ x_{1}+ 2x_{2} \leq 12$ *(knapsack)* | / | / | / |
| 32 | 5 | quadratic_knapsack, cardinality | $x_{4}+ x_{0}+ x_{3}+ x_{2}+ x_{1} = 0$ *(cardinality)* | $3x_{2}^2+ x_{2}x_{4}+ 3x_{2}x_{1}$<br>$+ x_{4}^2 + 4x_{4}x_{1}+ 3x_{1}^2 \leq 5$ *(quadratic_knapsack)* | / | / | / |
| 33 | 4 | quadratic_knapsack, cardinality | $x_{3}+ x_{1}+ x_{0}+ x_{2} \leq 1$ *(cardinality)* | $x_{3}^2+ 2x_{3}x_{1}+ x_{3}x_{2}$<br>$+ x_{1}^2 + 5x_{1}x_{2}+ x_{2}^2 \leq 6$ *(quadratic_knapsack)* | / | / | / |
| 34 | 5 | flow, quadratic_knapsack, knapsack | $x_{4}- x_{3}- x_{1} = 0$ *(flow)* | $5x_{0}^2+ 3x_{0}x_{2}+ x_{0}x_{1}$<br>$+ 5x_{0}x_{3} + 3x_{2}^2$<br>$+ x_{2}x_{1} + 4x_{2}x_{3}$<br>$+ 5x_{1}^2 + 5x_{1}x_{3}+ x_{3}^2 \leq 20$ *(quadratic_knapsack)*<br>$5x_{3}+ 10x_{1}+ x_{4}+ 9x_{0}$<br>$+ 6x_{2} \leq 19$ *(knapsack)* | / | / | / |
| 35 | 8 | cardinality, cardinality, cardinality | $x_{2}+ x_{6} \leq 0$ *(cardinality)*<br>$x_{7}+ x_{1}+ x_{4}+ x_{5}+ x_{0} \geq 1$ *(cardinality)* | $x_{6}+ x_{4}+ x_{7}+ x_{0}+ x_{5} \leq 1$ *(cardinality)* | / | / | / |
| 36 | 5 | quadratic_knapsack, knapsack | $2x_{0}^2+ 5x_{0}x_{2}+ 4x_{0}x_{1}$<br>$+ 4x_{0}x_{3} + 3x_{0}x_{4}$<br>$+ 4x_{2}^2 + 4x_{2}x_{1}$<br>$+ x_{2}x_{3} + 2x_{2}x_{4}$<br>$+ 3x_{1}^2 + x_{1}x_{3}$<br>$+ 4x_{1}x_{4} + 3x_{3}^2$<br>$+ 4x_{3}x_{4} + 3x_{4}^2 \leq 16$ *(quadratic_knapsack)* | $3x_{3}+ 8x_{2}+ 3x_{1} \leq 5$ *(knapsack)* | / | / | / |
| 37 | 8 | quadratic_knapsack, flow, quadratic_knapsack | $x_{0}+ x_{5}+ x_{4}- x_{7}- x_{2} = 0$ *(flow)* | $4x_{4}^2+ 5x_{4}x_{6}+ 2x_{4}x_{0}$<br>$+ 5x_{4}x_{3} + 4x_{6}^2$<br>$+ 3x_{6}x_{0} + 5x_{6}x_{3}$<br>$+ 3x_{0}^2 + 2x_{0}x_{3}+ 3x_{3}^2 \leq 22$ *(quadratic_knapsack)*<br>$5x_{3}^2+ 2x_{3}x_{5}+ 2x_{3}x_{0}$<br>$+ 3x_{3}x_{6} + x_{3}x_{4}+ x_{5}^2$<br>$+ 4x_{5}x_{0} + 4x_{5}x_{6}$<br>$+ 3x_{5}x_{4} + x_{0}^2$<br>$+ 3x_{0}x_{6} + x_{0}x_{4}$<br>$+ 5x_{6}^2 + x_{6}x_{4}+ 5x_{4}^2 \leq 23$ *(quadratic_knapsack)* | / | / | / |
| 38 | 5 | quadratic_knapsack, cardinality | $x_{3}+ x_{2} = 2$ *(cardinality)* | $2x_{1}^2+ 4x_{1}x_{4}+ 5x_{1}x_{3}$<br>$+ x_{1}x_{0} + 4x_{1}x_{2}$<br>$+ 2x_{4}^2 + 4x_{4}x_{3}$<br>$+ 4x_{4}x_{0} + x_{4}x_{2}$<br>$+ 2x_{3}^2 + 3x_{3}x_{0}$<br>$+ 3x_{3}x_{2} + 3x_{0}^2$<br>$+ 2x_{0}x_{2} + x_{2}^2 \leq 19$ *(quadratic_knapsack)* | / | / | / |
| 39 | 5 | knapsack, assignment | $x_{4}+ x_{3} = 1$ *(assignment)* | $10x_{1}+ 8x_{2}+ 2x_{0}+ x_{3}$<br>$+ 4x_{4} \leq 10$ *(knapsack)* | / | / | / |
| 40 | 7 | cardinality, knapsack | $x_{6}+ x_{5}+ x_{1} = 0$ *(cardinality)* | $9x_{0}+ 2x_{5}+ x_{2}+ 9x_{6}$<br>$+ 6x_{4} \leq 13$ *(knapsack)* | / | / | / |
| 41 | 6 | knapsack, knapsack, quadratic_knapsack | $7x_{0}+ 7x_{4}+ 10x_{2}+ 2x_{5} \leq 10$ *(knapsack)* | $6x_{3}+ 2x_{5}+ 2x_{0} \leq 3$ *(knapsack)*<br>$x_{3}^2+ 5x_{3}x_{2}+ 4x_{3}x_{5}$<br>$+ 2x_{2}^2 + 2x_{2}x_{5}+ 2x_{5}^2 \leq 8$ *(quadratic_knapsack)* | / | / | / |
| 42 | 6 | cardinality, quadratic_knapsack, knapsack | $x_{4}+ x_{5}+ x_{3}+ x_{1} = 1$ *(cardinality)* | $5x_{3}^2+ 3x_{3}x_{4}+ 5x_{3}x_{2}$<br>$+ 4x_{4}^2 + 5x_{4}x_{2}+ 3x_{2}^2 \leq 14$ *(quadratic_knapsack)*<br>$4x_{2}+ 6x_{1}+ x_{0}+ x_{4} \leq 6$ *(knapsack)* | / | / | / |
| 43 | 6 | quadratic_knapsack, knapsack | $4x_{4}^2+ 5x_{4}x_{3}+ 2x_{4}x_{2}$<br>$+ 5x_{4}x_{0} + 4x_{3}^2$<br>$+ 3x_{3}x_{2} + 5x_{3}x_{0}$<br>$+ 3x_{2}^2 + 2x_{2}x_{0}+ 3x_{0}^2 \leq 22$ *(quadratic_knapsack)* | $3x_{3}+ 8x_{2}+ 3x_{1} \leq 5$ *(knapsack)* | / | / | / |
| 44 | 7 | cardinality, quadratic_knapsack | $x_{2}+ x_{6}+ x_{4}+ x_{5}+ x_{3} \leq 3$ *(cardinality)* | $x_{0}^2+ 5x_{0}x_{4}+ 2x_{0}x_{1}$<br>$+ 2x_{0}x_{6} + 3x_{4}^2$<br>$+ 2x_{4}x_{1} + 3x_{4}x_{6}$<br>$+ 2x_{1}^2 + 4x_{1}x_{6}+ 5x_{6}^2 \leq 13$ *(quadratic_knapsack)* | / | / | / |
| 45 | 5 | knapsack, knapsack, quadratic_knapsack | $7x_{3}+ 10x_{4}+ 9x_{1} \leq 8$ *(knapsack)* | $3x_{0}+ x_{4}+ 4x_{1}+ 8x_{3} \leq 7$ *(knapsack)*<br>$x_{3}^2+ x_{3}x_{1}+ 3x_{3}x_{4}$<br>$+ 2x_{3}x_{2} + x_{1}^2$<br>$+ 2x_{1}x_{4} + 3x_{1}x_{2}$<br>$+ 2x_{4}^2 + 3x_{4}x_{2}+ 3x_{2}^2 \leq 7$ *(quadratic_knapsack)* | / | / | / |
| 46 | 6 | knapsack, quadratic_knapsack, knapsack | $6x_{2}+ 2x_{3}+ 2x_{5} \leq 3$ *(knapsack)* | $x_{2}^2+ 4x_{2}x_{5}+ x_{2}x_{1}$<br>$+ 5x_{2}x_{0} + 2x_{5}^2$<br>$+ 3x_{5}x_{1} + 3x_{5}x_{0}$<br>$+ 5x_{1}^2 + 4x_{1}x_{0}+ 5x_{0}^2 \leq 22$ *(quadratic_knapsack)*<br>$8x_{5}+ 10x_{1}+ x_{2}+ 2x_{0} \leq 12$ *(knapsack)* | / | / | / |
| 47 | 7 | cardinality, cardinality | $x_{2}+ x_{5}+ x_{4}+ x_{6} = 3$ *(cardinality)* | $x_{0}+ x_{1}+ x_{2}+ x_{5} \leq 1$ *(cardinality)* | / | / | / |
| 48 | 5 | cardinality, cardinality | $x_{2}+ x_{0}+ x_{3}+ x_{1}+ x_{4} = 2$ *(cardinality)* | $x_{3}+ x_{0} = 1$ *(cardinality)* | / | / | / |
| 49 | 5 | knapsack, knapsack, knapsack | $10x_{1}+ 8x_{0}+ 10x_{4} \leq 9$ *(knapsack)* | $10x_{3}+ x_{0}+ 9x_{4} \leq 7$ *(knapsack)*<br>$5x_{3}+ 4x_{4}+ x_{0} \leq 6$ *(knapsack)* | / | / | / |
| 50 | 6 | cardinality, quadratic_knapsack | $x_{2}+ x_{0}+ x_{1} \geq 2$ *(cardinality)* | $x_{0}^2+ 4x_{0}x_{4}+ x_{0}x_{3}$<br>$+ 2x_{0}x_{2} + 5x_{4}^2$<br>$+ 4x_{4}x_{3} + 3x_{4}x_{2}$<br>$+ 5x_{3}^2 + 4x_{3}x_{2}+ 2x_{2}^2 \leq 18$ *(quadratic_knapsack)* | / | / | / |
| 51 | 7 | quadratic_knapsack, knapsack | $4x_{3}^2+ x_{3}x_{6}+ 4x_{3}x_{2}$<br>$+ 3x_{3}x_{5} + 4x_{6}^2$<br>$+ 5x_{6}x_{2} + 4x_{6}x_{5}$<br>$+ 3x_{2}^2 + 5x_{2}x_{5}+ 5x_{5}^2 \leq 24$ *(quadratic_knapsack)* | $4x_{2}+ 6x_{4}+ x_{5}+ x_{0} \leq 6$ *(knapsack)* | / | / | / |
| 52 | 6 | flow, quadratic_knapsack | $x_{5}+ x_{1}+ x_{2}- x_{3}- x_{0} = 0$ *(flow)* | $x_{1}^2+ x_{1}x_{5}+ 3x_{1}x_{4}$<br>$+ 2x_{1}x_{2} + x_{5}^2$<br>$+ 2x_{5}x_{4} + 3x_{5}x_{2}$<br>$+ 2x_{4}^2 + 3x_{4}x_{2}+ 3x_{2}^2 \leq 7$ *(quadratic_knapsack)* | / | / | / |
| 53 | 6 | cardinality, cardinality | $x_{5}+ x_{4} = 0$ *(cardinality)* | $x_{3}+ x_{2}+ x_{0}+ x_{5}+ x_{1} \geq 4$ *(cardinality)* | / | / | / |
| 54 | 6 | cardinality, quadratic_knapsack, cardinality | $x_{5}+ x_{4}+ x_{1} = 0$ *(cardinality)* | $2x_{3}^2+ 5x_{3}x_{0}+ 2x_{3}x_{1}$<br>$+ 2x_{0}^2 + x_{0}x_{1}+ x_{1}^2 \leq 7$ *(quadratic_knapsack)*<br>$x_{5}+ x_{1}+ x_{2}+ x_{0} \leq 3$ *(cardinality)* | / | / | / |
| 55 | 6 | cardinality, quadratic_knapsack | $x_{4}+ x_{5}+ x_{2}+ x_{1} = 0$ *(cardinality)* | $4x_{0}^2+ 4x_{0}x_{3}+ x_{0}x_{2}$<br>$+ 5x_{0}x_{4} + 5x_{3}^2$<br>$+ 5x_{3}x_{2} + 5x_{3}x_{4}$<br>$+ x_{2}^2 + 4x_{2}x_{4}+ x_{4}^2 \leq 11$ *(quadratic_knapsack)* | / | / | / |
| 56 | 7 | knapsack, knapsack, knapsack | $9x_{0}+ 9x_{4}+ 4x_{6}+ x_{3} \leq 9$ *(knapsack)* | $x_{5}+ 3x_{2}+ 9x_{4}+ 8x_{6} \leq 10$ *(knapsack)*<br>$2x_{0}+ 6x_{6}+ x_{1} \leq 3$ *(knapsack)* | / | / | / |
| 57 | 5 | quadratic_knapsack, quadratic_knapsack | $x_{4}^2+ x_{4}x_{0}+ 3x_{4}x_{3}$<br>$+ 2x_{4}x_{1} + x_{0}^2$<br>$+ 2x_{0}x_{3} + 3x_{0}x_{1}$<br>$+ 2x_{3}^2 + 3x_{3}x_{1}+ 3x_{1}^2 \leq 7$ *(quadratic_knapsack)* | $x_{4}^2+ 5x_{4}x_{3}+ x_{4}x_{1}$<br>$+ 3x_{3}^2 + 3x_{3}x_{1}+ 5x_{1}^2 \leq 10$ *(quadratic_knapsack)* | / | / | / |
| 58 | 5 | knapsack, quadratic_knapsack, knapsack | $x_{3}+ 10x_{2}+ 3x_{1}+ 5x_{4}$<br>$+ x_{0} \leq 12$ *(knapsack)* | $3x_{0}^2+ x_{0}x_{1}+ 5x_{0}x_{3}$<br>$+ 2x_{0}x_{2} + 5x_{0}x_{4}$<br>$+ 2x_{1}^2 + 5x_{1}x_{3}$<br>$+ 5x_{1}x_{2} + 5x_{1}x_{4}$<br>$+ 2x_{3}^2 + 3x_{3}x_{2}$<br>$+ 3x_{3}x_{4} + 4x_{2}^2$<br>$+ x_{2}x_{4} + 5x_{4}^2 \leq 18$ *(quadratic_knapsack)*<br>$7x_{2}+ 10x_{3}+ 9x_{4} \leq 8$ *(knapsack)* | / | / | / |
| 59 | 7 | cardinality, cardinality, cardinality | $x_{0}+ x_{3}+ x_{4} = 2$ *(cardinality)* | $x_{6}+ x_{0}+ x_{4}+ x_{3}+ x_{1} = 4$ *(cardinality)*<br>$x_{0}+ x_{1}+ x_{4}+ x_{2} \geq 1$ *(cardinality)* | / | / | / |
| 60 | 5 | knapsack, knapsack, quadratic_knapsack | $2x_{1}+ 2x_{2}+ 9x_{3}+ 10x_{0}$<br>$+ 4x_{4} \leq 18$ *(knapsack)* | $5x_{3}+ 4x_{4}+ x_{0} \leq 6$ *(knapsack)*<br>$x_{0}^2+ x_{0}x_{1}+ 3x_{0}x_{2}$<br>$+ 2x_{0}x_{4} + x_{1}^2$<br>$+ 2x_{1}x_{2} + 3x_{1}x_{4}$<br>$+ 2x_{2}^2 + 3x_{2}x_{4}+ 3x_{4}^2 \leq 7$ *(quadratic_knapsack)* | / | / | / |
| 61 | 5 | knapsack, cardinality | $8x_{1}+ 10x_{0}+ x_{4}+ 2x_{3} \leq 12$ *(knapsack)* | $x_{1}+ x_{3}+ x_{2}+ x_{4}+ x_{0} \geq 4$ *(cardinality)* | / | / | / |
| 62 | 7 | independent_set, knapsack, cardinality | $x_{4}x_{1} = 0$ *(independent_set)* | $10x_{3}+ 8x_{1}+ 10x_{4} \leq 9$ *(knapsack)*<br>$x_{4}+ x_{5}+ x_{1}+ x_{0} \geq 0$ *(cardinality)* | / | / | / |
| 63 | 7 | cardinality, flow, quadratic_knapsack | $x_{6}+ x_{0}+ x_{2}+ x_{4}+ x_{3} \leq 1$ *(cardinality)* | $x_{4}+ x_{5}- x_{3}- x_{1}- x_{6} = 0$ *(flow)*<br>$5x_{4}^2+ x_{4}x_{3}+ 3x_{4}x_{2}$<br>$+ x_{4}x_{6} + 4x_{3}^2+ x_{3}x_{2}$<br>$+ 5x_{3}x_{6} + 4x_{2}^2$<br>$+ 2x_{2}x_{6} + 3x_{6}^2 \leq 19$ *(quadratic_knapsack)* | / | / | / |
| 64 | 5 | assignment, cardinality, cardinality | $x_{0}+ x_{3} = 1$ *(assignment)* | $x_{0}+ x_{3} \leq 1$ *(cardinality)*<br>$x_{0}+ x_{1}+ x_{3} \leq 3$ *(cardinality)* | / | / | / |
| 65 | 5 | knapsack, quadratic_knapsack | $10x_{2}+ 8x_{3}+ 10x_{4} \leq 9$ *(knapsack)* | $5x_{0}^2+ x_{0}x_{2}+ 3x_{0}x_{3}$<br>$+ x_{0}x_{1} + 4x_{2}^2+ x_{2}x_{3}$<br>$+ 5x_{2}x_{1} + 4x_{3}^2$<br>$+ 2x_{3}x_{1} + 3x_{1}^2 \leq 19$ *(quadratic_knapsack)* | / | / | / |
| 66 | 5 | quadratic_knapsack, knapsack, cardinality | $x_{2}+ x_{0} \leq 1$ *(cardinality)* | $4x_{1}^2+ x_{1}x_{2}+ 2x_{1}x_{0}$<br>$+ 4x_{1}x_{3} + x_{1}x_{4}$<br>$+ 2x_{2}^2 + 3x_{2}x_{0}$<br>$+ x_{2}x_{3} + x_{2}x_{4}+ 3x_{0}^2$<br>$+ 3x_{0}x_{3} + 4x_{0}x_{4}$<br>$+ x_{3}^2 + 3x_{3}x_{4}+ 2x_{4}^2 \leq 11$ *(quadratic_knapsack)*<br>$x_{0}+ 3x_{4}+ 9x_{1}+ 8x_{3} \leq 10$ *(knapsack)* | / | / | / |
| 67 | 7 | quadratic_knapsack, cardinality, cardinality | $x_{4}+ x_{5}+ x_{1}+ x_{6}+ x_{0} = 1$ *(cardinality)* | $4x_{2}^2+ 4x_{2}x_{3}+ x_{2}x_{1}$<br>$+ 5x_{2}x_{4} + 5x_{3}^2$<br>$+ 5x_{3}x_{1} + 5x_{3}x_{4}$<br>$+ x_{1}^2 + 4x_{1}x_{4}+ x_{4}^2 \leq 11$ *(quadratic_knapsack)*<br>$x_{4}+ x_{3} \geq 1$ *(cardinality)* | / | / | / |
| 68 | 8 | knapsack, quadratic_knapsack, cardinality | $x_{7}+ x_{2}+ x_{5}+ x_{4} \leq 4$ *(cardinality)* | $x_{2}+ 4x_{1}+ 8x_{3}+ 10x_{7}$<br>$+ 8x_{5} \leq 16$ *(knapsack)*<br>$2x_{4}^2+ 5x_{4}x_{5}+ 2x_{4}x_{0}$<br>$+ 2x_{5}^2 + x_{5}x_{0}+ x_{0}^2 \leq 7$ *(quadratic_knapsack)* | / | / | / |
| 69 | 5 | knapsack, flow | $x_{3}+ x_{2}+ x_{1}- x_{4} = 0$ *(flow)* | $5x_{4}+ 4x_{2}+ x_{3} \leq 6$ *(knapsack)* | / | / | / |
| 70 | 6 | quadratic_knapsack, knapsack, cardinality | $x_{5}+ x_{3}+ x_{2} \leq 0$ *(cardinality)* | $4x_{3}^2+ 3x_{3}x_{4}+ 3x_{3}x_{2}$<br>$+ 2x_{3}x_{5} + 2x_{3}x_{1}$<br>$+ 2x_{4}^2 + 3x_{4}x_{2}$<br>$+ 2x_{4}x_{5} + 3x_{4}x_{1}$<br>$+ 3x_{2}^2 + 5x_{2}x_{5}$<br>$+ 4x_{2}x_{1} + 3x_{5}^2$<br>$+ 5x_{5}x_{1} + x_{1}^2 \leq 22$ *(quadratic_knapsack)*<br>$10x_{5}+ x_{2}+ 9x_{0} \leq 7$ *(knapsack)* | / | / | / |
| 71 | 8 | flow, knapsack | $x_{4}- x_{6}- x_{1}- x_{7} = 0$ *(flow)* | $10x_{5}+ 8x_{1}+ 2x_{7}+ x_{6}$<br>$+ 4x_{2} \leq 10$ *(knapsack)* | / | / | / |
| 72 | 7 | flow, knapsack | $x_{0}- x_{2}- x_{3} = 0$ *(flow)* | $2x_{0}+ 2x_{4}+ 9x_{2}+ 10x_{1}$<br>$+ 4x_{6} \leq 18$ *(knapsack)* | / | / | / |
| 73 | 4 | cardinality, quadratic_knapsack | $x_{3}+ x_{2}+ x_{1} \geq 1$ *(cardinality)* | $4x_{0}^2+ 5x_{0}x_{1}+ 2x_{0}x_{3}$<br>$+ 5x_{0}x_{2} + 4x_{1}^2$<br>$+ 3x_{1}x_{3} + 5x_{1}x_{2}$<br>$+ 3x_{3}^2 + 2x_{3}x_{2}+ 3x_{2}^2 \leq 22$ *(quadratic_knapsack)* | / | / | / |
| 74 | 7 | knapsack, cardinality | $x_{4}+ x_{3}+ x_{2}+ x_{0}+ x_{6} \leq 5$ *(cardinality)* | $2x_{5}+ 5x_{6}+ 2x_{2} \leq 5$ *(knapsack)* | / | / | / |
| 75 | 7 | cardinality, knapsack, quadratic_knapsack | $x_{4}+ x_{1} \leq 0$ *(cardinality)* | $x_{5}+ 4x_{0}+ 8x_{3}+ 10x_{2}$<br>$+ 8x_{4} \leq 16$ *(knapsack)*<br>$5x_{3}^2+ 3x_{3}x_{0}+ 4x_{3}x_{1}$<br>$+ 5x_{3}x_{5} + x_{3}x_{6}+ x_{0}^2$<br>$+ 4x_{0}x_{1} + 2x_{0}x_{5}$<br>$+ 3x_{0}x_{6} + 4x_{1}^2$<br>$+ 5x_{1}x_{5} + x_{1}x_{6}+ x_{5}^2$<br>$+ 5x_{5}x_{6} + 2x_{6}^2 \leq 17$ *(quadratic_knapsack)* | / | / | / |
| 76 | 8 | cardinality, cardinality, cardinality | $x_{2}+ x_{3}+ x_{6} \leq 1$ *(cardinality)* | $x_{5}+ x_{7}+ x_{6} = 3$ *(cardinality)*<br>$x_{6}+ x_{4}+ x_{5}+ x_{1} \leq 4$ *(cardinality)* | / | / | / |
| 77 | 7 | quadratic_knapsack, knapsack | $5x_{3}^2+ 2x_{3}x_{2}+ 2x_{3}x_{0}$<br>$+ 3x_{3}x_{5} + x_{3}x_{4}+ x_{2}^2$<br>$+ 4x_{2}x_{0} + 4x_{2}x_{5}$<br>$+ 3x_{2}x_{4} + x_{0}^2$<br>$+ 3x_{0}x_{5} + x_{0}x_{4}$<br>$+ 5x_{5}^2 + x_{5}x_{4}+ 5x_{4}^2 \leq 23$ *(quadratic_knapsack)* | $7x_{3}+ 10x_{6}+ 9x_{1} \leq 8$ *(knapsack)* | / | / | / |
| 78 | 4 | independent_set, quadratic_knapsack | $x_{2}x_{0} = 0$ *(independent_set)* | $2x_{3}^2+ 5x_{3}x_{1}+ 2x_{3}x_{0}$<br>$+ 2x_{1}^2 + x_{1}x_{0}+ x_{0}^2 \leq 7$ *(quadratic_knapsack)* | / | / | / |
| 79 | 8 | cardinality, quadratic_knapsack, cardinality | $x_{0}+ x_{7}+ x_{2} = 2$ *(cardinality)*<br>$x_{3}^2+ 5x_{3}x_{1}+ 4x_{3}x_{4}$<br>$+ 2x_{1}^2 + 2x_{1}x_{4}+ 2x_{4}^2 \leq 8$ *(quadratic_knapsack)* | $x_{3}+ x_{0}+ x_{4}+ x_{1} \leq 1$ *(cardinality)* | / | / | / |
| 80 | 6 | quadratic_knapsack, quadratic_knapsack | $3x_{2}^2+ x_{2}x_{1}+ 5x_{2}x_{0}$<br>$+ 2x_{2}x_{4} + 5x_{2}x_{3}$<br>$+ 2x_{1}^2 + 5x_{1}x_{0}$<br>$+ 5x_{1}x_{4} + 5x_{1}x_{3}$<br>$+ 2x_{0}^2 + 3x_{0}x_{4}$<br>$+ 3x_{0}x_{3} + 4x_{4}^2$<br>$+ x_{4}x_{3} + 5x_{3}^2 \leq 18$ *(quadratic_knapsack)* | $4x_{1}^2+ x_{1}x_{4}+ 2x_{1}x_{2}$<br>$+ 4x_{1}x_{0} + x_{1}x_{3}$<br>$+ 2x_{4}^2 + 3x_{4}x_{2}$<br>$+ x_{4}x_{0} + x_{4}x_{3}+ 3x_{2}^2$<br>$+ 3x_{2}x_{0} + 4x_{2}x_{3}$<br>$+ x_{0}^2 + 3x_{0}x_{3}+ 2x_{3}^2 \leq 11$ *(quadratic_knapsack)* | / | / | / |
| 81 | 5 | cardinality, cardinality | $x_{4}+ x_{2} = 1$ *(cardinality)* | $x_{0}+ x_{1}+ x_{3}+ x_{4}+ x_{2} \leq 3$ *(cardinality)* | / | / | / |
| 82 | 5 | quadratic_knapsack, cardinality | $x_{3}^2+ 4x_{3}x_{2}+ x_{3}x_{4}$<br>$+ 5x_{3}x_{0} + 2x_{2}^2$<br>$+ 3x_{2}x_{4} + 3x_{2}x_{0}$<br>$+ 5x_{4}^2 + 4x_{4}x_{0}+ 5x_{0}^2 \leq 22$ *(quadratic_knapsack)* | $x_{3}+ x_{1} \geq 2$ *(cardinality)* | / | / | / |
| 83 | 6 | quadratic_knapsack, quadratic_knapsack | $5x_{4}^2+ 3x_{4}x_{2}+ 5x_{4}x_{3}$<br>$+ 4x_{2}^2 + 5x_{2}x_{3}+ 3x_{3}^2 \leq 14$ *(quadratic_knapsack)* | $5x_{5}^2+ x_{5}x_{0}+ 3x_{5}x_{4}$<br>$+ x_{5}x_{1} + 4x_{0}^2+ x_{0}x_{4}$<br>$+ 5x_{0}x_{1} + 4x_{4}^2$<br>$+ 2x_{4}x_{1} + 3x_{1}^2 \leq 19$ *(quadratic_knapsack)* | / | / | / |
| 84 | 8 | quadratic_knapsack, cardinality, knapsack | $2x_{7}^2+ 5x_{7}x_{0}+ 2x_{7}x_{6}$<br>$+ 2x_{0}^2 + x_{0}x_{6}+ x_{6}^2 \leq 7$ *(quadratic_knapsack)*<br>$x_{5}+ x_{4}+ x_{1} \leq 0$ *(cardinality)* | $7x_{1}+ 10x_{7}+ 9x_{2} \leq 8$ *(knapsack)* | / | / | / |
| 85 | 7 | cardinality, knapsack, knapsack | $x_{3}+ x_{4}+ x_{5}+ x_{6}+ x_{0} \geq 0$ *(cardinality)* | $x_{6}+ 9x_{5}+ 5x_{0}+ 2x_{1}$<br>$+ 10x_{3} \leq 10$ *(knapsack)*<br>$8x_{3}+ 5x_{1}+ 7x_{6}+ 4x_{2} \leq 11$ *(knapsack)* | / | / | / |
| 86 | 6 | cardinality, quadratic_knapsack, quadratic_knapsack | $x_{1}+ x_{4}+ x_{0} = 3$ *(cardinality)* | $x_{4}^2+ 4x_{4}x_{5}+ x_{4}x_{2}$<br>$+ 5x_{4}x_{3} + 2x_{5}^2$<br>$+ 3x_{5}x_{2} + 3x_{5}x_{3}$<br>$+ 5x_{2}^2 + 4x_{2}x_{3}+ 5x_{3}^2 \leq 22$ *(quadratic_knapsack)*<br>$3x_{0}^2+ x_{0}x_{3}+ 3x_{0}x_{2}$<br>$+ 4x_{0}x_{4} + 5x_{3}^2$<br>$+ 4x_{3}x_{2} + 2x_{3}x_{4}$<br>$+ 2x_{2}^2 + 4x_{2}x_{4}+ 4x_{4}^2 \leq 17$ *(quadratic_knapsack)* | / | / | / |
| 87 | 6 | assignment, cardinality | $x_{2}+ x_{0} = 1$ *(assignment)* | $x_{4}+ x_{5}+ x_{1}+ x_{0}+ x_{3} = 0$ *(cardinality)* | / | / | / |
| 88 | 7 | cardinality, quadratic_knapsack | $x_{0}+ x_{2}+ x_{4}+ x_{1}+ x_{3} \leq 2$ *(cardinality)* | $2x_{6}^2+ 4x_{6}x_{4}+ 4x_{6}x_{5}$<br>$+ 4x_{6}x_{0} + 5x_{6}x_{2}$<br>$+ x_{4}^2 + 2x_{4}x_{5}$<br>$+ 2x_{4}x_{0} + 5x_{4}x_{2}$<br>$+ x_{5}^2 + 3x_{5}x_{0}+ x_{5}x_{2}$<br>$+ 5x_{0}^2 + 4x_{0}x_{2}+ x_{2}^2 \leq 28$ *(quadratic_knapsack)* | / | / | / |
| 89 | 6 | cardinality, cardinality | $x_{2}+ x_{5}+ x_{0}+ x_{3} = 2$ *(cardinality)* | $x_{1}+ x_{4}+ x_{3}+ x_{0} \geq 3$ *(cardinality)* | / | / | / |
| 90 | 6 | cardinality, quadratic_knapsack | $x_{1}+ x_{3}+ x_{4} = 2$ *(cardinality)* | $x_{3}^2+ 4x_{3}x_{2}+ x_{3}x_{1}$<br>$+ 5x_{3}x_{5} + 2x_{2}^2$<br>$+ 3x_{2}x_{1} + 3x_{2}x_{5}$<br>$+ 5x_{1}^2 + 4x_{1}x_{5}+ 5x_{5}^2 \leq 22$ *(quadratic_knapsack)* | / | / | / |
| 91 | 8 | quadratic_knapsack, flow, knapsack | $x_{0}+ x_{5}+ x_{7}- x_{2} = 0$ *(flow)* | $4x_{1}^2+ x_{1}x_{0}+ 4x_{1}x_{7}$<br>$+ 3x_{1}x_{5} + 4x_{0}^2$<br>$+ 5x_{0}x_{7} + 4x_{0}x_{5}$<br>$+ 3x_{7}^2 + 5x_{7}x_{5}+ 5x_{5}^2 \leq 24$ *(quadratic_knapsack)*<br>$4x_{7}+ 2x_{5}+ 10x_{3}+ 8x_{6} \leq 13$ *(knapsack)* | / | / | / |
| 92 | 6 | flow, cardinality, quadratic_knapsack | $x_{2}+ x_{5}+ x_{4}- x_{0}- x_{3} = 0$ *(flow)* | $x_{3}+ x_{5}+ x_{1}+ x_{0} \leq 1$ *(cardinality)*<br>$5x_{3}^2+ 3x_{3}x_{4}+ x_{3}x_{1}$<br>$+ 5x_{3}x_{2} + 3x_{4}^2$<br>$+ x_{4}x_{1} + 4x_{4}x_{2}$<br>$+ 5x_{1}^2 + 5x_{1}x_{2}+ x_{2}^2 \leq 20$ *(quadratic_knapsack)* | / | / | / |
| 93 | 5 | cardinality, cardinality, quadratic_knapsack | $x_{3}+ x_{0}+ x_{1} = 0$ *(cardinality)* | $x_{0}+ x_{2}+ x_{3}+ x_{1}+ x_{4} \geq 2$ *(cardinality)*<br>$2x_{3}^2+ 5x_{3}x_{2}+ 4x_{3}x_{0}$<br>$+ 4x_{3}x_{1} + 3x_{3}x_{4}$<br>$+ 4x_{2}^2 + 4x_{2}x_{0}$<br>$+ x_{2}x_{1} + 2x_{2}x_{4}$<br>$+ 3x_{0}^2 + x_{0}x_{1}$<br>$+ 4x_{0}x_{4} + 3x_{1}^2$<br>$+ 4x_{1}x_{4} + 3x_{4}^2 \leq 16$ *(quadratic_knapsack)* | / | / | / |
| 94 | 8 | quadratic_knapsack, flow | $x_{3}+ x_{7}+ x_{4}- x_{6}- x_{0} = 0$ *(flow)* | $x_{5}^2+ 5x_{5}x_{0}+ 2x_{5}x_{3}$<br>$+ 2x_{5}x_{6} + 3x_{0}^2$<br>$+ 2x_{0}x_{3} + 3x_{0}x_{6}$<br>$+ 2x_{3}^2 + 4x_{3}x_{6}+ 5x_{6}^2 \leq 13$ *(quadratic_knapsack)* | / | / | / |
| 95 | 7 | quadratic_knapsack, cardinality, cardinality | $x_{4}+ x_{6}+ x_{1}+ x_{2}+ x_{5} \leq 1$ *(cardinality)* | $x_{6}^2+ 4x_{6}x_{1}+ x_{6}x_{0}$<br>$+ 5x_{6}x_{4} + 2x_{1}^2$<br>$+ 3x_{1}x_{0} + 3x_{1}x_{4}$<br>$+ 5x_{0}^2 + 4x_{0}x_{4}+ 5x_{4}^2 \leq 22$ *(quadratic_knapsack)*<br>$x_{5}+ x_{6}+ x_{1}+ x_{4} \leq 2$ *(cardinality)* | / | / | / |
| 96 | 5 | cardinality, cardinality | $x_{1}+ x_{3}+ x_{4}+ x_{2} = 4$ *(cardinality)* | $x_{4}+ x_{1}+ x_{2}+ x_{0} \geq 1$ *(cardinality)* | / | / | / |
| 97 | 7 | quadratic_knapsack, assignment, knapsack | $x_{5}+ x_{2}+ x_{6} = 1$ *(assignment)* | $2x_{1}^2+ 5x_{1}x_{3}+ 4x_{1}x_{0}$<br>$+ 4x_{1}x_{5} + 3x_{1}x_{6}$<br>$+ 4x_{3}^2 + 4x_{3}x_{0}$<br>$+ x_{3}x_{5} + 2x_{3}x_{6}$<br>$+ 3x_{0}^2 + x_{0}x_{5}$<br>$+ 4x_{0}x_{6} + 3x_{5}^2$<br>$+ 4x_{5}x_{6} + 3x_{6}^2 \leq 16$ *(quadratic_knapsack)*<br>$2x_{4}+ 2x_{0}+ 9x_{2}+ 10x_{5}$<br>$+ 4x_{1} \leq 18$ *(knapsack)* | / | / | / |
| 98 | 7 | cardinality, cardinality, cardinality | $x_{0}+ x_{5} \leq 1$ *(cardinality)* | $x_{6}+ x_{0} \geq 1$ *(cardinality)*<br>$x_{4}+ x_{6}+ x_{5}+ x_{3}+ x_{1} \leq 5$ *(cardinality)* | / | / | / |
| 99 | 6 | knapsack, flow | $x_{4}- x_{3}- x_{1}- x_{0} = 0$ *(flow)* | $2x_{5}+ 10x_{1}+ 4x_{2} \leq 6$ *(knapsack)* | / | / | / |
| 100 | 5 | quadratic_knapsack, flow | $x_{3}- x_{4}- x_{2} = 0$ *(flow)* | $x_{0}^2+ 4x_{0}x_{1}+ x_{0}x_{4}$<br>$+ 2x_{0}x_{2} + 5x_{1}^2$<br>$+ 4x_{1}x_{4} + 3x_{1}x_{2}$<br>$+ 5x_{4}^2 + 4x_{4}x_{2}+ 2x_{2}^2 \leq 18$ *(quadratic_knapsack)* | / | / | / |
| 101 | 7 | cardinality, cardinality | $x_{3}+ x_{6}+ x_{2}+ x_{0}+ x_{4} \leq 4$ *(cardinality)* | $x_{5}+ x_{2}+ x_{0}+ x_{6}+ x_{4} \leq 5$ *(cardinality)* | / | / | / |
| 102 | 6 | knapsack, quadratic_knapsack, cardinality | $x_{2}+ x_{1}+ x_{4}+ x_{3}+ x_{5} \leq 0$ *(cardinality)* | $6x_{5}+ 7x_{4}+ 8x_{0}+ 9x_{2}$<br>$+ 5x_{1} \leq 14$ *(knapsack)*<br>$4x_{4}^2+ 4x_{4}x_{2}+ x_{4}x_{0}$<br>$+ 5x_{4}x_{1} + 5x_{2}^2$<br>$+ 5x_{2}x_{0} + 5x_{2}x_{1}$<br>$+ x_{0}^2 + 4x_{0}x_{1}+ x_{1}^2 \leq 11$ *(quadratic_knapsack)* | / | / | / |
| 103 | 3 | cardinality, cardinality | $x_{1}+ x_{0}+ x_{2} \geq 0$ *(cardinality)* | $x_{2}+ x_{1}+ x_{0} \geq 2$ *(cardinality)* | / | / | / |
| 104 | 8 | cardinality, cardinality, cardinality | $x_{3}+ x_{7}+ x_{4}+ x_{2} = 2$ *(cardinality)* | $x_{2}+ x_{5}+ x_{3} \geq 2$ *(cardinality)*<br>$x_{0}+ x_{3}+ x_{2} \geq 3$ *(cardinality)* | / | / | / |
| 105 | 6 | knapsack, cardinality | $x_{4}+ x_{3}+ x_{5}+ x_{1} \leq 4$ *(cardinality)* | $7x_{3}+ 7x_{1}+ 10x_{2}+ 2x_{5} \leq 10$ *(knapsack)* | / | / | / |
| 106 | 6 | cardinality, quadratic_knapsack, cardinality | $x_{3}+ x_{4}+ x_{1} = 1$ *(cardinality)* | $x_{3}^2+ x_{3}x_{0}+ 3x_{3}x_{1}$<br>$+ 2x_{3}x_{5} + x_{0}^2$<br>$+ 2x_{0}x_{1} + 3x_{0}x_{5}$<br>$+ 2x_{1}^2 + 3x_{1}x_{5}+ 3x_{5}^2 \leq 7$ *(quadratic_knapsack)*<br>$x_{2}+ x_{0}+ x_{3} \geq 3$ *(cardinality)* | / | / | / |
| 107 | 8 | knapsack, cardinality, cardinality | $x_{5}+ x_{0}+ x_{7}+ x_{2}+ x_{1} \leq 5$ *(cardinality)* | $9x_{1}+ 9x_{2}+ 4x_{7}+ x_{0} \leq 9$ *(knapsack)*<br>$x_{2}+ x_{0}+ x_{7}+ x_{3} \leq 4$ *(cardinality)* | / | / | / |
| 108 | 5 | knapsack, knapsack, cardinality | $x_{2}+ x_{0} = 0$ *(cardinality)* | $2x_{0}+ 10x_{2}+ 4x_{4} \leq 6$ *(knapsack)*<br>$4x_{1}+ 5x_{3}+ 3x_{0}+ 7x_{4}$<br>$+ 6x_{2} \leq 11$ *(knapsack)* | / | / | / |
| 109 | 8 | cardinality, knapsack | $x_{1}+ x_{0}+ x_{3}+ x_{4}+ x_{6} \geq 5$ *(cardinality)* | $x_{3}+ 10x_{5}+ 3x_{1}+ 5x_{2}$<br>$+ x_{7} \leq 12$ *(knapsack)* | / | / | / |
| 110 | 5 | cardinality, quadratic_knapsack | $x_{2}+ x_{1}+ x_{4}+ x_{0}+ x_{3} \leq 3$ *(cardinality)* | $5x_{1}^2+ x_{1}x_{0}+ 3x_{1}x_{4}$<br>$+ x_{1}x_{2} + 4x_{0}^2+ x_{0}x_{4}$<br>$+ 5x_{0}x_{2} + 4x_{4}^2$<br>$+ 2x_{4}x_{2} + 3x_{2}^2 \leq 19$ *(quadratic_knapsack)* | / | / | / |
| 111 | 4 | independent_set, quadratic_knapsack, knapsack | $x_{1}x_{0} = 0$ *(independent_set)* | $5x_{3}^2+ x_{3}x_{0}+ 2x_{3}x_{1}$<br>$+ 5x_{0}^2 + x_{0}x_{1}+ 3x_{1}^2 \leq 6$ *(quadratic_knapsack)*<br>$10x_{1}+ x_{0}+ 9x_{2} \leq 7$ *(knapsack)* | / | / | / |
| 112 | 8 | flow, cardinality, knapsack | $x_{7}+ x_{2}- x_{0}- x_{1}- x_{6} = 0$ *(flow)* | $x_{5}+ x_{6}+ x_{0}+ x_{2}+ x_{4} = 3$ *(cardinality)*<br>$4x_{1}+ 6x_{6}+ 2x_{2}+ 10x_{3} \leq 7$ *(knapsack)* | / | / | / |
| 113 | 7 | flow, flow, cardinality | $x_{1}- x_{3} = 0$ *(flow)* | $x_{5}+ x_{3}+ x_{2}- x_{4}- x_{0}$<br>$- x_{1} = 0$ *(flow)*<br>$x_{0}+ x_{3}+ x_{4}+ x_{1}+ x_{2} \geq 1$ *(cardinality)* | / | / | / |
| 114 | 5 | cardinality, flow, knapsack | $x_{2}+ x_{3}+ x_{0}- x_{4}- x_{1} = 0$ *(flow)* | $x_{4}+ x_{1}+ x_{2}+ x_{3}+ x_{0} \geq 1$ *(cardinality)*<br>$4x_{3}+ 2x_{2}+ 10x_{4}+ 8x_{1} \leq 13$ *(knapsack)* | / | / | / |
| 115 | 4 | cardinality, quadratic_knapsack | $x_{3}+ x_{2} \leq 0$ *(cardinality)* | $3x_{3}^2+ x_{3}x_{2}+ 3x_{3}x_{0}$<br>$+ 4x_{3}x_{1} + 5x_{2}^2$<br>$+ 4x_{2}x_{0} + 2x_{2}x_{1}$<br>$+ 2x_{0}^2 + 4x_{0}x_{1}+ 4x_{1}^2 \leq 17$ *(quadratic_knapsack)* | / | / | / |
| 116 | 8 | cardinality, knapsack, cardinality | $x_{3}+ x_{0}+ x_{1}+ x_{6} \leq 1$ *(cardinality)* | $4x_{7}+ 5x_{1}+ 3x_{5}+ 7x_{3}$<br>$+ 6x_{0} \leq 11$ *(knapsack)*<br>$x_{5}+ x_{7}+ x_{2}+ x_{0}+ x_{6} = 0$ *(cardinality)* | / | / | / |
| 117 | 5 | quadratic_knapsack, quadratic_knapsack | $3x_{1}^2+ x_{1}x_{2}+ 3x_{1}x_{3}$<br>$+ x_{2}^2 + 4x_{2}x_{3}+ 3x_{3}^2 \leq 5$ *(quadratic_knapsack)* | $x_{1}^2+ 2x_{1}x_{0}+ x_{1}x_{3}$<br>$+ x_{0}^2 + 5x_{0}x_{3}+ x_{3}^2 \leq 6$ *(quadratic_knapsack)* | / | / | / |
| 118 | 4 | quadratic_knapsack, knapsack | $4x_{2}^2+ 4x_{2}x_{3}+ x_{2}x_{1}$<br>$+ 5x_{2}x_{0} + 5x_{3}^2$<br>$+ 5x_{3}x_{1} + 5x_{3}x_{0}$<br>$+ x_{1}^2 + 4x_{1}x_{0}+ x_{0}^2 \leq 11$ *(quadratic_knapsack)* | $2x_{1}+ 6x_{2}+ x_{0} \leq 3$ *(knapsack)* | / | / | / |
| 119 | 8 | cardinality, quadratic_knapsack, quadratic_knapsack | $x_{6}+ x_{7}+ x_{3}+ x_{4}+ x_{0} \geq 2$ *(cardinality)* | $3x_{2}^2+ x_{2}x_{7}+ 3x_{2}x_{5}$<br>$+ x_{7}^2 + 4x_{7}x_{5}+ 3x_{5}^2 \leq 5$ *(quadratic_knapsack)*<br>$3x_{1}^2+ x_{1}x_{3}+ 3x_{1}x_{6}$<br>$+ 5x_{3}^2 + 2x_{3}x_{6}+ 2x_{6}^2 \leq 8$ *(quadratic_knapsack)* | / | / | / |
| 120 | 6 | cardinality, quadratic_knapsack, quadratic_knapsack | $x_{1}+ x_{3}+ x_{0} \leq 3$ *(cardinality)* | $2x_{0}^2+ 5x_{0}x_{4}+ 2x_{0}x_{1}$<br>$+ x_{0}x_{3} + 4x_{0}x_{5}$<br>$+ 4x_{4}^2 + 2x_{4}x_{1}$<br>$+ 5x_{4}x_{3} + x_{4}x_{5}+ x_{1}^2$<br>$+ 4x_{1}x_{3} + 2x_{1}x_{5}$<br>$+ 5x_{3}^2 + 4x_{3}x_{5}+ x_{5}^2 \leq 23$ *(quadratic_knapsack)*<br>$5x_{1}^2+ 2x_{1}x_{2}+ 2x_{1}x_{5}$<br>$+ 3x_{1}x_{3} + x_{1}x_{4}+ x_{2}^2$<br>$+ 4x_{2}x_{5} + 4x_{2}x_{3}$<br>$+ 3x_{2}x_{4} + x_{5}^2$<br>$+ 3x_{5}x_{3} + x_{5}x_{4}$<br>$+ 5x_{3}^2 + x_{3}x_{4}+ 5x_{4}^2 \leq 23$ *(quadratic_knapsack)* | / | / | / |
| 121 | 5 | quadratic_knapsack, quadratic_knapsack | $2x_{0}^2+ 5x_{0}x_{4}+ 4x_{0}x_{2}$<br>$+ 2x_{0}x_{1} + 4x_{0}x_{3}$<br>$+ 5x_{4}^2 + 5x_{4}x_{2}$<br>$+ 2x_{4}x_{1} + x_{4}x_{3}$<br>$+ 4x_{2}^2 + x_{2}x_{1}$<br>$+ 2x_{2}x_{3} + 5x_{1}^2$<br>$+ 4x_{1}x_{3} + 2x_{3}^2 \leq 29$ *(quadratic_knapsack)* | $2x_{4}^2+ 5x_{4}x_{3}+ 4x_{4}x_{2}$<br>$+ 4x_{4}x_{0} + 3x_{4}x_{1}$<br>$+ 4x_{3}^2 + 4x_{3}x_{2}$<br>$+ x_{3}x_{0} + 2x_{3}x_{1}$<br>$+ 3x_{2}^2 + x_{2}x_{0}$<br>$+ 4x_{2}x_{1} + 3x_{0}^2$<br>$+ 4x_{0}x_{1} + 3x_{1}^2 \leq 16$ *(quadratic_knapsack)* | / | / | / |
| 122 | 7 | knapsack, flow, cardinality | $x_{0}+ x_{5}+ x_{6}- x_{1}- x_{3}$<br>$- x_{2} = 0$ *(flow)* | $6x_{1}+ 7x_{5}+ 8x_{3}+ 9x_{2}$<br>$+ 5x_{6} \leq 14$ *(knapsack)*<br>$x_{3}+ x_{6}+ x_{1}+ x_{2}+ x_{5} = 2$ *(cardinality)* | / | / | / |
| 123 | 8 | cardinality, flow | $x_{6}+ x_{1}+ x_{7}+ x_{2}+ x_{4} = 1$ *(cardinality)* | $x_{3}+ x_{6}+ x_{1}- x_{7}- x_{4} = 0$ *(flow)* | / | / | / |
| 124 | 6 | knapsack, cardinality, cardinality | $x_{1}+ x_{4} = 2$ *(cardinality)* | $2x_{4}+ 6x_{3}+ x_{1} \leq 3$ *(knapsack)*<br>$x_{2}+ x_{0}+ x_{4} = 2$ *(cardinality)* | / | / | / |
| 125 | 8 | knapsack, quadratic_knapsack, quadratic_knapsack | $x_{4}+ 9x_{7}+ 5x_{0}+ 2x_{5}$<br>$+ 10x_{3} \leq 10$ *(knapsack)* | $2x_{2}^2+ 5x_{2}x_{7}+ 2x_{2}x_{5}$<br>$+ x_{2}x_{6} + 4x_{2}x_{3}$<br>$+ 4x_{7}^2 + 2x_{7}x_{5}$<br>$+ 5x_{7}x_{6} + x_{7}x_{3}+ x_{5}^2$<br>$+ 4x_{5}x_{6} + 2x_{5}x_{3}$<br>$+ 5x_{6}^2 + 4x_{6}x_{3}+ x_{3}^2 \leq 23$ *(quadratic_knapsack)*<br>$2x_{5}^2+ 4x_{5}x_{0}+ 5x_{5}x_{3}$<br>$+ x_{5}x_{6} + 4x_{5}x_{1}$<br>$+ 2x_{0}^2 + 4x_{0}x_{3}$<br>$+ 4x_{0}x_{6} + x_{0}x_{1}$<br>$+ 2x_{3}^2 + 3x_{3}x_{6}$<br>$+ 3x_{3}x_{1} + 3x_{6}^2$<br>$+ 2x_{6}x_{1} + x_{1}^2 \leq 19$ *(quadratic_knapsack)* | / | / | / |
| 126 | 8 | cardinality, cardinality, knapsack | $x_{4}+ x_{6}+ x_{7} = 0$ *(cardinality)* | $x_{1}+ x_{7}+ x_{0}+ x_{3}+ x_{2} \geq 1$ *(cardinality)*<br>$9x_{5}+ 9x_{4}+ 4x_{2}+ x_{1} \leq 9$ *(knapsack)* | / | / | / |
| 127 | 5 | knapsack, cardinality, cardinality | $x_{1}+ x_{4} \leq 2$ *(cardinality)* | $9x_{2}+ 9x_{3}+ 4x_{4}+ x_{1} \leq 9$ *(knapsack)*<br>$x_{2}+ x_{4} \leq 1$ *(cardinality)* | / | / | / |
| 128 | 7 | knapsack, quadratic_knapsack, quadratic_knapsack | $4x_{4}+ 5x_{2}+ 3x_{6}+ 7x_{3}$<br>$+ 6x_{5} \leq 11$ *(knapsack)* | $3x_{4}^2+ x_{4}x_{2}+ 3x_{4}x_{3}$<br>$+ 4x_{4}x_{0} + 5x_{2}^2$<br>$+ 4x_{2}x_{3} + 2x_{2}x_{0}$<br>$+ 2x_{3}^2 + 4x_{3}x_{0}+ 4x_{0}^2 \leq 17$ *(quadratic_knapsack)*<br>$2x_{2}^2+ 4x_{2}x_{6}+ 5x_{2}x_{0}$<br>$+ x_{2}x_{3} + 4x_{2}x_{5}$<br>$+ 2x_{6}^2 + 4x_{6}x_{0}$<br>$+ 4x_{6}x_{3} + x_{6}x_{5}$<br>$+ 2x_{0}^2 + 3x_{0}x_{3}$<br>$+ 3x_{0}x_{5} + 3x_{3}^2$<br>$+ 2x_{3}x_{5} + x_{5}^2 \leq 19$ *(quadratic_knapsack)* | / | / | / |
| 129 | 6 | knapsack, knapsack | $x_{5}+ 9x_{4}+ 5x_{1}+ 2x_{2}$<br>$+ 10x_{0} \leq 10$ *(knapsack)* | $10x_{4}+ 8x_{3}+ 10x_{1} \leq 9$ *(knapsack)* | / | / | / |
| 130 | 6 | cardinality, quadratic_knapsack, quadratic_knapsack | $x_{0}+ x_{4}+ x_{5} \geq 1$ *(cardinality)* | $2x_{1}^2+ 5x_{1}x_{0}+ 4x_{1}x_{3}$<br>$+ 2x_{1}x_{5} + 4x_{1}x_{2}$<br>$+ 5x_{0}^2 + 5x_{0}x_{3}$<br>$+ 2x_{0}x_{5} + x_{0}x_{2}$<br>$+ 4x_{3}^2 + x_{3}x_{5}$<br>$+ 2x_{3}x_{2} + 5x_{5}^2$<br>$+ 4x_{5}x_{2} + 2x_{2}^2 \leq 29$ *(quadratic_knapsack)*<br>$2x_{1}^2+ 5x_{1}x_{3}+ 2x_{1}x_{5}$<br>$+ 2x_{3}^2 + x_{3}x_{5}+ x_{5}^2 \leq 7$ *(quadratic_knapsack)* | / | / | / |
| 131 | 5 | quadratic_knapsack, quadratic_knapsack, quadratic_knapsack | $4x_{1}^2+ x_{1}x_{4}+ 2x_{1}x_{2}$<br>$+ 4x_{1}x_{3} + x_{1}x_{0}$<br>$+ 2x_{4}^2 + 3x_{4}x_{2}$<br>$+ x_{4}x_{3} + x_{4}x_{0}+ 3x_{2}^2$<br>$+ 3x_{2}x_{3} + 4x_{2}x_{0}$<br>$+ x_{3}^2 + 3x_{3}x_{0}+ 2x_{0}^2 \leq 11$ *(quadratic_knapsack)* | $2x_{4}^2+ 4x_{4}x_{3}+ 4x_{4}x_{1}$<br>$+ 4x_{4}x_{0} + 5x_{4}x_{2}$<br>$+ x_{3}^2 + 2x_{3}x_{1}$<br>$+ 2x_{3}x_{0} + 5x_{3}x_{2}$<br>$+ x_{1}^2 + 3x_{1}x_{0}+ x_{1}x_{2}$<br>$+ 5x_{0}^2 + 4x_{0}x_{2}+ x_{2}^2 \leq 28$ *(quadratic_knapsack)*<br>$3x_{4}^2+ x_{4}x_{2}+ 3x_{4}x_{3}$<br>$+ 4x_{4}x_{1} + 5x_{2}^2$<br>$+ 4x_{2}x_{3} + 2x_{2}x_{1}$<br>$+ 2x_{3}^2 + 4x_{3}x_{1}+ 4x_{1}^2 \leq 17$ *(quadratic_knapsack)* | / | / | / |
| 132 | 7 | cardinality, flow, cardinality | $x_{4}+ x_{2}+ x_{1}+ x_{5} = 3$ *(cardinality)* | $x_{6}+ x_{2}+ x_{1}- x_{3}- x_{4}$<br>$- x_{5} = 0$ *(flow)*<br>$x_{2}+ x_{4}+ x_{6}+ x_{5} \geq 0$ *(cardinality)* | / | / | / |
| 133 | 7 | flow, independent_set, cardinality | $x_{6}- x_{1} = 0$ *(flow)*<br>$x_{2}x_{0} = 0$ *(independent_set)* | $x_{2}+ x_{5}+ x_{0}+ x_{6}+ x_{1} \geq 2$ *(cardinality)* | / | / | / |
| 134 | 4 | cardinality, cardinality | $x_{3}+ x_{0}+ x_{2} = 3$ *(cardinality)* | $x_{2}+ x_{3} \geq 0$ *(cardinality)* | / | / | / |
| 135 | 8 | cardinality, knapsack | $x_{7}+ x_{5}+ x_{1}+ x_{2}+ x_{4} \geq 4$ *(cardinality)* | $x_{4}+ 6x_{5}+ 9x_{0}+ 8x_{6}$<br>$+ 9x_{1} \leq 18$ *(knapsack)* | / | / | / |
| 136 | 4 | flow, flow | $x_{3}- x_{0}- x_{2}- x_{1} = 0$ *(flow)* | $x_{3}+ x_{0}- x_{1} = 0$ *(flow)* | / | / | / |
| 137 | 5 | cardinality, knapsack | $x_{2}+ x_{4}+ x_{1}+ x_{3}+ x_{0} = 2$ *(cardinality)* | $4x_{4}+ 2x_{3}+ 10x_{1}+ 8x_{0} \leq 13$ *(knapsack)* | / | / | / |
| 138 | 8 | cardinality, quadratic_knapsack, knapsack | $x_{4}+ x_{5}+ x_{1} \geq 3$ *(cardinality)* | $5x_{0}^2+ 3x_{0}x_{5}+ 5x_{0}x_{2}$<br>$+ 4x_{5}^2 + 5x_{5}x_{2}+ 3x_{2}^2 \leq 14$ *(quadratic_knapsack)*<br>$2x_{4}+ 5x_{7}+ 2x_{5} \leq 5$ *(knapsack)* | / | / | / |
| 139 | 8 | knapsack, quadratic_knapsack | $4x_{6}+ 5x_{4}+ 3x_{2}+ 7x_{0}$<br>$+ 6x_{3} \leq 11$ *(knapsack)* | $4x_{1}^2+ 3x_{1}x_{7}+ 3x_{1}x_{6}$<br>$+ 2x_{1}x_{3} + 2x_{1}x_{2}$<br>$+ 2x_{7}^2 + 3x_{7}x_{6}$<br>$+ 2x_{7}x_{3} + 3x_{7}x_{2}$<br>$+ 3x_{6}^2 + 5x_{6}x_{3}$<br>$+ 4x_{6}x_{2} + 3x_{3}^2$<br>$+ 5x_{3}x_{2} + x_{2}^2 \leq 22$ *(quadratic_knapsack)* | / | / | / |
| 140 | 6 | cardinality, cardinality | $x_{3}+ x_{2}+ x_{4}+ x_{1} \geq 2$ *(cardinality)* | $x_{4}+ x_{5}+ x_{0}+ x_{1}+ x_{2} \geq 1$ *(cardinality)* | / | / | / |
| 141 | 4 | quadratic_knapsack, cardinality | $x_{2}+ x_{3}+ x_{0} \leq 2$ *(cardinality)* | $x_{2}^2+ 5x_{2}x_{0}+ 3x_{2}x_{1}$<br>$+ 4x_{0}^2 + 2x_{0}x_{1}+ 4x_{1}^2 \leq 7$ *(quadratic_knapsack)* | / | / | / |
| 142 | 6 | knapsack, cardinality, flow | $x_{3}+ x_{4}+ x_{5}+ x_{2} \leq 2$ *(cardinality)* | $4x_{3}+ 5x_{1}+ 3x_{5}+ 7x_{2}$<br>$+ 6x_{4} \leq 11$ *(knapsack)*<br>$x_{5}+ x_{1}+ x_{4}- x_{2}- x_{0}$<br>$- x_{3} = 0$ *(flow)* | / | / | / |
| 143 | 4 | cardinality, flow, knapsack | $x_{0}+ x_{2}+ x_{3} = 0$ *(cardinality)* | $x_{1}+ x_{0}- x_{3} = 0$ *(flow)*<br>$4x_{3}+ 2x_{0}+ 10x_{2}+ 8x_{1} \leq 13$ *(knapsack)* | / | / | / |
| 144 | 5 | quadratic_knapsack, cardinality, knapsack | $x_{4}^2+ 4x_{4}x_{3}+ x_{4}x_{1}$<br>$+ 2x_{4}x_{0} + 5x_{3}^2$<br>$+ 4x_{3}x_{1} + 3x_{3}x_{0}$<br>$+ 5x_{1}^2 + 4x_{1}x_{0}+ 2x_{0}^2 \leq 18$ *(quadratic_knapsack)* | $x_{4}+ x_{3} \geq 1$ *(cardinality)*<br>$2x_{2}+ 5x_{0}+ 2x_{1} \leq 5$ *(knapsack)* | / | / | / |
| 145 | 4 | quadratic_knapsack, cardinality | $x_{0}+ x_{3}+ x_{1}+ x_{2} \leq 1$ *(cardinality)* | $x_{0}^2+ 4x_{0}x_{1}+ x_{0}x_{2}$<br>$+ 5x_{0}x_{3} + 2x_{1}^2$<br>$+ 3x_{1}x_{2} + 3x_{1}x_{3}$<br>$+ 5x_{2}^2 + 4x_{2}x_{3}+ 5x_{3}^2 \leq 22$ *(quadratic_knapsack)* | / | / | / |
| 146 | 7 | cardinality, quadratic_knapsack, cardinality | $x_{2}+ x_{3}+ x_{5}+ x_{4} = 2$ *(cardinality)* | $5x_{3}^2+ 3x_{3}x_{0}+ x_{3}x_{4}$<br>$+ 5x_{3}x_{1} + 3x_{0}^2$<br>$+ x_{0}x_{4} + 4x_{0}x_{1}$<br>$+ 5x_{4}^2 + 5x_{4}x_{1}+ x_{1}^2 \leq 20$ *(quadratic_knapsack)*<br>$x_{5}+ x_{2}+ x_{0} \geq 0$ *(cardinality)* | / | / | / |
| 147 | 5 | knapsack, cardinality, quadratic_knapsack | $x_{4}+ x_{3}+ x_{2} \leq 3$ *(cardinality)* | $8x_{0}+ 5x_{4}+ 7x_{1}+ 4x_{2} \leq 11$ *(knapsack)*<br>$5x_{3}^2+ 3x_{3}x_{1}+ 5x_{3}x_{0}$<br>$+ 4x_{1}^2 + 5x_{1}x_{0}+ 3x_{0}^2 \leq 14$ *(quadratic_knapsack)* | / | / | / |
| 148 | 4 | cardinality, quadratic_knapsack | $x_{3}+ x_{1}+ x_{0}+ x_{2} \geq 0$ *(cardinality)* | $3x_{0}^2+ x_{0}x_{3}+ 3x_{0}x_{1}$<br>$+ 4x_{0}x_{2} + 5x_{3}^2$<br>$+ 4x_{3}x_{1} + 2x_{3}x_{2}$<br>$+ 2x_{1}^2 + 4x_{1}x_{2}+ 4x_{2}^2 \leq 17$ *(quadratic_knapsack)* | / | / | / |
| 149 | 7 | knapsack, knapsack, quadratic_knapsack | $5x_{1}+ x_{2}+ 10x_{6}+ 4x_{4} \leq 10$ *(knapsack)* | $x_{3}+ 9x_{2}+ 5x_{5}+ 2x_{0}$<br>$+ 10x_{4} \leq 10$ *(knapsack)*<br>$5x_{4}^2+ 3x_{4}x_{5}+ x_{4}x_{1}$<br>$+ 5x_{4}x_{3} + 3x_{5}^2$<br>$+ x_{5}x_{1} + 4x_{5}x_{3}$<br>$+ 5x_{1}^2 + 5x_{1}x_{3}+ x_{3}^2 \leq 20$ *(quadratic_knapsack)* | / | / | / |
| 150 | 8 | cardinality, assignment, cardinality | $x_{5}+ x_{7} \geq 0$ *(cardinality)*<br>$x_{6}+ x_{3}+ x_{1} = 1$ *(assignment)* | $x_{4}+ x_{3}+ x_{1}+ x_{7}+ x_{6} \leq 2$ *(cardinality)* | / | / | / |
| 151 | 6 | cardinality, quadratic_knapsack | $x_{3}+ x_{0}+ x_{5}+ x_{1}+ x_{4} = 2$ *(cardinality)* | $4x_{3}^2+ 3x_{3}x_{1}+ 3x_{3}x_{2}$<br>$+ 2x_{3}x_{0} + 2x_{3}x_{4}$<br>$+ 2x_{1}^2 + 3x_{1}x_{2}$<br>$+ 2x_{1}x_{0} + 3x_{1}x_{4}$<br>$+ 3x_{2}^2 + 5x_{2}x_{0}$<br>$+ 4x_{2}x_{4} + 3x_{0}^2$<br>$+ 5x_{0}x_{4} + x_{4}^2 \leq 22$ *(quadratic_knapsack)* | / | / | / |
| 152 | 6 | quadratic_knapsack, cardinality, cardinality | $x_{0}+ x_{1}+ x_{4}+ x_{3}+ x_{5} \leq 5$ *(cardinality)* | $2x_{0}^2+ 4x_{0}x_{2}+ 4x_{0}x_{3}$<br>$+ 4x_{0}x_{4} + 5x_{0}x_{5}$<br>$+ x_{2}^2 + 2x_{2}x_{3}$<br>$+ 2x_{2}x_{4} + 5x_{2}x_{5}$<br>$+ x_{3}^2 + 3x_{3}x_{4}+ x_{3}x_{5}$<br>$+ 5x_{4}^2 + 4x_{4}x_{5}+ x_{5}^2 \leq 28$ *(quadratic_knapsack)*<br>$x_{2}+ x_{5}+ x_{0} = 2$ *(cardinality)* | / | / | / |
| 153 | 4 | assignment, quadratic_knapsack | $x_{3}+ x_{0} = 1$ *(assignment)* | $3x_{1}^2+ x_{1}x_{3}+ 3x_{1}x_{0}$<br>$+ 4x_{1}x_{2} + 5x_{3}^2$<br>$+ 4x_{3}x_{0} + 2x_{3}x_{2}$<br>$+ 2x_{0}^2 + 4x_{0}x_{2}+ 4x_{2}^2 \leq 17$ *(quadratic_knapsack)* | / | / | / |
| 154 | 7 | knapsack, knapsack, knapsack | $3x_{2}+ 8x_{6}+ 3x_{5} \leq 5$ *(knapsack)* | $7x_{4}+ 7x_{3}+ 10x_{5}+ 2x_{0} \leq 10$ *(knapsack)*<br>$x_{6}+ 6x_{5}+ 9x_{3}+ 8x_{2}$<br>$+ 9x_{0} \leq 18$ *(knapsack)* | / | / | / |
| 155 | 6 | cardinality, cardinality, knapsack | $x_{2}+ x_{1} \leq 2$ *(cardinality)* | $x_{5}+ x_{1}+ x_{0}+ x_{4}+ x_{2} \geq 4$ *(cardinality)*<br>$4x_{1}+ 2x_{2}+ 10x_{5}+ 8x_{3} \leq 13$ *(knapsack)* | / | / | / |
| 156 | 6 | knapsack, knapsack | $2x_{5}+ 6x_{2}+ x_{4} \leq 3$ *(knapsack)* | $8x_{3}+ 10x_{5}+ x_{4}+ 2x_{0} \leq 12$ *(knapsack)* | / | / | / |
| 157 | 5 | cardinality, quadratic_knapsack | $x_{3}+ x_{1}+ x_{2} \geq 0$ *(cardinality)* | $3x_{0}^2+ x_{0}x_{4}+ 3x_{0}x_{3}$<br>$+ 5x_{4}^2 + 2x_{4}x_{3}+ 2x_{3}^2 \leq 8$ *(quadratic_knapsack)* | / | / | / |
| 158 | 7 | cardinality, cardinality | $x_{1}+ x_{0}+ x_{6}+ x_{5}+ x_{3} \geq 4$ *(cardinality)* | $x_{4}+ x_{3}+ x_{2} \geq 0$ *(cardinality)* | / | / | / |
| 159 | 3 | cardinality, flow | $x_{2}+ x_{0}+ x_{1} \leq 0$ *(cardinality)* | $x_{0}+ x_{1}- x_{2} = 0$ *(flow)* | / | / | / |
| 160 | 7 | cardinality, cardinality, flow | $x_{3}+ x_{4}+ x_{1} = 1$ *(cardinality)* | $x_{5}+ x_{1}+ x_{2}+ x_{0}+ x_{4} \leq 5$ *(cardinality)*<br>$x_{6}+ x_{5}+ x_{0}- x_{4}- x_{1}$<br>$- x_{2} = 0$ *(flow)* | / | / | / |
| 161 | 6 | cardinality, knapsack, cardinality | $x_{0}+ x_{3} \geq 1$ *(cardinality)*<br>$x_{1}+ x_{5} = 1$ *(cardinality)* | $x_{3}+ 10x_{2}+ 3x_{0}+ 5x_{4}$<br>$+ x_{5} \leq 12$ *(knapsack)* | / | / | / |
| 162 | 4 | cardinality, cardinality, flow | $x_{2}+ x_{0} \leq 2$ *(cardinality)* | $x_{2}+ x_{3}+ x_{0}+ x_{1} \geq 3$ *(cardinality)*<br>$x_{1}+ x_{2}- x_{0}- x_{3} = 0$ *(flow)* | / | / | / |
| 163 | 8 | cardinality, knapsack, cardinality | $x_{1}+ x_{4}+ x_{5}+ x_{2} \leq 4$ *(cardinality)* | $x_{6}+ x_{5}+ x_{4} \geq 3$ *(cardinality)*<br>$4x_{5}+ 2x_{3}+ 10x_{7}+ 8x_{1} \leq 13$ *(knapsack)* | / | / | / |
| 164 | 4 | knapsack, cardinality | $x_{2}+ x_{0}+ x_{1}+ x_{3} \leq 3$ *(cardinality)* | $2x_{2}+ 7x_{3}+ 7x_{1} \leq 7$ *(knapsack)* | / | / | / |
| 165 | 8 | cardinality, knapsack | $x_{6}+ x_{1}+ x_{5}+ x_{4}+ x_{3} \leq 3$ *(cardinality)* | $8x_{1}+ 10x_{6}+ x_{4}+ 2x_{2} \leq 12$ *(knapsack)* | / | / | / |
| 166 | 5 | cardinality, flow | $x_{3}+ x_{2} = 0$ *(cardinality)* | $x_{2}+ x_{0}+ x_{3}- x_{4} = 0$ *(flow)* | / | / | / |
| 167 | 7 | knapsack, knapsack, knapsack | $2x_{0}+ 10x_{6}+ 4x_{5} \leq 6$ *(knapsack)* | $3x_{2}+ x_{4}+ 4x_{0}+ 8x_{6} \leq 7$ *(knapsack)*<br>$4x_{6}+ 6x_{3}+ 2x_{0}+ 10x_{2} \leq 7$ *(knapsack)* | / | / | / |
| 168 | 8 | quadratic_knapsack, cardinality, quadratic_knapsack | $x_{6}+ x_{1}+ x_{7} = 1$ *(cardinality)* | $4x_{3}^2+ x_{3}x_{4}+ 4x_{3}x_{6}$<br>$+ 3x_{3}x_{1} + 4x_{4}^2$<br>$+ 5x_{4}x_{6} + 4x_{4}x_{1}$<br>$+ 3x_{6}^2 + 5x_{6}x_{1}+ 5x_{1}^2 \leq 24$ *(quadratic_knapsack)*<br>$3x_{4}^2+ x_{4}x_{2}+ 3x_{4}x_{6}$<br>$+ 5x_{2}^2 + 2x_{2}x_{6}+ 2x_{6}^2 \leq 8$ *(quadratic_knapsack)* | / | / | / |
| 169 | 7 | flow, cardinality, quadratic_knapsack | $x_{4}+ x_{6}+ x_{1}- x_{3} = 0$ *(flow)* | $x_{0}+ x_{1}+ x_{2}+ x_{6} \geq 0$ *(cardinality)*<br>$3x_{3}^2+ x_{3}x_{2}+ 3x_{3}x_{4}$<br>$+ 5x_{2}^2 + 2x_{2}x_{4}+ 2x_{4}^2 \leq 8$ *(quadratic_knapsack)* | / | / | / |
| 170 | 6 | knapsack, cardinality, knapsack | $x_{3}+ x_{2}+ x_{1}+ x_{5}+ x_{4} \leq 0$ *(cardinality)* | $5x_{2}+ 4x_{5}+ x_{1} \leq 6$ *(knapsack)*<br>$2x_{2}+ 7x_{3}+ 7x_{1} \leq 7$ *(knapsack)* | / | / | / |
| 171 | 8 | quadratic_knapsack, knapsack, cardinality | $x_{4}^2+ 5x_{4}x_{1}+ x_{4}x_{2}$<br>$+ 3x_{1}^2 + 3x_{1}x_{2}+ 5x_{2}^2 \leq 10$ *(quadratic_knapsack)*<br>$x_{6}+ x_{0}+ x_{3} \geq 0$ *(cardinality)* | $2x_{2}+ 6x_{5}+ x_{3} \leq 3$ *(knapsack)* | / | / | / |
| 172 | 8 | knapsack, cardinality, knapsack | $5x_{0}+ 10x_{6}+ x_{2}+ 9x_{4}$<br>$+ 6x_{7} \leq 19$ *(knapsack)* | $x_{1}+ x_{6}+ x_{7}+ x_{2}+ x_{5} \geq 3$ *(cardinality)*<br>$7x_{0}+ 7x_{4}+ 10x_{5}+ 2x_{3} \leq 10$ *(knapsack)* | / | / | / |
| 173 | 4 | knapsack, cardinality | $x_{1}+ x_{3}+ x_{2} \leq 0$ *(cardinality)* | $10x_{3}+ x_{0}+ 9x_{2} \leq 7$ *(knapsack)* | / | / | / |
| 174 | 7 | quadratic_knapsack, cardinality, knapsack | $x_{5}+ x_{6}+ x_{3}+ x_{2} \leq 0$ *(cardinality)* | $5x_{2}^2+ x_{2}x_{3}+ 3x_{2}x_{0}$<br>$+ x_{2}x_{5} + 4x_{3}^2+ x_{3}x_{0}$<br>$+ 5x_{3}x_{5} + 4x_{0}^2$<br>$+ 2x_{0}x_{5} + 3x_{5}^2 \leq 19$ *(quadratic_knapsack)*<br>$4x_{0}+ 6x_{2}+ x_{6}+ x_{4} \leq 6$ *(knapsack)* | / | / | / |
| 175 | 6 | knapsack, cardinality, quadratic_knapsack | $x_{2}+ x_{4}+ x_{3} \leq 0$ *(cardinality)* | $6x_{5}+ 7x_{1}+ 8x_{4}+ 9x_{2}$<br>$+ 5x_{0} \leq 14$ *(knapsack)*<br>$x_{3}^2+ 5x_{3}x_{1}+ x_{3}x_{5}$<br>$+ 3x_{1}^2 + 3x_{1}x_{5}+ 5x_{5}^2 \leq 10$ *(quadratic_knapsack)* | / | / | / |
| 176 | 6 | knapsack, quadratic_knapsack | $x_{3}+ 10x_{0}+ 3x_{1}+ 5x_{5}$<br>$+ x_{2} \leq 12$ *(knapsack)* | $x_{3}^2+ 5x_{3}x_{0}+ 4x_{3}x_{1}$<br>$+ 2x_{0}^2 + 2x_{0}x_{1}+ 2x_{1}^2 \leq 8$ *(quadratic_knapsack)* | / | / | / |
| 177 | 4 | quadratic_knapsack, quadratic_knapsack, quadratic_knapsack | $4x_{0}^2+ 5x_{0}x_{2}+ 2x_{0}x_{3}$<br>$+ 5x_{0}x_{1} + 4x_{2}^2$<br>$+ 3x_{2}x_{3} + 5x_{2}x_{1}$<br>$+ 3x_{3}^2 + 2x_{3}x_{1}+ 3x_{1}^2 \leq 22$ *(quadratic_knapsack)* | $x_{2}^2+ 2x_{2}x_{0}+ x_{2}x_{1}$<br>$+ x_{0}^2 + 5x_{0}x_{1}+ x_{1}^2 \leq 6$ *(quadratic_knapsack)*<br>$x_{0}^2+ 4x_{0}x_{1}+ x_{0}x_{2}$<br>$+ 2x_{0}x_{3} + 5x_{1}^2$<br>$+ 4x_{1}x_{2} + 3x_{1}x_{3}$<br>$+ 5x_{2}^2 + 4x_{2}x_{3}+ 2x_{3}^2 \leq 18$ *(quadratic_knapsack)* | / | / | / |
| 178 | 5 | knapsack, cardinality, knapsack | $4x_{2}+ 5x_{4}+ 3x_{1}+ 7x_{0}$<br>$+ 6x_{3} \leq 11$ *(knapsack)* | $x_{4}+ x_{0}+ x_{2}+ x_{3}+ x_{1} \geq 2$ *(cardinality)*<br>$9x_{0}+ 2x_{1}+ x_{3}+ 9x_{4}$<br>$+ 6x_{2} \leq 13$ *(knapsack)* | / | / | / |
| 179 | 8 | knapsack, cardinality | $4x_{2}+ 6x_{4}+ 2x_{3}+ 10x_{5} \leq 7$ *(knapsack)* | $x_{2}+ x_{3}+ x_{5}+ x_{4}+ x_{1} \geq 1$ *(cardinality)* | / | / | / |
| 180 | 5 | knapsack, cardinality | $x_{4}+ x_{0}+ x_{1}+ x_{3}+ x_{2} = 0$ *(cardinality)* | $7x_{1}+ 10x_{2}+ 9x_{4} \leq 8$ *(knapsack)* | / | / | / |
| 181 | 8 | quadratic_knapsack, cardinality | $3x_{6}^2+ x_{6}x_{5}+ 5x_{6}x_{3}$<br>$+ 2x_{6}x_{7} + 5x_{6}x_{4}$<br>$+ 2x_{5}^2 + 5x_{5}x_{3}$<br>$+ 5x_{5}x_{7} + 5x_{5}x_{4}$<br>$+ 2x_{3}^2 + 3x_{3}x_{7}$<br>$+ 3x_{3}x_{4} + 4x_{7}^2$<br>$+ x_{7}x_{4} + 5x_{4}^2 \leq 18$ *(quadratic_knapsack)* | $x_{7}+ x_{1}+ x_{0}+ x_{4}+ x_{5} \geq 1$ *(cardinality)* | / | / | / |
| 182 | 8 | cardinality, cardinality, cardinality | $x_{4}+ x_{1} \leq 1$ *(cardinality)* | $x_{7}+ x_{4}+ x_{5}+ x_{3} \geq 4$ *(cardinality)*<br>$x_{1}+ x_{4}+ x_{0}+ x_{2} \leq 2$ *(cardinality)* | / | / | / |
| 183 | 5 | knapsack, quadratic_knapsack, quadratic_knapsack | $x_{3}+ 3x_{1}+ 9x_{4}+ 8x_{0} \leq 10$ *(knapsack)* | $2x_{1}^2+ 5x_{1}x_{2}+ 4x_{1}x_{0}$<br>$+ 2x_{1}x_{4} + 4x_{1}x_{3}$<br>$+ 5x_{2}^2 + 5x_{2}x_{0}$<br>$+ 2x_{2}x_{4} + x_{2}x_{3}$<br>$+ 4x_{0}^2 + x_{0}x_{4}$<br>$+ 2x_{0}x_{3} + 5x_{4}^2$<br>$+ 4x_{4}x_{3} + 2x_{3}^2 \leq 29$ *(quadratic_knapsack)*<br>$4x_{2}^2+ 5x_{2}x_{0}+ 2x_{2}x_{1}$<br>$+ 5x_{2}x_{3} + 4x_{0}^2$<br>$+ 3x_{0}x_{1} + 5x_{0}x_{3}$<br>$+ 3x_{1}^2 + 2x_{1}x_{3}+ 3x_{3}^2 \leq 22$ *(quadratic_knapsack)* | / | / | / |
| 184 | 6 | quadratic_knapsack, quadratic_knapsack | $x_{5}^2+ 4x_{5}x_{4}+ x_{5}x_{3}$<br>$+ 2x_{5}x_{0} + 5x_{4}^2$<br>$+ 4x_{4}x_{3} + 3x_{4}x_{0}$<br>$+ 5x_{3}^2 + 4x_{3}x_{0}+ 2x_{0}^2 \leq 18$ *(quadratic_knapsack)* | $3x_{4}^2+ x_{4}x_{1}+ 3x_{4}x_{5}$<br>$+ 4x_{4}x_{0} + 5x_{1}^2$<br>$+ 4x_{1}x_{5} + 2x_{1}x_{0}$<br>$+ 2x_{5}^2 + 4x_{5}x_{0}+ 4x_{0}^2 \leq 17$ *(quadratic_knapsack)* | / | / | / |
| 185 | 6 | cardinality, cardinality | $x_{2}+ x_{1}+ x_{3}+ x_{4} \geq 0$ *(cardinality)* | $x_{1}+ x_{4}+ x_{2} \geq 1$ *(cardinality)* | / | / | / |
| 186 | 5 | knapsack, cardinality, knapsack | $x_{2}+ x_{3}+ x_{1} = 0$ *(cardinality)* | $2x_{3}+ 10x_{2}+ 4x_{4} \leq 6$ *(knapsack)*<br>$x_{2}+ 10x_{0}+ 3x_{4}+ 5x_{3}$<br>$+ x_{1} \leq 12$ *(knapsack)* | / | / | / |
| 187 | 7 | quadratic_knapsack, knapsack | $4x_{1}^2+ 4x_{1}x_{3}+ x_{1}x_{6}$<br>$+ 5x_{1}x_{5} + 5x_{3}^2$<br>$+ 5x_{3}x_{6} + 5x_{3}x_{5}$<br>$+ x_{6}^2 + 4x_{6}x_{5}+ x_{5}^2 \leq 11$ *(quadratic_knapsack)* | $7x_{2}+ 7x_{6}+ 10x_{5}+ 2x_{4} \leq 10$ *(knapsack)* | / | / | / |
| 188 | 8 | knapsack, quadratic_knapsack | $4x_{5}+ 2x_{7}+ 10x_{0}+ 8x_{6} \leq 13$ *(knapsack)* | $4x_{1}^2+ 3x_{1}x_{5}+ 3x_{1}x_{3}$<br>$+ 2x_{1}x_{0} + 2x_{1}x_{4}$<br>$+ 2x_{5}^2 + 3x_{5}x_{3}$<br>$+ 2x_{5}x_{0} + 3x_{5}x_{4}$<br>$+ 3x_{3}^2 + 5x_{3}x_{0}$<br>$+ 4x_{3}x_{4} + 3x_{0}^2$<br>$+ 5x_{0}x_{4} + x_{4}^2 \leq 22$ *(quadratic_knapsack)* | / | / | / |
| 189 | 7 | knapsack, cardinality, flow | $x_{5}+ x_{2} = 0$ *(cardinality)* | $2x_{1}+ 5x_{5}+ 2x_{6} \leq 5$ *(knapsack)*<br>$x_{1}+ x_{2}+ x_{0}- x_{5}- x_{6}$<br>$- x_{4} = 0$ *(flow)* | / | / | / |
| 190 | 6 | cardinality, cardinality | $x_{5}+ x_{1} \geq 0$ *(cardinality)* | $x_{3}+ x_{0}+ x_{4}+ x_{5}+ x_{2} \geq 2$ *(cardinality)* | / | / | / |
| 191 | 5 | cardinality, knapsack | $x_{4}+ x_{3}+ x_{0} \leq 3$ *(cardinality)* | $9x_{0}+ 2x_{2}+ x_{4}+ 9x_{1}$<br>$+ 6x_{3} \leq 13$ *(knapsack)* | / | / | / |
| 192 | 5 | cardinality, cardinality | $x_{0}+ x_{4}+ x_{1}+ x_{2} \leq 3$ *(cardinality)* | $x_{2}+ x_{1}+ x_{3}+ x_{0}+ x_{4} \geq 1$ *(cardinality)* | / | / | / |
| 193 | 6 | knapsack, knapsack | $4x_{3}+ 5x_{2}+ 3x_{1}+ 7x_{4}$<br>$+ 6x_{0} \leq 11$ *(knapsack)* | $x_{3}+ 6x_{4}+ 9x_{5}+ 8x_{0}$<br>$+ 9x_{2} \leq 18$ *(knapsack)* | / | / | / |
| 194 | 6 | quadratic_knapsack, cardinality | $x_{2}+ x_{1}+ x_{4} \leq 2$ *(cardinality)* | $4x_{5}^2+ 5x_{5}x_{3}+ 2x_{5}x_{2}$<br>$+ 5x_{5}x_{1} + 4x_{3}^2$<br>$+ 3x_{3}x_{2} + 5x_{3}x_{1}$<br>$+ 3x_{2}^2 + 2x_{2}x_{1}+ 3x_{1}^2 \leq 22$ *(quadratic_knapsack)* | / | / | / |
| 195 | 6 | cardinality, knapsack | $x_{5}+ x_{0}+ x_{2}+ x_{4} \geq 0$ *(cardinality)* | $8x_{4}+ 10x_{5}+ x_{1}+ 2x_{2} \leq 12$ *(knapsack)* | / | / | / |
| 196 | 6 | knapsack, cardinality, quadratic_knapsack | $6x_{3}+ 7x_{5}+ 8x_{0}+ 9x_{4}$<br>$+ 5x_{1} \leq 14$ *(knapsack)* | $x_{5}+ x_{4} \geq 0$ *(cardinality)*<br>$5x_{0}^2+ x_{0}x_{4}+ 3x_{0}x_{5}$<br>$+ x_{0}x_{1} + 4x_{4}^2+ x_{4}x_{5}$<br>$+ 5x_{4}x_{1} + 4x_{5}^2$<br>$+ 2x_{5}x_{1} + 3x_{1}^2 \leq 19$ *(quadratic_knapsack)* | / | / | / |
| 197 | 8 | cardinality, quadratic_knapsack | $x_{5}+ x_{4}+ x_{7}+ x_{1}+ x_{3} \geq 1$ *(cardinality)* | $2x_{1}^2+ 5x_{1}x_{0}+ 4x_{1}x_{4}$<br>$+ 4x_{1}x_{6} + 3x_{1}x_{5}$<br>$+ 4x_{0}^2 + 4x_{0}x_{4}$<br>$+ x_{0}x_{6} + 2x_{0}x_{5}$<br>$+ 3x_{4}^2 + x_{4}x_{6}$<br>$+ 4x_{4}x_{5} + 3x_{6}^2$<br>$+ 4x_{6}x_{5} + 3x_{5}^2 \leq 16$ *(quadratic_knapsack)* | / | / | / |
| 198 | 4 | cardinality, flow | $x_{3}+ x_{2} \leq 2$ *(cardinality)* | $x_{2}+ x_{3}- x_{1}- x_{0} = 0$ *(flow)* | / | / | / |
| 199 | 6 | knapsack, knapsack | $x_{2}+ 10x_{0}+ 3x_{5}+ 5x_{4}$<br>$+ x_{3} \leq 12$ *(knapsack)* | $7x_{1}+ 7x_{4}+ 10x_{2}+ 2x_{5} \leq 10$ *(knapsack)* | / | / | / |
| 200 | 6 | cardinality, quadratic_knapsack | $x_{5}+ x_{1}+ x_{0}+ x_{3}+ x_{2} = 3$ *(cardinality)* | $5x_{3}^2+ x_{3}x_{0}+ 2x_{3}x_{5}$<br>$+ 5x_{0}^2 + x_{0}x_{5}+ 3x_{5}^2 \leq 6$ *(quadratic_knapsack)* | / | / | / |
| 201 | 7 | cardinality, quadratic_knapsack | $x_{5}+ x_{6}+ x_{1}+ x_{0}+ x_{3} = 3$ *(cardinality)* | $x_{5}^2+ 5x_{5}x_{1}+ 3x_{5}x_{0}$<br>$+ 4x_{1}^2 + 2x_{1}x_{0}+ 4x_{0}^2 \leq 7$ *(quadratic_knapsack)* | / | / | / |
| 202 | 7 | flow, cardinality, knapsack | $x_{1}+ x_{2}+ x_{6}- x_{5}- x_{3} = 0$ *(flow)* | $x_{0}+ x_{6}+ x_{2}+ x_{3}+ x_{1} \leq 2$ *(cardinality)*<br>$4x_{4}+ 6x_{0}+ x_{2}+ x_{5} \leq 6$ *(knapsack)* | / | / | / |
| 203 | 8 | knapsack, cardinality, cardinality | $x_{1}+ x_{4}+ x_{5}+ x_{0}+ x_{7} \leq 4$ *(cardinality)* | $4x_{2}+ 6x_{7}+ 2x_{1}+ 10x_{6} \leq 7$ *(knapsack)*<br>$x_{0}+ x_{7} \geq 1$ *(cardinality)* | / | / | / |
| 204 | 4 | cardinality, knapsack | $x_{3}+ x_{0} \leq 2$ *(cardinality)* | $10x_{1}+ 8x_{0}+ 10x_{3} \leq 9$ *(knapsack)* | / | / | / |
| 205 | 8 | quadratic_knapsack, knapsack, cardinality | $3x_{5}^2+ x_{5}x_{4}+ 3x_{5}x_{3}$<br>$+ 4x_{5}x_{1} + 5x_{4}^2$<br>$+ 4x_{4}x_{3} + 2x_{4}x_{1}$<br>$+ 2x_{3}^2 + 4x_{3}x_{1}+ 4x_{1}^2 \leq 17$ *(quadratic_knapsack)* | $x_{3}+ 4x_{1}+ 8x_{0}+ 10x_{7}$<br>$+ 8x_{5} \leq 16$ *(knapsack)*<br>$x_{3}+ x_{4}+ x_{2}+ x_{0} \geq 0$ *(cardinality)* | / | / | / |
| 206 | 5 | quadratic_knapsack, quadratic_knapsack | $x_{2}^2+ x_{2}x_{0}+ 3x_{2}x_{3}$<br>$+ 2x_{2}x_{4} + x_{0}^2$<br>$+ 2x_{0}x_{3} + 3x_{0}x_{4}$<br>$+ 2x_{3}^2 + 3x_{3}x_{4}+ 3x_{4}^2 \leq 7$ *(quadratic_knapsack)* | $x_{3}^2+ 5x_{3}x_{4}+ 4x_{3}x_{0}$<br>$+ 2x_{4}^2 + 2x_{4}x_{0}+ 2x_{0}^2 \leq 8$ *(quadratic_knapsack)* | / | / | / |
| 207 | 4 | cardinality, cardinality, flow | $x_{3}+ x_{2}+ x_{1} = 2$ *(cardinality)* | $x_{3}+ x_{2} \leq 2$ *(cardinality)*<br>$x_{0}+ x_{2}- x_{3} = 0$ *(flow)* | / | / | / |
| 208 | 8 | knapsack, quadratic_knapsack | $x_{1}+ 10x_{4}+ 3x_{6}+ 5x_{0}$<br>$+ x_{3} \leq 12$ *(knapsack)* | $3x_{2}^2+ x_{2}x_{1}+ 5x_{2}x_{7}$<br>$+ 2x_{2}x_{5} + 5x_{2}x_{4}$<br>$+ 2x_{1}^2 + 5x_{1}x_{7}$<br>$+ 5x_{1}x_{5} + 5x_{1}x_{4}$<br>$+ 2x_{7}^2 + 3x_{7}x_{5}$<br>$+ 3x_{7}x_{4} + 4x_{5}^2$<br>$+ x_{5}x_{4} + 5x_{4}^2 \leq 18$ *(quadratic_knapsack)* | / | / | / |
| 209 | 6 | cardinality, knapsack, cardinality | $x_{0}+ x_{1}+ x_{4}+ x_{2}+ x_{5} = 4$ *(cardinality)* | $2x_{1}+ 10x_{0}+ 4x_{5} \leq 6$ *(knapsack)*<br>$x_{3}+ x_{5}+ x_{2}+ x_{1} \geq 3$ *(cardinality)* | / | / | / |
| 210 | 7 | assignment, quadratic_knapsack, cardinality | $x_{2}+ x_{3} = 1$ *(assignment)* | $5x_{2}^2+ 3x_{2}x_{0}+ x_{2}x_{5}$<br>$+ 5x_{2}x_{1} + 3x_{0}^2$<br>$+ x_{0}x_{5} + 4x_{0}x_{1}$<br>$+ 5x_{5}^2 + 5x_{5}x_{1}+ x_{1}^2 \leq 20$ *(quadratic_knapsack)*<br>$x_{6}+ x_{5}+ x_{2}+ x_{0} \leq 0$ *(cardinality)* | / | / | / |
| 211 | 4 | cardinality, cardinality, quadratic_knapsack | $x_{2}+ x_{1}+ x_{0} = 2$ *(cardinality)* | $x_{2}+ x_{1}+ x_{0} \geq 1$ *(cardinality)*<br>$3x_{0}^2+ x_{0}x_{1}+ 3x_{0}x_{2}$<br>$+ x_{1}^2 + 4x_{1}x_{2}+ 3x_{2}^2 \leq 5$ *(quadratic_knapsack)* | / | / | / |
| 212 | 7 | quadratic_knapsack, cardinality | $2x_{5}^2+ 5x_{5}x_{6}+ 4x_{5}x_{0}$<br>$+ 2x_{5}x_{2} + 4x_{5}x_{4}$<br>$+ 5x_{6}^2 + 5x_{6}x_{0}$<br>$+ 2x_{6}x_{2} + x_{6}x_{4}$<br>$+ 4x_{0}^2 + x_{0}x_{2}$<br>$+ 2x_{0}x_{4} + 5x_{2}^2$<br>$+ 4x_{2}x_{4} + 2x_{4}^2 \leq 29$ *(quadratic_knapsack)* | $x_{0}+ x_{6}+ x_{1}+ x_{3}+ x_{5} \geq 3$ *(cardinality)* | / | / | / |
| 213 | 7 | cardinality, cardinality, flow | $x_{0}+ x_{1}+ x_{4} \leq 2$ *(cardinality)*<br>$x_{3}+ x_{5} \leq 2$ *(cardinality)* | $x_{3}+ x_{6}+ x_{0}- x_{5} = 0$ *(flow)* | / | / | / |
| 214 | 6 | cardinality, cardinality | $x_{2}+ x_{5}+ x_{3}+ x_{0}+ x_{4} \leq 3$ *(cardinality)* | $x_{4}+ x_{5}+ x_{1}+ x_{2} \leq 1$ *(cardinality)* | / | / | / |
| 215 | 8 | cardinality, quadratic_knapsack, knapsack | $x_{0}+ x_{7}+ x_{2}+ x_{3}+ x_{1} \geq 4$ *(cardinality)* | $2x_{6}^2+ 5x_{6}x_{1}+ 2x_{6}x_{5}$<br>$+ x_{6}x_{4} + 4x_{6}x_{7}$<br>$+ 4x_{1}^2 + 2x_{1}x_{5}$<br>$+ 5x_{1}x_{4} + x_{1}x_{7}+ x_{5}^2$<br>$+ 4x_{5}x_{4} + 2x_{5}x_{7}$<br>$+ 5x_{4}^2 + 4x_{4}x_{7}+ x_{7}^2 \leq 23$ *(quadratic_knapsack)*<br>$8x_{0}+ 10x_{3}+ x_{5}+ 2x_{2} \leq 12$ *(knapsack)* | / | / | / |
| 216 | 7 | cardinality, flow, knapsack | $x_{3}+ x_{2}+ x_{1}+ x_{0}+ x_{4} \leq 2$ *(cardinality)* | $x_{4}- x_{1}- x_{0}- x_{2} = 0$ *(flow)*<br>$4x_{0}+ 2x_{3}+ 10x_{4}+ 8x_{2} \leq 13$ *(knapsack)* | / | / | / |
| 217 | 6 | cardinality, flow, quadratic_knapsack | $x_{1}+ x_{5}+ x_{2}+ x_{3}+ x_{4} \leq 3$ *(cardinality)* | $x_{4}+ x_{0}- x_{2}- x_{3}- x_{5} = 0$ *(flow)*<br>$2x_{4}^2+ 4x_{4}x_{1}+ 5x_{4}x_{3}$<br>$+ x_{4}x_{2} + 4x_{4}x_{5}$<br>$+ 2x_{1}^2 + 4x_{1}x_{3}$<br>$+ 4x_{1}x_{2} + x_{1}x_{5}$<br>$+ 2x_{3}^2 + 3x_{3}x_{2}$<br>$+ 3x_{3}x_{5} + 3x_{2}^2$<br>$+ 2x_{2}x_{5} + x_{5}^2 \leq 19$ *(quadratic_knapsack)* | / | / | / |
| 218 | 4 | cardinality, cardinality, cardinality | $x_{1}+ x_{0} \leq 1$ *(cardinality)* | $x_{3}+ x_{0}+ x_{1} = 2$ *(cardinality)*<br>$x_{1}+ x_{0} \geq 1$ *(cardinality)* | / | / | / |
| 219 | 5 | flow, quadratic_knapsack | $x_{0}+ x_{4}- x_{3} = 0$ *(flow)* | $x_{4}^2+ 5x_{4}x_{1}+ 2x_{4}x_{0}$<br>$+ 2x_{4}x_{3} + 3x_{1}^2$<br>$+ 2x_{1}x_{0} + 3x_{1}x_{3}$<br>$+ 2x_{0}^2 + 4x_{0}x_{3}+ 5x_{3}^2 \leq 13$ *(quadratic_knapsack)* | / | / | / |
| 220 | 7 | quadratic_knapsack, flow, quadratic_knapsack | $x_{2}+ x_{6}+ x_{1}- x_{4} = 0$ *(flow)* | $x_{4}^2+ 2x_{4}x_{0}+ x_{4}x_{2}$<br>$+ x_{0}^2 + 5x_{0}x_{2}+ x_{2}^2 \leq 6$ *(quadratic_knapsack)*<br>$x_{5}^2+ 5x_{5}x_{3}+ 4x_{5}x_{6}$<br>$+ 2x_{3}^2 + 2x_{3}x_{6}+ 2x_{6}^2 \leq 8$ *(quadratic_knapsack)* | / | / | / |
| 221 | 4 | quadratic_knapsack, assignment | $x_{3}+ x_{1} = 1$ *(assignment)* | $x_{1}^2+ 2x_{1}x_{2}+ x_{1}x_{0}$<br>$+ x_{2}^2 + 5x_{2}x_{0}+ x_{0}^2 \leq 6$ *(quadratic_knapsack)* | / | / | / |
| 222 | 4 | cardinality, quadratic_knapsack | $x_{3}+ x_{2}+ x_{0}+ x_{1} = 1$ *(cardinality)* | $5x_{2}^2+ 3x_{2}x_{3}+ 5x_{2}x_{0}$<br>$+ 4x_{3}^2 + 5x_{3}x_{0}+ 3x_{0}^2 \leq 14$ *(quadratic_knapsack)* | / | / | / |
| 223 | 5 | flow, knapsack | $x_{2}+ x_{0}- x_{3}- x_{1}- x_{4} = 0$ *(flow)* | $2x_{1}+ 6x_{0}+ x_{2} \leq 3$ *(knapsack)* | / | / | / |
| 224 | 8 | quadratic_knapsack, cardinality | $5x_{6}^2+ 3x_{6}x_{2}+ 4x_{6}x_{4}$<br>$+ 5x_{6}x_{3} + x_{6}x_{0}+ x_{2}^2$<br>$+ 4x_{2}x_{4} + 2x_{2}x_{3}$<br>$+ 3x_{2}x_{0} + 4x_{4}^2$<br>$+ 5x_{4}x_{3} + x_{4}x_{0}+ x_{3}^2$<br>$+ 5x_{3}x_{0} + 2x_{0}^2 \leq 17$ *(quadratic_knapsack)* | $x_{7}+ x_{6}+ x_{4}+ x_{1}+ x_{5} \geq 4$ *(cardinality)* | / | / | / |
| 225 | 5 | quadratic_knapsack, quadratic_knapsack | $x_{2}^2+ 5x_{2}x_{0}+ 2x_{2}x_{1}$<br>$+ 2x_{2}x_{4} + 3x_{0}^2$<br>$+ 2x_{0}x_{1} + 3x_{0}x_{4}$<br>$+ 2x_{1}^2 + 4x_{1}x_{4}+ 5x_{4}^2 \leq 13$ *(quadratic_knapsack)* | $2x_{0}^2+ 4x_{0}x_{4}+ 4x_{0}x_{1}$<br>$+ 4x_{0}x_{3} + 5x_{0}x_{2}$<br>$+ x_{4}^2 + 2x_{4}x_{1}$<br>$+ 2x_{4}x_{3} + 5x_{4}x_{2}$<br>$+ x_{1}^2 + 3x_{1}x_{3}+ x_{1}x_{2}$<br>$+ 5x_{3}^2 + 4x_{3}x_{2}+ x_{2}^2 \leq 28$ *(quadratic_knapsack)* | / | / | / |
| 226 | 4 | knapsack, flow | $x_{3}+ x_{2}+ x_{0}- x_{1} = 0$ *(flow)* | $8x_{2}+ 10x_{3}+ x_{1}+ 2x_{0} \leq 12$ *(knapsack)* | / | / | / |
| 227 | 4 | cardinality, cardinality | $x_{3}+ x_{0}+ x_{2} = 3$ *(cardinality)* | $x_{2}+ x_{0} \geq 0$ *(cardinality)* | / | / | / |
| 228 | 8 | cardinality, quadratic_knapsack | $x_{3}+ x_{0}+ x_{6}+ x_{1}+ x_{4} = 1$ *(cardinality)* | $2x_{4}^2+ 5x_{4}x_{0}+ 4x_{4}x_{3}$<br>$+ 2x_{4}x_{6} + 4x_{4}x_{2}$<br>$+ 5x_{0}^2 + 5x_{0}x_{3}$<br>$+ 2x_{0}x_{6} + x_{0}x_{2}$<br>$+ 4x_{3}^2 + x_{3}x_{6}$<br>$+ 2x_{3}x_{2} + 5x_{6}^2$<br>$+ 4x_{6}x_{2} + 2x_{2}^2 \leq 29$ *(quadratic_knapsack)* | / | / | / |
| 229 | 5 | knapsack, flow, quadratic_knapsack | $x_{2}- x_{3} = 0$ *(flow)* | $7x_{0}+ 10x_{3}+ 9x_{4} \leq 8$ *(knapsack)*<br>$x_{2}^2+ 5x_{2}x_{1}+ 4x_{2}x_{3}$<br>$+ 2x_{1}^2 + 2x_{1}x_{3}+ 2x_{3}^2 \leq 8$ *(quadratic_knapsack)* | / | / | / |
| 230 | 5 | knapsack, knapsack, cardinality | $5x_{2}+ 4x_{0}+ x_{3} \leq 6$ *(knapsack)* | $3x_{0}+ 8x_{4}+ 3x_{2} \leq 5$ *(knapsack)*<br>$x_{2}+ x_{3}+ x_{4}+ x_{0} \geq 2$ *(cardinality)* | / | / | / |
| 231 | 8 | quadratic_knapsack, cardinality | $2x_{7}^2+ 4x_{7}x_{4}+ 5x_{7}x_{1}$<br>$+ x_{7}x_{3} + 4x_{7}x_{2}$<br>$+ 2x_{4}^2 + 4x_{4}x_{1}$<br>$+ 4x_{4}x_{3} + x_{4}x_{2}$<br>$+ 2x_{1}^2 + 3x_{1}x_{3}$<br>$+ 3x_{1}x_{2} + 3x_{3}^2$<br>$+ 2x_{3}x_{2} + x_{2}^2 \leq 19$ *(quadratic_knapsack)* | $x_{1}+ x_{3}+ x_{5}+ x_{0}+ x_{7} \geq 1$ *(cardinality)* | / | / | / |
| 232 | 6 | cardinality, flow, independent_set | $x_{1}+ x_{5}+ x_{4}+ x_{0} = 1$ *(cardinality)* | $x_{3}+ x_{0}+ x_{5}- x_{4}- x_{2} = 0$ *(flow)*<br>$x_{4}x_{2} = 0$ *(independent_set)* | / | / | / |
| 233 | 7 | cardinality, cardinality | $x_{3}+ x_{4}+ x_{2} = 2$ *(cardinality)* | $x_{6}+ x_{2}+ x_{4}+ x_{3}+ x_{0} \geq 1$ *(cardinality)* | / | / | / |
| 234 | 5 | knapsack, knapsack, knapsack | $5x_{1}+ x_{2}+ 10x_{4}+ 4x_{0} \leq 10$ *(knapsack)* | $10x_{4}+ 8x_{3}+ 2x_{0}+ x_{1}$<br>$+ 4x_{2} \leq 10$ *(knapsack)*<br>$10x_{1}+ 8x_{0}+ 10x_{2} \leq 9$ *(knapsack)* | / | / | / |
| 235 | 7 | flow, cardinality, cardinality | $x_{6}- x_{2} = 0$ *(flow)* | $x_{2}+ x_{6}+ x_{3}+ x_{0}+ x_{4} \leq 5$ *(cardinality)*<br>$x_{1}+ x_{2} = 2$ *(cardinality)* | / | / | / |
| 236 | 5 | quadratic_knapsack, quadratic_knapsack | $4x_{1}^2+ x_{1}x_{4}+ 4x_{1}x_{0}$<br>$+ 3x_{1}x_{2} + 4x_{4}^2$<br>$+ 5x_{4}x_{0} + 4x_{4}x_{2}$<br>$+ 3x_{0}^2 + 5x_{0}x_{2}+ 5x_{2}^2 \leq 24$ *(quadratic_knapsack)* | $2x_{1}^2+ 4x_{1}x_{0}+ 4x_{1}x_{4}$<br>$+ 4x_{1}x_{3} + 5x_{1}x_{2}$<br>$+ x_{0}^2 + 2x_{0}x_{4}$<br>$+ 2x_{0}x_{3} + 5x_{0}x_{2}$<br>$+ x_{4}^2 + 3x_{4}x_{3}+ x_{4}x_{2}$<br>$+ 5x_{3}^2 + 4x_{3}x_{2}+ x_{2}^2 \leq 28$ *(quadratic_knapsack)* | / | / | / |
| 237 | 8 | cardinality, quadratic_knapsack, knapsack | $x_{1}+ x_{5}+ x_{0}+ x_{7}+ x_{2} \leq 5$ *(cardinality)* | $2x_{7}^2+ 5x_{7}x_{4}+ 4x_{7}x_{3}$<br>$+ 2x_{7}x_{1} + 4x_{7}x_{5}$<br>$+ 5x_{4}^2 + 5x_{4}x_{3}$<br>$+ 2x_{4}x_{1} + x_{4}x_{5}$<br>$+ 4x_{3}^2 + x_{3}x_{1}$<br>$+ 2x_{3}x_{5} + 5x_{1}^2$<br>$+ 4x_{1}x_{5} + 2x_{5}^2 \leq 29$ *(quadratic_knapsack)*<br>$3x_{5}+ 8x_{2}+ 3x_{1} \leq 5$ *(knapsack)* | / | / | / |
| 238 | 6 | cardinality, cardinality | $x_{0}+ x_{1}+ x_{4}+ x_{5}+ x_{2} = 0$ *(cardinality)* | $x_{5}+ x_{1} \leq 0$ *(cardinality)* | / | / | / |
| 239 | 7 | flow, cardinality, quadratic_knapsack | $x_{3}- x_{2} = 0$ *(flow)* | $x_{1}+ x_{3}+ x_{2}+ x_{5}+ x_{4} \geq 0$ *(cardinality)*<br>$4x_{0}^2+ 3x_{0}x_{6}+ 3x_{0}x_{4}$<br>$+ 2x_{0}x_{2} + 2x_{0}x_{3}$<br>$+ 2x_{6}^2 + 3x_{6}x_{4}$<br>$+ 2x_{6}x_{2} + 3x_{6}x_{3}$<br>$+ 3x_{4}^2 + 5x_{4}x_{2}$<br>$+ 4x_{4}x_{3} + 3x_{2}^2$<br>$+ 5x_{2}x_{3} + x_{3}^2 \leq 22$ *(quadratic_knapsack)* | / | / | / |
| 240 | 8 | cardinality, quadratic_knapsack, cardinality | $x_{1}+ x_{2}+ x_{6}+ x_{5}+ x_{4} = 2$ *(cardinality)* | $2x_{4}^2+ 5x_{4}x_{5}+ 2x_{4}x_{6}$<br>$+ 2x_{5}^2 + x_{5}x_{6}+ x_{6}^2 \leq 7$ *(quadratic_knapsack)*<br>$x_{4}+ x_{1}+ x_{6} \leq 0$ *(cardinality)* | / | / | / |
| 241 | 4 | flow, cardinality | $x_{2}- x_{3}- x_{0} = 0$ *(flow)* | $x_{0}+ x_{2} \geq 1$ *(cardinality)* | / | / | / |
| 242 | 7 | quadratic_knapsack, quadratic_knapsack | $2x_{1}^2+ 4x_{1}x_{2}+ 5x_{1}x_{0}$<br>$+ x_{1}x_{3} + 4x_{1}x_{4}$<br>$+ 2x_{2}^2 + 4x_{2}x_{0}$<br>$+ 4x_{2}x_{3} + x_{2}x_{4}$<br>$+ 2x_{0}^2 + 3x_{0}x_{3}$<br>$+ 3x_{0}x_{4} + 3x_{3}^2$<br>$+ 2x_{3}x_{4} + x_{4}^2 \leq 19$ *(quadratic_knapsack)* | $x_{5}^2+ 5x_{5}x_{2}+ 4x_{5}x_{4}$<br>$+ 2x_{2}^2 + 2x_{2}x_{4}+ 2x_{4}^2 \leq 8$ *(quadratic_knapsack)* | / | / | / |
| 243 | 7 | knapsack, cardinality, cardinality | $x_{2}+ x_{6}+ x_{4} = 2$ *(cardinality)* | $x_{1}+ 6x_{5}+ 9x_{2}+ 8x_{3}$<br>$+ 9x_{6} \leq 18$ *(knapsack)*<br>$x_{2}+ x_{3}+ x_{0}+ x_{5} \leq 2$ *(cardinality)* | / | / | / |
| 244 | 5 | knapsack, quadratic_knapsack | $10x_{4}+ 8x_{0}+ 10x_{2} \leq 9$ *(knapsack)* | $5x_{3}^2+ 3x_{3}x_{2}+ 4x_{3}x_{1}$<br>$+ 5x_{3}x_{0} + x_{3}x_{4}+ x_{2}^2$<br>$+ 4x_{2}x_{1} + 2x_{2}x_{0}$<br>$+ 3x_{2}x_{4} + 4x_{1}^2$<br>$+ 5x_{1}x_{0} + x_{1}x_{4}+ x_{0}^2$<br>$+ 5x_{0}x_{4} + 2x_{4}^2 \leq 17$ *(quadratic_knapsack)* | / | / | / |
| 245 | 4 | cardinality, knapsack | $x_{1}+ x_{0} = 0$ *(cardinality)* | $2x_{1}+ 10x_{3}+ 4x_{0} \leq 6$ *(knapsack)* | / | / | / |
| 246 | 8 | knapsack, knapsack, cardinality | $x_{7}+ 10x_{3}+ 3x_{0}+ 5x_{5}$<br>$+ x_{1} \leq 12$ *(knapsack)* | $6x_{5}+ 2x_{0}+ 2x_{2} \leq 3$ *(knapsack)*<br>$x_{3}+ x_{0}+ x_{5}+ x_{6}+ x_{1} \geq 3$ *(cardinality)* | / | / | / |
| 247 | 8 | cardinality, cardinality, cardinality | $x_{4}+ x_{1}+ x_{0}+ x_{5}+ x_{2} = 3$ *(cardinality)* | $x_{4}+ x_{3}+ x_{5} \leq 3$ *(cardinality)*<br>$x_{5}+ x_{3}+ x_{7} \leq 0$ *(cardinality)* | / | / | / |
| 248 | 6 | flow, knapsack, cardinality | $x_{1}- x_{3}- x_{0} = 0$ *(flow)* | $7x_{1}+ 10x_{5}+ 9x_{4} \leq 8$ *(knapsack)*<br>$x_{1}+ x_{0}+ x_{5}+ x_{4}+ x_{3} \leq 3$ *(cardinality)* | / | / | / |
| 249 | 5 | quadratic_knapsack, knapsack, cardinality | $2x_{4}^2+ 5x_{4}x_{0}+ 2x_{4}x_{2}$<br>$+ x_{4}x_{3} + 4x_{4}x_{1}$<br>$+ 4x_{0}^2 + 2x_{0}x_{2}$<br>$+ 5x_{0}x_{3} + x_{0}x_{1}+ x_{2}^2$<br>$+ 4x_{2}x_{3} + 2x_{2}x_{1}$<br>$+ 5x_{3}^2 + 4x_{3}x_{1}+ x_{1}^2 \leq 23$ *(quadratic_knapsack)* | $x_{0}+ 10x_{4}+ 3x_{1}+ 5x_{2}$<br>$+ x_{3} \leq 12$ *(knapsack)*<br>$x_{0}+ x_{2}+ x_{1}+ x_{4} \geq 1$ *(cardinality)* | / | / | / |

---

## Results

`H` = HybridQAOA, `P` = PenaltyQAOA. Columns: AR$_f$ = AR$_{\text{feas}}$, $P_f$ = $P(\text{feas})$, $P_o$ = $P(\text{opt})$.

| COP | $n_x$ | Method | $p=1$ AR$_f$ | $p=1$ $P_f$ | $p=1$ $P_o$ | $p=2$ AR$_f$ | $p=2$ $P_f$ | $p=2$ $P_o$ | $p=3$ AR$_f$ | $p=3$ $P_f$ | $p=3$ $P_o$ | $p=4$ AR$_f$ | $p=4$ $P_f$ | $p=4$ $P_o$ | $p=5$ AR$_f$ | $p=5$ $P_f$ | $p=5$ $P_o$ |
|-----|-------|--------|--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---|
| 0 | 5 | `H` | 0.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| 0 | 5 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 1 | 4 | `H` |  | 0.000 | 0.000 |  | 0.000 | 0.000 |  | 0.000 | 0.000 |  | 0.000 | 0.000 |  | 0.000 | 0.000 |
| 1 | 4 | `P` |  | 0.493 | 0.493 |  | 0.980 | 0.980 |  | 0.997 | 0.997 |  | 0.997 | 0.997 |  | 0.998 | 0.998 |
| 2 | 7 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 2 | 7 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 3 | 3 | `H` |  | 0.530 | 0.530 |  | 0.905 | 0.905 |  | 0.999 | 0.999 |  | 1.000 | 1.000 |  | 1.000 | 1.000 |
| 3 | 3 | `P` |  | 0.127 | 0.127 |  | 0.234 | 0.234 |  | 0.126 | 0.126 |  | 0.013 | 0.013 |  | 0.239 | 0.239 |
| 4 | 6 | `H` |  | 0.000 | 0.000 |  | 0.000 | 0.000 |  | 0.000 | 0.000 |  | 0.000 | 0.000 |  | 0.000 | 0.000 |
| 4 | 6 | `P` | 0.278 | 0.437 | 0.034 | 0.292 | 0.670 | 0.010 | 0.245 | 0.386 | 0.057 | 0.262 | 0.367 | 0.172 | 0.367 | 0.670 | 0.155 |
| 5 | 8 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 5 | 8 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 6 | 5 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 6 | 5 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 7 | 3 | `H` | 0.531 | 0.832 | 0.300 | 0.521 | 0.833 | 0.306 | 0.430 | 0.857 | 0.146 | 0.590 | 0.768 | 0.390 | 0.544 | 0.847 | 0.311 |
| 7 | 3 | `P` | 0.458 | 0.687 | 0.251 | 0.438 | 0.654 | 0.257 | 0.439 | 0.627 | 0.258 | 0.406 | 0.624 | 0.244 | 0.426 | 0.618 | 0.244 |
| 8 | 8 | `H` | 0.006 | 0.951 | 0.000 | 0.045 | 0.993 | 0.000 | 0.005 | 0.978 | 0.000 | 0.024 | 0.963 | 0.000 | 0.077 | 0.959 | 0.000 |
| 8 | 8 | `P` | 0.059 | 0.115 | 0.031 | 0.020 | 0.054 | 0.003 | 0.021 | 0.051 | 0.004 | 0.021 | 0.055 | 0.003 | 0.024 | 0.056 | 0.005 |
| 9 | 6 | `H` | 0.099 | 0.572 | 0.000 | 0.059 | 0.480 | 0.000 | 0.006 | 0.022 | 0.000 | 0.081 | 0.911 | 0.000 | 0.024 | 0.102 | 0.000 |
| 9 | 6 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 10 | 8 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 10 | 8 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 11 | 5 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 11 | 5 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 12 | 6 | `H` | 0.426 | 0.751 | 0.057 | 0.430 | 0.755 | 0.066 | 0.438 | 0.763 | 0.067 | 0.432 | 0.763 | 0.070 | 0.431 | 0.757 | 0.060 |
| 12 | 6 | `P` | 0.218 | 0.419 | 0.019 | 0.205 | 0.378 | 0.023 | 0.324 | 0.554 | 0.094 | 0.342 | 0.579 | 0.067 | 0.179 | 0.331 | 0.041 |
| 13 | 5 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 13 | 5 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 14 | 8 | `H` | 0.199 | 0.289 | 0.000 | 0.224 | 0.321 | 0.000 | 0.249 | 0.353 | 0.000 | 0.243 | 0.349 | 0.000 | 0.256 | 0.371 | 0.000 |
| 14 | 8 | `P` | 0.258 | 0.331 | 0.006 | 0.368 | 0.487 | 0.041 | 0.422 | 0.544 | 0.110 | 0.342 | 0.502 | 0.000 | 0.140 | 0.211 | 0.016 |
| 15 | 8 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 15 | 8 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 16 | 6 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 16 | 6 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 17 | 6 | `H` | 0.384 | 0.802 | 0.000 | 0.398 | 0.791 | 0.000 | 0.400 | 0.791 | 0.000 | 0.394 | 0.791 | 0.000 | 0.366 | 0.771 | 0.000 |
| 17 | 6 | `P` | 0.562 | 0.970 | 0.125 | 0.515 | 0.838 | 0.023 | 0.620 | 0.898 | 0.017 | 0.743 | 0.963 | 0.220 | 0.771 | 0.987 | 0.184 |
| 18 | 5 | `H` | 0.097 | 0.206 | 0.000 | 0.038 | 0.082 | 0.000 | 0.299 | 0.635 | 0.000 | 0.022 | 0.048 | 0.000 | 0.026 | 0.055 | 0.000 |
| 18 | 5 | `P` | 0.066 | 0.160 | 0.019 | 0.058 | 0.142 | 0.022 | 0.066 | 0.148 | 0.031 | 0.069 | 0.162 | 0.028 | 0.068 | 0.153 | 0.036 |
| 19 | 7 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 19 | 7 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 20 | 6 | `H` | 0.243 | 0.523 | 0.064 | 0.230 | 0.532 | 0.050 | 0.270 | 0.557 | 0.061 | 0.252 | 0.569 | 0.050 | 0.265 | 0.546 | 0.100 |
| 20 | 6 | `P` | 0.119 | 0.323 | 0.021 | 0.140 | 0.324 | 0.038 | 0.098 | 0.254 | 0.021 | 0.090 | 0.239 | 0.019 | 0.097 | 0.264 | 0.016 |
| 21 | 5 | `H` | 0.430 | 1.000 | 0.000 | 0.497 | 0.999 | 0.000 | 0.499 | 1.000 | 0.000 | 0.499 | 1.000 | 0.000 | 0.499 | 0.999 | 0.000 |
| 21 | 5 | `P` | 0.399 | 0.663 | 0.267 | 0.660 | 0.883 | 0.474 | 0.981 | 0.988 | 0.978 | 0.453 | 0.988 | 0.184 | 0.897 | 0.961 | 0.870 |
| 22 | 5 | `H` | 0.472 | 1.000 | 0.126 | 0.463 | 1.000 | 0.113 | 0.465 | 1.000 | 0.113 | 0.471 | 1.000 | 0.130 | 0.466 | 1.000 | 0.126 |
| 22 | 5 | `P` | 0.184 | 0.401 | 0.012 | 0.116 | 0.254 | 0.012 | 0.160 | 0.336 | 0.032 | 0.108 | 0.246 | 0.028 | 0.148 | 0.326 | 0.033 |
| 23 | 7 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 23 | 7 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 24 | 7 | `H` | 0.103 | 0.498 | 0.000 | 0.099 | 0.487 | 0.000 | 0.101 | 0.495 | 0.000 | 0.102 | 0.499 | 0.000 | 0.100 | 0.492 | 0.000 |
| 24 | 7 | `P` | 0.035 | 0.132 | 0.004 | 0.053 | 0.121 | 0.008 | 0.041 | 0.101 | 0.006 | 0.048 | 0.112 | 0.010 | 0.047 | 0.112 | 0.009 |
| 25 | 5 | `H` | 0.022 | 0.810 | 0.000 | 0.023 | 0.816 | 0.000 | 0.021 | 0.802 | 0.000 | 0.021 | 0.811 | 0.000 | 0.022 | 0.811 | 0.000 |
| 25 | 5 | `P` | 0.501 | 1.000 | 0.002 | 0.547 | 0.999 | 0.236 | 0.554 | 0.992 | 0.009 | 0.630 | 0.995 | 0.248 | 0.406 | 0.996 | 0.001 |
| 26 | 7 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 26 | 7 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 27 | 5 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 27 | 5 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 28 | 6 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 28 | 6 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 29 | 4 | `H` | 0.271 | 1.000 | 0.000 | 0.249 | 1.000 | 0.000 | 0.249 | 0.999 | 0.000 | 0.252 | 0.999 | 0.000 | 0.271 | 0.999 | 0.000 |
| 29 | 4 | `P` | 0.166 | 0.448 | 0.073 | 0.154 | 0.449 | 0.056 | 0.163 | 0.459 | 0.061 | 0.158 | 0.446 | 0.064 | 0.164 | 0.431 | 0.061 |
| 30 | 8 | `H` | 0.058 | 0.547 | 0.000 | 0.060 | 0.554 | 0.000 | 0.057 | 0.546 | 0.000 | 0.058 | 0.552 | 0.000 | 0.060 | 0.550 | 0.000 |
| 30 | 8 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 31 | 8 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 31 | 8 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 32 | 5 | `H` |  | 1.000 | 1.000 |  | 1.000 | 1.000 |  | 1.000 | 1.000 |  | 1.000 | 1.000 |  | 1.000 | 1.000 |
| 32 | 5 | `P` |  | 0.086 | 0.086 |  | 0.150 | 0.150 |  | 0.078 | 0.078 |  | 0.154 | 0.154 |  | 0.021 | 0.021 |
| 33 | 4 | `H` | 0.541 | 1.000 | 0.395 | 0.512 | 1.000 | 0.369 | 0.503 | 1.000 | 0.365 | 0.519 | 1.000 | 0.380 | 0.550 | 1.000 | 0.416 |
| 33 | 4 | `P` | 0.198 | 0.371 | 0.129 | 0.221 | 0.506 | 0.139 | 0.201 | 0.457 | 0.135 | 0.233 | 0.477 | 0.143 | 0.243 | 0.473 | 0.200 |
| 34 | 5 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 34 | 5 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 35 | 8 | `H` | 0.199 | 0.495 | 0.063 | 0.246 | 0.589 | 0.097 | 0.256 | 0.619 | 0.101 | 0.285 | 0.612 | 0.107 | 0.309 | 0.618 | 0.123 |
| 35 | 8 | `P` | 0.123 | 0.348 | 0.000 | 0.138 | 0.387 | 0.000 | 0.222 | 0.501 | 0.065 | 0.202 | 0.513 | 0.028 | 0.124 | 0.300 | 0.010 |
| 36 | 5 | `H` | 0.258 | 0.713 | 0.007 | 0.230 | 0.657 | 0.048 | 0.284 | 0.691 | 0.005 | 0.145 | 0.589 | 0.007 | 0.365 | 0.883 | 0.057 |
| 36 | 5 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 37 | 8 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 37 | 8 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 38 | 5 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 38 | 5 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 39 | 5 | `H` | 0.416 | 1.000 | 0.000 | 0.418 | 1.000 | 0.000 | 0.419 | 1.000 | 0.000 | 0.416 | 1.000 | 0.000 | 0.424 | 1.000 | 0.000 |
| 39 | 5 | `P` | 0.107 | 0.125 | 0.056 | 0.097 | 0.166 | 0.040 | 0.077 | 0.146 | 0.026 | 0.089 | 0.159 | 0.032 | 0.109 | 0.179 | 0.038 |
| 40 | 7 | `H` | 0.062 | 1.000 | 0.000 | 0.062 | 1.000 | 0.000 | 0.062 | 1.000 | 0.000 | 0.062 | 1.000 | 0.000 | 0.062 | 1.000 | 0.000 |
| 40 | 7 | `P` | 0.055 | 0.134 | 0.009 | 0.059 | 0.137 | 0.013 | 0.048 | 0.113 | 0.011 | 0.042 | 0.097 | 0.008 | 0.046 | 0.104 | 0.011 |
| 41 | 6 | `H` | 0.239 | 0.814 | 0.000 | 0.388 | 0.951 | 0.000 | 0.304 | 0.900 | 0.000 | 0.334 | 0.972 | 0.000 | 0.354 | 0.752 | 0.000 |
| 41 | 6 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 42 | 6 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 42 | 6 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 43 | 6 | `H` | 0.180 | 0.500 | 0.000 | 0.182 | 0.504 | 0.000 | 0.177 | 0.493 | 0.000 | 0.177 | 0.491 | 0.000 | 0.189 | 0.505 | 0.000 |
| 43 | 6 | `P` | 0.174 | 0.350 | 0.022 | 0.127 | 0.340 | 0.010 | 0.174 | 0.346 | 0.027 | 0.170 | 0.346 | 0.017 | 0.155 | 0.327 | 0.017 |
| 44 | 7 | `H` | 0.480 | 1.000 | 0.000 | 0.479 | 1.000 | 0.000 | 0.482 | 1.000 | 0.000 | 0.480 | 1.000 | 0.000 | 0.480 | 1.000 | 0.000 |
| 44 | 7 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 45 | 5 | `H` | 0.488 | 0.488 | 0.488 | 0.494 | 0.494 | 0.494 | 0.501 | 0.501 | 0.501 | 0.495 | 0.495 | 0.495 | 0.494 | 0.494 | 0.494 |
| 45 | 5 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 46 | 6 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 46 | 6 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 47 | 7 | `H` | 0.485 | 0.749 | 0.000 | 0.531 | 0.739 | 0.000 | 0.537 | 0.745 | 0.000 | 0.538 | 0.745 | 0.000 | 0.539 | 0.750 | 0.000 |
| 47 | 7 | `P` | 0.287 | 0.569 | 0.167 | 0.476 | 0.931 | 0.222 | 0.527 | 0.986 | 0.208 | 0.547 | 0.995 | 0.268 | 0.575 | 0.996 | 0.461 |
| 48 | 5 | `H` | 0.639 | 0.996 | 0.320 | 0.819 | 0.998 | 0.566 | 0.959 | 0.999 | 0.914 | 0.985 | 0.999 | 0.967 | 0.964 | 1.000 | 0.893 |
| 48 | 5 | `P` | 0.573 | 0.970 | 0.242 | 0.614 | 0.981 | 0.185 | 0.592 | 0.953 | 0.233 | 0.821 | 0.945 | 0.504 | 0.574 | 0.979 | 0.243 |
| 49 | 5 | `H` | 0.170 | 0.999 | 0.000 | 0.172 | 0.999 | 0.000 | 0.168 | 0.998 | 0.000 | 0.171 | 1.000 | 0.000 | 0.169 | 1.000 | 0.000 |
| 49 | 5 | `P` | 0.041 | 0.121 | 0.027 | 0.056 | 0.128 | 0.028 | 0.056 | 0.119 | 0.039 | 0.046 | 0.126 | 0.025 | 0.054 | 0.126 | 0.033 |
| 50 | 6 | `H` | 0.252 | 1.000 | 0.000 | 0.250 | 1.000 | 0.000 | 0.248 | 1.000 | 0.000 | 0.254 | 1.000 | 0.000 | 0.248 | 1.000 | 0.000 |
| 50 | 6 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 51 | 7 | `H` | 0.356 | 1.000 | 0.000 | 0.335 | 1.000 | 0.000 | 0.356 | 1.000 | 0.000 | 0.357 | 1.000 | 0.000 | 0.369 | 1.000 | 0.000 |
| 51 | 7 | `P` | 0.212 | 0.500 | 0.010 | 0.220 | 0.517 | 0.007 | 0.215 | 0.501 | 0.009 | 0.212 | 0.507 | 0.007 | 0.216 | 0.512 | 0.007 |
| 52 | 6 | `H` | 0.368 | 1.000 | 0.091 | 0.354 | 1.000 | 0.091 | 0.394 | 1.000 | 0.127 | 0.369 | 1.000 | 0.100 | 0.398 | 1.000 | 0.127 |
| 52 | 6 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 53 | 6 | `H` |  | 0.000 | 0.000 |  | 0.000 | 0.000 |  | 0.000 | 0.000 |  | 0.000 | 0.000 |  | 0.000 | 0.000 |
| 53 | 6 | `P` |  | 0.761 | 0.761 |  | 0.961 | 0.961 |  | 0.993 | 0.993 |  | 0.996 | 0.996 |  | 0.994 | 0.994 |
| 54 | 6 | `H` | 0.500 | 1.000 | 0.000 | 0.500 | 1.000 | 0.000 | 0.500 | 1.000 | 0.000 | 0.500 | 1.000 | 0.000 | 0.500 | 1.000 | 0.000 |
| 54 | 6 | `P` | 0.168 | 0.339 | 0.076 | 0.170 | 0.343 | 0.085 | 0.168 | 0.393 | 0.053 | 0.121 | 0.315 | 0.037 | 0.084 | 0.206 | 0.027 |
| 55 | 6 | `H` | 0.444 | 1.000 | 0.000 | 0.444 | 1.000 | 0.000 | 0.444 | 1.000 | 0.000 | 0.444 | 1.000 | 0.000 | 0.444 | 1.000 | 0.000 |
| 55 | 6 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 56 | 7 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 56 | 7 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 57 | 5 | `H` | 0.212 | 1.000 | 0.000 | 0.220 | 1.000 | 0.000 | 0.213 | 1.000 | 0.000 | 0.135 | 1.000 | 0.000 | 0.209 | 0.999 | 0.000 |
| 57 | 5 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 58 | 5 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 58 | 5 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 59 | 7 | `H` |  | 0.000 | 0.000 |  | 0.000 | 0.000 |  | 0.000 | 0.000 |  | 0.000 | 0.000 |  | 0.000 | 0.000 |
| 59 | 7 | `P` | 0.277 | 0.402 | 0.107 | 0.583 | 0.676 | 0.422 | 0.526 | 0.630 | 0.123 | 0.195 | 0.225 | 0.142 | 0.366 | 0.447 | 0.098 |
| 60 | 5 | `H` | 0.278 | 0.605 | 0.041 | 0.305 | 0.567 | 0.064 | 0.271 | 0.588 | 0.038 | 0.273 | 0.602 | 0.038 | 0.270 | 0.599 | 0.038 |
| 60 | 5 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 61 | 5 | `H` |  | 0.000 | 0.000 |  | 0.000 | 0.000 |  | 0.000 | 0.000 |  | 0.000 | 0.000 |  | 0.000 | 0.000 |
| 61 | 5 | `P` |  | 0.084 | 0.084 |  | 0.082 | 0.082 |  | 0.077 | 0.077 |  | 0.060 | 0.060 |  | 0.021 | 0.021 |
| 62 | 7 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 62 | 7 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 63 | 7 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 63 | 7 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 64 | 5 | `H` | 0.140 | 1.000 | 0.000 | 0.142 | 1.000 | 0.000 | 0.140 | 1.000 | 0.000 | 0.141 | 1.000 | 0.000 | 0.140 | 1.000 | 0.000 |
| 64 | 5 | `P` | 0.423 | 0.944 | 0.116 | 0.494 | 0.994 | 0.146 | 0.594 | 0.999 | 0.225 | 0.550 | 0.951 | 0.212 | 0.615 | 0.984 | 0.231 |
| 65 | 5 | `H` | 0.200 | 1.000 | 0.000 | 0.200 | 1.000 | 0.000 | 0.200 | 1.000 | 0.000 | 0.200 | 1.000 | 0.000 | 0.200 | 1.000 | 0.000 |
| 65 | 5 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 66 | 5 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 66 | 5 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 67 | 7 | `H` | 0.049 | 0.213 | 0.000 | 0.048 | 0.209 | 0.000 | 0.049 | 0.210 | 0.000 | 0.051 | 0.221 | 0.000 | 0.051 | 0.222 | 0.000 |
| 67 | 7 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 68 | 8 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 68 | 8 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 69 | 5 | `H` | 0.572 | 0.775 | 0.234 | 0.566 | 0.772 | 0.223 | 0.568 | 0.776 | 0.221 | 0.560 | 0.763 | 0.223 | 0.563 | 0.764 | 0.229 |
| 69 | 5 | `P` | 0.202 | 0.319 | 0.077 | 0.189 | 0.264 | 0.042 | 0.317 | 0.479 | 0.080 | 0.275 | 0.316 | 0.129 | 0.223 | 0.244 | 0.146 |
| 70 | 6 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 70 | 6 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 71 | 8 | `H` | 0.452 | 1.000 | 0.000 | 0.451 | 1.000 | 0.000 | 0.448 | 1.000 | 0.000 | 0.452 | 1.000 | 0.000 | 0.451 | 1.000 | 0.000 |
| 71 | 8 | `P` | 0.062 | 0.130 | 0.008 | 0.050 | 0.117 | 0.002 | 0.061 | 0.126 | 0.006 | 0.056 | 0.125 | 0.002 | 0.055 | 0.121 | 0.004 |
| 72 | 7 | `H` | 0.513 | 1.000 | 0.000 | 0.512 | 1.000 | 0.000 | 0.514 | 1.000 | 0.000 | 0.513 | 1.000 | 0.000 | 0.512 | 1.000 | 0.000 |
| 72 | 7 | `P` | 0.192 | 0.312 | 0.006 | 0.190 | 0.309 | 0.008 | 0.186 | 0.302 | 0.010 | 0.197 | 0.323 | 0.007 | 0.184 | 0.304 | 0.009 |
| 73 | 4 | `H` | 0.737 | 1.000 | 0.284 | 0.738 | 1.000 | 0.291 | 0.726 | 1.000 | 0.274 | 0.727 | 1.000 | 0.276 | 0.734 | 1.000 | 0.291 |
| 73 | 4 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 74 | 7 | `H` | 0.429 | 0.758 | 0.065 | 0.409 | 0.728 | 0.060 | 0.421 | 0.740 | 0.065 | 0.416 | 0.727 | 0.069 | 0.413 | 0.738 | 0.060 |
| 74 | 7 | `P` | 0.322 | 0.624 | 0.015 | 0.312 | 0.647 | 0.004 | 0.267 | 0.601 | 0.002 | 0.290 | 0.633 | 0.010 | 0.303 | 0.617 | 0.013 |
| 75 | 7 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 75 | 7 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 76 | 8 | `H` |  | 0.000 | 0.000 |  | 0.000 | 0.000 |  | 0.000 | 0.000 |  | 0.000 | 0.000 |  | 0.000 | 0.000 |
| 76 | 8 | `P` | 0.043 | 0.108 | 0.018 | 0.007 | 0.020 | 0.002 | 0.160 | 0.369 | 0.054 | 0.077 | 0.210 | 0.022 | 0.056 | 0.157 | 0.010 |
| 77 | 7 | `H` | 0.539 | 1.000 | 0.038 | 0.528 | 1.000 | 0.036 | 0.538 | 1.000 | 0.039 | 0.536 | 1.000 | 0.038 | 0.533 | 1.000 | 0.035 |
| 77 | 7 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 78 | 4 | `H` | 0.127 | 1.000 | 0.000 | 0.129 | 1.000 | 0.000 | 0.121 | 1.000 | 0.000 | 0.123 | 1.000 | 0.000 | 0.123 | 1.000 | 0.000 |
| 78 | 4 | `P` | 0.176 | 0.579 | 0.058 | 0.168 | 0.588 | 0.027 | 0.198 | 0.650 | 0.074 | 0.171 | 0.496 | 0.050 | 0.272 | 0.598 | 0.150 |
| 79 | 8 | `H` | 0.234 | 0.528 | 0.000 | 0.261 | 0.591 | 0.000 | 0.277 | 0.627 | 0.000 | 0.271 | 0.629 | 0.000 | 0.278 | 0.630 | 0.000 |
| 79 | 8 | `P` | 0.051 | 0.109 | 0.001 | 0.047 | 0.103 | 0.003 | 0.038 | 0.081 | 0.003 | 0.044 | 0.104 | 0.003 | 0.044 | 0.097 | 0.003 |
| 80 | 6 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 80 | 6 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 81 | 5 | `H` | 0.900 | 1.000 | 0.000 | 0.900 | 1.000 | 0.000 | 0.900 | 1.000 | 0.000 | 0.900 | 1.000 | 0.000 | 0.900 | 1.000 | 0.000 |
| 81 | 5 | `P` | 0.943 | 0.995 | 0.739 | 0.985 | 0.991 | 0.964 | 0.995 | 0.997 | 0.984 | 0.991 | 0.996 | 0.980 | 0.990 | 0.995 | 0.983 |
| 82 | 5 | `H` |  | 0.000 | 0.000 |  | 0.000 | 0.000 |  | 0.000 | 0.000 |  | 0.000 | 0.000 |  | 0.000 | 0.000 |
| 82 | 5 | `P` | 0.081 | 0.193 | 0.011 | 0.107 | 0.205 | 0.028 | 0.113 | 0.224 | 0.024 | 0.130 | 0.231 | 0.032 | 0.110 | 0.200 | 0.030 |
| 83 | 6 | `H` | 0.267 | 1.000 | 0.000 | 0.268 | 1.000 | 0.000 | 0.267 | 1.000 | 0.000 | 0.268 | 1.000 | 0.000 | 0.267 | 1.000 | 0.000 |
| 83 | 6 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 84 | 8 | `H` | 0.170 | 0.660 | 0.000 | 0.171 | 0.655 | 0.000 | 0.173 | 0.656 | 0.000 | 0.170 | 0.650 | 0.000 | 0.109 | 0.416 | 0.000 |
| 84 | 8 | `P` | 0.017 | 0.041 | 0.004 | 0.010 | 0.030 | 0.002 | 0.014 | 0.031 | 0.003 | 0.014 | 0.033 | 0.004 | 0.016 | 0.034 | 0.005 |
| 85 | 7 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 85 | 7 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 86 | 6 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 86 | 6 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 87 | 6 | `H` |  | 1.000 | 1.000 |  | 1.000 | 1.000 |  | 1.000 | 1.000 |  | 1.000 | 1.000 |  | 1.000 | 1.000 |
| 87 | 6 | `P` |  | 0.787 | 0.787 |  | 0.998 | 0.998 |  | 0.999 | 0.999 |  | 1.000 | 1.000 |  | 0.998 | 0.998 |
| 88 | 7 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 88 | 7 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 89 | 6 | `H` |  | 0.000 | 0.000 |  | 0.000 | 0.000 |  | 0.000 | 0.000 |  | 0.000 | 0.000 |  | 0.000 | 0.000 |
| 89 | 6 | `P` | 0.579 | 0.764 | 0.382 | 0.955 | 0.967 | 0.910 | 0.941 | 0.947 | 0.932 | 0.985 | 0.990 | 0.979 | 0.988 | 0.991 | 0.980 |
| 90 | 6 | `H` | 0.576 | 1.000 | 0.000 | 0.571 | 1.000 | 0.000 | 0.568 | 1.000 | 0.000 | 0.573 | 1.000 | 0.000 | 0.575 | 1.000 | 0.000 |
| 90 | 6 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 91 | 8 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 91 | 8 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 92 | 6 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 92 | 6 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 93 | 5 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 93 | 5 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 94 | 8 | `H` | 0.305 | 0.787 | 0.000 | 0.312 | 0.798 | 0.000 | 0.312 | 0.801 | 0.000 | 0.312 | 0.790 | 0.000 | 0.304 | 0.788 | 0.000 |
| 94 | 8 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 95 | 7 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 95 | 7 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 96 | 5 | `H` | 0.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| 96 | 5 | `P` | 0.464 | 0.903 | 0.464 | 0.457 | 0.967 | 0.457 | 0.431 | 0.953 | 0.431 | 0.594 | 0.889 | 0.594 | 0.045 | 0.458 | 0.045 |
| 97 | 7 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 97 | 7 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 98 | 7 | `H` | 0.093 | 0.318 | 0.000 | 0.092 | 0.315 | 0.000 | 0.090 | 0.309 | 0.000 | 0.091 | 0.310 | 0.000 | 0.091 | 0.313 | 0.000 |
| 98 | 7 | `P` | 0.221 | 0.477 | 0.009 | 0.273 | 0.568 | 0.005 | 0.296 | 0.560 | 0.031 | 0.233 | 0.464 | 0.018 | 0.323 | 0.538 | 0.089 |
| 99 | 6 | `H` | 0.133 | 0.688 | 0.000 | 0.131 | 0.670 | 0.000 | 0.129 | 0.681 | 0.000 | 0.128 | 0.675 | 0.000 | 0.132 | 0.681 | 0.000 |
| 99 | 6 | `P` | 0.123 | 0.234 | 0.025 | 0.171 | 0.402 | 0.061 | 0.122 | 0.230 | 0.058 | 0.078 | 0.198 | 0.022 | 0.079 | 0.186 | 0.032 |
| 100 | 5 | `H` | 0.403 | 1.000 | 0.000 | 0.403 | 1.000 | 0.000 | 0.406 | 1.000 | 0.000 | 0.405 | 1.000 | 0.000 | 0.407 | 1.000 | 0.000 |
| 100 | 5 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 101 | 7 | `H` | 0.302 | 1.000 | 0.000 | 0.299 | 1.000 | 0.000 | 0.300 | 1.000 | 0.000 | 0.300 | 1.000 | 0.000 | 0.299 | 1.000 | 0.000 |
| 101 | 7 | `P` | 0.477 | 0.983 | 0.006 | 0.495 | 0.973 | 0.006 | 0.483 | 0.990 | 0.005 | 0.543 | 0.965 | 0.010 | 0.497 | 0.969 | 0.008 |
| 102 | 6 | `H` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 102 | 6 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 103 | 3 | `H` | 0.327 | 0.746 | 0.295 | 0.466 | 0.750 | 0.435 | 0.593 | 0.755 | 0.562 | 0.592 | 0.743 | 0.559 | 0.584 | 0.747 | 0.553 |
| 103 | 3 | `P` | 0.508 | 0.983 | 0.508 | 0.946 | 0.998 | 0.903 | 0.995 | 1.000 | 0.992 | 0.994 | 0.998 | 0.992 | 0.996 | 0.999 | 0.996 |
| 104 | 8 | `H` |  | 0.000 | 0.000 |  | 0.000 | 0.000 |  | 0.000 | 0.000 |  | 0.000 | 0.000 |  | 0.000 | 0.000 |
| 104 | 8 | `P` | 0.228 | 0.272 | 0.177 | 0.623 | 0.670 | 0.545 | 0.820 | 0.853 | 0.771 | 0.739 | 0.934 | 0.501 | 0.937 | 0.970 | 0.892 |
| 105 | 6 | `H` | 0.218 | 0.750 | 0.000 | 0.216 | 0.758 | 0.000 | 0.220 | 0.764 | 0.000 | 0.216 | 0.745 | 0.000 | 0.221 | 0.763 | 0.000 |
| 105 | 6 | `P` | 0.224 | 0.440 | 0.020 | 0.220 | 0.454 | 0.017 | 0.227 | 0.445 | 0.014 | 0.218 | 0.438 | 0.018 | 0.221 | 0.439 | 0.014 |
| 106 | 6 | `H` |  | 0.000 | 0.000 |  | 0.000 | 0.000 |  | 0.000 | 0.000 |  | 0.000 | 0.000 |  | 0.000 | 0.000 |
| 106 | 6 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 107 | 8 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 107 | 8 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 108 | 5 | `H` | 0.100 | 1.000 | 0.000 | 0.100 | 1.000 | 0.000 | 0.100 | 1.000 | 0.000 | 0.100 | 1.000 | 0.000 | 0.100 | 1.000 | 0.000 |
| 108 | 5 | `P` | 0.072 | 0.189 | 0.024 | 0.090 | 0.235 | 0.028 | 0.079 | 0.197 | 0.029 | 0.085 | 0.192 | 0.035 | 0.077 | 0.198 | 0.032 |
| 109 | 8 | `H` | 0.059 | 1.000 | 0.000 | 0.059 | 1.000 | 0.000 | 0.059 | 1.000 | 0.000 | 0.059 | 1.000 | 0.000 | 0.059 | 1.000 | 0.000 |
| 109 | 8 | `P` | 0.014 | 0.028 | 0.009 | 0.017 | 0.019 | 0.015 | 0.034 | 0.062 | 0.025 | 0.014 | 0.044 | 0.011 | 0.003 | 0.010 | 0.001 |
| 110 | 5 | `H` | 0.437 | 1.000 | 0.034 | 0.441 | 1.000 | 0.040 | 0.440 | 1.000 | 0.035 | 0.448 | 1.000 | 0.042 | 0.445 | 1.000 | 0.044 |
| 110 | 5 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 111 | 4 | `H` | 0.336 | 0.669 | 0.336 | 0.331 | 0.661 | 0.331 | 0.341 | 0.677 | 0.341 | 0.339 | 0.666 | 0.339 | 0.326 | 0.658 | 0.326 |
| 111 | 4 | `P` | 0.171 | 0.280 | 0.111 | 0.108 | 0.209 | 0.070 | 0.107 | 0.175 | 0.070 | 0.115 | 0.199 | 0.075 | 0.110 | 0.197 | 0.065 |
| 112 | 8 | `H` |  | 0.000 | 0.000 |  | 0.000 | 0.000 |  | 0.000 | 0.000 |  | 0.000 | 0.000 |  | 0.000 | 0.000 |
| 112 | 8 | `P` | 0.020 | 0.026 | 0.005 | 0.018 | 0.035 | 0.007 | 0.030 | 0.044 | 0.009 | 0.014 | 0.029 | 0.008 | 0.025 | 0.032 | 0.011 |
| 113 | 7 | `H` | 0.305 | 0.559 | 0.000 | 0.298 | 0.546 | 0.000 | 0.292 | 0.535 | 0.000 | 0.296 | 0.543 | 0.000 | 0.251 | 0.460 | 0.000 |
| 113 | 7 | `P` | 0.142 | 0.268 | 0.001 | 0.235 | 0.369 | 0.004 | 0.122 | 0.216 | 0.029 | 0.142 | 0.277 | 0.008 | 0.123 | 0.227 | 0.008 |
| 114 | 5 | `H` | 0.251 | 0.500 | 0.100 | 0.253 | 0.505 | 0.098 | 0.247 | 0.504 | 0.102 | 0.247 | 0.507 | 0.094 | 0.251 | 0.501 | 0.103 |
| 114 | 5 | `P` | 0.090 | 0.167 | 0.037 | 0.075 | 0.152 | 0.031 | 0.079 | 0.154 | 0.037 | 0.074 | 0.157 | 0.026 | 0.069 | 0.145 | 0.023 |
| 115 | 4 | `H` | 0.333 | 1.000 | 0.000 | 0.333 | 1.000 | 0.000 | 0.333 | 1.000 | 0.000 | 0.333 | 1.000 | 0.000 | 0.333 | 1.000 | 0.000 |
| 115 | 4 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 116 | 8 | `H` | 0.234 | 0.630 | 0.000 | 0.225 | 0.618 | 0.000 | 0.239 | 0.626 | 0.000 | 0.229 | 0.600 | 0.000 | 0.208 | 0.576 | 0.000 |
| 116 | 8 | `P` | 0.039 | 0.069 | 0.013 | 0.044 | 0.085 | 0.018 | 0.049 | 0.087 | 0.020 | 0.012 | 0.024 | 0.004 | 0.015 | 0.030 | 0.004 |
| 117 | 5 | `H` | 0.134 | 1.000 | 0.000 | 0.127 | 1.000 | 0.000 | 0.119 | 1.000 | 0.000 | 0.134 | 1.000 | 0.000 | 0.128 | 1.000 | 0.000 |
| 117 | 5 | `P` | 0.210 | 0.548 | 0.028 | 0.210 | 0.576 | 0.002 | 0.272 | 0.567 | 0.066 | 0.238 | 0.578 | 0.022 | 0.213 | 0.571 | 0.033 |
| 118 | 4 | `H` | 0.141 | 0.827 | 0.050 | 0.156 | 0.823 | 0.053 | 0.166 | 0.823 | 0.079 | 0.142 | 0.827 | 0.047 | 0.137 | 0.827 | 0.048 |
| 118 | 4 | `P` | 0.161 | 0.446 | 0.079 | 0.149 | 0.447 | 0.062 | 0.130 | 0.431 | 0.050 | 0.163 | 0.439 | 0.080 | 0.141 | 0.434 | 0.050 |
| 119 | 8 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 119 | 8 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 120 | 6 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 120 | 6 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 121 | 5 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 121 | 5 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 122 | 7 | `H` | 0.148 | 0.216 | 0.053 | 0.136 | 0.197 | 0.052 | 0.147 | 0.212 | 0.052 | 0.129 | 0.191 | 0.047 | 0.136 | 0.200 | 0.047 |
| 122 | 7 | `P` | 0.042 | 0.067 | 0.016 | 0.034 | 0.054 | 0.011 | 0.054 | 0.088 | 0.015 | 0.047 | 0.083 | 0.008 | 0.034 | 0.061 | 0.006 |
| 123 | 8 | `H` | 0.054 | 0.868 | 0.000 | 0.062 | 0.999 | 0.000 | 0.062 | 0.999 | 0.000 | 0.062 | 0.999 | 0.000 | 0.062 | 0.999 | 0.000 |
| 123 | 8 | `P` | 0.144 | 0.269 | 0.032 | 0.359 | 0.639 | 0.171 | 0.523 | 0.854 | 0.350 | 0.579 | 0.975 | 0.357 | 0.619 | 0.996 | 0.481 |
| 124 | 6 | `H` |  | 0.000 | 0.000 |  | 0.000 | 0.000 |  | 0.000 | 0.000 |  | 0.000 | 0.000 |  | 0.000 | 0.000 |
| 124 | 6 | `P` | 0.111 | 0.214 | 0.028 | 0.001 | 0.002 | 0.000 | 0.000 | 0.000 | 0.000 | 0.011 | 0.013 | 0.009 | 0.500 | 0.500 | 0.500 |
| 125 | 8 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 125 | 8 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 126 | 8 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 126 | 8 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 127 | 5 | `H` | 0.284 | 1.000 | 0.000 | 0.287 | 1.000 | 0.000 | 0.288 | 1.000 | 0.000 | 0.282 | 1.000 | 0.000 | 0.282 | 1.000 | 0.000 |
| 127 | 5 | `P` | 0.164 | 0.355 | 0.027 | 0.180 | 0.374 | 0.035 | 0.169 | 0.368 | 0.031 | 0.186 | 0.379 | 0.035 | 0.170 | 0.379 | 0.029 |
| 128 | 7 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 128 | 7 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 129 | 6 | `H` | 0.522 | 0.892 | 0.000 | 0.415 | 0.575 | 0.000 | 0.192 | 0.294 | 0.000 | 0.361 | 0.491 | 0.000 | 0.396 | 0.578 | 0.000 |
| 129 | 6 | `P` | 0.089 | 0.148 | 0.016 | 0.090 | 0.154 | 0.015 | 0.089 | 0.154 | 0.015 | 0.095 | 0.165 | 0.017 | 0.092 | 0.159 | 0.018 |
| 130 | 6 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 130 | 6 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 131 | 5 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 131 | 5 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 132 | 7 | `H` |  | 0.000 | 0.000 |  | 0.000 | 0.000 |  | 0.000 | 0.000 |  | 0.000 | 0.000 |  | 0.000 | 0.000 |
| 132 | 7 | `P` | 0.007 | 0.014 | 0.000 | 0.053 | 0.074 | 0.003 | 0.485 | 0.535 | 0.334 | 0.026 | 0.146 | 0.001 | 0.236 | 0.366 | 0.000 |
| 133 | 7 | `H` | 0.397 | 0.591 | 0.000 | 0.414 | 0.615 | 0.000 | 0.424 | 0.625 | 0.000 | 0.422 | 0.622 | 0.000 | 0.420 | 0.621 | 0.000 |
| 133 | 7 | `P` | 0.543 | 0.818 | 0.141 | 0.759 | 0.983 | 0.204 | 0.872 | 0.999 | 0.242 | 0.879 | 0.997 | 0.253 | 0.416 | 0.499 | 0.001 |
| 134 | 4 | `H` | 0.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| 134 | 4 | `P` | 0.507 | 0.507 | 0.507 | 0.947 | 0.972 | 0.947 | 0.994 | 0.994 | 0.994 | 0.977 | 0.977 | 0.977 | 0.978 | 0.978 | 0.978 |
| 135 | 8 | `H` | 0.150 | 1.000 | 0.000 | 0.149 | 1.000 | 0.000 | 0.148 | 1.000 | 0.000 | 0.147 | 1.000 | 0.000 | 0.147 | 1.000 | 0.000 |
| 135 | 8 | `P` | 0.097 | 0.143 | 0.018 | 0.046 | 0.075 | 0.007 | 0.037 | 0.074 | 0.005 | 0.041 | 0.071 | 0.005 | 0.028 | 0.070 | 0.002 |
| 136 | 4 | `H` | 0.724 | 0.992 | 0.724 | 0.979 | 0.999 | 0.979 | 0.999 | 0.999 | 0.999 | 0.998 | 1.000 | 0.998 | 0.996 | 1.000 | 0.996 |
| 136 | 4 | `P` | 0.989 | 0.989 | 0.989 | 0.994 | 0.995 | 0.994 | 0.993 | 0.996 | 0.993 | 0.999 | 0.999 | 0.999 | 0.995 | 0.998 | 0.995 |
| 137 | 5 | `H` | 0.240 | 0.797 | 0.098 | 0.240 | 0.802 | 0.092 | 0.241 | 0.802 | 0.099 | 0.249 | 0.802 | 0.095 | 0.256 | 0.807 | 0.104 |
| 137 | 5 | `P` | 0.088 | 0.298 | 0.032 | 0.080 | 0.277 | 0.022 | 0.131 | 0.285 | 0.053 | 0.072 | 0.246 | 0.031 | 0.087 | 0.291 | 0.036 |
| 138 | 8 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 138 | 8 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 139 | 8 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 139 | 8 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 140 | 6 | `H` | 0.485 | 1.000 | 0.000 | 0.482 | 1.000 | 0.000 | 0.485 | 1.000 | 0.000 | 0.488 | 1.000 | 0.000 | 0.484 | 1.000 | 0.000 |
| 140 | 6 | `P` | 0.412 | 0.952 | 0.003 | 0.359 | 0.922 | 0.014 | 0.549 | 0.912 | 0.027 | 0.332 | 0.869 | 0.109 | 0.284 | 0.944 | 0.016 |
| 141 | 4 | `H` | 0.290 | 0.857 | 0.150 | 0.281 | 0.852 | 0.154 | 0.269 | 0.826 | 0.137 | 0.260 | 0.794 | 0.126 | 0.261 | 0.857 | 0.130 |
| 141 | 4 | `P` | 0.159 | 0.514 | 0.050 | 0.169 | 0.504 | 0.053 | 0.178 | 0.518 | 0.081 | 0.217 | 0.511 | 0.127 | 0.158 | 0.494 | 0.052 |
| 142 | 6 | `H` | 0.208 | 0.364 | 0.000 | 0.209 | 0.372 | 0.000 | 0.212 | 0.371 | 0.000 | 0.197 | 0.349 | 0.000 | 0.210 | 0.368 | 0.000 |
| 142 | 6 | `P` | 0.089 | 0.155 | 0.028 | 0.056 | 0.118 | 0.013 | 0.058 | 0.115 | 0.014 | 0.065 | 0.130 | 0.016 | 0.062 | 0.130 | 0.012 |
| 143 | 4 | `H` |  | 1.000 | 1.000 |  | 1.000 | 1.000 |  | 1.000 | 1.000 |  | 1.000 | 1.000 |  | 1.000 | 1.000 |
| 143 | 4 | `P` |  | 0.027 | 0.027 |  | 0.080 | 0.080 |  | 0.055 | 0.055 |  | 0.059 | 0.059 |  | 0.066 | 0.066 |
| 144 | 5 | `H` | 0.270 | 0.592 | 0.000 | 0.259 | 0.585 | 0.000 | 0.275 | 0.600 | 0.000 | 0.265 | 0.585 | 0.000 | 0.261 | 0.597 | 0.000 |
| 144 | 5 | `P` | 0.216 | 0.417 | 0.024 | 0.215 | 0.406 | 0.029 | 0.218 | 0.410 | 0.033 | 0.213 | 0.409 | 0.031 | 0.214 | 0.405 | 0.031 |
| 145 | 4 | `H` | 0.367 | 1.000 | 0.200 | 0.376 | 1.000 | 0.210 | 0.369 | 1.000 | 0.205 | 0.376 | 1.000 | 0.206 | 0.373 | 1.000 | 0.205 |
| 145 | 4 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 146 | 7 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 146 | 7 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 147 | 5 | `H` | 0.415 | 1.000 | 0.000 | 0.418 | 1.000 | 0.000 | 0.412 | 1.000 | 0.000 | 0.423 | 1.000 | 0.000 | 0.418 | 1.000 | 0.000 |
| 147 | 5 | `P` | 0.208 | 0.417 | 0.031 | 0.211 | 0.437 | 0.029 | 0.209 | 0.435 | 0.032 | 0.214 | 0.437 | 0.033 | 0.216 | 0.439 | 0.034 |
| 148 | 4 | `H` | 0.283 | 0.696 | 0.063 | 0.275 | 0.685 | 0.060 | 0.278 | 0.687 | 0.064 | 0.282 | 0.684 | 0.065 | 0.284 | 0.695 | 0.065 |
| 148 | 4 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 149 | 7 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 149 | 7 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 150 | 8 | `H` | 0.346 | 1.000 | 0.000 | 0.342 | 1.000 | 0.000 | 0.334 | 1.000 | 0.000 | 0.343 | 1.000 | 0.000 | 0.340 | 1.000 | 0.000 |
| 150 | 8 | `P` | 0.268 | 0.533 | 0.002 | 0.306 | 0.571 | 0.000 | 0.333 | 0.544 | 0.009 | 0.376 | 0.599 | 0.102 | 0.459 | 0.649 | 0.135 |
| 151 | 6 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 151 | 6 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 152 | 6 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 152 | 6 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 153 | 4 | `H` | 0.045 | 1.000 | 0.000 | 0.047 | 1.000 | 0.000 | 0.045 | 1.000 | 0.000 | 0.047 | 1.000 | 0.000 | 0.047 | 1.000 | 0.000 |
| 153 | 4 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 154 | 7 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 154 | 7 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 155 | 6 | `H` |  | 0.000 | 0.000 |  | 0.000 | 0.000 |  | 0.000 | 0.000 |  | 0.000 | 0.000 |  | 0.000 | 0.000 |
| 155 | 6 | `P` | 0.015 | 0.052 | 0.015 | 0.023 | 0.044 | 0.023 | 0.022 | 0.031 | 0.022 | 0.015 | 0.037 | 0.015 | 0.016 | 0.034 | 0.017 |
| 156 | 6 | `H` | 0.414 | 1.000 | 0.000 | 0.413 | 1.000 | 0.000 | 0.416 | 1.000 | 0.000 | 0.418 | 1.000 | 0.000 | 0.419 | 1.000 | 0.000 |
| 156 | 6 | `P` | 0.148 | 0.346 | 0.009 | 0.145 | 0.323 | 0.004 | 0.172 | 0.370 | 0.011 | 0.161 | 0.360 | 0.005 | 0.158 | 0.354 | 0.010 |
| 157 | 5 | `H` | 0.233 | 1.000 | 0.000 | 0.243 | 1.000 | 0.000 | 0.237 | 1.000 | 0.000 | 0.235 | 1.000 | 0.000 | 0.240 | 1.000 | 0.000 |
| 157 | 5 | `P` | 0.209 | 0.628 | 0.030 | 0.238 | 0.627 | 0.018 | 0.221 | 0.629 | 0.028 | 0.206 | 0.638 | 0.020 | 0.218 | 0.648 | 0.021 |
| 158 | 7 | `H` | 0.355 | 1.000 | 0.000 | 0.331 | 1.000 | 0.000 | 0.327 | 1.000 | 0.000 | 0.328 | 1.000 | 0.000 | 0.326 | 1.000 | 0.000 |
| 158 | 7 | `P` | 0.492 | 0.844 | 0.008 | 0.654 | 0.928 | 0.002 | 0.502 | 0.996 | 0.000 | 0.376 | 0.491 | 0.000 | 0.421 | 0.993 | 0.000 |
| 159 | 3 | `H` |  | 1.000 | 1.000 |  | 1.000 | 1.000 |  | 1.000 | 1.000 |  | 1.000 | 1.000 |  | 1.000 | 1.000 |
| 159 | 3 | `P` |  | 0.996 | 0.996 |  | 1.000 | 1.000 |  | 1.000 | 1.000 |  | 1.000 | 1.000 |  | 1.000 | 1.000 |
| 160 | 7 | `H` | 0.069 | 0.297 | 0.000 | 0.070 | 0.300 | 0.000 | 0.068 | 0.291 | 0.000 | 0.069 | 0.294 | 0.000 | 0.068 | 0.289 | 0.000 |
| 160 | 7 | `P` | 0.102 | 0.198 | 0.005 | 0.107 | 0.208 | 0.011 | 0.067 | 0.158 | 0.007 | 0.074 | 0.167 | 0.012 | 0.074 | 0.168 | 0.016 |
| 161 | 6 | `H` | 0.446 | 1.000 | 0.000 | 0.448 | 1.000 | 0.000 | 0.457 | 1.000 | 0.000 | 0.450 | 1.000 | 0.000 | 0.453 | 1.000 | 0.000 |
| 161 | 6 | `P` | 0.180 | 0.327 | 0.027 | 0.147 | 0.252 | 0.022 | 0.142 | 0.250 | 0.024 | 0.164 | 0.305 | 0.030 | 0.121 | 0.255 | 0.013 |
| 162 | 4 | `H` |  | 0.000 | 0.000 |  | 0.000 | 0.000 |  | 0.000 | 0.000 |  | 0.000 | 0.000 |  | 0.000 | 0.000 |
| 162 | 4 | `P` |  | 0.969 | 0.969 |  | 0.962 | 0.962 |  | 0.934 | 0.934 |  | 0.481 | 0.481 |  | 0.969 | 0.969 |
| 163 | 8 | `H` |  | 0.000 | 0.000 |  | 0.000 | 0.000 |  | 0.000 | 0.000 |  | 0.000 | 0.000 |  | 0.000 | 0.000 |
| 163 | 8 | `P` | 0.031 | 0.060 | 0.007 | 0.036 | 0.075 | 0.006 | 0.023 | 0.052 | 0.004 | 0.025 | 0.059 | 0.005 | 0.033 | 0.076 | 0.007 |
| 164 | 4 | `H` | 0.255 | 0.527 | 0.062 | 0.267 | 0.561 | 0.075 | 0.261 | 0.535 | 0.070 | 0.265 | 0.540 | 0.072 | 0.255 | 0.516 | 0.068 |
| 164 | 4 | `P` | 0.236 | 0.461 | 0.028 | 0.206 | 0.443 | 0.052 | 0.233 | 0.502 | 0.058 | 0.253 | 0.511 | 0.079 | 0.262 | 0.498 | 0.078 |
| 165 | 8 | `H` | 0.350 | 0.848 | 0.000 | 0.348 | 0.842 | 0.000 | 0.356 | 0.848 | 0.000 | 0.351 | 0.846 | 0.000 | 0.347 | 0.842 | 0.000 |
| 165 | 8 | `P` | 0.295 | 0.645 | 0.005 | 0.290 | 0.644 | 0.002 | 0.293 | 0.646 | 0.005 | 0.291 | 0.638 | 0.004 | 0.296 | 0.646 | 0.005 |
| 166 | 5 | `H` | 0.167 | 1.000 | 0.000 | 0.167 | 1.000 | 0.000 | 0.167 | 1.000 | 0.000 | 0.167 | 1.000 | 0.000 | 0.167 | 1.000 | 0.000 |
| 166 | 5 | `P` | 0.440 | 0.926 | 0.324 | 0.896 | 0.986 | 0.866 | 0.989 | 0.995 | 0.988 | 0.986 | 0.998 | 0.986 | 0.992 | 0.998 | 0.991 |
| 167 | 7 | `H` | 0.660 | 1.000 | 0.243 | 0.663 | 1.000 | 0.249 | 0.662 | 1.000 | 0.245 | 0.661 | 1.000 | 0.244 | 0.668 | 1.000 | 0.261 |
| 167 | 7 | `P` | 0.090 | 0.194 | 0.011 | 0.088 | 0.194 | 0.008 | 0.079 | 0.182 | 0.008 | 0.086 | 0.189 | 0.007 | 0.086 | 0.192 | 0.009 |
| 168 | 8 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 168 | 8 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 169 | 7 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 169 | 7 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 170 | 6 | `H` | 0.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| 170 | 6 | `P` | 0.029 | 0.034 | 0.029 | 0.064 | 0.067 | 0.064 | 0.030 | 0.079 | 0.030 | 0.024 | 0.056 | 0.024 | 0.023 | 0.044 | 0.023 |
| 171 | 8 | `H` | 0.374 | 1.000 | 0.000 | 0.336 | 1.000 | 0.000 | 0.354 | 1.000 | 0.000 | 0.353 | 1.000 | 0.000 | 0.355 | 1.000 | 0.000 |
| 171 | 8 | `P` | 0.151 | 0.376 | 0.003 | 0.146 | 0.360 | 0.004 | 0.143 | 0.371 | 0.004 | 0.145 | 0.369 | 0.004 | 0.148 | 0.375 | 0.004 |
| 172 | 8 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 172 | 8 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 173 | 4 | `H` | 0.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| 173 | 4 | `P` | 0.083 | 0.174 | 0.083 | 0.090 | 0.177 | 0.090 | 0.073 | 0.171 | 0.073 | 0.070 | 0.145 | 0.070 | 0.065 | 0.128 | 0.065 |
| 174 | 7 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 174 | 7 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 175 | 6 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 175 | 6 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 176 | 6 | `H` | 0.475 | 1.000 | 0.046 | 0.481 | 1.000 | 0.052 | 0.475 | 1.000 | 0.046 | 0.484 | 1.000 | 0.071 | 0.479 | 1.000 | 0.054 |
| 176 | 6 | `P` | 0.268 | 0.625 | 0.013 | 0.280 | 0.609 | 0.019 | 0.269 | 0.614 | 0.016 | 0.273 | 0.623 | 0.018 | 0.280 | 0.626 | 0.017 |
| 177 | 4 | `H` | 0.241 | 0.600 | 0.042 | 0.302 | 0.803 | 0.080 | 0.301 | 0.798 | 0.079 | 0.303 | 0.801 | 0.083 | 0.301 | 0.800 | 0.077 |
| 177 | 4 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 178 | 5 | `H` | 0.065 | 0.107 | 0.056 | 0.006 | 0.036 | 0.000 | 0.032 | 0.042 | 0.025 | 0.009 | 0.046 | 0.003 | 0.022 | 0.061 | 0.001 |
| 178 | 5 | `P` | 0.074 | 0.184 | 0.032 | 0.073 | 0.189 | 0.030 | 0.079 | 0.191 | 0.034 | 0.074 | 0.181 | 0.031 | 0.076 | 0.192 | 0.030 |
| 179 | 8 | `H` | 0.079 | 0.342 | 0.000 | 0.296 | 0.768 | 0.000 | 0.151 | 0.601 | 0.000 | 0.166 | 0.636 | 0.000 | 0.162 | 0.722 | 0.000 |
| 179 | 8 | `P` | 0.114 | 0.279 | 0.005 | 0.118 | 0.285 | 0.006 | 0.117 | 0.286 | 0.004 | 0.118 | 0.278 | 0.003 | 0.118 | 0.290 | 0.004 |
| 180 | 5 | `H` |  | 1.000 | 1.000 |  | 1.000 | 1.000 |  | 1.000 | 1.000 |  | 1.000 | 1.000 |  | 1.000 | 1.000 |
| 180 | 5 | `P` |  | 0.051 | 0.051 |  | 0.028 | 0.028 |  | 0.044 | 0.044 |  | 0.025 | 0.025 |  | 0.026 | 0.026 |
| 181 | 8 | `H` | 0.319 | 0.924 | 0.000 | 0.252 | 0.676 | 0.000 | 0.203 | 0.728 | 0.000 | 0.089 | 0.188 | 0.000 | 0.265 | 0.725 | 0.000 |
| 181 | 8 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 182 | 8 | `H` |  | 0.000 | 0.000 |  | 0.000 | 0.000 |  | 0.000 | 0.000 |  | 0.000 | 0.000 |  | 0.000 | 0.000 |
| 182 | 8 | `P` | 0.081 | 0.123 | 0.054 | 0.083 | 0.129 | 0.028 | 0.107 | 0.143 | 0.068 | 0.070 | 0.111 | 0.028 | 0.139 | 0.188 | 0.081 |
| 183 | 5 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 183 | 5 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 184 | 6 | `H` | 0.246 | 0.913 | 0.000 | 0.244 | 0.917 | 0.000 | 0.246 | 0.911 | 0.000 | 0.247 | 0.919 | 0.000 | 0.245 | 0.911 | 0.000 |
| 184 | 6 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 185 | 6 | `H` | 0.264 | 0.897 | 0.000 | 0.269 | 0.903 | 0.000 | 0.270 | 0.913 | 0.000 | 0.268 | 0.904 | 0.000 | 0.267 | 0.903 | 0.000 |
| 185 | 6 | `P` | 0.833 | 1.000 | 0.603 | 0.890 | 1.000 | 0.643 | 0.852 | 0.997 | 0.432 | 0.970 | 1.000 | 0.849 | 0.699 | 0.886 | 0.254 |
| 186 | 5 | `H` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 186 | 5 | `P` | 0.033 | 0.080 | 0.022 | 0.035 | 0.090 | 0.025 | 0.040 | 0.095 | 0.029 | 0.044 | 0.101 | 0.033 | 0.046 | 0.097 | 0.036 |
| 187 | 7 | `H` | 0.410 | 0.856 | 0.000 | 0.438 | 0.999 | 0.000 | 0.443 | 0.999 | 0.000 | 0.477 | 0.916 | 0.000 | 0.445 | 0.999 | 0.000 |
| 187 | 7 | `P` | 0.162 | 0.326 | 0.010 | 0.165 | 0.333 | 0.006 | 0.166 | 0.329 | 0.007 | 0.162 | 0.324 | 0.007 | 0.164 | 0.329 | 0.006 |
| 188 | 8 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 188 | 8 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 189 | 7 | `H` | 0.667 | 1.000 | 0.000 | 0.667 | 1.000 | 0.000 | 0.667 | 1.000 | 0.000 | 0.667 | 1.000 | 0.000 | 0.667 | 1.000 | 0.000 |
| 189 | 7 | `P` | 0.126 | 0.186 | 0.031 | 0.093 | 0.193 | 0.005 | 0.151 | 0.303 | 0.005 | 0.256 | 0.415 | 0.010 | 0.086 | 0.178 | 0.009 |
| 190 | 6 | `H` |  | 0.000 | 0.000 |  | 0.000 | 0.000 |  | 0.000 | 0.000 |  | 0.000 | 0.000 |  | 0.000 | 0.000 |
| 190 | 6 | `P` | 0.466 | 0.997 | 0.006 | 0.455 | 0.998 | 0.009 | 0.645 | 0.996 | 0.026 | 0.605 | 0.979 | 0.008 | 0.637 | 0.995 | 0.001 |
| 191 | 5 | `H` | 0.291 | 0.727 | 0.000 | 0.296 | 0.747 | 0.000 | 0.287 | 0.733 | 0.000 | 0.294 | 0.739 | 0.000 | 0.294 | 0.743 | 0.000 |
| 191 | 5 | `P` | 0.231 | 0.519 | 0.024 | 0.226 | 0.497 | 0.028 | 0.226 | 0.517 | 0.033 | 0.231 | 0.513 | 0.031 | 0.228 | 0.509 | 0.034 |
| 192 | 5 | `H` | 0.290 | 0.939 | 0.000 | 0.300 | 0.945 | 0.000 | 0.303 | 0.944 | 0.000 | 0.291 | 0.943 | 0.000 | 0.296 | 0.935 | 0.000 |
| 192 | 5 | `P` | 0.316 | 0.793 | 0.019 | 0.357 | 0.734 | 0.023 | 0.543 | 0.689 | 0.207 | 0.367 | 0.985 | 0.024 | 0.588 | 0.727 | 0.306 |
| 193 | 6 | `H` | 0.257 | 0.999 | 0.000 | 0.282 | 0.999 | 0.000 | 0.215 | 0.999 | 0.000 | 0.173 | 1.000 | 0.000 | 0.152 | 0.999 | 0.000 |
| 193 | 6 | `P` | 0.152 | 0.410 | 0.017 | 0.152 | 0.416 | 0.014 | 0.146 | 0.401 | 0.016 | 0.154 | 0.414 | 0.016 | 0.152 | 0.407 | 0.018 |
| 194 | 6 | `H` | 0.577 | 1.000 | 0.000 | 0.578 | 1.000 | 0.000 | 0.577 | 1.000 | 0.000 | 0.581 | 1.000 | 0.000 | 0.580 | 1.000 | 0.000 |
| 194 | 6 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 195 | 6 | `H` | 0.308 | 0.746 | 0.000 | 0.307 | 0.748 | 0.000 | 0.314 | 0.765 | 0.000 | 0.303 | 0.739 | 0.000 | 0.315 | 0.756 | 0.000 |
| 195 | 6 | `P` | 0.298 | 0.674 | 0.022 | 0.341 | 0.686 | 0.026 | 0.303 | 0.690 | 0.015 | 0.309 | 0.696 | 0.013 | 0.308 | 0.692 | 0.015 |
| 196 | 6 | `H` | 0.581 | 1.000 | 0.000 | 0.696 | 0.999 | 0.000 | 0.622 | 0.999 | 0.000 | 0.607 | 1.000 | 0.000 | 0.637 | 0.999 | 0.000 |
| 196 | 6 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 197 | 8 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 197 | 8 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 198 | 4 | `H` | 0.264 | 0.952 | 0.000 | 0.278 | 1.000 | 0.000 | 0.278 | 1.000 | 0.000 | 0.278 | 0.999 | 0.000 | 0.278 | 1.000 | 0.000 |
| 198 | 4 | `P` | 0.641 | 0.993 | 0.505 | 0.981 | 0.983 | 0.981 | 0.911 | 0.977 | 0.886 | 0.989 | 0.995 | 0.987 | 0.960 | 0.969 | 0.956 |
| 199 | 6 | `H` | 0.361 | 0.689 | 0.052 | 0.362 | 0.685 | 0.048 | 0.358 | 0.679 | 0.052 | 0.359 | 0.672 | 0.056 | 0.363 | 0.686 | 0.048 |
| 199 | 6 | `P` | 0.121 | 0.290 | 0.015 | 0.115 | 0.296 | 0.024 | 0.131 | 0.317 | 0.027 | 0.126 | 0.307 | 0.012 | 0.134 | 0.319 | 0.016 |
| 200 | 6 | `H` | 0.033 | 0.289 | 0.000 | 0.033 | 0.300 | 0.000 | 0.034 | 0.305 | 0.000 | 0.036 | 0.326 | 0.000 | 0.032 | 0.321 | 0.000 |
| 200 | 6 | `P` | 0.050 | 0.123 | 0.015 | 0.057 | 0.120 | 0.025 | 0.033 | 0.089 | 0.011 | 0.039 | 0.085 | 0.023 | 0.040 | 0.084 | 0.012 |
| 201 | 7 | `H` | 0.216 | 0.282 | 0.097 | 0.244 | 0.312 | 0.124 | 0.255 | 0.328 | 0.115 | 0.242 | 0.306 | 0.125 | 0.231 | 0.303 | 0.113 |
| 201 | 7 | `P` | 0.060 | 0.109 | 0.005 | 0.059 | 0.116 | 0.003 | 0.055 | 0.140 | 0.012 | 0.056 | 0.117 | 0.007 | 0.040 | 0.084 | 0.009 |
| 202 | 7 | `H` | 0.238 | 0.694 | 0.000 | 0.239 | 0.684 | 0.000 | 0.243 | 0.699 | 0.000 | 0.234 | 0.689 | 0.000 | 0.236 | 0.690 | 0.000 |
| 202 | 7 | `P` | 0.087 | 0.179 | 0.010 | 0.096 | 0.161 | 0.023 | 0.147 | 0.257 | 0.038 | 0.141 | 0.249 | 0.044 | 0.057 | 0.117 | 0.005 |
| 203 | 8 | `H` | 0.300 | 0.520 | 0.000 | 0.297 | 0.511 | 0.000 | 0.300 | 0.521 | 0.000 | 0.297 | 0.509 | 0.000 | 0.309 | 0.524 | 0.000 |
| 203 | 8 | `P` | 0.107 | 0.213 | 0.004 | 0.087 | 0.169 | 0.004 | 0.096 | 0.184 | 0.003 | 0.092 | 0.184 | 0.004 | 0.101 | 0.191 | 0.004 |
| 204 | 4 | `H` | 0.429 | 0.491 | 0.244 | 0.422 | 0.483 | 0.240 | 0.445 | 0.511 | 0.247 | 0.455 | 0.526 | 0.242 | 0.441 | 0.510 | 0.234 |
| 204 | 4 | `P` | 0.194 | 0.326 | 0.030 | 0.166 | 0.241 | 0.052 | 0.198 | 0.263 | 0.072 | 0.151 | 0.263 | 0.030 | 0.171 | 0.245 | 0.082 |
| 205 | 8 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 205 | 8 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 206 | 5 | `H` | 0.297 | 1.000 | 0.000 | 0.302 | 0.999 | 0.000 | 0.307 | 1.000 | 0.000 | 0.297 | 1.000 | 0.000 | 0.299 | 0.999 | 0.000 |
| 206 | 5 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 207 | 4 | `H` | 0.504 | 0.504 | 0.504 | 0.501 | 0.501 | 0.501 | 0.497 | 0.496 | 0.496 | 0.491 | 0.491 | 0.491 | 0.499 | 0.499 | 0.499 |
| 207 | 4 | `P` | 0.496 | 0.990 | 0.496 | 0.976 | 0.977 | 0.976 | 0.999 | 0.999 | 0.999 | 1.000 | 1.000 | 1.000 | 0.999 | 1.000 | 0.999 |
| 208 | 8 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 208 | 8 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 209 | 6 | `H` | 0.000 | 0.238 | 0.000 | 0.000 | 0.218 | 0.000 | 0.000 | 0.215 | 0.000 | 0.000 | 0.209 | 0.000 | 0.000 | 0.203 | 0.000 |
| 209 | 6 | `P` | 0.087 | 0.114 | 0.087 | 0.008 | 0.022 | 0.009 | 0.008 | 0.021 | 0.008 | 0.018 | 0.042 | 0.018 | 0.013 | 0.023 | 0.013 |
| 210 | 7 | `H` | 0.503 | 0.502 | 0.502 | 0.482 | 0.482 | 0.482 | 0.489 | 0.489 | 0.489 | 0.483 | 0.483 | 0.483 | 0.490 | 0.490 | 0.490 |
| 210 | 7 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 211 | 4 | `H` | 0.342 | 0.342 | 0.342 | 0.349 | 0.349 | 0.349 | 0.347 | 0.347 | 0.347 | 0.352 | 0.352 | 0.352 | 0.345 | 0.345 | 0.345 |
| 211 | 4 | `P` | 0.053 | 0.079 | 0.053 | 0.000 | 0.000 | 0.000 | 0.001 | 0.002 | 0.001 | 0.004 | 0.005 | 0.004 | 0.165 | 0.188 | 0.165 |
| 212 | 7 | `H` | 0.021 | 0.046 | 0.000 | 0.029 | 0.064 | 0.000 | 0.030 | 0.067 | 0.000 | 0.031 | 0.068 | 0.000 | 0.034 | 0.073 | 0.000 |
| 212 | 7 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 213 | 7 | `H` | 0.175 | 0.624 | 0.000 | 0.334 | 0.903 | 0.000 | 0.369 | 0.960 | 0.000 | 0.381 | 0.981 | 0.000 | 0.378 | 0.986 | 0.000 |
| 213 | 7 | `P` | 0.280 | 0.584 | 0.004 | 0.365 | 0.715 | 0.000 | 0.522 | 0.888 | 0.000 | 0.341 | 0.695 | 0.000 | 0.390 | 0.690 | 0.000 |
| 214 | 6 | `H` | 0.377 | 0.760 | 0.000 | 0.414 | 0.798 | 0.000 | 0.437 | 0.801 | 0.000 | 0.451 | 0.799 | 0.000 | 0.468 | 0.804 | 0.000 |
| 214 | 6 | `P` | 0.367 | 0.935 | 0.000 | 0.582 | 0.982 | 0.001 | 0.556 | 0.997 | 0.000 | 0.570 | 0.998 | 0.000 | 0.566 | 0.997 | 0.000 |
| 215 | 8 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 215 | 8 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 216 | 7 | `H` | 0.081 | 0.174 | 0.000 | 0.088 | 0.186 | 0.000 | 0.091 | 0.190 | 0.000 | 0.085 | 0.184 | 0.000 | 0.083 | 0.182 | 0.000 |
| 216 | 7 | `P` | 0.065 | 0.129 | 0.027 | 0.066 | 0.113 | 0.012 | 0.060 | 0.100 | 0.015 | 0.049 | 0.089 | 0.010 | 0.048 | 0.092 | 0.007 |
| 217 | 6 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 217 | 6 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 218 | 4 | `H` |  | 0.000 | 0.000 |  | 0.000 | 0.000 |  | 0.000 | 0.000 |  | 0.000 | 0.000 |  | 0.000 | 0.000 |
| 218 | 4 | `P` | 0.780 | 0.851 | 0.410 | 0.924 | 0.990 | 0.499 | 0.941 | 0.999 | 0.504 | 0.945 | 0.999 | 0.507 | 0.942 | 0.998 | 0.499 |
| 219 | 5 | `H` | 0.444 | 1.000 | 0.305 | 0.460 | 1.000 | 0.326 | 0.462 | 1.000 | 0.327 | 0.469 | 1.000 | 0.337 | 0.457 | 1.000 | 0.321 |
| 219 | 5 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 220 | 7 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 220 | 7 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 221 | 4 | `H` | 0.262 | 1.000 | 0.000 | 0.258 | 1.000 | 0.000 | 0.259 | 1.000 | 0.000 | 0.255 | 1.000 | 0.000 | 0.256 | 1.000 | 0.000 |
| 221 | 4 | `P` | 0.158 | 0.405 | 0.091 | 0.104 | 0.356 | 0.060 | 0.128 | 0.436 | 0.054 | 0.166 | 0.433 | 0.100 | 0.183 | 0.405 | 0.114 |
| 222 | 4 | `H` | 0.619 | 1.000 | 0.250 | 0.614 | 1.000 | 0.253 | 0.627 | 1.000 | 0.251 | 0.592 | 1.000 | 0.239 | 0.612 | 1.000 | 0.232 |
| 222 | 4 | `P` | 0.188 | 0.363 | 0.094 | 0.169 | 0.251 | 0.102 | 0.119 | 0.200 | 0.042 | 0.200 | 0.312 | 0.079 | 0.178 | 0.305 | 0.087 |
| 223 | 5 | `H` | 0.188 | 0.415 | 0.095 | 0.213 | 0.465 | 0.127 | 0.195 | 0.432 | 0.094 | 0.230 | 0.479 | 0.124 | 0.243 | 0.461 | 0.139 |
| 223 | 5 | `P` | 0.029 | 0.129 | 0.003 | 0.330 | 0.335 | 0.329 | 0.464 | 0.465 | 0.464 | 0.476 | 0.477 | 0.476 | 0.090 | 0.262 | 0.083 |
| 224 | 8 | `H` |  | 0.000 | 0.000 |  | 0.000 | 0.000 |  | 0.000 | 0.000 |  | 0.000 | 0.000 |  | 0.000 | 0.000 |
| 224 | 8 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 225 | 5 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 225 | 5 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 226 | 4 | `H` | 0.360 | 1.000 | 0.264 | 0.365 | 1.000 | 0.270 | 0.348 | 1.000 | 0.251 | 0.336 | 1.000 | 0.237 | 0.326 | 1.000 | 0.236 |
| 226 | 4 | `P` | 0.072 | 0.198 | 0.021 | 0.097 | 0.297 | 0.068 | 0.095 | 0.271 | 0.063 | 0.080 | 0.228 | 0.047 | 0.089 | 0.253 | 0.054 |
| 227 | 4 | `H` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 227 | 4 | `P` | 0.708 | 0.891 | 0.708 | 0.948 | 0.978 | 0.948 | 0.996 | 0.996 | 0.996 | 0.997 | 0.998 | 0.998 | 0.986 | 0.987 | 0.986 |
| 228 | 8 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 228 | 8 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 229 | 5 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 229 | 5 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 230 | 5 | `H` | 0.169 | 0.309 | 0.000 | 0.168 | 0.310 | 0.000 | 0.165 | 0.305 | 0.000 | 0.195 | 0.408 | 0.000 | 0.205 | 0.315 | 0.000 |
| 230 | 5 | `P` | 0.123 | 0.165 | 0.095 | 0.135 | 0.143 | 0.124 | 0.059 | 0.077 | 0.048 | 0.078 | 0.120 | 0.057 | 0.069 | 0.137 | 0.043 |
| 231 | 8 | `H` | 0.249 | 0.848 | 0.000 | 0.251 | 0.851 | 0.000 | 0.251 | 0.849 | 0.000 | 0.252 | 0.851 | 0.000 | 0.254 | 0.859 | 0.000 |
| 231 | 8 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 232 | 6 | `H` | 0.540 | 0.918 | 0.000 | 0.587 | 0.999 | 0.000 | 0.588 | 1.000 | 0.000 | 0.588 | 1.000 | 0.000 | 0.588 | 1.000 | 0.000 |
| 232 | 6 | `P` | 0.717 | 0.896 | 0.461 | 0.921 | 0.973 | 0.846 | 0.992 | 0.997 | 0.985 | 0.997 | 0.997 | 0.997 | 0.991 | 0.992 | 0.989 |
| 233 | 7 | `H` | 0.191 | 1.000 | 0.000 | 0.191 | 1.000 | 0.000 | 0.190 | 1.000 | 0.000 | 0.190 | 1.000 | 0.000 | 0.190 | 1.000 | 0.000 |
| 233 | 7 | `P` | 0.231 | 0.484 | 0.027 | 0.401 | 0.639 | 0.078 | 0.404 | 0.652 | 0.013 | 0.268 | 0.490 | 0.002 | 0.463 | 0.677 | 0.056 |
| 234 | 5 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 234 | 5 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 235 | 7 | `H` |  | 0.000 | 0.000 |  | 0.000 | 0.000 |  | 0.000 | 0.000 |  | 0.000 | 0.000 |  | 0.000 | 0.000 |
| 235 | 7 | `P` | 0.085 | 0.174 | 0.003 | 0.103 | 0.198 | 0.017 | 0.094 | 0.156 | 0.006 | 0.122 | 0.206 | 0.022 | 0.098 | 0.223 | 0.017 |
| 236 | 5 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 236 | 5 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 237 | 8 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 237 | 8 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 238 | 6 | `H` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 238 | 6 | `P` | 0.295 | 0.528 | 0.295 | 0.662 | 0.955 | 0.662 | 0.997 | 0.997 | 0.997 | 0.999 | 0.999 | 0.999 | 0.999 | 0.999 | 0.999 |
| 239 | 7 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 239 | 7 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 240 | 8 | `H` | 0.012 | 0.100 | 0.000 | 0.012 | 0.104 | 0.000 | 0.011 | 0.094 | 0.000 | 0.011 | 0.094 | 0.000 | 0.011 | 0.094 | 0.000 |
| 240 | 8 | `P` | 0.021 | 0.038 | 0.009 | 0.012 | 0.037 | 0.003 | 0.011 | 0.027 | 0.006 | 0.012 | 0.034 | 0.005 | 0.013 | 0.041 | 0.005 |
| 241 | 4 | `H` | 0.096 | 0.835 | 0.000 | 0.095 | 0.828 | 0.000 | 0.094 | 0.829 | 0.000 | 0.095 | 0.832 | 0.000 | 0.096 | 0.834 | 0.000 |
| 241 | 4 | `P` | 0.806 | 0.868 | 0.795 | 0.985 | 0.985 | 0.985 | 0.999 | 0.999 | 0.999 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 242 | 7 | `H` | 0.507 | 1.000 | 0.044 | 0.504 | 1.000 | 0.041 | 0.507 | 1.000 | 0.043 | 0.510 | 1.000 | 0.041 | 0.507 | 1.000 | 0.043 |
| 242 | 7 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 243 | 7 | `H` | 0.379 | 1.000 | 0.000 | 0.376 | 1.000 | 0.000 | 0.375 | 1.000 | 0.000 | 0.381 | 1.000 | 0.000 | 0.379 | 1.000 | 0.000 |
| 243 | 7 | `P` | 0.097 | 0.183 | 0.009 | 0.092 | 0.172 | 0.010 | 0.086 | 0.160 | 0.009 | 0.088 | 0.170 | 0.007 | 0.088 | 0.170 | 0.006 |
| 244 | 5 | `H` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 244 | 5 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 245 | 4 | `H` | 0.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| 245 | 4 | `P` | 0.110 | 0.229 | 0.110 | 0.082 | 0.235 | 0.082 | 0.034 | 0.204 | 0.034 | 0.035 | 0.160 | 0.035 | 0.056 | 0.134 | 0.056 |
| 246 | 8 | `H` |  | 0.000 | 0.000 |  | 0.000 | 0.000 |  | 0.000 | 0.000 |  | 0.000 | 0.000 |  | 0.000 | 0.000 |
| 246 | 8 | `P` | 0.029 | 0.064 | 0.006 | 0.023 | 0.051 | 0.005 | 0.019 | 0.044 | 0.004 | 0.020 | 0.043 | 0.003 | 0.021 | 0.044 | 0.004 |
| 247 | 8 | `H` | 0.118 | 0.360 | 0.000 | 0.115 | 0.306 | 0.000 | 0.114 | 0.310 | 0.000 | 0.115 | 0.309 | 0.000 | 0.122 | 0.326 | 0.000 |
| 247 | 8 | `P` | 0.043 | 0.086 | 0.003 | 0.075 | 0.260 | 0.000 | 0.324 | 0.451 | 0.000 | 0.631 | 0.664 | 0.309 | 0.398 | 0.526 | 0.000 |
| 248 | 6 | `H` | 0.250 | 1.000 | 0.000 | 0.248 | 1.000 | 0.000 | 0.253 | 1.000 | 0.000 | 0.251 | 1.000 | 0.000 | 0.251 | 1.000 | 0.000 |
| 248 | 6 | `P` | 0.071 | 0.117 | 0.058 | 0.076 | 0.102 | 0.066 | 0.070 | 0.102 | 0.059 | 0.052 | 0.099 | 0.037 | 0.051 | 0.090 | 0.037 |
| 249 | 5 | `H` | 0.319 | 0.657 | 0.038 | 0.307 | 0.649 | 0.037 | 0.312 | 0.654 | 0.037 | 0.312 | 0.648 | 0.040 | 0.311 | 0.649 | 0.037 |
| 249 | 5 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

---

*Generated by `analyze_results/generate_results_markdown.py`.*
