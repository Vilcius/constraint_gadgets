# Disjoint Problem Set — Raw Results

**250 problem instances** (disjoint constraint variable supports).
Empty cells indicate runs that did not complete (transient cluster errors).

---

## Problem Definitions

| COP | $n_x$ | Families | Structural constraints | Penalized constraints | Qubits (H/P) | SP gates (H/P) | Layer gates (H/P) |
|-----|-------|----------|------------------------|-----------------------|-------------|---------------|------------------|
| 0 | 6 | cardinality, quadratic_knapsack | $x_{0}+ x_{1} \leq 0$ *(cardinality)*<br>$3x_{2}^2+ x_{2}x_{3}+ 3x_{2}x_{4}$<br>$+ 4x_{2}x_{5} + 5x_{3}^2$<br>$+ 4x_{3}x_{4} + 2x_{3}x_{5}$<br>$+ 2x_{4}^2 + 4x_{4}x_{5}+ 4x_{5}^2 \leq 17$ *(quadratic_knapsack)* | — | 6/11 | 19/5 | 101/664 |
| 1 | 7 | cardinality, cardinality | $x_{0}+ x_{1}+ x_{2}+ x_{3} \leq 2$ *(cardinality)*<br>$x_{4}+ x_{5}+ x_{6} \geq 0$ *(cardinality)* | — | 7/11 | 19/4 | 126/130 |
| 2 | 8 | knapsack, knapsack | $9x_{0}+ 2x_{1}+ x_{2}+ 9x_{3}$<br>$+ 6x_{4} \leq 13$ *(knapsack)*<br>$6x_{5}+ 2x_{6}+ 2x_{7} \leq 3$ *(knapsack)* | — | 8/14 | 67/6 | 239/211 |
| 3 | 7 | cardinality, cardinality | $x_{0}+ x_{1}+ x_{2} = 2$ *(cardinality)*<br>$x_{3}+ x_{4}+ x_{5}+ x_{6} \leq 1$ *(cardinality)* | — | 7/8 | 24/1 | 129/87 |
| 4 | 8 | knapsack, cardinality | $3x_{0}+ x_{1}+ 4x_{2}+ 8x_{3} \leq 7$ *(knapsack)*<br>$x_{4}+ x_{5}+ x_{6}+ x_{7} = 4$ *(cardinality)* | — | 8/11 | 61/3 | 229/147 |
| 5 | 6 | quadratic_knapsack, knapsack | $5x_{0}^2+ x_{0}x_{1}+ 2x_{0}x_{2}$<br>$+ 5x_{1}^2 + x_{1}x_{2}+ 3x_{2}^2 \leq 6$ *(quadratic_knapsack)*<br>$6x_{3}+ 2x_{4}+ 2x_{5} \leq 3$ *(knapsack)* | — | 6/11 | 31/5 | 121/272 |
| 6 | 6 | quadratic_knapsack, knapsack | $5x_{0}^2+ 3x_{0}x_{1}+ 5x_{0}x_{2}$<br>$+ 4x_{1}^2 + 5x_{1}x_{2}+ 3x_{2}^2 \leq 14$ *(quadratic_knapsack)*<br>$2x_{3}+ 5x_{4}+ 2x_{5} \leq 5$ *(knapsack)* | — | 6/13 | 75/7 | 219/336 |
| 7 | 7 | assignment, flow | $x_{0}+ x_{1}+ x_{2} = 1$ *(assignment)*<br>$x_{3}+ x_{4}- x_{5}- x_{6} = 0$ *(flow)* | — | 7/7 | 17/0 | 119/70 |
| 8 | 5 | cardinality, quadratic_knapsack | $x_{0}+ x_{1} \leq 2$ *(cardinality)*<br>$x_{2}^2+ 5x_{2}x_{3}+ 4x_{2}x_{4}$<br>$+ 2x_{3}^2 + 2x_{3}x_{4}+ 2x_{4}^2 \leq 8$ *(quadratic_knapsack)* | — | 5/11 | 18/6 | 86/297 |
| 9 | 7 | quadratic_knapsack, cardinality | $x_{0}^2+ 5x_{0}x_{1}+ 3x_{0}x_{2}$<br>$+ 4x_{1}^2 + 2x_{1}x_{2}+ 4x_{2}^2 \leq 7$ *(quadratic_knapsack)*<br>$x_{3}+ x_{4}+ x_{5}+ x_{6} \geq 3$ *(cardinality)* | — | 7/11 | 8/4 | 104/282 |
| 10 | 7 | knapsack, quadratic_knapsack | $7x_{0}+ 7x_{1}+ 10x_{2}+ 2x_{3} \leq 10$ *(knapsack)*<br>$5x_{4}^2+ 3x_{4}x_{5}+ 5x_{4}x_{6}$<br>$+ 4x_{5}^2 + 5x_{5}x_{6}+ 3x_{6}^2 \leq 14$ *(quadratic_knapsack)* | — | 7/15 | 281/8 | 643/384 |
| 11 | 8 | knapsack, cardinality | $3x_{0}+ 8x_{1}+ 3x_{2} \leq 5$ *(knapsack)*<br>$x_{3}+ x_{4}+ x_{5}+ x_{6}+ x_{7} \geq 0$ *(cardinality)* | — | 8/14 | 103/6 | 318/197 |
| 12 | 8 | quadratic_knapsack, cardinality | $2x_{0}^2+ 5x_{0}x_{1}+ 2x_{0}x_{2}$<br>$+ 2x_{1}^2 + x_{1}x_{2}+ x_{2}^2 \leq 7$ *(quadratic_knapsack)*<br>$x_{3}+ x_{4}+ x_{5}+ x_{6}+ x_{7} = 5$ *(cardinality)* | — | 8/11 | 56/3 | 217/286 |
| 13 | 7 | quadratic_knapsack, cardinality | $2x_{0}^2+ 5x_{0}x_{1}+ 4x_{0}x_{2}$<br>$+ 2x_{0}x_{3} + 4x_{0}x_{4}$<br>$+ 5x_{1}^2 + 5x_{1}x_{2}$<br>$+ 2x_{1}x_{3} + x_{1}x_{4}$<br>$+ 4x_{2}^2 + x_{2}x_{3}$<br>$+ 2x_{2}x_{4} + 5x_{3}^2$<br>$+ 4x_{3}x_{4} + 2x_{4}^2 \leq 29$ *(quadratic_knapsack)*<br>$x_{5}+ x_{6} \leq 1$ *(cardinality)* | — | 7/13 | 37/6 | 155/1332 |
| 14 | 8 | quadratic_knapsack, knapsack | $x_{0}^2+ 4x_{0}x_{1}+ x_{0}x_{2}$<br>$+ 5x_{0}x_{3} + 2x_{1}^2$<br>$+ 3x_{1}x_{2} + 3x_{1}x_{3}$<br>$+ 5x_{2}^2 + 4x_{2}x_{3}+ 5x_{3}^2 \leq 22$ *(quadratic_knapsack)*<br>$3x_{4}+ x_{5}+ 4x_{6}+ 8x_{7} \leq 7$ *(knapsack)* | — | 8/16 | 80/8 | 267/755 |
| 15 | 8 | cardinality, knapsack | $x_{0}+ x_{1}+ x_{2} \geq 2$ *(cardinality)*<br>$2x_{3}+ 2x_{4}+ 9x_{5}+ 10x_{6}$<br>$+ 4x_{7} \leq 18$ *(knapsack)* | — | 8/14 | 35/6 | 180/229 |
| 16 | 6 | quadratic_knapsack, quadratic_knapsack | $2x_{0}^2+ 5x_{0}x_{1}+ 2x_{0}x_{2}$<br>$+ 2x_{1}^2 + x_{1}x_{2}+ x_{2}^2 \leq 7$ *(quadratic_knapsack)*<br>$3x_{3}^2+ x_{3}x_{4}+ 3x_{3}x_{5}$<br>$+ 5x_{4}^2 + 2x_{4}x_{5}+ 2x_{5}^2 \leq 8$ *(quadratic_knapsack)* | — | 6/13 | 74/7 | 214/482 |
| 17 | 8 | cardinality, knapsack | $x_{0}+ x_{1}+ x_{2}+ x_{3}+ x_{4} = 1$ *(cardinality)*<br>$2x_{5}+ 7x_{6}+ 7x_{7} \leq 7$ *(knapsack)* | — | 8/11 | 17/3 | 144/142 |
| 18 | 8 | knapsack, flow | $9x_{0}+ 9x_{1}+ 4x_{2}+ x_{3} \leq 9$ *(knapsack)*<br>$x_{4}- x_{5}- x_{6}- x_{7} = 0$ *(flow)* | — | 8/12 | 75/4 | 263/169 |
| 19 | 8 | cardinality, cardinality | $x_{0}+ x_{1}+ x_{2} \geq 2$ *(cardinality)*<br>$x_{3}+ x_{4}+ x_{5}+ x_{6}+ x_{7} \leq 5$ *(cardinality)* | — | 8/12 | 37/4 | 184/172 |
| 20 | 6 | cardinality, quadratic_knapsack | $x_{0}+ x_{1}+ x_{2} = 1$ *(cardinality)*<br>$5x_{3}^2+ 3x_{3}x_{4}+ 5x_{3}x_{5}$<br>$+ 4x_{4}^2 + 5x_{4}x_{5}+ 3x_{5}^2 \leq 14$ *(quadratic_knapsack)* | — | 6/10 | 17/4 | 97/289 |
| 21 | 6 | independent_set, quadratic_knapsack | $x_{0}x_{1} = 0$ *(independent_set)*<br>$x_{2}^2+ x_{2}x_{3}+ 3x_{2}x_{4}$<br>$+ 2x_{2}x_{5} + x_{3}^2$<br>$+ 2x_{3}x_{4} + 3x_{3}x_{5}$<br>$+ 2x_{4}^2 + 3x_{4}x_{5}+ 3x_{5}^2 \leq 7$ *(quadratic_knapsack)* | — | 6/9 | 128/3 | 325/529 |
| 22 | 6 | cardinality, quadratic_knapsack | $x_{0}+ x_{1}+ x_{2} \leq 3$ *(cardinality)*<br>$x_{3}^2+ 5x_{3}x_{4}+ 4x_{3}x_{5}$<br>$+ 2x_{4}^2 + 2x_{4}x_{5}+ 2x_{5}^2 \leq 8$ *(quadratic_knapsack)* | — | 6/12 | 25/6 | 113/316 |
| 23 | 8 | cardinality, knapsack | $x_{0}+ x_{1}+ x_{2}+ x_{3} \geq 2$ *(cardinality)*<br>$8x_{4}+ 5x_{5}+ 7x_{6}+ 4x_{7} \leq 11$ *(knapsack)* | — | 8/14 | 57/6 | 221/204 |
| 24 | 8 | quadratic_knapsack, cardinality | $3x_{0}^2+ x_{0}x_{1}+ 3x_{0}x_{2}$<br>$+ x_{1}^2 + 4x_{1}x_{2}+ 3x_{2}^2 \leq 5$ *(quadratic_knapsack)*<br>$x_{3}+ x_{4}+ x_{5}+ x_{6}+ x_{7} = 3$ *(cardinality)* | — | 8/11 | 114/3 | 338/289 |
| 25 | 7 | cardinality, cardinality | $x_{0}+ x_{1}+ x_{2}+ x_{3}+ x_{4} \leq 1$ *(cardinality)*<br>$x_{5}+ x_{6} = 2$ *(cardinality)* | — | 7/8 | 17/1 | 122/99 |
| 26 | 6 | cardinality, quadratic_knapsack | $x_{0}+ x_{1}+ x_{2} = 0$ *(cardinality)*<br>$x_{3}^2+ 2x_{3}x_{4}+ x_{3}x_{5}$<br>$+ x_{4}^2 + 5x_{4}x_{5}+ x_{5}^2 \leq 6$ *(quadratic_knapsack)* | — | 6/9 | 43/3 | 152/245 |
| 27 | 5 | cardinality, flow | $x_{0}+ x_{1} \geq 2$ *(cardinality)*<br>$x_{2}- x_{3}- x_{4} = 0$ *(flow)* | — | 5/5 | 8/0 | 66/39 |
| 28 | 8 | quadratic_knapsack, flow | $3x_{0}^2+ x_{0}x_{1}+ 3x_{0}x_{2}$<br>$+ 5x_{1}^2 + 2x_{1}x_{2}+ 2x_{2}^2 \leq 8$ *(quadratic_knapsack)*<br>$x_{3}+ x_{4}- x_{5}- x_{6}- x_{7} = 0$ *(flow)* | — | 8/12 | 41/4 | 189/325 |
| 29 | 7 | cardinality, cardinality | $x_{0}+ x_{1} \geq 1$ *(cardinality)*<br>$x_{2}+ x_{3}+ x_{4}+ x_{5}+ x_{6} \geq 3$ *(cardinality)* | — | 7/10 | 0/3 | 81/126 |
| 30 | 8 | cardinality, cardinality | $x_{0}+ x_{1}+ x_{2}+ x_{3} \geq 0$ *(cardinality)*<br>$x_{4}+ x_{5}+ x_{6}+ x_{7} = 3$ *(cardinality)* | — | 8/11 | 23/3 | 162/143 |
| 31 | 7 | knapsack, quadratic_knapsack | $5x_{0}+ 4x_{1}+ x_{2} \leq 6$ *(knapsack)*<br>$x_{3}^2+ 4x_{3}x_{4}+ x_{3}x_{5}$<br>$+ 2x_{3}x_{6} + 5x_{4}^2$<br>$+ 4x_{4}x_{5} + 3x_{4}x_{6}$<br>$+ 5x_{5}^2 + 4x_{5}x_{6}+ 2x_{6}^2 \leq 18$ *(quadratic_knapsack)* | — | 7/15 | 386/8 | 851/725 |
| 32 | 7 | cardinality, knapsack | $x_{0}+ x_{1} \geq 0$ *(cardinality)*<br>$x_{2}+ 6x_{3}+ 9x_{4}+ 8x_{5}$<br>$+ 9x_{6} \leq 18$ *(knapsack)* | — | 7/14 | 139/7 | 357/215 |
| 33 | 6 | assignment, cardinality | $x_{0}+ x_{1}+ x_{2} = 1$ *(assignment)*<br>$x_{3}+ x_{4}+ x_{5} = 2$ *(cardinality)* | — | 6/6 | 17/0 | 95/53 |
| 34 | 8 | cardinality, knapsack | $x_{0}+ x_{1}+ x_{2}+ x_{3}+ x_{4} \geq 4$ *(cardinality)*<br>$7x_{5}+ 10x_{6}+ 9x_{7} \leq 8$ *(knapsack)* | — | 8/13 | 11/5 | 128/179 |
| 35 | 4 | assignment, cardinality | $x_{0}+ x_{1} = 1$ *(assignment)*<br>$x_{2}+ x_{3} \geq 1$ *(cardinality)* | — | 4/5 | 3/1 | 39/36 |
| 36 | 7 | cardinality, knapsack | $x_{0}+ x_{1} \geq 0$ *(cardinality)*<br>$5x_{2}+ 10x_{3}+ x_{4}+ 9x_{5}$<br>$+ 6x_{6} \leq 19$ *(knapsack)* | — | 7/14 | 407/7 | 899/217 |
| 37 | 8 | cardinality, flow | $x_{0}+ x_{1} \geq 1$ *(cardinality)*<br>$x_{2}+ x_{3}+ x_{4}- x_{5}- x_{6}$<br>$- x_{7} = 0$ *(flow)* | — | 8/9 | 26/1 | 159/108 |
| 38 | 7 | knapsack, cardinality | $x_{0}+ 9x_{1}+ 5x_{2}+ 2x_{3}$<br>$+ 10x_{4} \leq 10$ *(knapsack)*<br>$x_{5}+ x_{6} \geq 2$ *(cardinality)* | — | 7/11 | 141/4 | 363/168 |
| 39 | 8 | knapsack, cardinality | $x_{0}+ 4x_{1}+ 8x_{2}+ 10x_{3}$<br>$+ 8x_{4} \leq 16$ *(knapsack)*<br>$x_{5}+ x_{6}+ x_{7} = 1$ *(cardinality)* | — | 8/13 | 412/5 | 934/217 |
| 40 | 8 | quadratic_knapsack, knapsack | $x_{0}^2+ 5x_{0}x_{1}+ 3x_{0}x_{2}$<br>$+ 4x_{1}^2 + 2x_{1}x_{2}+ 4x_{2}^2 \leq 7$ *(quadratic_knapsack)*<br>$6x_{3}+ 7x_{4}+ 8x_{5}+ 9x_{6}$<br>$+ 5x_{7} \leq 14$ *(knapsack)* | — | 8/15 | 511/7 | 1134/379 |
| 41 | 5 | cardinality, knapsack | $x_{0}+ x_{1} = 1$ *(cardinality)*<br>$2x_{2}+ 5x_{3}+ 2x_{4} \leq 5$ *(knapsack)* | — | 5/8 | 66/3 | 182/87 |
| 42 | 7 | cardinality, knapsack | $x_{0}+ x_{1}+ x_{2} = 0$ *(cardinality)*<br>$7x_{3}+ 7x_{4}+ 10x_{5}+ 2x_{6} \leq 10$ *(knapsack)* | — | 7/11 | 269/4 | 619/147 |
| 43 | 8 | quadratic_knapsack, quadratic_knapsack | $x_{0}^2+ 5x_{0}x_{1}+ 3x_{0}x_{2}$<br>$+ 4x_{1}^2 + 2x_{1}x_{2}+ 4x_{2}^2 \leq 7$ *(quadratic_knapsack)*<br>$3x_{3}^2+ x_{3}x_{4}+ 5x_{3}x_{5}$<br>$+ 2x_{3}x_{6} + 5x_{3}x_{7}$<br>$+ 2x_{4}^2 + 5x_{4}x_{5}$<br>$+ 5x_{4}x_{6} + 5x_{4}x_{7}$<br>$+ 2x_{5}^2 + 3x_{5}x_{6}$<br>$+ 3x_{5}x_{7} + 4x_{6}^2$<br>$+ x_{6}x_{7} + 5x_{7}^2 \leq 18$ *(quadratic_knapsack)* | — | 8/16 | 415/8 | 937/1533 |
| 44 | 7 | cardinality, cardinality | $x_{0}+ x_{1}+ x_{2}+ x_{3} \geq 2$ *(cardinality)*<br>$x_{4}+ x_{5}+ x_{6} \leq 1$ *(cardinality)* | — | 7/10 | 8/3 | 104/120 |
| 45 | 8 | quadratic_knapsack, cardinality | $3x_{0}^2+ x_{0}x_{1}+ 3x_{0}x_{2}$<br>$+ x_{1}^2 + 4x_{1}x_{2}+ 3x_{2}^2 \leq 5$ *(quadratic_knapsack)*<br>$x_{3}+ x_{4}+ x_{5}+ x_{6}+ x_{7} \leq 1$ *(cardinality)* | — | 8/12 | 98/4 | 312/309 |
| 46 | 8 | cardinality, knapsack | $x_{0}+ x_{1}+ x_{2} = 2$ *(cardinality)*<br>$x_{3}+ 10x_{4}+ 3x_{5}+ 5x_{6}$<br>$+ x_{7} \leq 12$ *(knapsack)* | — | 8/12 | 83/4 | 276/187 |
| 47 | 7 | knapsack, independent_set | $x_{0}+ 10x_{1}+ 3x_{2}+ 5x_{3}$<br>$+ x_{4} \leq 12$ *(knapsack)*<br>$x_{5}x_{6} = 0$ *(independent_set)* | — | 7/11 | 71/4 | 224/165 |
| 48 | 7 | quadratic_knapsack, knapsack | $3x_{0}^2+ x_{0}x_{1}+ 3x_{0}x_{2}$<br>$+ 4x_{0}x_{3} + 5x_{1}^2$<br>$+ 4x_{1}x_{2} + 2x_{1}x_{3}$<br>$+ 2x_{2}^2 + 4x_{2}x_{3}+ 4x_{3}^2 \leq 17$ *(quadratic_knapsack)*<br>$5x_{4}+ 4x_{5}+ x_{6} \leq 6$ *(knapsack)* | — | 7/15 | 30/8 | 139/725 |
| 49 | 6 | knapsack, cardinality | $x_{0}+ 3x_{1}+ 9x_{2}+ 8x_{3} \leq 10$ *(knapsack)*<br>$x_{4}+ x_{5} \leq 1$ *(cardinality)* | — | 6/11 | 105/5 | 269/143 |
| 50 | 6 | knapsack, cardinality | $2x_{0}+ 7x_{1}+ 7x_{2} \leq 7$ *(knapsack)*<br>$x_{3}+ x_{4}+ x_{5} \leq 3$ *(cardinality)* | — | 6/11 | 21/5 | 108/128 |
| 51 | 8 | flow, cardinality | $x_{0}- x_{1}- x_{2} = 0$ *(flow)*<br>$x_{3}+ x_{4}+ x_{5}+ x_{6}+ x_{7} \leq 3$ *(cardinality)* | — | 8/10 | 38/2 | 192/135 |
| 52 | 6 | quadratic_knapsack, cardinality | $x_{0}^2+ 5x_{0}x_{1}+ 3x_{0}x_{2}$<br>$+ 4x_{1}^2 + 2x_{1}x_{2}+ 4x_{2}^2 \leq 7$ *(quadratic_knapsack)*<br>$x_{3}+ x_{4}+ x_{5} \leq 0$ *(cardinality)* | — | 6/9 | 8/3 | 75/242 |
| 53 | 8 | quadratic_knapsack, cardinality | $2x_{0}^2+ 5x_{0}x_{1}+ 4x_{0}x_{2}$<br>$+ 4x_{0}x_{3} + 3x_{0}x_{4}$<br>$+ 4x_{1}^2 + 4x_{1}x_{2}$<br>$+ x_{1}x_{3} + 2x_{1}x_{4}$<br>$+ 3x_{2}^2 + x_{2}x_{3}$<br>$+ 4x_{2}x_{4} + 3x_{3}^2$<br>$+ 4x_{3}x_{4} + 3x_{4}^2 \leq 16$ *(quadratic_knapsack)*<br>$x_{5}+ x_{6}+ x_{7} \geq 3$ *(cardinality)* | — | 8/13 | 1080/5 | 2270/1342 |
| 54 | 7 | cardinality, cardinality | $x_{0}+ x_{1}+ x_{2}+ x_{3} \geq 3$ *(cardinality)*<br>$x_{4}+ x_{5}+ x_{6} \geq 3$ *(cardinality)* | — | 7/8 | 3/1 | 94/90 |
| 55 | 6 | cardinality, cardinality | $x_{0}+ x_{1}+ x_{2}+ x_{3} \leq 2$ *(cardinality)*<br>$x_{4}+ x_{5} \leq 0$ *(cardinality)* | — | 6/8 | 19/2 | 99/87 |
| 56 | 6 | assignment, quadratic_knapsack | $x_{0}+ x_{1} = 1$ *(assignment)*<br>$5x_{2}^2+ 3x_{2}x_{3}+ x_{2}x_{4}$<br>$+ 5x_{2}x_{5} + 3x_{3}^2$<br>$+ x_{3}x_{4} + 4x_{3}x_{5}$<br>$+ 5x_{4}^2 + 5x_{4}x_{5}+ x_{5}^2 \leq 20$ *(quadratic_knapsack)* | — | 6/11 | 378/5 | 819/666 |
| 57 | 8 | cardinality, flow | $x_{0}+ x_{1}+ x_{2}+ x_{3}+ x_{4} \geq 1$ *(cardinality)*<br>$x_{5}- x_{6}- x_{7} = 0$ *(flow)* | — | 8/11 | 6/3 | 119/156 |
| 58 | 6 | cardinality, cardinality | $x_{0}+ x_{1}+ x_{2}+ x_{3} \geq 1$ *(cardinality)*<br>$x_{4}+ x_{5} \leq 2$ *(cardinality)* | — | 6/10 | 6/4 | 75/112 |
| 59 | 8 | knapsack, flow | $4x_{0}+ 6x_{1}+ x_{2}+ x_{3} \leq 6$ *(knapsack)*<br>$x_{4}+ x_{5}- x_{6}- x_{7} = 0$ *(flow)* | — | 8/11 | 281/3 | 669/143 |
| 60 | 8 | cardinality, knapsack | $x_{0}+ x_{1}+ x_{2}+ x_{3}+ x_{4} \leq 4$ *(cardinality)*<br>$2x_{5}+ 7x_{6}+ 7x_{7} \leq 7$ *(knapsack)* | — | 8/14 | 44/6 | 201/206 |
| 61 | 6 | quadratic_knapsack, flow | $3x_{0}^2+ x_{0}x_{1}+ 3x_{0}x_{2}$<br>$+ 5x_{1}^2 + 2x_{1}x_{2}+ 2x_{2}^2 \leq 8$ *(quadratic_knapsack)*<br>$x_{3}- x_{4}- x_{5} = 0$ *(flow)* | — | 6/10 | 29/4 | 121/286 |
| 62 | 7 | knapsack, quadratic_knapsack | $2x_{0}+ 7x_{1}+ 7x_{2} \leq 7$ *(knapsack)*<br>$5x_{3}^2+ x_{3}x_{4}+ 3x_{3}x_{5}$<br>$+ x_{3}x_{6} + 4x_{4}^2+ x_{4}x_{5}$<br>$+ 5x_{4}x_{6} + 4x_{5}^2$<br>$+ 2x_{5}x_{6} + 3x_{6}^2 \leq 19$ *(quadratic_knapsack)* | — | 7/15 | 31/8 | 150/728 |
| 63 | 7 | knapsack, knapsack | $4x_{0}+ 6x_{1}+ 2x_{2}+ 10x_{3} \leq 7$ *(knapsack)*<br>$10x_{4}+ 8x_{5}+ 10x_{6} \leq 9$ *(knapsack)* | — | 7/14 | 68/7 | 219/192 |
| 64 | 8 | quadratic_knapsack, cardinality | $x_{0}^2+ 5x_{0}x_{1}+ x_{0}x_{2}$<br>$+ 3x_{1}^2 + 3x_{1}x_{2}+ 5x_{2}^2 \leq 10$ *(quadratic_knapsack)*<br>$x_{3}+ x_{4}+ x_{5}+ x_{6}+ x_{7} = 3$ *(cardinality)* | — | 8/12 | 42/4 | 200/333 |
| 65 | 5 | cardinality, cardinality | $x_{0}+ x_{1}+ x_{2} \geq 1$ *(cardinality)*<br>$x_{3}+ x_{4} \leq 0$ *(cardinality)* | — | 5/7 | 0/2 | 50/67 |
| 66 | 5 | independent_set, knapsack | $x_{0}x_{1} = 0$ *(independent_set)*<br>$10x_{2}+ 8x_{3}+ 10x_{4} \leq 9$ *(knapsack)* | — | 5/9 | 11/4 | 72/104 |
| 67 | 8 | knapsack, cardinality | $2x_{0}+ 5x_{1}+ 2x_{2} \leq 5$ *(knapsack)*<br>$x_{3}+ x_{4}+ x_{5}+ x_{6}+ x_{7} \geq 5$ *(cardinality)* | — | 8/11 | 68/3 | 243/141 |
| 68 | 6 | quadratic_knapsack, quadratic_knapsack | $3x_{0}^2+ x_{0}x_{1}+ 3x_{0}x_{2}$<br>$+ x_{1}^2 + 4x_{1}x_{2}+ 3x_{2}^2 \leq 5$ *(quadratic_knapsack)*<br>$5x_{3}^2+ x_{3}x_{4}+ 2x_{3}x_{5}$<br>$+ 5x_{4}^2 + x_{4}x_{5}+ 3x_{5}^2 \leq 6$ *(quadratic_knapsack)* | — | 6/12 | 91/6 | 245/439 |
| 69 | 6 | flow, quadratic_knapsack | $x_{0}- x_{1} = 0$ *(flow)*<br>$4x_{2}^2+ 5x_{2}x_{3}+ 2x_{2}x_{4}$<br>$+ 5x_{2}x_{5} + 4x_{3}^2$<br>$+ 3x_{3}x_{4} + 5x_{3}x_{5}$<br>$+ 3x_{4}^2 + 2x_{4}x_{5}+ 3x_{5}^2 \leq 22$ *(quadratic_knapsack)* | — | 6/11 | 27/5 | 120/665 |
| 70 | 6 | knapsack, cardinality | $2x_{0}+ 5x_{1}+ 2x_{2} \leq 5$ *(knapsack)*<br>$x_{3}+ x_{4}+ x_{5} \geq 0$ *(cardinality)* | — | 6/11 | 63/5 | 189/122 |
| 71 | 7 | cardinality, cardinality | $x_{0}+ x_{1}+ x_{2} \geq 2$ *(cardinality)*<br>$x_{3}+ x_{4}+ x_{5}+ x_{6} \leq 3$ *(cardinality)* | — | 7/10 | 23/3 | 131/119 |
| 72 | 7 | quadratic_knapsack, cardinality | $x_{0}^2+ 2x_{0}x_{1}+ x_{0}x_{2}$<br>$+ x_{1}^2 + 5x_{1}x_{2}+ x_{2}^2 \leq 6$ *(quadratic_knapsack)*<br>$x_{3}+ x_{4}+ x_{5}+ x_{6} \leq 1$ *(cardinality)* | — | 7/11 | 55/4 | 198/282 |
| 73 | 7 | knapsack, cardinality | $7x_{0}+ 10x_{1}+ 9x_{2} \leq 8$ *(knapsack)*<br>$x_{3}+ x_{4}+ x_{5}+ x_{6} \leq 2$ *(cardinality)* | — | 7/13 | 30/6 | 142/172 |
| 74 | 8 | flow, cardinality, independent_set | $x_{0}+ x_{1}- x_{2} = 0$ *(flow)*<br>$x_{3}+ x_{4}+ x_{5} \leq 3$ *(cardinality)*<br>$x_{6}x_{7} = 0$ *(independent_set)* | — | 8/10 | 19/2 | 145/98 |
| 75 | 8 | cardinality, knapsack | $x_{0}+ x_{1}+ x_{2} = 1$ *(cardinality)*<br>$6x_{3}+ 7x_{4}+ 8x_{5}+ 9x_{6}$<br>$+ 5x_{7} \leq 14$ *(knapsack)* | — | 8/12 | 508/4 | 1126/187 |
| 76 | 8 | quadratic_knapsack, knapsack | $2x_{0}^2+ 5x_{0}x_{1}+ 4x_{0}x_{2}$<br>$+ 4x_{0}x_{3} + 3x_{0}x_{4}$<br>$+ 4x_{1}^2 + 4x_{1}x_{2}$<br>$+ x_{1}x_{3} + 2x_{1}x_{4}$<br>$+ 3x_{2}^2 + x_{2}x_{3}$<br>$+ 4x_{2}x_{4} + 3x_{3}^2$<br>$+ 4x_{3}x_{4} + 3x_{4}^2 \leq 16$ *(quadratic_knapsack)*<br>$3x_{5}+ 8x_{6}+ 3x_{7} \leq 5$ *(knapsack)* | — | 8/16 | 1180/8 | 2470/1387 |
| 77 | 7 | knapsack, cardinality | $x_{0}+ 6x_{1}+ 9x_{2}+ 8x_{3}$<br>$+ 9x_{4} \leq 18$ *(knapsack)*<br>$x_{5}+ x_{6} \geq 2$ *(cardinality)* | — | 7/12 | 141/5 | 366/199 |
| 78 | 7 | quadratic_knapsack, cardinality | $2x_{0}^2+ 4x_{0}x_{1}+ 5x_{0}x_{2}$<br>$+ x_{0}x_{3} + 4x_{0}x_{4}$<br>$+ 2x_{1}^2 + 4x_{1}x_{2}$<br>$+ 4x_{1}x_{3} + x_{1}x_{4}$<br>$+ 2x_{2}^2 + 3x_{2}x_{3}$<br>$+ 3x_{2}x_{4} + 3x_{3}^2$<br>$+ 2x_{3}x_{4} + x_{4}^2 \leq 19$ *(quadratic_knapsack)*<br>$x_{5}+ x_{6} = 2$ *(cardinality)* | — | 7/12 | 34/5 | 156/1326 |
| 79 | 7 | cardinality, cardinality | $x_{0}+ x_{1}+ x_{2}+ x_{3} \leq 4$ *(cardinality)*<br>$x_{4}+ x_{5}+ x_{6} \leq 3$ *(cardinality)* | — | 7/12 | 37/5 | 159/155 |
| 80 | 6 | knapsack, quadratic_knapsack | $3x_{0}+ 8x_{1}+ 3x_{2} \leq 5$ *(knapsack)*<br>$5x_{3}^2+ x_{3}x_{4}+ 2x_{3}x_{5}$<br>$+ 5x_{4}^2 + x_{4}x_{5}+ 3x_{5}^2 \leq 6$ *(quadratic_knapsack)* | — | 6/12 | 111/6 | 285/292 |
| 81 | 7 | assignment, cardinality | $x_{0}+ x_{1} = 1$ *(assignment)*<br>$x_{2}+ x_{3}+ x_{4}+ x_{5}+ x_{6} = 3$ *(cardinality)* | — | 7/7 | 34/0 | 156/81 |
| 82 | 8 | cardinality, knapsack | $x_{0}+ x_{1}+ x_{2} = 0$ *(cardinality)*<br>$9x_{3}+ 2x_{4}+ x_{5}+ 9x_{6}$<br>$+ 6x_{7} \leq 13$ *(knapsack)* | — | 8/12 | 44/4 | 204/186 |
| 83 | 7 | quadratic_knapsack, quadratic_knapsack | $5x_{0}^2+ x_{0}x_{1}+ 2x_{0}x_{2}$<br>$+ 5x_{1}^2 + x_{1}x_{2}+ 3x_{2}^2 \leq 6$ *(quadratic_knapsack)*<br>$5x_{3}^2+ 3x_{3}x_{4}+ x_{3}x_{5}$<br>$+ 5x_{3}x_{6} + 3x_{4}^2$<br>$+ x_{4}x_{5} + 4x_{4}x_{6}$<br>$+ 5x_{5}^2 + 5x_{5}x_{6}+ x_{6}^2 \leq 20$ *(quadratic_knapsack)* | — | 7/15 | 383/8 | 848/873 |
| 84 | 6 | quadratic_knapsack, independent_set | $4x_{0}^2+ 4x_{0}x_{1}+ x_{0}x_{2}$<br>$+ 5x_{0}x_{3} + 5x_{1}^2$<br>$+ 5x_{1}x_{2} + 5x_{1}x_{3}$<br>$+ x_{2}^2 + 4x_{2}x_{3}+ x_{3}^2 \leq 11$ *(quadratic_knapsack)*<br>$x_{4}x_{5} = 0$ *(independent_set)* | — | 6/10 | 221/4 | 496/590 |
| 85 | 7 | quadratic_knapsack, knapsack | $5x_{0}^2+ 3x_{0}x_{1}+ 5x_{0}x_{2}$<br>$+ 4x_{1}^2 + 5x_{1}x_{2}+ 3x_{2}^2 \leq 14$ *(quadratic_knapsack)*<br>$4x_{3}+ 6x_{4}+ x_{5}+ x_{6} \leq 6$ *(knapsack)* | — | 7/14 | 281/7 | 644/361 |
| 86 | 7 | cardinality, cardinality | $x_{0}+ x_{1}+ x_{2} = 1$ *(cardinality)*<br>$x_{3}+ x_{4}+ x_{5}+ x_{6} \leq 4$ *(cardinality)* | — | 7/10 | 29/3 | 143/128 |
| 87 | 8 | cardinality, quadratic_knapsack | $x_{0}+ x_{1}+ x_{2}+ x_{3}+ x_{4} = 3$ *(cardinality)*<br>$x_{5}^2+ 5x_{5}x_{6}+ 4x_{5}x_{7}$<br>$+ 2x_{6}^2 + 2x_{6}x_{7}+ 2x_{7}^2 \leq 8$ *(quadratic_knapsack)* | — | 8/12 | 43/4 | 193/330 |
| 88 | 7 | quadratic_knapsack, cardinality | $3x_{0}^2+ x_{0}x_{1}+ 3x_{0}x_{2}$<br>$+ 5x_{1}^2 + 2x_{1}x_{2}+ 2x_{2}^2 \leq 8$ *(quadratic_knapsack)*<br>$x_{3}+ x_{4}+ x_{5}+ x_{6} = 3$ *(cardinality)* | — | 7/11 | 46/4 | 173/306 |
| 89 | 8 | quadratic_knapsack, knapsack | $x_{0}^2+ 4x_{0}x_{1}+ x_{0}x_{2}$<br>$+ 5x_{0}x_{3} + 2x_{1}^2$<br>$+ 3x_{1}x_{2} + 3x_{1}x_{3}$<br>$+ 5x_{2}^2 + 4x_{2}x_{3}+ 5x_{3}^2 \leq 22$ *(quadratic_knapsack)*<br>$7x_{4}+ 7x_{5}+ 10x_{6}+ 2x_{7} \leq 10$ *(knapsack)* | — | 8/17 | 292/9 | 694/780 |
| 90 | 8 | cardinality, quadratic_knapsack | $x_{0}+ x_{1}+ x_{2}+ x_{3} \leq 1$ *(cardinality)*<br>$3x_{4}^2+ x_{4}x_{5}+ 3x_{4}x_{6}$<br>$+ 4x_{4}x_{7} + 5x_{5}^2$<br>$+ 4x_{5}x_{6} + 2x_{5}x_{7}$<br>$+ 2x_{6}^2 + 4x_{6}x_{7}+ 4x_{7}^2 \leq 17$ *(quadratic_knapsack)* | — | 8/14 | 31/6 | 168/715 |
| 91 | 6 | cardinality, cardinality | $x_{0}+ x_{1}+ x_{2}+ x_{3} \geq 4$ *(cardinality)*<br>$x_{4}+ x_{5} \leq 2$ *(cardinality)* | — | 6/8 | 10/2 | 86/80 |
| 92 | 8 | cardinality, assignment, cardinality | $x_{0}+ x_{1}+ x_{2}+ x_{3} = 2$ *(cardinality)*<br>$x_{4}+ x_{5} = 1$ *(assignment)*<br>$x_{6}+ x_{7} \geq 0$ *(cardinality)* | — | 8/10 | 21/2 | 154/99 |
| 93 | 7 | cardinality, quadratic_knapsack | $x_{0}+ x_{1}+ x_{2} \geq 2$ *(cardinality)*<br>$x_{3}^2+ 4x_{3}x_{4}+ x_{3}x_{5}$<br>$+ 2x_{3}x_{6} + 5x_{4}^2$<br>$+ 4x_{4}x_{5} + 3x_{4}x_{6}$<br>$+ 5x_{5}^2 + 4x_{5}x_{6}+ 2x_{6}^2 \leq 18$ *(quadratic_knapsack)* | — | 7/13 | 375/6 | 838/695 |
| 94 | 8 | quadratic_knapsack, knapsack | $4x_{0}^2+ x_{0}x_{1}+ 4x_{0}x_{2}$<br>$+ 3x_{0}x_{3} + 4x_{1}^2$<br>$+ 5x_{1}x_{2} + 4x_{1}x_{3}$<br>$+ 3x_{2}^2 + 5x_{2}x_{3}+ 5x_{3}^2 \leq 24$ *(quadratic_knapsack)*<br>$4x_{4}+ 6x_{5}+ x_{6}+ x_{7} \leq 6$ *(knapsack)* | — | 8/16 | 441/8 | 992/756 |
| 95 | 8 | cardinality, knapsack | $x_{0}+ x_{1}+ x_{2}+ x_{3} = 1$ *(cardinality)*<br>$8x_{4}+ 10x_{5}+ x_{6}+ 2x_{7} \leq 12$ *(knapsack)* | — | 8/12 | 64/4 | 241/173 |
| 96 | 7 | cardinality, cardinality | $x_{0}+ x_{1} = 0$ *(cardinality)*<br>$x_{2}+ x_{3}+ x_{4}+ x_{5}+ x_{6} \geq 4$ *(cardinality)* | — | 7/8 | 0/1 | 85/96 |
| 97 | 6 | knapsack, cardinality | $6x_{0}+ 2x_{1}+ 2x_{2} \leq 3$ *(knapsack)*<br>$x_{3}+ x_{4}+ x_{5} \leq 1$ *(cardinality)* | — | 6/9 | 31/3 | 131/96 |
| 98 | 8 | knapsack, cardinality | $6x_{0}+ 7x_{1}+ 8x_{2}+ 9x_{3}$<br>$+ 5x_{4} \leq 14$ *(knapsack)*<br>$x_{5}+ x_{6}+ x_{7} \geq 2$ *(cardinality)* | — | 8/13 | 503/5 | 1118/199 |
| 99 | 5 | cardinality, cardinality | $x_{0}+ x_{1}+ x_{2} = 0$ *(cardinality)*<br>$x_{3}+ x_{4} \geq 1$ *(cardinality)* | — | 5/6 | 0/1 | 47/47 |
| 100 | 7 | quadratic_knapsack, quadratic_knapsack | $x_{0}^2+ x_{0}x_{1}+ 3x_{0}x_{2}$<br>$+ 2x_{0}x_{3} + x_{1}^2$<br>$+ 2x_{1}x_{2} + 3x_{1}x_{3}$<br>$+ 2x_{2}^2 + 3x_{2}x_{3}+ 3x_{3}^2 \leq 7$ *(quadratic_knapsack)*<br>$x_{4}^2+ 5x_{4}x_{5}+ x_{4}x_{6}$<br>$+ 3x_{5}^2 + 3x_{5}x_{6}+ 5x_{6}^2 \leq 10$ *(quadratic_knapsack)* | — | 7/14 | 139/7 | 360/780 |
| 101 | 8 | quadratic_knapsack, quadratic_knapsack | $x_{0}^2+ x_{0}x_{1}+ 3x_{0}x_{2}$<br>$+ 2x_{0}x_{3} + x_{1}^2$<br>$+ 2x_{1}x_{2} + 3x_{1}x_{3}$<br>$+ 2x_{2}^2 + 3x_{2}x_{3}+ 3x_{3}^2 \leq 7$ *(quadratic_knapsack)*<br>$4x_{4}^2+ 5x_{4}x_{5}+ 2x_{4}x_{6}$<br>$+ 5x_{4}x_{7} + 4x_{5}^2$<br>$+ 3x_{5}x_{6} + 5x_{5}x_{7}$<br>$+ 3x_{6}^2 + 2x_{6}x_{7}+ 3x_{7}^2 \leq 22$ *(quadratic_knapsack)* | — | 8/16 | 153/8 | 416/1175 |
| 102 | 8 | quadratic_knapsack, cardinality | $4x_{0}^2+ 3x_{0}x_{1}+ 3x_{0}x_{2}$<br>$+ 2x_{0}x_{3} + 2x_{0}x_{4}$<br>$+ 2x_{1}^2 + 3x_{1}x_{2}$<br>$+ 2x_{1}x_{3} + 3x_{1}x_{4}$<br>$+ 3x_{2}^2 + 5x_{2}x_{3}$<br>$+ 4x_{2}x_{4} + 3x_{3}^2$<br>$+ 5x_{3}x_{4} + x_{4}^2 \leq 22$ *(quadratic_knapsack)*<br>$x_{5}+ x_{6}+ x_{7} \geq 3$ *(cardinality)* | — | 8/13 | 35/5 | 180/1342 |
| 103 | 6 | knapsack, cardinality | $2x_{0}+ 5x_{1}+ 2x_{2} \leq 5$ *(knapsack)*<br>$x_{3}+ x_{4}+ x_{5} = 2$ *(cardinality)* | — | 6/9 | 75/3 | 209/98 |
| 104 | 8 | quadratic_knapsack, quadratic_knapsack | $x_{0}^2+ 5x_{0}x_{1}+ 4x_{0}x_{2}$<br>$+ 2x_{1}^2 + 2x_{1}x_{2}+ 2x_{2}^2 \leq 8$ *(quadratic_knapsack)*<br>$4x_{3}^2+ x_{3}x_{4}+ 2x_{3}x_{5}$<br>$+ 4x_{3}x_{6} + x_{3}x_{7}$<br>$+ 2x_{4}^2 + 3x_{4}x_{5}$<br>$+ x_{4}x_{6} + x_{4}x_{7}+ 3x_{5}^2$<br>$+ 3x_{5}x_{6} + 4x_{5}x_{7}$<br>$+ x_{6}^2 + 3x_{6}x_{7}+ 2x_{7}^2 \leq 11$ *(quadratic_knapsack)* | — | 8/16 | 191/8 | 489/1475 |
| 105 | 8 | cardinality, cardinality, assignment | $x_{0}+ x_{1}+ x_{2} \leq 2$ *(cardinality)*<br>$x_{3}+ x_{4} \leq 0$ *(cardinality)*<br>$x_{5}+ x_{6}+ x_{7} = 1$ *(assignment)* | — | 8/10 | 17/2 | 141/103 |
| 106 | 6 | quadratic_knapsack, cardinality | $x_{0}^2+ 2x_{0}x_{1}+ x_{0}x_{2}$<br>$+ x_{1}^2 + 5x_{1}x_{2}+ x_{2}^2 \leq 6$ *(quadratic_knapsack)*<br>$x_{3}+ x_{4}+ x_{5} \geq 3$ *(cardinality)* | — | 6/9 | 46/3 | 155/247 |
| 107 | 8 | knapsack, flow | $3x_{0}+ 8x_{1}+ 3x_{2} \leq 5$ *(knapsack)*<br>$x_{3}+ x_{4}+ x_{5}- x_{6}- x_{7} = 0$ *(flow)* | — | 8/11 | 121/3 | 355/138 |
| 108 | 8 | knapsack, flow | $2x_{0}+ 6x_{1}+ x_{2} \leq 3$ *(knapsack)*<br>$x_{3}+ x_{4}- x_{5}- x_{6}- x_{7} = 0$ *(flow)* | — | 8/10 | 25/2 | 162/119 |
| 109 | 7 | quadratic_knapsack, cardinality | $2x_{0}^2+ 5x_{0}x_{1}+ 4x_{0}x_{2}$<br>$+ 4x_{0}x_{3} + 3x_{0}x_{4}$<br>$+ 4x_{1}^2 + 4x_{1}x_{2}$<br>$+ x_{1}x_{3} + 2x_{1}x_{4}$<br>$+ 3x_{2}^2 + x_{2}x_{3}$<br>$+ 4x_{2}x_{4} + 3x_{3}^2$<br>$+ 4x_{3}x_{4} + 3x_{4}^2 \leq 16$ *(quadratic_knapsack)*<br>$x_{5}+ x_{6} \geq 2$ *(cardinality)* | — | 7/12 | 1079/5 | 2240/1324 |
| 110 | 5 | independent_set, quadratic_knapsack | $x_{0}x_{1} = 0$ *(independent_set)*<br>$5x_{2}^2+ x_{2}x_{3}+ 2x_{2}x_{4}$<br>$+ 5x_{3}^2 + x_{3}x_{4}+ 3x_{4}^2 \leq 6$ *(quadratic_knapsack)* | — | 5/8 | 8/3 | 66/230 |
| 111 | 5 | knapsack, assignment | $2x_{0}+ 6x_{1}+ x_{2} \leq 3$ *(knapsack)*<br>$x_{3}+ x_{4} = 1$ *(assignment)* | — | 5/7 | 10/2 | 70/69 |
| 112 | 6 | knapsack, assignment | $9x_{0}+ 9x_{1}+ 4x_{2}+ x_{3} \leq 9$ *(knapsack)*<br>$x_{4}+ x_{5} = 1$ *(assignment)* | — | 6/10 | 69/4 | 192/133 |
| 113 | 8 | cardinality, quadratic_knapsack, cardinality | $x_{0}+ x_{1} = 1$ *(cardinality)*<br>$3x_{2}^2+ x_{2}x_{3}+ 3x_{2}x_{4}$<br>$+ 5x_{3}^2 + 2x_{3}x_{4}+ 2x_{4}^2 \leq 8$ *(quadratic_knapsack)*<br>$x_{5}+ x_{6}+ x_{7} \geq 1$ *(cardinality)* | — | 8/14 | 26/6 | 162/340 |
| 114 | 6 | cardinality, knapsack | $x_{0}+ x_{1}+ x_{2} = 0$ *(cardinality)*<br>$10x_{3}+ 8x_{4}+ 10x_{5} \leq 9$ *(knapsack)* | — | 6/10 | 11/4 | 85/118 |
| 115 | 7 | knapsack, cardinality | $5x_{0}+ 4x_{1}+ x_{2} \leq 6$ *(knapsack)*<br>$x_{3}+ x_{4}+ x_{5}+ x_{6} = 0$ *(cardinality)* | — | 7/10 | 11/3 | 104/114 |
| 116 | 7 | cardinality, cardinality | $x_{0}+ x_{1} \leq 2$ *(cardinality)*<br>$x_{2}+ x_{3}+ x_{4}+ x_{5}+ x_{6} = 5$ *(cardinality)* | — | 7/9 | 11/2 | 106/100 |
| 117 | 8 | cardinality, knapsack | $x_{0}+ x_{1}+ x_{2}+ x_{3} \geq 0$ *(cardinality)*<br>$8x_{4}+ 5x_{5}+ 7x_{6}+ 4x_{7} \leq 11$ *(knapsack)* | — | 8/15 | 57/7 | 224/219 |
| 118 | 8 | cardinality, cardinality | $x_{0}+ x_{1}+ x_{2}+ x_{3} = 0$ *(cardinality)*<br>$x_{4}+ x_{5}+ x_{6}+ x_{7} = 1$ *(cardinality)* | — | 8/8 | 7/0 | 127/91 |
| 119 | 8 | knapsack, knapsack | $3x_{0}+ 8x_{1}+ 3x_{2} \leq 5$ *(knapsack)*<br>$x_{3}+ 10x_{4}+ 3x_{5}+ 5x_{6}$<br>$+ x_{7} \leq 12$ *(knapsack)* | — | 8/15 | 174/7 | 460/232 |
| 120 | 6 | cardinality, knapsack | $x_{0}+ x_{1}+ x_{2} = 0$ *(cardinality)*<br>$5x_{3}+ 4x_{4}+ x_{5} \leq 6$ *(knapsack)* | — | 6/9 | 11/3 | 81/95 |
| 121 | 8 | knapsack, cardinality | $2x_{0}+ 2x_{1}+ 9x_{2}+ 10x_{3}$<br>$+ 4x_{4} \leq 18$ *(knapsack)*<br>$x_{5}+ x_{6}+ x_{7} = 3$ *(cardinality)* | — | 8/13 | 38/5 | 181/214 |
| 122 | 7 | cardinality, cardinality | $x_{0}+ x_{1}+ x_{2}+ x_{3} = 4$ *(cardinality)*<br>$x_{4}+ x_{5}+ x_{6} \leq 3$ *(cardinality)* | — | 7/9 | 17/2 | 113/99 |
| 123 | 7 | cardinality, quadratic_knapsack | $x_{0}+ x_{1}+ x_{2} \geq 1$ *(cardinality)*<br>$4x_{3}^2+ 5x_{3}x_{4}+ 2x_{3}x_{5}$<br>$+ 5x_{3}x_{6} + 4x_{4}^2$<br>$+ 3x_{4}x_{5} + 5x_{4}x_{6}$<br>$+ 3x_{5}^2 + 2x_{5}x_{6}+ 3x_{6}^2 \leq 22$ *(quadratic_knapsack)* | — | 7/14 | 25/7 | 131/707 |
| 124 | 8 | cardinality, knapsack | $x_{0}+ x_{1}+ x_{2}+ x_{3} \geq 2$ *(cardinality)*<br>$7x_{4}+ 7x_{5}+ 10x_{6}+ 2x_{7} \leq 10$ *(knapsack)* | — | 8/14 | 269/6 | 651/206 |
| 125 | 8 | cardinality, cardinality, cardinality | $x_{0}+ x_{1}+ x_{2} \leq 1$ *(cardinality)*<br>$x_{3}+ x_{4}+ x_{5} \geq 0$ *(cardinality)*<br>$x_{6}+ x_{7} \leq 2$ *(cardinality)* | — | 8/13 | 14/5 | 133/131 |
| 126 | 8 | cardinality, knapsack | $x_{0}+ x_{1}+ x_{2}+ x_{3}+ x_{4} \geq 1$ *(cardinality)*<br>$2x_{5}+ 5x_{6}+ 2x_{7} \leq 5$ *(knapsack)* | — | 8/14 | 63/6 | 236/205 |
| 127 | 8 | quadratic_knapsack, knapsack | $5x_{0}^2+ 2x_{0}x_{1}+ 2x_{0}x_{2}$<br>$+ 3x_{0}x_{3} + x_{0}x_{4}+ x_{1}^2$<br>$+ 4x_{1}x_{2} + 4x_{1}x_{3}$<br>$+ 3x_{1}x_{4} + x_{2}^2$<br>$+ 3x_{2}x_{3} + x_{2}x_{4}$<br>$+ 5x_{3}^2 + x_{3}x_{4}+ 5x_{4}^2 \leq 23$ *(quadratic_knapsack)*<br>$7x_{5}+ 10x_{6}+ 9x_{7} \leq 8$ *(knapsack)* | — | 8/17 | 43/9 | 193/1407 |
| 128 | 8 | quadratic_knapsack, cardinality | $4x_{0}^2+ 4x_{0}x_{1}+ 2x_{0}x_{2}$<br>$+ 5x_{1}^2 + x_{1}x_{2}+ 2x_{2}^2 \leq 9$ *(quadratic_knapsack)*<br>$x_{3}+ x_{4}+ x_{5}+ x_{6}+ x_{7} \leq 5$ *(cardinality)* | — | 8/15 | 56/7 | 228/396 |
| 129 | 8 | cardinality, independent_set, quadratic_knapsack | $x_{0}+ x_{1}+ x_{2} = 3$ *(cardinality)*<br>$x_{3}x_{4} = 0$ *(independent_set)*<br>$2x_{5}^2+ 5x_{5}x_{6}+ 2x_{5}x_{7}$<br>$+ 2x_{6}^2 + x_{6}x_{7}+ x_{7}^2 \leq 7$ *(quadratic_knapsack)* | — | 8/11 | 54/3 | 218/267 |
| 130 | 5 | assignment, cardinality | $x_{0}+ x_{1} = 1$ *(assignment)*<br>$x_{2}+ x_{3}+ x_{4} \leq 2$ *(cardinality)* | — | 5/7 | 15/2 | 68/65 |
| 131 | 7 | quadratic_knapsack, cardinality | $x_{0}^2+ 5x_{0}x_{1}+ 4x_{0}x_{2}$<br>$+ 2x_{1}^2 + 2x_{1}x_{2}+ 2x_{2}^2 \leq 8$ *(quadratic_knapsack)*<br>$x_{3}+ x_{4}+ x_{5}+ x_{6} \leq 1$ *(cardinality)* | — | 7/12 | 24/5 | 136/324 |
| 132 | 6 | cardinality, quadratic_knapsack | $x_{0}+ x_{1} \geq 0$ *(cardinality)*<br>$5x_{2}^2+ x_{2}x_{3}+ 3x_{2}x_{4}$<br>$+ x_{2}x_{5} + 4x_{3}^2+ x_{3}x_{4}$<br>$+ 5x_{3}x_{5} + 4x_{4}^2$<br>$+ 2x_{4}x_{5} + 3x_{5}^2 \leq 19$ *(quadratic_knapsack)* | — | 6/13 | 23/7 | 105/681 |
| 133 | 7 | cardinality, quadratic_knapsack | $x_{0}+ x_{1}+ x_{2} \leq 3$ *(cardinality)*<br>$5x_{3}^2+ x_{3}x_{4}+ 3x_{3}x_{5}$<br>$+ x_{3}x_{6} + 4x_{4}^2+ x_{4}x_{5}$<br>$+ 5x_{4}x_{6} + 4x_{5}^2$<br>$+ 2x_{5}x_{6} + 3x_{6}^2 \leq 19$ *(quadratic_knapsack)* | — | 7/14 | 36/7 | 160/710 |
| 134 | 7 | cardinality, quadratic_knapsack | $x_{0}+ x_{1} \geq 2$ *(cardinality)*<br>$5x_{2}^2+ 2x_{2}x_{3}+ 2x_{2}x_{4}$<br>$+ 3x_{2}x_{5} + x_{2}x_{6}+ x_{3}^2$<br>$+ 4x_{3}x_{4} + 4x_{3}x_{5}$<br>$+ 3x_{3}x_{6} + x_{4}^2$<br>$+ 3x_{4}x_{5} + x_{4}x_{6}$<br>$+ 5x_{5}^2 + x_{5}x_{6}+ 5x_{6}^2 \leq 23$ *(quadratic_knapsack)* | — | 7/12 | 34/5 | 147/1323 |
| 135 | 7 | quadratic_knapsack, flow | $5x_{0}^2+ x_{0}x_{1}+ 3x_{0}x_{2}$<br>$+ x_{0}x_{3} + 4x_{1}^2+ x_{1}x_{2}$<br>$+ 5x_{1}x_{3} + 4x_{2}^2$<br>$+ 2x_{2}x_{3} + 3x_{3}^2 \leq 19$ *(quadratic_knapsack)*<br>$x_{4}- x_{5}- x_{6} = 0$ *(flow)* | — | 7/12 | 29/5 | 146/680 |
| 136 | 7 | cardinality, knapsack | $x_{0}+ x_{1}+ x_{2} \leq 3$ *(cardinality)*<br>$x_{3}+ 3x_{4}+ 9x_{5}+ 8x_{6} \leq 10$ *(knapsack)* | — | 7/13 | 113/6 | 308/178 |
| 137 | 6 | cardinality, quadratic_knapsack | $x_{0}+ x_{1}+ x_{2} \leq 1$ *(cardinality)*<br>$x_{3}^2+ 5x_{3}x_{4}+ 4x_{3}x_{5}$<br>$+ 2x_{4}^2 + 2x_{4}x_{5}+ 2x_{5}^2 \leq 8$ *(quadratic_knapsack)* | — | 6/11 | 20/5 | 106/302 |
| 138 | 7 | cardinality, cardinality | $x_{0}+ x_{1} = 1$ *(cardinality)*<br>$x_{2}+ x_{3}+ x_{4}+ x_{5}+ x_{6} \leq 3$ *(cardinality)* | — | 7/9 | 35/2 | 151/117 |
| 139 | 7 | cardinality, quadratic_knapsack | $x_{0}+ x_{1}+ x_{2} \leq 3$ *(cardinality)*<br>$3x_{3}^2+ x_{3}x_{4}+ 3x_{3}x_{5}$<br>$+ 4x_{3}x_{6} + 5x_{4}^2$<br>$+ 4x_{4}x_{5} + 2x_{4}x_{6}$<br>$+ 2x_{5}^2 + 4x_{5}x_{6}+ 4x_{6}^2 \leq 17$ *(quadratic_knapsack)* | — | 7/14 | 32/7 | 149/709 |
| 140 | 6 | quadratic_knapsack, cardinality | $x_{0}^2+ 4x_{0}x_{1}+ x_{0}x_{2}$<br>$+ 5x_{0}x_{3} + 2x_{1}^2$<br>$+ 3x_{1}x_{2} + 3x_{1}x_{3}$<br>$+ 5x_{2}^2 + 4x_{2}x_{3}+ 5x_{3}^2 \leq 22$ *(quadratic_knapsack)*<br>$x_{4}+ x_{5} \geq 2$ *(cardinality)* | — | 6/11 | 25/5 | 113/666 |
| 141 | 7 | knapsack, knapsack | $2x_{0}+ 10x_{1}+ 4x_{2} \leq 6$ *(knapsack)*<br>$9x_{3}+ 9x_{4}+ 4x_{5}+ x_{6} \leq 9$ *(knapsack)* | — | 7/14 | 73/7 | 227/195 |
| 142 | 7 | quadratic_knapsack, quadratic_knapsack | $5x_{0}^2+ 3x_{0}x_{1}+ 5x_{0}x_{2}$<br>$+ 4x_{1}^2 + 5x_{1}x_{2}+ 3x_{2}^2 \leq 14$ *(quadratic_knapsack)*<br>$5x_{3}^2+ x_{3}x_{4}+ 3x_{3}x_{5}$<br>$+ x_{3}x_{6} + 4x_{4}^2+ x_{4}x_{5}$<br>$+ 5x_{4}x_{6} + 4x_{5}^2$<br>$+ 2x_{5}x_{6} + 3x_{6}^2 \leq 19$ *(quadratic_knapsack)* | — | 7/16 | 35/9 | 149/914 |
| 143 | 5 | cardinality, knapsack | $x_{0}+ x_{1} \geq 1$ *(cardinality)*<br>$10x_{2}+ x_{3}+ 9x_{4} \leq 7$ *(knapsack)* | — | 5/9 | 11/4 | 71/95 |
| 144 | 6 | knapsack, knapsack | $2x_{0}+ 7x_{1}+ 7x_{2} \leq 7$ *(knapsack)*<br>$2x_{3}+ 10x_{4}+ 4x_{5} \leq 6$ *(knapsack)* | — | 6/12 | 15/6 | 91/143 |
| 145 | 6 | cardinality, knapsack | $x_{0}+ x_{1}+ x_{2} = 2$ *(cardinality)*<br>$2x_{3}+ 7x_{4}+ 7x_{5} \leq 7$ *(knapsack)* | — | 6/9 | 20/3 | 106/101 |
| 146 | 7 | assignment, cardinality | $x_{0}+ x_{1} = 1$ *(assignment)*<br>$x_{2}+ x_{3}+ x_{4}+ x_{5}+ x_{6} = 0$ *(cardinality)* | — | 7/7 | 3/0 | 94/76 |
| 147 | 6 | cardinality, quadratic_knapsack | $x_{0}+ x_{1}+ x_{2} \geq 2$ *(cardinality)*<br>$x_{3}^2+ 5x_{3}x_{4}+ 4x_{3}x_{5}$<br>$+ 2x_{4}^2 + 2x_{4}x_{5}+ 2x_{5}^2 \leq 8$ *(quadratic_knapsack)* | — | 6/11 | 12/5 | 87/301 |
| 148 | 8 | quadratic_knapsack, cardinality | $2x_{0}^2+ 4x_{0}x_{1}+ 4x_{0}x_{2}$<br>$+ 4x_{0}x_{3} + 5x_{0}x_{4}$<br>$+ x_{1}^2 + 2x_{1}x_{2}$<br>$+ 2x_{1}x_{3} + 5x_{1}x_{4}$<br>$+ x_{2}^2 + 3x_{2}x_{3}+ x_{2}x_{4}$<br>$+ 5x_{3}^2 + 4x_{3}x_{4}+ x_{4}^2 \leq 28$ *(quadratic_knapsack)*<br>$x_{5}+ x_{6}+ x_{7} \leq 0$ *(cardinality)* | — | 8/13 | 809/5 | 1728/1339 |
| 149 | 7 | quadratic_knapsack, knapsack | $x_{0}^2+ 4x_{0}x_{1}+ x_{0}x_{2}$<br>$+ 2x_{0}x_{3} + 5x_{1}^2$<br>$+ 4x_{1}x_{2} + 3x_{1}x_{3}$<br>$+ 5x_{2}^2 + 4x_{2}x_{3}+ 2x_{3}^2 \leq 18$ *(quadratic_knapsack)*<br>$6x_{4}+ 2x_{5}+ 2x_{6} \leq 3$ *(knapsack)* | — | 7/14 | 398/7 | 875/707 |
| 150 | 7 | knapsack, quadratic_knapsack | $4x_{0}+ 2x_{1}+ 10x_{2}+ 8x_{3} \leq 13$ *(knapsack)*<br>$x_{4}^2+ 5x_{4}x_{5}+ 4x_{4}x_{6}$<br>$+ 2x_{5}^2 + 2x_{5}x_{6}+ 2x_{6}^2 \leq 8$ *(quadratic_knapsack)* | — | 7/15 | 69/8 | 223/386 |
| 151 | 7 | cardinality, quadratic_knapsack | $x_{0}+ x_{1}+ x_{2} \geq 1$ *(cardinality)*<br>$3x_{3}^2+ x_{3}x_{4}+ 3x_{3}x_{5}$<br>$+ 4x_{3}x_{6} + 5x_{4}^2$<br>$+ 4x_{4}x_{5} + 2x_{4}x_{6}$<br>$+ 2x_{5}^2 + 4x_{5}x_{6}+ 4x_{6}^2 \leq 17$ *(quadratic_knapsack)* | — | 7/14 | 19/7 | 122/708 |
| 152 | 8 | quadratic_knapsack, knapsack | $x_{0}^2+ 4x_{0}x_{1}+ x_{0}x_{2}$<br>$+ 5x_{0}x_{3} + 2x_{1}^2$<br>$+ 3x_{1}x_{2} + 3x_{1}x_{3}$<br>$+ 5x_{2}^2 + 4x_{2}x_{3}+ 5x_{3}^2 \leq 22$ *(quadratic_knapsack)*<br>$8x_{4}+ 5x_{5}+ 7x_{6}+ 4x_{7} \leq 11$ *(knapsack)* | — | 8/17 | 80/9 | 276/782 |
| 153 | 7 | quadratic_knapsack, cardinality | $5x_{0}^2+ 3x_{0}x_{1}+ 4x_{0}x_{2}$<br>$+ 5x_{0}x_{3} + x_{0}x_{4}+ x_{1}^2$<br>$+ 4x_{1}x_{2} + 2x_{1}x_{3}$<br>$+ 3x_{1}x_{4} + 4x_{2}^2$<br>$+ 5x_{2}x_{3} + x_{2}x_{4}+ x_{3}^2$<br>$+ 5x_{3}x_{4} + 2x_{4}^2 \leq 17$ *(quadratic_knapsack)*<br>$x_{5}+ x_{6} \geq 0$ *(cardinality)* | — | 7/14 | 145/7 | 373/1340 |
| 154 | 6 | quadratic_knapsack, cardinality | $x_{0}^2+ 5x_{0}x_{1}+ x_{0}x_{2}$<br>$+ 3x_{1}^2 + 3x_{1}x_{2}+ 5x_{2}^2 \leq 10$ *(quadratic_knapsack)*<br>$x_{3}+ x_{4}+ x_{5} = 1$ *(cardinality)* | — | 6/10 | 16/4 | 95/289 |
| 155 | 8 | cardinality, knapsack | $x_{0}+ x_{1}+ x_{2}+ x_{3}+ x_{4} \leq 2$ *(cardinality)*<br>$2x_{5}+ 7x_{6}+ 7x_{7} \leq 7$ *(knapsack)* | — | 8/13 | 33/5 | 179/182 |
| 156 | 6 | cardinality, cardinality | $x_{0}+ x_{1} = 0$ *(cardinality)*<br>$x_{2}+ x_{3}+ x_{4}+ x_{5} \leq 0$ *(cardinality)* | — | 6/6 | 0/0 | 63/52 |
| 157 | 7 | quadratic_knapsack, cardinality | $5x_{0}^2+ 3x_{0}x_{1}+ x_{0}x_{2}$<br>$+ 5x_{0}x_{3} + 3x_{1}^2$<br>$+ x_{1}x_{2} + 4x_{1}x_{3}$<br>$+ 5x_{2}^2 + 5x_{2}x_{3}+ x_{3}^2 \leq 20$ *(quadratic_knapsack)*<br>$x_{4}+ x_{5}+ x_{6} \geq 0$ *(cardinality)* | — | 7/14 | 375/7 | 832/703 |
| 158 | 8 | knapsack, knapsack | $9x_{0}+ 9x_{1}+ 4x_{2}+ x_{3} \leq 9$ *(knapsack)*<br>$3x_{4}+ x_{5}+ 4x_{6}+ 8x_{7} \leq 7$ *(knapsack)* | — | 8/15 | 123/7 | 356/226 |
| 159 | 5 | knapsack, cardinality | $2x_{0}+ 7x_{1}+ 7x_{2} \leq 7$ *(knapsack)*<br>$x_{3}+ x_{4} = 2$ *(cardinality)* | — | 5/8 | 10/3 | 58/83 |
| 160 | 7 | assignment, cardinality | $x_{0}+ x_{1} = 1$ *(assignment)*<br>$x_{2}+ x_{3}+ x_{4}+ x_{5}+ x_{6} = 5$ *(cardinality)* | — | 7/7 | 8/0 | 99/78 |
| 161 | 7 | cardinality, knapsack | $x_{0}+ x_{1}+ x_{2}+ x_{3} \geq 0$ *(cardinality)*<br>$2x_{4}+ 7x_{5}+ 7x_{6} \leq 7$ *(knapsack)* | — | 7/13 | 8/6 | 104/167 |
| 162 | 8 | knapsack, knapsack | $8x_{0}+ 5x_{1}+ 7x_{2}+ 4x_{3} \leq 11$ *(knapsack)*<br>$4x_{4}+ 6x_{5}+ 2x_{6}+ 10x_{7} \leq 7$ *(knapsack)* | — | 8/15 | 114/7 | 344/228 |
| 163 | 7 | cardinality, cardinality | $x_{0}+ x_{1}+ x_{2}+ x_{3} \leq 4$ *(cardinality)*<br>$x_{4}+ x_{5}+ x_{6} \leq 0$ *(cardinality)* | — | 7/10 | 24/3 | 131/123 |
| 164 | 8 | cardinality, knapsack, cardinality | $x_{0}+ x_{1}+ x_{2} \geq 2$ *(cardinality)*<br>$5x_{3}+ 4x_{4}+ x_{5} \leq 6$ *(knapsack)*<br>$x_{6}+ x_{7} \geq 2$ *(cardinality)* | — | 8/12 | 13/4 | 132/134 |
| 165 | 8 | cardinality, knapsack | $x_{0}+ x_{1}+ x_{2}+ x_{3}+ x_{4} = 3$ *(cardinality)*<br>$3x_{5}+ 8x_{6}+ 3x_{7} \leq 5$ *(knapsack)* | — | 8/11 | 134/3 | 374/140 |
| 166 | 7 | cardinality, cardinality | $x_{0}+ x_{1} \geq 1$ *(cardinality)*<br>$x_{2}+ x_{3}+ x_{4}+ x_{5}+ x_{6} \leq 5$ *(cardinality)* | — | 7/11 | 37/4 | 156/151 |
| 167 | 7 | knapsack, flow | $2x_{0}+ 10x_{1}+ 4x_{2} \leq 6$ *(knapsack)*<br>$x_{3}- x_{4}- x_{5}- x_{6} = 0$ *(flow)* | — | 7/10 | 16/3 | 120/116 |
| 168 | 6 | knapsack, cardinality | $2x_{0}+ 10x_{1}+ 4x_{2} \leq 6$ *(knapsack)*<br>$x_{3}+ x_{4}+ x_{5} \leq 0$ *(cardinality)* | — | 6/9 | 7/3 | 73/95 |
| 169 | 6 | cardinality, cardinality | $x_{0}+ x_{1} \leq 2$ *(cardinality)*<br>$x_{2}+ x_{3}+ x_{4}+ x_{5} \geq 0$ *(cardinality)* | — | 6/11 | 6/5 | 75/126 |
| 170 | 8 | cardinality, cardinality | $x_{0}+ x_{1}+ x_{2}+ x_{3}+ x_{4} \leq 0$ *(cardinality)*<br>$x_{5}+ x_{6}+ x_{7} = 1$ *(cardinality)* | — | 8/8 | 5/0 | 117/91 |
| 171 | 8 | flow, cardinality | $x_{0}+ x_{1}- x_{2} = 0$ *(flow)*<br>$x_{3}+ x_{4}+ x_{5}+ x_{6}+ x_{7} \leq 2$ *(cardinality)* | — | 8/10 | 31/2 | 174/133 |
| 172 | 6 | quadratic_knapsack, quadratic_knapsack | $x_{0}^2+ 5x_{0}x_{1}+ 3x_{0}x_{2}$<br>$+ 4x_{1}^2 + 2x_{1}x_{2}+ 4x_{2}^2 \leq 7$ *(quadratic_knapsack)*<br>$5x_{3}^2+ x_{3}x_{4}+ 2x_{3}x_{5}$<br>$+ 5x_{4}^2 + x_{4}x_{5}+ 3x_{5}^2 \leq 6$ *(quadratic_knapsack)* | — | 6/12 | 16/6 | 101/441 |
| 173 | 8 | knapsack, quadratic_knapsack | $6x_{0}+ 7x_{1}+ 8x_{2}+ 9x_{3}$<br>$+ 5x_{4} \leq 14$ *(knapsack)*<br>$x_{5}^2+ 5x_{5}x_{6}+ 3x_{5}x_{7}$<br>$+ 4x_{6}^2 + 2x_{6}x_{7}+ 4x_{7}^2 \leq 7$ *(quadratic_knapsack)* | — | 8/15 | 511/7 | 1132/379 |
| 174 | 8 | cardinality, cardinality | $x_{0}+ x_{1}+ x_{2} = 0$ *(cardinality)*<br>$x_{3}+ x_{4}+ x_{5}+ x_{6}+ x_{7} = 1$ *(cardinality)* | — | 8/8 | 9/0 | 131/95 |
| 175 | 8 | quadratic_knapsack, cardinality | $x_{0}^2+ 5x_{0}x_{1}+ 4x_{0}x_{2}$<br>$+ 2x_{1}^2 + 2x_{1}x_{2}+ 2x_{2}^2 \leq 8$ *(quadratic_knapsack)*<br>$x_{3}+ x_{4}+ x_{5}+ x_{6}+ x_{7} \leq 1$ *(cardinality)* | — | 8/13 | 27/5 | 161/348 |
| 176 | 7 | cardinality, cardinality | $x_{0}+ x_{1}+ x_{2} \geq 3$ *(cardinality)*<br>$x_{3}+ x_{4}+ x_{5}+ x_{6} = 2$ *(cardinality)* | — | 7/7 | 21/0 | 130/75 |
| 177 | 6 | quadratic_knapsack, knapsack | $3x_{0}^2+ x_{0}x_{1}+ 3x_{0}x_{2}$<br>$+ 5x_{1}^2 + 2x_{1}x_{2}+ 2x_{2}^2 \leq 8$ *(quadratic_knapsack)*<br>$5x_{3}+ 4x_{4}+ x_{5} \leq 6$ *(knapsack)* | — | 6/13 | 34/7 | 134/335 |
| 178 | 6 | assignment, knapsack | $x_{0}+ x_{1}+ x_{2} = 1$ *(assignment)*<br>$5x_{3}+ 4x_{4}+ x_{5} \leq 6$ *(knapsack)* | — | 6/9 | 16/3 | 86/97 |
| 179 | 8 | knapsack, knapsack | $6x_{0}+ 7x_{1}+ 8x_{2}+ 9x_{3}$<br>$+ 5x_{4} \leq 14$ *(knapsack)*<br>$10x_{5}+ x_{6}+ 9x_{7} \leq 7$ *(knapsack)* | — | 8/15 | 514/7 | 1138/232 |
| 180 | 8 | quadratic_knapsack, quadratic_knapsack | $3x_{0}^2+ x_{0}x_{1}+ 3x_{0}x_{2}$<br>$+ x_{1}^2 + 4x_{1}x_{2}+ 3x_{2}^2 \leq 5$ *(quadratic_knapsack)*<br>$2x_{3}^2+ 4x_{3}x_{4}+ 4x_{3}x_{5}$<br>$+ 4x_{3}x_{6} + 5x_{3}x_{7}$<br>$+ x_{4}^2 + 2x_{4}x_{5}$<br>$+ 2x_{4}x_{6} + 5x_{4}x_{7}$<br>$+ x_{5}^2 + 3x_{5}x_{6}+ x_{5}x_{7}$<br>$+ 5x_{6}^2 + 4x_{6}x_{7}+ x_{7}^2 \leq 28$ *(quadratic_knapsack)* | — | 8/16 | 892/8 | 1891/1533 |
| 181 | 8 | cardinality, cardinality | $x_{0}+ x_{1}+ x_{2}+ x_{3}+ x_{4} = 0$ *(cardinality)*<br>$x_{5}+ x_{6}+ x_{7} = 1$ *(cardinality)* | — | 8/8 | 5/0 | 120/92 |
| 182 | 8 | quadratic_knapsack, knapsack | $5x_{0}^2+ 2x_{0}x_{1}+ 2x_{0}x_{2}$<br>$+ 3x_{0}x_{3} + x_{0}x_{4}+ x_{1}^2$<br>$+ 4x_{1}x_{2} + 4x_{1}x_{3}$<br>$+ 3x_{1}x_{4} + x_{2}^2$<br>$+ 3x_{2}x_{3} + x_{2}x_{4}$<br>$+ 5x_{3}^2 + x_{3}x_{4}+ 5x_{4}^2 \leq 23$ *(quadratic_knapsack)*<br>$2x_{5}+ 7x_{6}+ 7x_{7} \leq 7$ *(knapsack)* | — | 8/16 | 40/8 | 193/1388 |
| 183 | 8 | quadratic_knapsack, knapsack | $3x_{0}^2+ x_{0}x_{1}+ 5x_{0}x_{2}$<br>$+ 2x_{0}x_{3} + 5x_{0}x_{4}$<br>$+ 2x_{1}^2 + 5x_{1}x_{2}$<br>$+ 5x_{1}x_{3} + 5x_{1}x_{4}$<br>$+ 2x_{2}^2 + 3x_{2}x_{3}$<br>$+ 3x_{2}x_{4} + 4x_{3}^2$<br>$+ x_{3}x_{4} + 5x_{4}^2 \leq 18$ *(quadratic_knapsack)*<br>$2x_{5}+ 10x_{6}+ 4x_{7} \leq 6$ *(knapsack)* | — | 8/16 | 414/8 | 938/1387 |
| 184 | 7 | cardinality, cardinality | $x_{0}+ x_{1}+ x_{2}+ x_{3}+ x_{4} \geq 5$ *(cardinality)*<br>$x_{5}+ x_{6} \geq 0$ *(cardinality)* | — | 7/9 | 5/2 | 95/97 |
| 185 | 8 | quadratic_knapsack, quadratic_knapsack | $2x_{0}^2+ 4x_{0}x_{1}+ 5x_{0}x_{2}$<br>$+ x_{0}x_{3} + 4x_{0}x_{4}$<br>$+ 2x_{1}^2 + 4x_{1}x_{2}$<br>$+ 4x_{1}x_{3} + x_{1}x_{4}$<br>$+ 2x_{2}^2 + 3x_{2}x_{3}$<br>$+ 3x_{2}x_{4} + 3x_{3}^2$<br>$+ 2x_{3}x_{4} + x_{4}^2 \leq 19$ *(quadratic_knapsack)*<br>$3x_{5}^2+ x_{5}x_{6}+ 3x_{5}x_{7}$<br>$+ x_{6}^2 + 4x_{6}x_{7}+ 3x_{7}^2 \leq 5$ *(quadratic_knapsack)* | — | 8/16 | 115/8 | 340/1534 |
| 186 | 5 | cardinality, cardinality | $x_{0}+ x_{1}+ x_{2} \geq 0$ *(cardinality)*<br>$x_{3}+ x_{4} \leq 1$ *(cardinality)* | — | 5/8 | 5/3 | 60/73 |
| 187 | 6 | cardinality, cardinality | $x_{0}+ x_{1} = 2$ *(cardinality)*<br>$x_{2}+ x_{3}+ x_{4}+ x_{5} \geq 1$ *(cardinality)* | — | 6/8 | 2/2 | 65/89 |
| 188 | 8 | knapsack, flow | $8x_{0}+ 10x_{1}+ x_{2}+ 2x_{3} \leq 12$ *(knapsack)*<br>$x_{4}+ x_{5}+ x_{6}- x_{7} = 0$ *(flow)* | — | 8/12 | 66/4 | 239/167 |
| 189 | 8 | cardinality, cardinality | $x_{0}+ x_{1}+ x_{2}+ x_{3}+ x_{4} = 2$ *(cardinality)*<br>$x_{5}+ x_{6}+ x_{7} \leq 2$ *(cardinality)* | — | 8/10 | 36/2 | 177/121 |
| 190 | 6 | quadratic_knapsack, cardinality | $3x_{0}^2+ x_{0}x_{1}+ 3x_{0}x_{2}$<br>$+ x_{1}^2 + 4x_{1}x_{2}+ 3x_{2}^2 \leq 5$ *(quadratic_knapsack)*<br>$x_{3}+ x_{4}+ x_{5} \geq 1$ *(cardinality)* | — | 6/11 | 83/5 | 229/274 |
| 191 | 6 | quadratic_knapsack, cardinality | $x_{0}^2+ 5x_{0}x_{1}+ 3x_{0}x_{2}$<br>$+ 4x_{1}^2 + 2x_{1}x_{2}+ 4x_{2}^2 \leq 7$ *(quadratic_knapsack)*<br>$x_{3}+ x_{4}+ x_{5} = 1$ *(cardinality)* | — | 6/9 | 13/3 | 95/249 |
| 192 | 6 | quadratic_knapsack, quadratic_knapsack | $5x_{0}^2+ 3x_{0}x_{1}+ 5x_{0}x_{2}$<br>$+ 4x_{1}^2 + 5x_{1}x_{2}+ 3x_{2}^2 \leq 14$ *(quadratic_knapsack)*<br>$x_{3}^2+ 5x_{3}x_{4}+ 4x_{3}x_{5}$<br>$+ 2x_{4}^2 + 2x_{4}x_{5}+ 2x_{5}^2 \leq 8$ *(quadratic_knapsack)* | — | 6/14 | 24/8 | 114/524 |
| 193 | 8 | independent_set, quadratic_knapsack, knapsack | $x_{0}x_{1} = 0$ *(independent_set)*<br>$5x_{2}^2+ x_{2}x_{3}+ 2x_{2}x_{4}$<br>$+ 5x_{3}^2 + x_{3}x_{4}+ 3x_{4}^2 \leq 6$ *(quadratic_knapsack)*<br>$10x_{5}+ x_{6}+ 9x_{7} \leq 7$ *(knapsack)* | — | 8/14 | 19/6 | 148/312 |
| 194 | 5 | cardinality, quadratic_knapsack | $x_{0}+ x_{1} \geq 2$ *(cardinality)*<br>$3x_{2}^2+ x_{2}x_{3}+ 3x_{2}x_{4}$<br>$+ x_{3}^2 + 4x_{3}x_{4}+ 3x_{4}^2 \leq 5$ *(quadratic_knapsack)* | — | 5/8 | 85/3 | 208/230 |
| 195 | 8 | quadratic_knapsack, cardinality | $4x_{0}^2+ 4x_{0}x_{1}+ x_{0}x_{2}$<br>$+ 5x_{0}x_{3} + 5x_{1}^2$<br>$+ 5x_{1}x_{2} + 5x_{1}x_{3}$<br>$+ x_{2}^2 + 4x_{2}x_{3}+ x_{3}^2 \leq 11$ *(quadratic_knapsack)*<br>$x_{4}+ x_{5}+ x_{6}+ x_{7} = 4$ *(cardinality)* | — | 8/12 | 225/4 | 555/630 |
| 196 | 8 | knapsack, cardinality, quadratic_knapsack | $2x_{0}+ 10x_{1}+ 4x_{2} \leq 6$ *(knapsack)*<br>$x_{3}+ x_{4} \leq 2$ *(cardinality)*<br>$5x_{5}^2+ 3x_{5}x_{6}+ 5x_{5}x_{7}$<br>$+ 4x_{6}^2 + 5x_{6}x_{7}+ 3x_{7}^2 \leq 14$ *(quadratic_knapsack)* | — | 8/17 | 25/9 | 160/379 |
| 197 | 6 | cardinality, quadratic_knapsack | $x_{0}+ x_{1}+ x_{2} = 0$ *(cardinality)*<br>$x_{3}^2+ 5x_{3}x_{4}+ 3x_{3}x_{5}$<br>$+ 4x_{4}^2 + 2x_{4}x_{5}+ 4x_{5}^2 \leq 7$ *(quadratic_knapsack)* | — | 6/9 | 8/3 | 79/244 |
| 198 | 7 | flow, cardinality | $x_{0}+ x_{1}+ x_{2}- x_{3}- x_{4} = 0$ *(flow)*<br>$x_{5}+ x_{6} \leq 0$ *(cardinality)* | — | 7/7 | 18/0 | 119/71 |
| 199 | 7 | cardinality, cardinality, cardinality | $x_{0}+ x_{1} \leq 1$ *(cardinality)*<br>$x_{2}+ x_{3}+ x_{4} \leq 3$ *(cardinality)*<br>$x_{5}+ x_{6} \geq 2$ *(cardinality)* | — | 7/10 | 20/3 | 121/96 |
| 200 | 8 | knapsack, cardinality | $10x_{0}+ 8x_{1}+ 2x_{2}+ x_{3}$<br>$+ 4x_{4} \leq 10$ *(knapsack)*<br>$x_{5}+ x_{6}+ x_{7} \geq 2$ *(cardinality)* | — | 8/13 | 161/5 | 428/197 |
| 201 | 7 | knapsack, cardinality | $5x_{0}+ x_{1}+ 10x_{2}+ 4x_{3} \leq 10$ *(knapsack)*<br>$x_{4}+ x_{5}+ x_{6} \geq 0$ *(cardinality)* | — | 7/13 | 57/6 | 193/172 |
| 202 | 8 | quadratic_knapsack, cardinality | $3x_{0}^2+ x_{0}x_{1}+ 3x_{0}x_{2}$<br>$+ 4x_{0}x_{3} + 5x_{1}^2$<br>$+ 4x_{1}x_{2} + 2x_{1}x_{3}$<br>$+ 2x_{2}^2 + 4x_{2}x_{3}+ 4x_{3}^2 \leq 17$ *(quadratic_knapsack)*<br>$x_{4}+ x_{5}+ x_{6}+ x_{7} = 2$ *(cardinality)* | — | 8/13 | 37/5 | 180/700 |
| 203 | 7 | knapsack, cardinality | $2x_{0}+ 10x_{1}+ 4x_{2} \leq 6$ *(knapsack)*<br>$x_{3}+ x_{4}+ x_{5}+ x_{6} = 1$ *(cardinality)* | — | 7/10 | 14/3 | 113/119 |
| 204 | 7 | quadratic_knapsack, quadratic_knapsack | $x_{0}^2+ 4x_{0}x_{1}+ x_{0}x_{2}$<br>$+ 2x_{0}x_{3} + 5x_{1}^2$<br>$+ 4x_{1}x_{2} + 3x_{1}x_{3}$<br>$+ 5x_{2}^2 + 4x_{2}x_{3}+ 2x_{3}^2 \leq 18$ *(quadratic_knapsack)*<br>$2x_{4}^2+ 5x_{4}x_{5}+ 2x_{4}x_{6}$<br>$+ 2x_{5}^2 + x_{5}x_{6}+ x_{6}^2 \leq 7$ *(quadratic_knapsack)* | — | 7/15 | 426/8 | 933/872 |
| 205 | 6 | quadratic_knapsack, quadratic_knapsack | $3x_{0}^2+ x_{0}x_{1}+ 3x_{0}x_{2}$<br>$+ x_{1}^2 + 4x_{1}x_{2}+ 3x_{2}^2 \leq 5$ *(quadratic_knapsack)*<br>$x_{3}^2+ 2x_{3}x_{4}+ x_{3}x_{5}$<br>$+ x_{4}^2 + 5x_{4}x_{5}+ x_{5}^2 \leq 6$ *(quadratic_knapsack)* | — | 6/12 | 126/6 | 315/439 |
| 206 | 7 | quadratic_knapsack, cardinality | $4x_{0}^2+ 4x_{0}x_{1}+ x_{0}x_{2}$<br>$+ 5x_{0}x_{3} + 5x_{1}^2$<br>$+ 5x_{1}x_{2} + 5x_{1}x_{3}$<br>$+ x_{2}^2 + 4x_{2}x_{3}+ x_{3}^2 \leq 11$ *(quadratic_knapsack)*<br>$x_{4}+ x_{5}+ x_{6} \geq 0$ *(cardinality)* | — | 7/13 | 221/6 | 523/633 |
| 207 | 7 | cardinality, cardinality | $x_{0}+ x_{1} \geq 2$ *(cardinality)*<br>$x_{2}+ x_{3}+ x_{4}+ x_{5}+ x_{6} = 3$ *(cardinality)* | — | 7/7 | 33/0 | 151/80 |
| 208 | 8 | assignment, cardinality | $x_{0}+ x_{1}+ x_{2} = 1$ *(assignment)*<br>$x_{3}+ x_{4}+ x_{5}+ x_{6}+ x_{7} = 4$ *(cardinality)* | — | 8/8 | 43/0 | 199/98 |
| 209 | 6 | quadratic_knapsack, assignment | $5x_{0}^2+ 3x_{0}x_{1}+ x_{0}x_{2}$<br>$+ 5x_{0}x_{3} + 3x_{1}^2$<br>$+ x_{1}x_{2} + 4x_{1}x_{3}$<br>$+ 5x_{2}^2 + 5x_{2}x_{3}+ x_{3}^2 \leq 20$ *(quadratic_knapsack)*<br>$x_{4}+ x_{5} = 1$ *(assignment)* | — | 6/11 | 378/5 | 819/666 |
| 210 | 8 | quadratic_knapsack, cardinality | $5x_{0}^2+ 2x_{0}x_{1}+ 2x_{0}x_{2}$<br>$+ 3x_{0}x_{3} + x_{0}x_{4}+ x_{1}^2$<br>$+ 4x_{1}x_{2} + 4x_{1}x_{3}$<br>$+ 3x_{1}x_{4} + x_{2}^2$<br>$+ 3x_{2}x_{3} + x_{2}x_{4}$<br>$+ 5x_{3}^2 + x_{3}x_{4}+ 5x_{4}^2 \leq 23$ *(quadratic_knapsack)*<br>$x_{5}+ x_{6}+ x_{7} \leq 2$ *(cardinality)* | — | 8/15 | 44/7 | 195/1368 |
| 211 | 7 | cardinality, quadratic_knapsack | $x_{0}+ x_{1}+ x_{2}+ x_{3} = 0$ *(cardinality)*<br>$x_{4}^2+ 5x_{4}x_{5}+ 4x_{4}x_{6}$<br>$+ 2x_{5}^2 + 2x_{5}x_{6}+ 2x_{6}^2 \leq 8$ *(quadratic_knapsack)* | — | 7/11 | 12/4 | 112/305 |
| 212 | 5 | cardinality, cardinality | $x_{0}+ x_{1}+ x_{2} = 1$ *(cardinality)*<br>$x_{3}+ x_{4} \leq 2$ *(cardinality)* | — | 5/7 | 11/2 | 72/63 |
| 213 | 6 | cardinality, knapsack | $x_{0}+ x_{1}+ x_{2} \geq 3$ *(cardinality)*<br>$5x_{3}+ 4x_{4}+ x_{5} \leq 6$ *(knapsack)* | — | 6/9 | 14/3 | 91/100 |
| 214 | 6 | cardinality, cardinality | $x_{0}+ x_{1}+ x_{2} \geq 1$ *(cardinality)*<br>$x_{3}+ x_{4}+ x_{5} = 3$ *(cardinality)* | — | 6/8 | 3/2 | 72/83 |
| 215 | 8 | knapsack, cardinality, cardinality | $2x_{0}+ 6x_{1}+ x_{2} \leq 3$ *(knapsack)*<br>$x_{3}+ x_{4} = 2$ *(cardinality)*<br>$x_{5}+ x_{6}+ x_{7} = 2$ *(cardinality)* | — | 8/10 | 21/2 | 152/106 |
| 216 | 6 | knapsack, cardinality | $2x_{0}+ 7x_{1}+ 7x_{2} \leq 7$ *(knapsack)*<br>$x_{3}+ x_{4}+ x_{5} \geq 0$ *(cardinality)* | — | 6/11 | 8/5 | 79/122 |
| 217 | 7 | cardinality, cardinality | $x_{0}+ x_{1}+ x_{2}+ x_{3}+ x_{4} \geq 3$ *(cardinality)*<br>$x_{5}+ x_{6} \geq 0$ *(cardinality)* | — | 7/11 | 0/4 | 88/137 |
| 218 | 6 | cardinality, assignment | $x_{0}+ x_{1}+ x_{2} \leq 3$ *(cardinality)*<br>$x_{3}+ x_{4}+ x_{5} = 1$ *(assignment)* | — | 6/8 | 18/2 | 102/83 |
| 219 | 8 | knapsack, cardinality, cardinality | $9x_{0}+ 9x_{1}+ 4x_{2}+ x_{3} \leq 9$ *(knapsack)*<br>$x_{4}+ x_{5} \leq 2$ *(cardinality)*<br>$x_{6}+ x_{7} \leq 1$ *(cardinality)* | — | 8/15 | 77/7 | 264/190 |
| 220 | 8 | quadratic_knapsack, cardinality, independent_set | $x_{0}^2+ 5x_{0}x_{1}+ x_{0}x_{2}$<br>$+ 3x_{1}^2 + 3x_{1}x_{2}+ 5x_{2}^2 \leq 10$ *(quadratic_knapsack)*<br>$x_{3}+ x_{4}+ x_{5} \leq 1$ *(cardinality)*<br>$x_{6}x_{7} = 0$ *(independent_set)* | — | 8/13 | 19/5 | 148/321 |
| 221 | 8 | knapsack, knapsack | $x_{0}+ 9x_{1}+ 5x_{2}+ 2x_{3}$<br>$+ 10x_{4} \leq 10$ *(knapsack)*<br>$10x_{5}+ 8x_{6}+ 10x_{7} \leq 9$ *(knapsack)* | — | 8/16 | 150/8 | 412/253 |
| 222 | 5 | quadratic_knapsack, cardinality | $3x_{0}^2+ x_{0}x_{1}+ 3x_{0}x_{2}$<br>$+ 5x_{1}^2 + 2x_{1}x_{2}+ 2x_{2}^2 \leq 8$ *(quadratic_knapsack)*<br>$x_{3}+ x_{4} \geq 0$ *(cardinality)* | — | 5/11 | 23/6 | 96/293 |
| 223 | 6 | cardinality, knapsack | $x_{0}+ x_{1} \leq 1$ *(cardinality)*<br>$5x_{2}+ x_{3}+ 10x_{4}+ 4x_{5} \leq 10$ *(knapsack)* | — | 6/11 | 62/5 | 190/146 |
| 224 | 5 | cardinality, knapsack | $x_{0}+ x_{1} = 2$ *(cardinality)*<br>$5x_{2}+ 4x_{3}+ x_{4} \leq 6$ *(knapsack)* | — | 5/8 | 13/3 | 72/85 |
| 225 | 6 | quadratic_knapsack, cardinality | $x_{0}^2+ 2x_{0}x_{1}+ x_{0}x_{2}$<br>$+ x_{1}^2 + 5x_{1}x_{2}+ x_{2}^2 \leq 6$ *(quadratic_knapsack)*<br>$x_{3}+ x_{4}+ x_{5} \geq 0$ *(cardinality)* | — | 6/11 | 43/5 | 155/271 |
| 226 | 7 | knapsack, flow | $2x_{0}+ 5x_{1}+ 2x_{2} \leq 5$ *(knapsack)*<br>$x_{3}- x_{4}- x_{5}- x_{6} = 0$ *(flow)* | — | 7/10 | 72/3 | 223/113 |
| 227 | 8 | cardinality, knapsack | $x_{0}+ x_{1}+ x_{2}+ x_{3} \geq 4$ *(cardinality)*<br>$3x_{4}+ x_{5}+ 4x_{6}+ 8x_{7} \leq 7$ *(knapsack)* | — | 8/11 | 61/3 | 234/148 |
| 228 | 7 | flow, assignment | $x_{0}+ x_{1}+ x_{2}- x_{3} = 0$ *(flow)*<br>$x_{4}+ x_{5}+ x_{6} = 1$ *(assignment)* | — | 7/7 | 14/0 | 110/69 |
| 229 | 7 | cardinality, cardinality | $x_{0}+ x_{1}+ x_{2}+ x_{3}+ x_{4} = 1$ *(cardinality)*<br>$x_{5}+ x_{6} \leq 2$ *(cardinality)* | — | 7/9 | 15/2 | 112/100 |
| 230 | 6 | cardinality, cardinality | $x_{0}+ x_{1}+ x_{2}+ x_{3} \geq 0$ *(cardinality)*<br>$x_{4}+ x_{5} \geq 0$ *(cardinality)* | — | 6/11 | 0/5 | 69/124 |
| 231 | 8 | cardinality, cardinality | $x_{0}+ x_{1}+ x_{2} = 3$ *(cardinality)*<br>$x_{3}+ x_{4}+ x_{5}+ x_{6}+ x_{7} \geq 4$ *(cardinality)* | — | 8/9 | 3/1 | 116/115 |
| 232 | 4 | cardinality, assignment | $x_{0}+ x_{1} = 0$ *(cardinality)*<br>$x_{2}+ x_{3} = 1$ *(assignment)* | — | 4/4 | 3/0 | 40/26 |
| 233 | 8 | flow, quadratic_knapsack | $x_{0}- x_{1}- x_{2}- x_{3} = 0$ *(flow)*<br>$5x_{4}^2+ 3x_{4}x_{5}+ x_{4}x_{6}$<br>$+ 5x_{4}x_{7} + 3x_{5}^2$<br>$+ x_{5}x_{6} + 4x_{5}x_{7}$<br>$+ 5x_{6}^2 + 5x_{6}x_{7}+ x_{7}^2 \leq 20$ *(quadratic_knapsack)* | — | 8/13 | 384/5 | 881/699 |
| 234 | 8 | knapsack, quadratic_knapsack | $7x_{0}+ 7x_{1}+ 10x_{2}+ 2x_{3} \leq 10$ *(knapsack)*<br>$3x_{4}^2+ x_{4}x_{5}+ 3x_{4}x_{6}$<br>$+ 4x_{4}x_{7} + 5x_{5}^2$<br>$+ 4x_{5}x_{6} + 2x_{5}x_{7}$<br>$+ 2x_{6}^2 + 4x_{6}x_{7}+ 4x_{7}^2 \leq 17$ *(quadratic_knapsack)* | — | 8/17 | 288/9 | 681/777 |
| 235 | 8 | flow, flow | $x_{0}+ x_{1}- x_{2}- x_{3}- x_{4} = 0$ *(flow)*<br>$x_{5}+ x_{6}- x_{7} = 0$ *(flow)* | — | 8/8 | 24/0 | 153/86 |
| 236 | 8 | cardinality, cardinality | $x_{0}+ x_{1}+ x_{2}+ x_{3}+ x_{4} \leq 0$ *(cardinality)*<br>$x_{5}+ x_{6}+ x_{7} \geq 3$ *(cardinality)* | — | 8/8 | 3/0 | 112/90 |
| 237 | 6 | cardinality, independent_set | $x_{0}+ x_{1}+ x_{2}+ x_{3} = 2$ *(cardinality)*<br>$x_{4}x_{5} = 0$ *(independent_set)* | — | 6/6 | 18/0 | 99/54 |
| 238 | 7 | knapsack, quadratic_knapsack | $2x_{0}+ 10x_{1}+ 4x_{2} \leq 6$ *(knapsack)*<br>$4x_{3}^2+ 5x_{3}x_{4}+ 2x_{3}x_{5}$<br>$+ 5x_{3}x_{6} + 4x_{4}^2$<br>$+ 3x_{4}x_{5} + 5x_{4}x_{6}$<br>$+ 3x_{5}^2 + 2x_{5}x_{6}+ 3x_{6}^2 \leq 22$ *(quadratic_knapsack)* | — | 7/15 | 32/8 | 146/726 |
| 239 | 8 | knapsack, cardinality | $3x_{0}+ 8x_{1}+ 3x_{2} \leq 5$ *(knapsack)*<br>$x_{3}+ x_{4}+ x_{5}+ x_{6}+ x_{7} = 3$ *(cardinality)* | — | 8/11 | 134/3 | 378/142 |
| 240 | 8 | quadratic_knapsack, cardinality | $x_{0}^2+ x_{0}x_{1}+ 3x_{0}x_{2}$<br>$+ 2x_{0}x_{3} + x_{1}^2$<br>$+ 2x_{1}x_{2} + 3x_{1}x_{3}$<br>$+ 2x_{2}^2 + 3x_{2}x_{3}+ 3x_{3}^2 \leq 7$ *(quadratic_knapsack)*<br>$x_{4}+ x_{5}+ x_{6}+ x_{7} \geq 0$ *(cardinality)* | — | 8/14 | 128/6 | 363/613 |
| 241 | 7 | knapsack, knapsack | $8x_{0}+ 5x_{1}+ 7x_{2}+ 4x_{3} \leq 11$ *(knapsack)*<br>$2x_{4}+ 10x_{5}+ 4x_{6} \leq 6$ *(knapsack)* | — | 7/14 | 64/7 | 207/195 |
| 242 | 7 | cardinality, quadratic_knapsack | $x_{0}+ x_{1} \geq 1$ *(cardinality)*<br>$5x_{2}^2+ 2x_{2}x_{3}+ 2x_{2}x_{4}$<br>$+ 3x_{2}x_{5} + x_{2}x_{6}+ x_{3}^2$<br>$+ 4x_{3}x_{4} + 4x_{3}x_{5}$<br>$+ 3x_{3}x_{6} + x_{4}^2$<br>$+ 3x_{4}x_{5} + x_{4}x_{6}$<br>$+ 5x_{5}^2 + x_{5}x_{6}+ 5x_{6}^2 \leq 23$ *(quadratic_knapsack)* | — | 7/13 | 32/6 | 143/1332 |
| 243 | 8 | knapsack, knapsack | $5x_{0}+ x_{1}+ 10x_{2}+ 4x_{3} \leq 10$ *(knapsack)*<br>$7x_{4}+ 7x_{5}+ 10x_{6}+ 2x_{7} \leq 10$ *(knapsack)* | — | 8/16 | 326/8 | 762/250 |
| 244 | 8 | quadratic_knapsack, cardinality | $x_{0}^2+ 4x_{0}x_{1}+ x_{0}x_{2}$<br>$+ 5x_{0}x_{3} + 2x_{1}^2$<br>$+ 3x_{1}x_{2} + 3x_{1}x_{3}$<br>$+ 5x_{2}^2 + 4x_{2}x_{3}+ 5x_{3}^2 \leq 22$ *(quadratic_knapsack)*<br>$x_{4}+ x_{5}+ x_{6}+ x_{7} \leq 1$ *(cardinality)* | — | 8/14 | 35/6 | 177/716 |
| 245 | 5 | cardinality, knapsack | $x_{0}+ x_{1} \leq 2$ *(cardinality)*<br>$6x_{2}+ 2x_{3}+ 2x_{4} \leq 3$ *(knapsack)* | — | 5/9 | 29/4 | 104/88 |
| 246 | 6 | knapsack, cardinality | $x_{0}+ 3x_{1}+ 9x_{2}+ 8x_{3} \leq 10$ *(knapsack)*<br>$x_{4}+ x_{5} \geq 1$ *(cardinality)* | — | 6/11 | 100/5 | 266/146 |
| 247 | 7 | quadratic_knapsack, cardinality | $5x_{0}^2+ 2x_{0}x_{1}+ 2x_{0}x_{2}$<br>$+ 3x_{0}x_{3} + x_{0}x_{4}+ x_{1}^2$<br>$+ 4x_{1}x_{2} + 4x_{1}x_{3}$<br>$+ 3x_{1}x_{4} + x_{2}^2$<br>$+ 3x_{2}x_{3} + x_{2}x_{4}$<br>$+ 5x_{3}^2 + x_{3}x_{4}+ 5x_{4}^2 \leq 23$ *(quadratic_knapsack)*<br>$x_{5}+ x_{6} = 0$ *(cardinality)* | — | 7/12 | 32/5 | 148/1322 |
| 248 | 7 | cardinality, knapsack | $x_{0}+ x_{1}+ x_{2}+ x_{3} \geq 0$ *(cardinality)*<br>$2x_{4}+ 5x_{5}+ 2x_{6} \leq 5$ *(knapsack)* | — | 7/13 | 63/6 | 207/164 |
| 249 | 8 | cardinality, cardinality | $x_{0}+ x_{1}+ x_{2}+ x_{3} \geq 4$ *(cardinality)*<br>$x_{4}+ x_{5}+ x_{6}+ x_{7} = 3$ *(cardinality)* | — | 8/8 | 27/0 | 167/95 |

---

## Results

`H` = HybridQAOA, `P` = PenaltyQAOA. Columns: AR$_f$ = AR$_{\text{feas}}$, $P_f$ = $P(\text{feas})$, $P_o$ = $P(\text{opt})$.

| COP | $n_x$ | Method | $p=1$ AR$_f$ | $p=1$ $P_f$ | $p=1$ $P_o$ | $p=2$ AR$_f$ | $p=2$ $P_f$ | $p=2$ $P_o$ | $p=3$ AR$_f$ | $p=3$ $P_f$ | $p=3$ $P_o$ | $p=4$ AR$_f$ | $p=4$ $P_f$ | $p=4$ $P_o$ | $p=5$ AR$_f$ | $p=5$ $P_f$ | $p=5$ $P_o$ |
|-----|-------|--------|--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---|
| 0 | 6 | `H` | 0.810 | 1.000 | 0.241 | 0.911 | 1.000 | 0.501 | 0.968 | 1.000 | 0.747 | 0.998 | 1.000 | 0.982 | 1.000 | 1.000 | 1.000 |
| 0 | 6 | `P` | 0.142 | 0.248 | 0.027 | 0.183 | 0.302 | 0.043 | 0.157 | 0.262 | 0.017 | 0.079 | 0.141 | 0.011 | 0.105 | 0.179 | 0.010 |
| 1 | 7 | `H` | 0.464 | 1.000 | 0.025 | 0.558 | 1.000 | 0.044 | 0.671 | 1.000 | 0.115 | 0.770 | 1.000 | 0.201 | 0.839 | 1.000 | 0.276 |
| 1 | 7 | `P` | 0.315 | 0.818 | 0.003 | 0.387 | 0.919 | 0.006 | 0.485 | 0.971 | 0.077 | 0.125 | 0.910 | 0.000 | 0.669 | 0.982 | 0.002 |
| 2 | 8 | `H` | 0.771 | 1.000 | 0.117 | 0.866 | 1.000 | 0.157 | 0.919 | 1.000 | 0.235 | 0.942 | 1.000 | 0.330 | 0.954 | 1.000 | 0.483 |
| 2 | 8 | `P` | 0.111 | 0.195 | 0.002 | 0.107 | 0.200 | 0.003 | 0.107 | 0.185 | 0.004 | 0.107 | 0.183 | 0.007 | 0.113 | 0.196 | 0.005 |
| 3 | 7 | `H` | 0.647 | 1.000 | 0.047 | 0.825 | 1.000 | 0.220 | 0.974 | 1.000 | 0.520 | 0.989 | 1.000 | 0.755 | 0.997 | 1.000 | 0.948 |
| 3 | 7 | `P` | 0.428 | 0.620 | 0.002 | 0.631 | 0.921 | 0.000 | 0.663 | 0.965 | 0.000 | 0.777 | 0.966 | 0.434 | 0.827 | 0.983 | 0.486 |
| 4 | 8 | `H` | 0.867 | 1.000 | 0.299 | 0.952 | 1.000 | 0.636 | 0.995 | 1.000 | 0.969 | 1.000 | 1.000 | 0.999 | 1.000 | 1.000 | 1.000 |
| 4 | 8 | `P` | 0.028 | 0.062 | 0.009 | 0.021 | 0.038 | 0.007 | 0.031 | 0.055 | 0.004 | 0.016 | 0.028 | 0.007 | 0.014 | 0.030 | 0.004 |
| 5 | 6 | `H` | 0.539 | 1.000 | 0.232 | 0.723 | 1.000 | 0.614 | 0.950 | 1.000 | 0.919 | 0.996 | 1.000 | 0.995 | 1.000 | 1.000 | 1.000 |
| 5 | 6 | `P` | 0.067 | 0.191 | 0.031 | 0.069 | 0.177 | 0.037 | 0.057 | 0.168 | 0.025 | 0.063 | 0.209 | 0.013 | 0.067 | 0.225 | 0.016 |
| 6 | 6 | `H` | 0.566 | 1.000 | 0.123 | 0.722 | 1.000 | 0.207 | 0.832 | 1.000 | 0.345 | 0.911 | 1.000 | 0.517 | 0.959 | 1.000 | 0.766 |
| 6 | 6 | `P` | 0.209 | 0.547 | 0.018 | 0.214 | 0.557 | 0.015 | 0.220 | 0.566 | 0.019 | 0.221 | 0.568 | 0.018 | 0.208 | 0.551 | 0.015 |
| 7 | 7 | `H` | 0.749 | 1.000 | 0.283 | 0.932 | 1.000 | 0.543 | 0.986 | 1.000 | 0.765 | 0.998 | 1.000 | 0.971 | 1.000 | 1.000 | 1.000 |
| 7 | 7 | `P` | 0.233 | 0.579 | 0.131 | 0.796 | 0.880 | 0.058 | 0.563 | 0.936 | 0.006 | 0.945 | 0.958 | 0.911 | 0.598 | 0.958 | 0.012 |
| 8 | 5 | `H` | 0.584 | 1.000 | 0.081 | 0.718 | 1.000 | 0.214 | 0.807 | 1.000 | 0.297 | 0.890 | 1.000 | 0.432 | 0.928 | 1.000 | 0.602 |
| 8 | 5 | `P` | 0.409 | 0.882 | 0.036 | 0.430 | 0.901 | 0.042 | 0.415 | 0.875 | 0.034 | 0.437 | 0.894 | 0.042 | 0.402 | 0.866 | 0.021 |
| 9 | 7 | `H` | 0.795 | 1.000 | 0.167 | 0.910 | 1.000 | 0.225 | 0.944 | 1.000 | 0.317 | 0.963 | 1.000 | 0.466 | 0.980 | 1.000 | 0.701 |
| 9 | 7 | `P` | 0.141 | 0.346 | 0.004 | 0.076 | 0.143 | 0.004 | 0.081 | 0.157 | 0.008 | 0.094 | 0.162 | 0.014 | 0.099 | 0.181 | 0.007 |
| 10 | 7 | `H` | 0.665 | 1.000 | 0.019 | 0.765 | 1.000 | 0.068 | 0.826 | 1.000 | 0.103 | 0.884 | 1.000 | 0.174 | 0.925 | 1.000 | 0.222 |
| 10 | 7 | `P` | 0.211 | 0.378 | 0.009 | 0.221 | 0.405 | 0.008 | 0.213 | 0.382 | 0.008 | 0.216 | 0.396 | 0.010 | 0.218 | 0.399 | 0.009 |
| 11 | 8 | `H` | 0.415 | 1.000 | 0.026 | 0.516 | 1.000 | 0.059 | 0.629 | 1.000 | 0.087 | 0.732 | 1.000 | 0.131 | 0.795 | 1.000 | 0.169 |
| 11 | 8 | `P` | 0.133 | 0.370 | 0.004 | 0.127 | 0.360 | 0.007 | 0.136 | 0.381 | 0.003 | 0.136 | 0.379 | 0.002 | 0.130 | 0.374 | 0.003 |
| 12 | 8 | `H` | 0.944 | 1.000 | 0.932 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 12 | 8 | `P` | 0.067 | 0.167 | 0.052 | 0.022 | 0.080 | 0.015 | 0.025 | 0.067 | 0.021 | 0.038 | 0.055 | 0.035 | 0.023 | 0.052 | 0.019 |
| 13 | 7 | `H` | 0.575 | 1.000 | 0.015 | 0.701 | 1.000 | 0.066 | 0.821 | 1.000 | 0.148 | 0.866 | 1.000 | 0.177 | 0.895 | 1.000 | 0.221 |
| 13 | 7 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 14 | 8 | `H` | 0.433 | 1.000 | 0.017 | 0.517 | 1.000 | 0.020 | 0.615 | 1.000 | 0.035 | 0.690 | 1.000 | 0.061 | 0.752 | 1.000 | 0.095 |
| 14 | 8 | `P` | 0.157 | 0.407 | 0.003 | 0.159 | 0.421 | 0.003 | 0.161 | 0.417 | 0.004 | 0.158 | 0.410 | 0.005 | 0.156 | 0.412 | 0.004 |
| 15 | 8 | `H` | 0.488 | 1.000 | 0.032 | 0.548 | 1.000 | 0.054 | 0.655 | 1.000 | 0.108 | 0.737 | 1.000 | 0.156 | 0.794 | 1.000 | 0.209 |
| 15 | 8 | `P` | 0.189 | 0.472 | 0.006 | 0.167 | 0.432 | 0.002 | 0.159 | 0.393 | 0.004 | 0.149 | 0.356 | 0.007 | 0.154 | 0.376 | 0.004 |
| 16 | 6 | `H` | 0.568 | 1.000 | 0.115 | 0.708 | 1.000 | 0.248 | 0.865 | 1.000 | 0.424 | 0.958 | 1.000 | 0.530 | 0.985 | 1.000 | 0.603 |
| 16 | 6 | `P` | 0.172 | 0.449 | 0.015 | 0.177 | 0.465 | 0.017 | 0.185 | 0.478 | 0.018 | 0.175 | 0.466 | 0.012 | 0.170 | 0.470 | 0.017 |
| 17 | 8 | `H` | 0.706 | 1.000 | 0.193 | 0.878 | 1.000 | 0.354 | 0.950 | 1.000 | 0.603 | 0.985 | 1.000 | 0.884 | 0.999 | 1.000 | 0.992 |
| 17 | 8 | `P` | 0.075 | 0.237 | 0.000 | 0.082 | 0.282 | 0.000 | 0.038 | 0.081 | 0.005 | 0.042 | 0.089 | 0.007 | 0.035 | 0.077 | 0.005 |
| 18 | 8 | `H` | 0.618 | 1.000 | 0.068 | 0.772 | 1.000 | 0.145 | 0.835 | 1.000 | 0.235 | 0.875 | 1.000 | 0.346 | 0.909 | 1.000 | 0.548 |
| 18 | 8 | `P` | 0.058 | 0.155 | 0.003 | 0.051 | 0.103 | 0.008 | 0.053 | 0.109 | 0.004 | 0.047 | 0.095 | 0.005 | 0.039 | 0.086 | 0.002 |
| 19 | 8 | `H` | 0.448 | 1.000 | 0.009 | 0.523 | 1.000 | 0.032 | 0.617 | 1.000 | 0.044 | 0.679 | 1.000 | 0.061 | 0.740 | 1.000 | 0.095 |
| 19 | 8 | `P` | 0.197 | 0.528 | 0.001 | 0.254 | 0.670 | 0.000 | 0.266 | 0.623 | 0.000 | 0.208 | 0.512 | 0.003 | 0.192 | 0.494 | 0.003 |
| 20 | 6 | `H` | 0.676 | 1.000 | 0.175 | 0.815 | 1.000 | 0.270 | 0.898 | 1.000 | 0.358 | 0.938 | 1.000 | 0.507 | 0.964 | 1.000 | 0.691 |
| 20 | 6 | `P` | 0.224 | 0.510 | 0.013 | 0.173 | 0.422 | 0.011 | 0.121 | 0.277 | 0.018 | 0.141 | 0.330 | 0.015 | 0.147 | 0.330 | 0.018 |
| 21 | 6 | `H` | 0.730 | 1.000 | 0.269 | 0.893 | 1.000 | 0.429 | 0.958 | 1.000 | 0.601 | 0.982 | 1.000 | 0.767 | 0.995 | 1.000 | 0.939 |
| 21 | 6 | `P` | 0.255 | 0.562 | 0.041 | 0.209 | 0.472 | 0.020 | 0.206 | 0.475 | 0.028 | 0.217 | 0.497 | 0.030 | 0.249 | 0.507 | 0.062 |
| 22 | 6 | `H` | 0.557 | 1.000 | 0.041 | 0.658 | 1.000 | 0.060 | 0.754 | 1.000 | 0.108 | 0.817 | 1.000 | 0.118 | 0.879 | 1.000 | 0.161 |
| 22 | 6 | `P` | 0.443 | 0.877 | 0.014 | 0.447 | 0.874 | 0.025 | 0.442 | 0.877 | 0.013 | 0.422 | 0.874 | 0.013 | 0.428 | 0.882 | 0.014 |
| 23 | 8 | `H` | 0.539 | 1.000 | 0.015 | 0.619 | 1.000 | 0.032 | 0.696 | 1.000 | 0.045 | 0.742 | 1.000 | 0.054 | 0.781 | 1.000 | 0.079 |
| 23 | 8 | `P` | 0.197 | 0.388 | 0.006 | 0.179 | 0.371 | 0.004 | 0.147 | 0.314 | 0.004 | 0.148 | 0.303 | 0.004 | 0.144 | 0.300 | 0.004 |
| 24 | 8 | `H` | 0.677 | 1.000 | 0.131 | 0.849 | 1.000 | 0.148 | 0.920 | 1.000 | 0.334 | 0.960 | 1.000 | 0.511 | 0.978 | 1.000 | 0.700 |
| 24 | 8 | `P` | 0.213 | 0.309 | 0.097 | 0.132 | 0.260 | 0.014 | 0.122 | 0.232 | 0.036 | 0.121 | 0.224 | 0.024 | 0.108 | 0.211 | 0.019 |
| 25 | 7 | `H` | 0.978 | 1.000 | 0.962 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 25 | 7 | `P` | 0.365 | 0.650 | 0.355 | 0.735 | 0.893 | 0.014 | 0.808 | 0.941 | 0.003 | 0.750 | 0.949 | 0.576 | 0.842 | 0.978 | 0.016 |
| 26 | 6 | `H` | 0.808 | 1.000 | 0.645 | 0.956 | 1.000 | 0.927 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 26 | 6 | `P` | 0.070 | 0.128 | 0.029 | 0.062 | 0.122 | 0.015 | 0.071 | 0.134 | 0.029 | 0.052 | 0.092 | 0.020 | 0.041 | 0.078 | 0.015 |
| 27 | 5 | `H` | 0.997 | 1.000 | 0.988 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 27 | 5 | `P` | 0.301 | 0.717 | 0.000 | 0.720 | 0.994 | 0.000 | 0.795 | 0.987 | 0.000 | 0.985 | 0.989 | 0.970 | 0.979 | 0.980 | 0.977 |
| 28 | 8 | `H` | 0.499 | 1.000 | 0.053 | 0.675 | 1.000 | 0.119 | 0.778 | 1.000 | 0.180 | 0.853 | 1.000 | 0.278 | 0.904 | 1.000 | 0.358 |
| 28 | 8 | `P` | 0.214 | 0.417 | 0.018 | 0.079 | 0.196 | 0.003 | 0.077 | 0.194 | 0.003 | 0.077 | 0.195 | 0.005 | 0.080 | 0.200 | 0.004 |
| 29 | 7 | `H` | 0.556 | 1.000 | 0.042 | 0.651 | 1.000 | 0.084 | 0.757 | 1.000 | 0.142 | 0.838 | 1.000 | 0.207 | 0.902 | 1.000 | 0.322 |
| 29 | 7 | `P` | 0.419 | 0.792 | 0.000 | 0.542 | 0.930 | 0.002 | 0.530 | 0.974 | 0.003 | 0.557 | 0.992 | 0.000 | 0.631 | 0.984 | 0.001 |
| 30 | 8 | `H` | 0.576 | 1.000 | 0.024 | 0.642 | 1.000 | 0.037 | 0.758 | 1.000 | 0.083 | 0.842 | 1.000 | 0.160 | 0.897 | 1.000 | 0.238 |
| 30 | 8 | `P` | 0.176 | 0.396 | 0.005 | 0.352 | 0.556 | 0.004 | 0.406 | 0.755 | 0.007 | 0.391 | 0.743 | 0.013 | 0.747 | 0.923 | 0.009 |
| 31 | 7 | `H` | 0.498 | 1.000 | 0.003 | 0.629 | 1.000 | 0.014 | 0.742 | 1.000 | 0.023 | 0.781 | 1.000 | 0.032 | 0.808 | 1.000 | 0.037 |
| 31 | 7 | `P` | 0.298 | 0.607 | 0.009 | 0.312 | 0.609 | 0.006 | 0.306 | 0.603 | 0.009 | 0.305 | 0.613 | 0.007 | 0.317 | 0.621 | 0.008 |
| 32 | 7 | `H` | 0.548 | 1.000 | 0.005 | 0.647 | 1.000 | 0.070 | 0.707 | 1.000 | 0.100 | 0.768 | 1.000 | 0.153 | 0.835 | 1.000 | 0.254 |
| 32 | 7 | `P` | 0.336 | 0.665 | 0.017 | 0.351 | 0.674 | 0.019 | 0.312 | 0.659 | 0.015 | 0.338 | 0.658 | 0.016 | 0.335 | 0.664 | 0.016 |
| 33 | 6 | `H` | 0.938 | 1.000 | 0.722 | 0.998 | 1.000 | 0.988 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 33 | 6 | `P` | 0.760 | 0.881 | 0.444 | 0.859 | 0.997 | 0.500 | 0.861 | 0.999 | 0.501 | 0.925 | 0.981 | 0.784 | 0.985 | 0.994 | 0.967 |
| 34 | 8 | `H` | 0.886 | 1.000 | 0.261 | 0.967 | 1.000 | 0.502 | 0.982 | 1.000 | 0.674 | 0.992 | 1.000 | 0.861 | 0.999 | 1.000 | 0.988 |
| 34 | 8 | `P` | 0.102 | 0.141 | 0.003 | 0.034 | 0.047 | 0.008 | 0.036 | 0.056 | 0.004 | 0.028 | 0.045 | 0.004 | 0.029 | 0.042 | 0.005 |
| 35 | 4 | `H` | 0.760 | 1.000 | 0.720 | 0.993 | 1.000 | 0.992 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 35 | 4 | `P` | 0.384 | 0.984 | 0.247 | 0.969 | 0.982 | 0.965 | 0.991 | 0.999 | 0.989 | 0.993 | 0.998 | 0.992 | 0.999 | 1.000 | 0.999 |
| 36 | 7 | `H` | 0.444 | 1.000 | 0.005 | 0.530 | 1.000 | 0.005 | 0.602 | 1.000 | 0.011 | 0.669 | 1.000 | 0.021 | 0.707 | 1.000 | 0.027 |
| 36 | 7 | `P` | 0.274 | 0.653 | 0.006 | 0.269 | 0.655 | 0.009 | 0.269 | 0.660 | 0.005 | 0.268 | 0.646 | 0.010 | 0.270 | 0.661 | 0.006 |
| 37 | 8 | `H` | 0.462 | 1.000 | 0.037 | 0.595 | 1.000 | 0.083 | 0.713 | 1.000 | 0.145 | 0.807 | 1.000 | 0.230 | 0.864 | 1.000 | 0.295 |
| 37 | 8 | `P` | 0.368 | 0.852 | 0.049 | 0.348 | 0.952 | 0.000 | 0.346 | 0.975 | 0.000 | 0.382 | 0.967 | 0.002 | 0.399 | 0.923 | 0.058 |
| 38 | 7 | `H` | 0.796 | 1.000 | 0.371 | 0.935 | 1.000 | 0.509 | 0.985 | 1.000 | 0.829 | 0.999 | 1.000 | 0.996 | 1.000 | 1.000 | 1.000 |
| 38 | 7 | `P` | 0.083 | 0.195 | 0.014 | 0.018 | 0.047 | 0.003 | 0.064 | 0.147 | 0.012 | 0.020 | 0.050 | 0.003 | 0.044 | 0.100 | 0.010 |
| 39 | 8 | `H` | 0.648 | 1.000 | 0.214 | 0.833 | 1.000 | 0.428 | 0.941 | 1.000 | 0.585 | 0.972 | 1.000 | 0.725 | 0.989 | 1.000 | 0.897 |
| 39 | 8 | `P` | 0.116 | 0.236 | 0.007 | 0.100 | 0.224 | 0.010 | 0.088 | 0.204 | 0.007 | 0.086 | 0.193 | 0.008 | 0.087 | 0.195 | 0.007 |
| 40 | 8 | `H` | 0.506 | 1.000 | 0.018 | 0.621 | 1.000 | 0.044 | 0.722 | 1.000 | 0.081 | 0.797 | 1.000 | 0.113 | 0.860 | 1.000 | 0.149 |
| 40 | 8 | `P` | 0.086 | 0.183 | 0.004 | 0.090 | 0.190 | 0.005 | 0.085 | 0.185 | 0.004 | 0.086 | 0.185 | 0.004 | 0.088 | 0.189 | 0.003 |
| 41 | 5 | `H` | 0.817 | 1.000 | 0.315 | 0.920 | 1.000 | 0.444 | 0.975 | 1.000 | 0.887 | 0.999 | 1.000 | 0.997 | 1.000 | 1.000 | 0.999 |
| 41 | 5 | `P` | 0.292 | 0.575 | 0.050 | 0.272 | 0.450 | 0.029 | 0.222 | 0.438 | 0.044 | 0.279 | 0.430 | 0.091 | 0.202 | 0.347 | 0.057 |
| 42 | 7 | `H` | 0.875 | 1.000 | 0.821 | 0.992 | 1.000 | 0.989 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 42 | 7 | `P` | 0.030 | 0.072 | 0.006 | 0.047 | 0.104 | 0.013 | 0.025 | 0.055 | 0.009 | 0.029 | 0.067 | 0.010 | 0.020 | 0.050 | 0.008 |
| 43 | 8 | `H` | 0.413 | 1.000 | 0.022 | 0.513 | 1.000 | 0.037 | 0.573 | 1.000 | 0.061 | 0.655 | 1.000 | 0.102 | 0.736 | 1.000 | 0.160 |
| 43 | 8 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 44 | 7 | `H` | 0.484 | 1.000 | 0.036 | 0.659 | 1.000 | 0.106 | 0.758 | 1.000 | 0.171 | 0.856 | 1.000 | 0.218 | 0.903 | 1.000 | 0.301 |
| 44 | 7 | `P` | 0.414 | 0.764 | 0.003 | 0.564 | 0.974 | 0.000 | 0.676 | 0.987 | 0.002 | 0.696 | 0.994 | 0.000 | 0.695 | 0.995 | 0.003 |
| 45 | 8 | `H` | 0.716 | 1.000 | 0.173 | 0.818 | 1.000 | 0.356 | 0.915 | 1.000 | 0.627 | 0.975 | 1.000 | 0.901 | 0.997 | 1.000 | 0.988 |
| 45 | 8 | `P` | 0.229 | 0.388 | 0.010 | 0.196 | 0.391 | 0.005 | 0.097 | 0.168 | 0.004 | 0.065 | 0.111 | 0.009 | 0.073 | 0.112 | 0.011 |
| 46 | 8 | `H` | 0.739 | 1.000 | 0.109 | 0.848 | 1.000 | 0.174 | 0.896 | 1.000 | 0.236 | 0.939 | 1.000 | 0.294 | 0.957 | 1.000 | 0.369 |
| 46 | 8 | `P` | 0.173 | 0.313 | 0.006 | 0.130 | 0.243 | 0.009 | 0.120 | 0.224 | 0.007 | 0.119 | 0.227 | 0.010 | 0.124 | 0.243 | 0.007 |
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
| 56 | 6 | `H` | 0.764 | 1.000 | 0.338 | 0.880 | 1.000 | 0.586 | 0.935 | 1.000 | 0.691 | 0.977 | 1.000 | 0.853 | 0.991 | 1.000 | 0.929 |
| 56 | 6 | `P` | 0.367 | 0.608 | 0.074 | 0.299 | 0.485 | 0.059 | 0.286 | 0.463 | 0.070 | 0.245 | 0.412 | 0.053 | 0.242 | 0.420 | 0.043 |
| 57 | 8 | `H` | 0.478 | 1.000 | 0.003 | 0.599 | 1.000 | 0.031 | 0.702 | 1.000 | 0.073 | 0.793 | 1.000 | 0.101 | 0.852 | 1.000 | 0.135 |
| 57 | 8 | `P` | 0.335 | 0.567 | 0.004 | 0.398 | 0.772 | 0.006 | 0.566 | 0.947 | 0.142 | 0.677 | 0.990 | 0.186 | 0.569 | 0.997 | 0.183 |
| 58 | 6 | `H` | 0.605 | 1.000 | 0.085 | 0.693 | 1.000 | 0.128 | 0.756 | 1.000 | 0.166 | 0.831 | 1.000 | 0.256 | 0.902 | 1.000 | 0.368 |
| 58 | 6 | `P` | 0.590 | 0.980 | 0.080 | 0.649 | 0.999 | 0.037 | 0.607 | 0.997 | 0.009 | 0.578 | 0.999 | 0.025 | 0.615 | 0.999 | 0.001 |
| 59 | 8 | `H` | 0.453 | 1.000 | 0.022 | 0.584 | 1.000 | 0.056 | 0.685 | 1.000 | 0.088 | 0.776 | 1.000 | 0.127 | 0.842 | 1.000 | 0.198 |
| 59 | 8 | `P` | 0.118 | 0.329 | 0.002 | 0.086 | 0.234 | 0.005 | 0.072 | 0.211 | 0.001 | 0.070 | 0.193 | 0.002 | 0.078 | 0.206 | 0.008 |
| 60 | 8 | `H` | 0.544 | 1.000 | 0.038 | 0.606 | 1.000 | 0.059 | 0.692 | 1.000 | 0.093 | 0.768 | 1.000 | 0.153 | 0.825 | 1.000 | 0.203 |
| 60 | 8 | `P` | 0.248 | 0.485 | 0.014 | 0.249 | 0.482 | 0.012 | 0.250 | 0.497 | 0.010 | 0.252 | 0.497 | 0.012 | 0.242 | 0.480 | 0.010 |
| 61 | 6 | `H` | 0.746 | 1.000 | 0.218 | 0.897 | 1.000 | 0.398 | 0.957 | 1.000 | 0.674 | 0.996 | 1.000 | 0.978 | 1.000 | 1.000 | 1.000 |
| 61 | 6 | `P` | 0.237 | 0.320 | 0.033 | 0.242 | 0.365 | 0.018 | 0.168 | 0.280 | 0.019 | 0.198 | 0.305 | 0.019 | 0.131 | 0.265 | 0.014 |
| 62 | 7 | `H` | 0.441 | 1.000 | 0.043 | 0.536 | 1.000 | 0.085 | 0.645 | 1.000 | 0.143 | 0.735 | 1.000 | 0.218 | 0.789 | 1.000 | 0.287 |
| 62 | 7 | `P` | 0.158 | 0.450 | 0.009 | 0.159 | 0.461 | 0.009 | 0.171 | 0.481 | 0.010 | 0.161 | 0.481 | 0.007 | 0.162 | 0.471 | 0.009 |
| 63 | 7 | `H` | 0.661 | 1.000 | 0.169 | 0.751 | 1.000 | 0.299 | 0.859 | 1.000 | 0.567 | 0.944 | 1.000 | 0.842 | 0.999 | 1.000 | 0.998 |
| 63 | 7 | `P` | 0.023 | 0.071 | 0.004 | 0.033 | 0.079 | 0.007 | 0.035 | 0.081 | 0.012 | 0.036 | 0.086 | 0.008 | 0.032 | 0.077 | 0.005 |
| 64 | 8 | `H` | 0.543 | 1.000 | 0.044 | 0.662 | 1.000 | 0.103 | 0.767 | 1.000 | 0.155 | 0.833 | 1.000 | 0.240 | 0.894 | 1.000 | 0.335 |
| 64 | 8 | `P` | 0.203 | 0.393 | 0.011 | 0.099 | 0.236 | 0.003 | 0.089 | 0.206 | 0.002 | 0.099 | 0.233 | 0.003 | 0.094 | 0.226 | 0.003 |
| 65 | 5 | `H` | 0.772 | 1.000 | 0.624 | 0.989 | 1.000 | 0.986 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 65 | 5 | `P` | 0.185 | 0.523 | 0.003 | 0.397 | 0.855 | 0.004 | 0.684 | 0.923 | 0.442 | 0.548 | 0.993 | 0.490 | 0.271 | 0.997 | 0.000 |
| 66 | 5 | `H` | 0.899 | 1.000 | 0.457 | 0.968 | 1.000 | 0.685 | 0.991 | 1.000 | 0.906 | 1.000 | 1.000 | 0.999 | 1.000 | 1.000 | 1.000 |
| 66 | 5 | `P` | 0.099 | 0.198 | 0.029 | 0.114 | 0.263 | 0.019 | 0.095 | 0.174 | 0.032 | 0.111 | 0.194 | 0.031 | 0.106 | 0.221 | 0.031 |
| 67 | 8 | `H` | 0.971 | 1.000 | 0.844 | 0.989 | 1.000 | 0.928 | 0.998 | 1.000 | 0.989 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 67 | 8 | `P` | 0.021 | 0.068 | 0.013 | 0.044 | 0.083 | 0.010 | 0.056 | 0.110 | 0.024 | 0.038 | 0.079 | 0.020 | 0.016 | 0.031 | 0.007 |
| 68 | 6 | `H` | 0.606 | 1.000 | 0.045 | 0.778 | 1.000 | 0.081 | 0.866 | 1.000 | 0.148 | 0.885 | 1.000 | 0.184 | 0.904 | 1.000 | 0.284 |
| 68 | 6 | `P` | 0.132 | 0.309 | 0.009 | 0.149 | 0.325 | 0.017 | 0.151 | 0.319 | 0.016 | 0.150 | 0.322 | 0.018 | 0.150 | 0.322 | 0.014 |
| 69 | 6 | `H` | 0.454 | 1.000 | 0.072 | 0.616 | 1.000 | 0.081 | 0.789 | 1.000 | 0.242 | 0.857 | 1.000 | 0.348 | 0.905 | 1.000 | 0.547 |
| 69 | 6 | `P` | 0.177 | 0.554 | 0.016 | 0.152 | 0.438 | 0.023 | 0.136 | 0.443 | 0.011 | 0.153 | 0.463 | 0.018 | 0.144 | 0.448 | 0.011 |
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
| 75 | 8 | `H` | 0.650 | 1.000 | 0.094 | 0.754 | 1.000 | 0.174 | 0.849 | 1.000 | 0.317 | 0.935 | 1.000 | 0.488 | 0.967 | 1.000 | 0.596 |
| 75 | 8 | `P` | 0.080 | 0.175 | 0.006 | 0.073 | 0.150 | 0.004 | 0.070 | 0.152 | 0.004 | 0.062 | 0.135 | 0.002 | 0.071 | 0.154 | 0.005 |
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
| 82 | 8 | `H` | 0.498 | 1.000 | 0.127 | 0.661 | 1.000 | 0.259 | 0.781 | 1.000 | 0.420 | 0.877 | 1.000 | 0.635 | 0.952 | 1.000 | 0.882 |
| 82 | 8 | `P` | 0.053 | 0.180 | 0.010 | 0.010 | 0.033 | 0.002 | 0.023 | 0.071 | 0.005 | 0.019 | 0.058 | 0.004 | 0.024 | 0.078 | 0.005 |
| 83 | 7 | `H` | 0.599 | 1.000 | 0.036 | 0.699 | 1.000 | 0.074 | 0.792 | 1.000 | 0.117 | 0.847 | 1.000 | 0.174 | 0.882 | 1.000 | 0.253 |
| 83 | 7 | `P` | 0.223 | 0.409 | 0.007 | 0.224 | 0.413 | 0.009 | 0.210 | 0.394 | 0.008 | 0.223 | 0.414 | 0.007 | 0.219 | 0.408 | 0.008 |
| 84 | 6 | `H` | 0.535 | 1.000 | 0.017 | 0.675 | 1.000 | 0.033 | 0.756 | 1.000 | 0.057 | 0.809 | 1.000 | 0.071 | 0.846 | 1.000 | 0.117 |
| 84 | 6 | `P` | 0.208 | 0.548 | 0.009 | 0.222 | 0.494 | 0.016 | 0.229 | 0.515 | 0.018 | 0.242 | 0.569 | 0.017 | 0.210 | 0.502 | 0.013 |
| 85 | 7 | `H` | 0.566 | 1.000 | 0.029 | 0.645 | 1.000 | 0.061 | 0.740 | 1.000 | 0.100 | 0.813 | 1.000 | 0.147 | 0.872 | 1.000 | 0.204 |
| 85 | 7 | `P` | 0.263 | 0.494 | 0.013 | 0.310 | 0.539 | 0.011 | 0.269 | 0.496 | 0.007 | 0.311 | 0.536 | 0.015 | 0.266 | 0.492 | 0.006 |
| 86 | 7 | `H` | 0.574 | 1.000 | 0.047 | 0.691 | 1.000 | 0.086 | 0.781 | 1.000 | 0.145 | 0.866 | 1.000 | 0.250 | 0.918 | 1.000 | 0.347 |
| 86 | 7 | `P` | 0.334 | 0.749 | 0.012 | 0.391 | 0.829 | 0.013 | 0.418 | 0.871 | 0.012 | 0.387 | 0.916 | 0.015 | 0.435 | 0.939 | 0.005 |
| 87 | 8 | `H` | 0.600 | 1.000 | 0.040 | 0.684 | 1.000 | 0.076 | 0.745 | 1.000 | 0.132 | 0.796 | 1.000 | 0.212 | 0.841 | 1.000 | 0.302 |
| 87 | 8 | `P` | 0.242 | 0.485 | 0.013 | 0.182 | 0.367 | 0.002 | 0.122 | 0.243 | 0.003 | 0.120 | 0.242 | 0.004 | 0.128 | 0.263 | 0.005 |
| 88 | 7 | `H` | 0.785 | 1.000 | 0.137 | 0.876 | 1.000 | 0.253 | 0.933 | 1.000 | 0.392 | 0.979 | 1.000 | 0.688 | 0.994 | 1.000 | 0.900 |
| 88 | 7 | `P` | 0.121 | 0.234 | 0.014 | 0.100 | 0.188 | 0.015 | 0.081 | 0.145 | 0.007 | 0.079 | 0.144 | 0.008 | 0.084 | 0.151 | 0.009 |
| 89 | 8 | `H` | 0.511 | 1.000 | 0.061 | 0.610 | 1.000 | 0.122 | 0.691 | 1.000 | 0.156 | 0.781 | 1.000 | 0.289 | 0.850 | 1.000 | 0.395 |
| 89 | 8 | `P` | 0.173 | 0.411 | 0.008 | 0.175 | 0.413 | 0.010 | 0.171 | 0.409 | 0.008 | 0.176 | 0.415 | 0.008 | 0.173 | 0.414 | 0.010 |
| 90 | 8 | `H` | 0.714 | 1.000 | 0.044 | 0.775 | 1.000 | 0.077 | 0.824 | 1.000 | 0.116 | 0.872 | 1.000 | 0.159 | 0.919 | 1.000 | 0.256 |
| 90 | 8 | `P` | 0.125 | 0.218 | 0.002 | 0.142 | 0.242 | 0.003 | 0.122 | 0.204 | 0.005 | 0.128 | 0.215 | 0.004 | 0.119 | 0.201 | 0.003 |
| 91 | 6 | `H` | 0.865 | 1.000 | 0.754 | 0.987 | 1.000 | 0.972 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 91 | 6 | `P` | 0.025 | 0.733 | 0.015 | 0.468 | 0.957 | 0.465 | 0.511 | 0.982 | 0.509 | 0.620 | 0.967 | 0.148 | 0.720 | 0.973 | 0.527 |
| 92 | 8 | `H` | 0.531 | 1.000 | 0.046 | 0.650 | 1.000 | 0.124 | 0.754 | 1.000 | 0.209 | 0.831 | 1.000 | 0.324 | 0.870 | 1.000 | 0.396 |
| 92 | 8 | `P` | 0.249 | 0.497 | 0.000 | 0.357 | 0.906 | 0.000 | 0.428 | 0.993 | 0.000 | 0.424 | 0.984 | 0.000 | 0.428 | 0.994 | 0.000 |
| 93 | 7 | `H` | 0.600 | 1.000 | 0.081 | 0.674 | 1.000 | 0.134 | 0.771 | 1.000 | 0.220 | 0.852 | 1.000 | 0.338 | 0.891 | 1.000 | 0.435 |
| 93 | 7 | `P` | 0.304 | 0.591 | 0.014 | 0.185 | 0.396 | 0.009 | 0.181 | 0.400 | 0.007 | 0.193 | 0.417 | 0.009 | 0.199 | 0.412 | 0.008 |
| 94 | 8 | `H` | 0.600 | 1.000 | 0.001 | 0.675 | 1.000 | 0.007 | 0.765 | 1.000 | 0.020 | 0.821 | 1.000 | 0.033 | 0.863 | 1.000 | 0.044 |
| 94 | 8 | `P` | 0.255 | 0.479 | 0.006 | 0.271 | 0.496 | 0.008 | 0.266 | 0.493 | 0.007 | 0.265 | 0.498 | 0.007 | 0.266 | 0.491 | 0.008 |
| 95 | 8 | `H` | 0.580 | 1.000 | 0.053 | 0.699 | 1.000 | 0.144 | 0.810 | 1.000 | 0.232 | 0.902 | 1.000 | 0.367 | 0.954 | 1.000 | 0.475 |
| 95 | 8 | `P` | 0.090 | 0.214 | 0.002 | 0.072 | 0.163 | 0.003 | 0.073 | 0.174 | 0.003 | 0.071 | 0.167 | 0.003 | 0.075 | 0.182 | 0.005 |
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
| 110 | 5 | `H` | 0.637 | 1.000 | 0.233 | 0.790 | 1.000 | 0.440 | 0.919 | 1.000 | 0.775 | 0.993 | 1.000 | 0.982 | 1.000 | 1.000 | 1.000 |
| 110 | 5 | `P` | 0.217 | 0.443 | 0.032 | 0.238 | 0.440 | 0.042 | 0.261 | 0.462 | 0.050 | 0.262 | 0.487 | 0.014 | 0.164 | 0.346 | 0.014 |
| 111 | 5 | `H` | 0.906 | 1.000 | 0.497 | 0.956 | 1.000 | 0.692 | 0.992 | 1.000 | 0.948 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 111 | 5 | `P` | 0.107 | 0.314 | 0.007 | 0.174 | 0.391 | 0.028 | 0.080 | 0.295 | 0.004 | 0.130 | 0.255 | 0.025 | 0.086 | 0.184 | 0.034 |
| 112 | 6 | `H` | 0.916 | 1.000 | 0.743 | 0.979 | 1.000 | 0.930 | 0.998 | 1.000 | 0.995 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 112 | 6 | `P` | 0.120 | 0.231 | 0.036 | 0.152 | 0.255 | 0.051 | 0.128 | 0.221 | 0.041 | 0.093 | 0.172 | 0.023 | 0.118 | 0.202 | 0.042 |
| 113 | 8 | `H` | 0.663 | 1.000 | 0.033 | 0.798 | 1.000 | 0.095 | 0.854 | 1.000 | 0.162 | 0.892 | 1.000 | 0.190 | 0.925 | 1.000 | 0.294 |
| 113 | 8 | `P` | 0.227 | 0.400 | 0.010 | 0.156 | 0.286 | 0.006 | 0.150 | 0.270 | 0.007 | 0.159 | 0.286 | 0.009 | 0.156 | 0.280 | 0.008 |
| 114 | 6 | `H` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 114 | 6 | `P` | 0.017 | 0.040 | 0.017 | 0.024 | 0.055 | 0.024 | 0.036 | 0.067 | 0.036 | 0.018 | 0.034 | 0.018 | 0.015 | 0.032 | 0.015 |
| 115 | 7 | `H` | 0.844 | 1.000 | 0.237 | 0.924 | 1.000 | 0.593 | 0.986 | 1.000 | 0.926 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 115 | 7 | `P` | 0.072 | 0.129 | 0.015 | 0.104 | 0.168 | 0.049 | 0.145 | 0.235 | 0.021 | 0.119 | 0.304 | 0.037 | 0.082 | 0.142 | 0.003 |
| 116 | 7 | `H` | 0.922 | 1.000 | 0.780 | 0.991 | 1.000 | 0.986 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 116 | 7 | `P` | 0.275 | 0.359 | 0.167 | 0.283 | 0.913 | 0.002 | 0.751 | 0.933 | 0.444 | 0.277 | 0.783 | 0.004 | 0.510 | 0.905 | 0.196 |
| 117 | 8 | `H` | 0.558 | 1.000 | 0.040 | 0.623 | 1.000 | 0.085 | 0.700 | 1.000 | 0.135 | 0.779 | 1.000 | 0.206 | 0.848 | 1.000 | 0.259 |
| 117 | 8 | `P` | 0.202 | 0.440 | 0.001 | 0.220 | 0.439 | 0.015 | 0.202 | 0.442 | 0.006 | 0.201 | 0.439 | 0.008 | 0.211 | 0.445 | 0.009 |
| 118 | 8 | `H` | 0.978 | 1.000 | 0.922 | 1.000 | 1.000 | 0.999 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 118 | 8 | `P` | 0.179 | 0.268 | 0.113 | 0.342 | 0.797 | 0.000 | 0.441 | 0.932 | 0.020 | 0.561 | 0.932 | 0.000 | 0.405 | 0.920 | 0.001 |
| 119 | 8 | `H` | 0.388 | 1.000 | 0.050 | 0.503 | 1.000 | 0.097 | 0.626 | 1.000 | 0.187 | 0.743 | 1.000 | 0.285 | 0.819 | 1.000 | 0.409 |
| 119 | 8 | `P` | 0.072 | 0.236 | 0.005 | 0.072 | 0.239 | 0.003 | 0.073 | 0.236 | 0.006 | 0.071 | 0.235 | 0.003 | 0.072 | 0.239 | 0.003 |
| 120 | 6 | `H` | 0.900 | 1.000 | 0.529 | 0.963 | 1.000 | 0.825 | 0.996 | 1.000 | 0.983 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 120 | 6 | `P` | 0.221 | 0.390 | 0.036 | 0.280 | 0.547 | 0.006 | 0.364 | 0.639 | 0.069 | 0.417 | 0.669 | 0.069 | 0.338 | 0.707 | 0.037 |
| 121 | 8 | `H` | 0.667 | 1.000 | 0.002 | 0.747 | 1.000 | 0.011 | 0.836 | 1.000 | 0.077 | 0.883 | 1.000 | 0.229 | 0.930 | 1.000 | 0.383 |
| 121 | 8 | `P` | 0.046 | 0.106 | 0.005 | 0.037 | 0.088 | 0.003 | 0.037 | 0.082 | 0.003 | 0.039 | 0.094 | 0.003 | 0.038 | 0.092 | 0.003 |
| 122 | 7 | `H` | 0.935 | 1.000 | 0.429 | 0.972 | 1.000 | 0.576 | 0.982 | 1.000 | 0.695 | 0.999 | 1.000 | 0.977 | 1.000 | 1.000 | 1.000 |
| 122 | 7 | `P` | 0.225 | 0.554 | 0.145 | 0.632 | 0.837 | 0.392 | 0.287 | 0.873 | 0.031 | 0.467 | 0.917 | 0.430 | 0.445 | 0.929 | 0.358 |
| 123 | 7 | `H` | 0.590 | 1.000 | 0.020 | 0.668 | 1.000 | 0.043 | 0.736 | 1.000 | 0.064 | 0.788 | 1.000 | 0.066 | 0.830 | 1.000 | 0.055 |
| 123 | 7 | `P` | 0.384 | 0.761 | 0.006 | 0.382 | 0.766 | 0.009 | 0.384 | 0.769 | 0.007 | 0.379 | 0.765 | 0.008 | 0.383 | 0.769 | 0.006 |
| 124 | 8 | `H` | 0.528 | 1.000 | 0.018 | 0.624 | 1.000 | 0.043 | 0.692 | 1.000 | 0.057 | 0.746 | 1.000 | 0.086 | 0.780 | 1.000 | 0.129 |
| 124 | 8 | `P` | 0.133 | 0.312 | 0.001 | 0.152 | 0.336 | 0.002 | 0.150 | 0.336 | 0.001 | 0.129 | 0.294 | 0.003 | 0.137 | 0.308 | 0.004 |
| 125 | 8 | `H` | 0.559 | 1.000 | 0.005 | 0.623 | 1.000 | 0.013 | 0.689 | 1.000 | 0.032 | 0.743 | 1.000 | 0.045 | 0.798 | 1.000 | 0.092 |
| 125 | 8 | `P` | 0.456 | 0.773 | 0.009 | 0.442 | 0.832 | 0.002 | 0.596 | 0.963 | 0.000 | 0.615 | 0.982 | 0.000 | 0.565 | 0.994 | 0.055 |
| 126 | 8 | `H` | 0.486 | 1.000 | 0.011 | 0.563 | 1.000 | 0.024 | 0.634 | 1.000 | 0.052 | 0.698 | 1.000 | 0.086 | 0.748 | 1.000 | 0.113 |
| 126 | 8 | `P` | 0.281 | 0.607 | 0.007 | 0.286 | 0.604 | 0.007 | 0.296 | 0.619 | 0.008 | 0.284 | 0.602 | 0.009 | 0.280 | 0.599 | 0.007 |
| 127 | 8 | `H` | 0.570 | 1.000 | 0.054 | 0.696 | 1.000 | 0.105 | 0.773 | 1.000 | 0.147 | 0.827 | 1.000 | 0.211 | 0.878 | 1.000 | 0.299 |
| 127 | 8 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 128 | 8 | `H` | 0.482 | 1.000 | 0.005 | 0.526 | 1.000 | 0.008 | 0.563 | 1.000 | 0.016 | 0.610 | 1.000 | 0.025 | 0.665 | 1.000 | 0.044 |
| 128 | 8 | `P` | 0.341 | 0.764 | 0.007 | 0.335 | 0.742 | 0.003 | 0.335 | 0.753 | 0.004 | 0.336 | 0.748 | 0.003 | 0.337 | 0.748 | 0.004 |
| 129 | 8 | `H` | 0.705 | 1.000 | 0.203 | 0.844 | 1.000 | 0.405 | 0.936 | 1.000 | 0.592 | 0.963 | 1.000 | 0.783 | 0.996 | 1.000 | 0.987 |
| 129 | 8 | `P` | 0.093 | 0.178 | 0.013 | 0.050 | 0.101 | 0.013 | 0.032 | 0.071 | 0.004 | 0.028 | 0.064 | 0.005 | 0.032 | 0.070 | 0.004 |
| 130 | 5 | `H` | 0.586 | 1.000 | 0.094 | 0.805 | 1.000 | 0.265 | 0.868 | 1.000 | 0.361 | 0.920 | 1.000 | 0.613 | 0.988 | 1.000 | 0.938 |
| 130 | 5 | `P` | 0.516 | 0.980 | 0.141 | 0.751 | 0.978 | 0.437 | 0.915 | 0.999 | 0.784 | 0.827 | 1.000 | 0.539 | 0.958 | 0.977 | 0.931 |
| 131 | 7 | `H` | 0.654 | 1.000 | 0.157 | 0.790 | 1.000 | 0.289 | 0.867 | 1.000 | 0.401 | 0.932 | 1.000 | 0.618 | 0.973 | 1.000 | 0.753 |
| 131 | 7 | `P` | 0.170 | 0.340 | 0.004 | 0.147 | 0.280 | 0.019 | 0.141 | 0.284 | 0.011 | 0.151 | 0.290 | 0.014 | 0.135 | 0.272 | 0.011 |
| 132 | 6 | `H` | 0.560 | 1.000 | 0.019 | 0.664 | 1.000 | 0.046 | 0.745 | 1.000 | 0.089 | 0.803 | 1.000 | 0.130 | 0.840 | 1.000 | 0.180 |
| 132 | 6 | `P` | 0.561 | 0.946 | 0.023 | 0.513 | 0.942 | 0.025 | 0.496 | 0.943 | 0.016 | 0.504 | 0.946 | 0.018 | 0.510 | 0.941 | 0.022 |
| 133 | 7 | `H` | 0.493 | 1.000 | 0.019 | 0.591 | 1.000 | 0.026 | 0.647 | 1.000 | 0.051 | 0.699 | 1.000 | 0.081 | 0.736 | 1.000 | 0.112 |
| 133 | 7 | `P` | 0.415 | 0.933 | 0.011 | 0.415 | 0.942 | 0.009 | 0.409 | 0.941 | 0.007 | 0.420 | 0.947 | 0.008 | 0.416 | 0.938 | 0.008 |
| 134 | 7 | `H` | 0.686 | 1.000 | 0.136 | 0.769 | 1.000 | 0.203 | 0.825 | 1.000 | 0.315 | 0.878 | 1.000 | 0.477 | 0.948 | 1.000 | 0.829 |
| 134 | 7 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 135 | 7 | `H` | 0.550 | 1.000 | 0.053 | 0.640 | 1.000 | 0.051 | 0.762 | 1.000 | 0.119 | 0.862 | 1.000 | 0.211 | 0.917 | 1.000 | 0.302 |
| 135 | 7 | `P` | 0.194 | 0.457 | 0.016 | 0.179 | 0.448 | 0.010 | 0.160 | 0.386 | 0.007 | 0.145 | 0.353 | 0.007 | 0.143 | 0.350 | 0.009 |
| 136 | 7 | `H` | 0.471 | 1.000 | 0.016 | 0.549 | 1.000 | 0.036 | 0.621 | 1.000 | 0.060 | 0.691 | 1.000 | 0.102 | 0.768 | 1.000 | 0.163 |
| 136 | 7 | `P` | 0.218 | 0.500 | 0.006 | 0.227 | 0.506 | 0.009 | 0.233 | 0.520 | 0.009 | 0.229 | 0.510 | 0.009 | 0.216 | 0.492 | 0.007 |
| 137 | 6 | `H` | 0.672 | 1.000 | 0.192 | 0.813 | 1.000 | 0.354 | 0.914 | 1.000 | 0.470 | 0.962 | 1.000 | 0.662 | 0.982 | 1.000 | 0.849 |
| 137 | 6 | `P` | 0.348 | 0.598 | 0.058 | 0.228 | 0.459 | 0.029 | 0.217 | 0.430 | 0.038 | 0.230 | 0.435 | 0.035 | 0.241 | 0.451 | 0.048 |
| 138 | 7 | `H` | 0.477 | 1.000 | 0.031 | 0.618 | 1.000 | 0.073 | 0.738 | 1.000 | 0.108 | 0.859 | 1.000 | 0.171 | 0.909 | 1.000 | 0.227 |
| 138 | 7 | `P` | 0.461 | 0.848 | 0.012 | 0.439 | 0.982 | 0.000 | 0.536 | 0.993 | 0.123 | 0.545 | 0.997 | 0.180 | 0.447 | 0.992 | 0.000 |
| 139 | 7 | `H` | 0.540 | 1.000 | 0.016 | 0.605 | 1.000 | 0.029 | 0.700 | 1.000 | 0.058 | 0.769 | 1.000 | 0.098 | 0.816 | 1.000 | 0.151 |
| 139 | 7 | `P` | 0.307 | 0.678 | 0.008 | 0.310 | 0.678 | 0.007 | 0.318 | 0.693 | 0.007 | 0.313 | 0.693 | 0.007 | 0.309 | 0.682 | 0.008 |
| 140 | 6 | `H` | 0.617 | 1.000 | 0.217 | 0.731 | 1.000 | 0.331 | 0.846 | 1.000 | 0.528 | 0.914 | 1.000 | 0.725 | 0.987 | 1.000 | 0.963 |
| 140 | 6 | `P` | 0.100 | 0.271 | 0.018 | 0.107 | 0.288 | 0.012 | 0.101 | 0.238 | 0.029 | 0.086 | 0.223 | 0.011 | 0.074 | 0.191 | 0.011 |
| 141 | 7 | `H` | 0.603 | 1.000 | 0.129 | 0.722 | 1.000 | 0.219 | 0.814 | 1.000 | 0.298 | 0.865 | 1.000 | 0.466 | 0.914 | 1.000 | 0.685 |
| 141 | 7 | `P` | 0.086 | 0.191 | 0.005 | 0.085 | 0.189 | 0.006 | 0.085 | 0.187 | 0.007 | 0.091 | 0.196 | 0.009 | 0.086 | 0.188 | 0.008 |
| 142 | 7 | `H` | 0.509 | 1.000 | 0.014 | 0.588 | 1.000 | 0.033 | 0.682 | 1.000 | 0.060 | 0.736 | 1.000 | 0.094 | 0.776 | 1.000 | 0.116 |
| 142 | 7 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 143 | 5 | `H` | 0.685 | 1.000 | 0.461 | 0.942 | 1.000 | 0.869 | 0.994 | 1.000 | 0.989 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 143 | 5 | `P` | 0.120 | 0.262 | 0.019 | 0.132 | 0.281 | 0.020 | 0.122 | 0.231 | 0.043 | 0.130 | 0.285 | 0.033 | 0.144 | 0.229 | 0.088 |
| 144 | 6 | `H` | 0.569 | 1.000 | 0.119 | 0.727 | 1.000 | 0.283 | 0.816 | 1.000 | 0.426 | 0.908 | 1.000 | 0.684 | 0.982 | 1.000 | 0.958 |
| 144 | 6 | `P` | 0.106 | 0.285 | 0.009 | 0.098 | 0.250 | 0.014 | 0.088 | 0.248 | 0.008 | 0.097 | 0.234 | 0.017 | 0.102 | 0.254 | 0.013 |
| 145 | 6 | `H` | 0.834 | 1.000 | 0.299 | 0.905 | 1.000 | 0.608 | 0.977 | 1.000 | 0.908 | 1.000 | 1.000 | 0.998 | 1.000 | 1.000 | 1.000 |
| 145 | 6 | `P` | 0.142 | 0.280 | 0.029 | 0.075 | 0.150 | 0.012 | 0.082 | 0.190 | 0.012 | 0.108 | 0.203 | 0.019 | 0.112 | 0.209 | 0.016 |
| 146 | 7 | `H` | 0.999 | 1.000 | 0.999 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 146 | 7 | `P` | 0.305 | 0.598 | 0.305 | 0.508 | 0.908 | 0.508 | 0.776 | 0.984 | 0.776 | 0.771 | 0.986 | 0.771 | 0.984 | 0.990 | 0.984 |
| 147 | 6 | `H` | 0.638 | 1.000 | 0.095 | 0.765 | 1.000 | 0.174 | 0.859 | 1.000 | 0.242 | 0.941 | 1.000 | 0.422 | 0.978 | 1.000 | 0.556 |
| 147 | 6 | `P` | 0.285 | 0.593 | 0.008 | 0.268 | 0.498 | 0.015 | 0.241 | 0.475 | 0.015 | 0.207 | 0.443 | 0.014 | 0.193 | 0.410 | 0.016 |
| 148 | 8 | `H` | 0.588 | 1.000 | 0.026 | 0.711 | 1.000 | 0.050 | 0.765 | 1.000 | 0.080 | 0.794 | 1.000 | 0.104 | 0.806 | 1.000 | 0.152 |
| 148 | 8 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 149 | 7 | `H` | 0.698 | 1.000 | 0.139 | 0.793 | 1.000 | 0.230 | 0.865 | 1.000 | 0.311 | 0.915 | 1.000 | 0.441 | 0.942 | 1.000 | 0.594 |
| 149 | 7 | `P` | 0.183 | 0.356 | 0.017 | 0.155 | 0.308 | 0.004 | 0.174 | 0.331 | 0.009 | 0.175 | 0.319 | 0.012 | 0.158 | 0.289 | 0.008 |
| 150 | 7 | `H` | 0.561 | 1.000 | 0.060 | 0.656 | 1.000 | 0.097 | 0.770 | 1.000 | 0.176 | 0.845 | 1.000 | 0.254 | 0.894 | 1.000 | 0.325 |
| 150 | 7 | `P` | 0.206 | 0.492 | 0.007 | 0.210 | 0.502 | 0.007 | 0.226 | 0.532 | 0.007 | 0.207 | 0.478 | 0.008 | 0.204 | 0.480 | 0.008 |
| 151 | 7 | `H` | 0.488 | 1.000 | 0.028 | 0.603 | 1.000 | 0.066 | 0.697 | 1.000 | 0.096 | 0.740 | 1.000 | 0.131 | 0.781 | 1.000 | 0.172 |
| 151 | 7 | `P` | 0.280 | 0.671 | 0.003 | 0.286 | 0.659 | 0.014 | 0.266 | 0.615 | 0.011 | 0.269 | 0.622 | 0.008 | 0.265 | 0.615 | 0.005 |
| 152 | 8 | `H` | 0.490 | 1.000 | 0.000 | 0.532 | 1.000 | 0.000 | 0.579 | 1.000 | 0.000 | 0.618 | 1.000 | 0.000 | 0.649 | 1.000 | 0.000 |
| 152 | 8 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 153 | 7 | `H` | 0.488 | 1.000 | 0.023 | 0.583 | 1.000 | 0.039 | 0.647 | 1.000 | 0.055 | 0.717 | 1.000 | 0.116 | 0.757 | 1.000 | 0.151 |
| 153 | 7 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 154 | 6 | `H` | 0.625 | 1.000 | 0.177 | 0.789 | 1.000 | 0.327 | 0.912 | 1.000 | 0.528 | 0.967 | 1.000 | 0.774 | 0.992 | 1.000 | 0.960 |
| 154 | 6 | `P` | 0.166 | 0.384 | 0.003 | 0.140 | 0.366 | 0.019 | 0.126 | 0.316 | 0.013 | 0.125 | 0.328 | 0.012 | 0.130 | 0.306 | 0.015 |
| 155 | 8 | `H` | 0.551 | 1.000 | 0.039 | 0.652 | 1.000 | 0.058 | 0.716 | 1.000 | 0.095 | 0.772 | 1.000 | 0.153 | 0.835 | 1.000 | 0.260 |
| 155 | 8 | `P` | 0.150 | 0.387 | 0.000 | 0.133 | 0.282 | 0.007 | 0.124 | 0.250 | 0.007 | 0.122 | 0.258 | 0.009 | 0.121 | 0.259 | 0.006 |
| 156 | 6 | `H` |  | 1.000 | 1.000 |  | 1.000 | 1.000 |  | 1.000 | 1.000 |  | 1.000 | 1.000 |  | 1.000 | 1.000 |
| 156 | 6 | `P` |  | 0.810 | 0.810 |  | 0.974 | 0.974 |  | 0.998 | 0.998 |  | 0.999 | 0.999 |  | 1.000 | 1.000 |
| 157 | 7 | `H` | 0.566 | 1.000 | 0.034 | 0.630 | 1.000 | 0.064 | 0.704 | 1.000 | 0.087 | 0.779 | 1.000 | 0.126 | 0.826 | 1.000 | 0.177 |
| 157 | 7 | `P` | 0.400 | 0.826 | 0.004 | 0.403 | 0.813 | 0.006 | 0.408 | 0.830 | 0.006 | 0.394 | 0.810 | 0.007 | 0.397 | 0.813 | 0.006 |
| 158 | 8 | `H` | 0.636 | 1.000 | 0.059 | 0.782 | 1.000 | 0.145 | 0.899 | 1.000 | 0.235 | 0.942 | 1.000 | 0.343 | 0.967 | 1.000 | 0.394 |
| 158 | 8 | `P` | 0.079 | 0.156 | 0.004 | 0.087 | 0.169 | 0.003 | 0.081 | 0.158 | 0.004 | 0.080 | 0.163 | 0.003 | 0.082 | 0.167 | 0.003 |
| 159 | 5 | `H` | 0.976 | 1.000 | 0.957 | 0.999 | 1.000 | 0.998 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 159 | 5 | `P` | 0.175 | 0.322 | 0.050 | 0.160 | 0.297 | 0.085 | 0.116 | 0.229 | 0.064 | 0.064 | 0.127 | 0.033 | 0.093 | 0.168 | 0.065 |
| 160 | 7 | `H` | 0.996 | 1.000 | 0.996 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 160 | 7 | `P` | 0.208 | 0.397 | 0.208 | 0.479 | 0.948 | 0.479 | 0.529 | 0.983 | 0.529 | 0.515 | 0.994 | 0.515 | 0.558 | 0.996 | 0.558 |
| 161 | 7 | `H` | 0.612 | 1.000 | 0.028 | 0.684 | 1.000 | 0.058 | 0.785 | 1.000 | 0.114 | 0.854 | 1.000 | 0.162 | 0.914 | 1.000 | 0.222 |
| 161 | 7 | `P` | 0.292 | 0.500 | 0.006 | 0.313 | 0.509 | 0.053 | 0.279 | 0.500 | 0.019 | 0.268 | 0.498 | 0.005 | 0.274 | 0.505 | 0.010 |
| 162 | 8 | `H` | 0.647 | 1.000 | 0.006 | 0.744 | 1.000 | 0.023 | 0.833 | 1.000 | 0.043 | 0.869 | 1.000 | 0.062 | 0.909 | 1.000 | 0.089 |
| 162 | 8 | `P` | 0.066 | 0.129 | 0.004 | 0.069 | 0.140 | 0.003 | 0.069 | 0.134 | 0.005 | 0.068 | 0.134 | 0.006 | 0.063 | 0.130 | 0.003 |
| 163 | 7 | `H` | 0.557 | 1.000 | 0.153 | 0.724 | 1.000 | 0.337 | 0.858 | 1.000 | 0.662 | 0.971 | 1.000 | 0.933 | 1.000 | 1.000 | 0.999 |
| 163 | 7 | `P` | 0.086 | 0.175 | 0.006 | 0.259 | 0.365 | 0.129 | 0.361 | 0.724 | 0.001 | 0.220 | 0.882 | 0.000 | 0.383 | 0.932 | 0.004 |
| 164 | 8 | `H` | 0.690 | 1.000 | 0.129 | 0.826 | 1.000 | 0.274 | 0.927 | 1.000 | 0.439 | 0.971 | 1.000 | 0.602 | 0.993 | 1.000 | 0.848 |
| 164 | 8 | `P` | 0.126 | 0.208 | 0.038 | 0.091 | 0.151 | 0.008 | 0.063 | 0.107 | 0.007 | 0.112 | 0.187 | 0.015 | 0.228 | 0.335 | 0.034 |
| 165 | 8 | `H` | 0.679 | 1.000 | 0.127 | 0.780 | 1.000 | 0.200 | 0.852 | 1.000 | 0.368 | 0.925 | 1.000 | 0.672 | 0.977 | 1.000 | 0.901 |
| 165 | 8 | `P` | 0.164 | 0.248 | 0.002 | 0.071 | 0.129 | 0.004 | 0.065 | 0.113 | 0.001 | 0.064 | 0.124 | 0.007 | 0.063 | 0.117 | 0.005 |
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
| 173 | 8 | `H` | 0.533 | 1.000 | 0.009 | 0.625 | 1.000 | 0.005 | 0.700 | 1.000 | 0.000 | 0.759 | 1.000 | 0.003 | 0.790 | 1.000 | 0.004 |
| 173 | 8 | `P` | 0.090 | 0.196 | 0.003 | 0.087 | 0.194 | 0.005 | 0.090 | 0.199 | 0.003 | 0.086 | 0.190 | 0.003 | 0.090 | 0.195 | 0.005 |
| 174 | 8 | `H` | 0.946 | 1.000 | 0.839 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 174 | 8 | `P` | 0.181 | 0.288 | 0.081 | 0.737 | 0.846 | 0.424 | 0.680 | 0.898 | 0.048 | 0.883 | 0.946 | 0.730 | 0.727 | 0.977 | 0.009 |
| 175 | 8 | `H` | 0.638 | 1.000 | 0.057 | 0.767 | 1.000 | 0.146 | 0.870 | 1.000 | 0.237 | 0.910 | 1.000 | 0.300 | 0.936 | 1.000 | 0.420 |
| 175 | 8 | `P` | 0.075 | 0.134 | 0.001 | 0.122 | 0.228 | 0.006 | 0.089 | 0.174 | 0.004 | 0.094 | 0.174 | 0.005 | 0.090 | 0.175 | 0.003 |
| 176 | 7 | `H` | 0.848 | 1.000 | 0.631 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 176 | 7 | `P` | 0.215 | 0.432 | 0.001 | 0.413 | 0.827 | 0.001 | 0.454 | 0.924 | 0.003 | 0.474 | 0.953 | 0.000 | 0.722 | 0.884 | 0.424 |
| 177 | 6 | `H` | 0.466 | 1.000 | 0.080 | 0.620 | 1.000 | 0.162 | 0.725 | 1.000 | 0.257 | 0.833 | 1.000 | 0.436 | 0.927 | 1.000 | 0.719 |
| 177 | 6 | `P` | 0.189 | 0.470 | 0.009 | 0.183 | 0.498 | 0.018 | 0.164 | 0.464 | 0.014 | 0.175 | 0.469 | 0.014 | 0.180 | 0.485 | 0.020 |
| 178 | 6 | `H` | 0.556 | 1.000 | 0.124 | 0.752 | 1.000 | 0.275 | 0.830 | 1.000 | 0.427 | 0.919 | 1.000 | 0.706 | 0.974 | 1.000 | 0.909 |
| 178 | 6 | `P` | 0.158 | 0.437 | 0.002 | 0.319 | 0.598 | 0.078 | 0.382 | 0.713 | 0.123 | 0.358 | 0.737 | 0.057 | 0.391 | 0.740 | 0.090 |
| 179 | 8 | `H` | 0.541 | 1.000 | 0.032 | 0.638 | 1.000 | 0.060 | 0.757 | 1.000 | 0.104 | 0.824 | 1.000 | 0.137 | 0.845 | 1.000 | 0.189 |
| 179 | 8 | `P` | 0.039 | 0.095 | 0.003 | 0.040 | 0.096 | 0.004 | 0.039 | 0.090 | 0.005 | 0.043 | 0.100 | 0.004 | 0.041 | 0.097 | 0.004 |
| 180 | 8 | `H` | 0.319 | 1.000 | 0.007 | 0.377 | 1.000 | 0.015 | 0.447 | 1.000 | 0.019 | 0.514 | 1.000 | 0.038 | 0.558 | 1.000 | 0.063 |
| 180 | 8 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 181 | 8 | `H` | 0.983 | 1.000 | 0.964 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 181 | 8 | `P` | 0.192 | 0.227 | 0.116 | 0.745 | 0.867 | 0.439 | 0.844 | 0.984 | 0.494 | 0.822 | 0.955 | 0.491 | 0.499 | 0.963 | 0.499 |
| 182 | 8 | `H` | 0.601 | 1.000 | 0.056 | 0.684 | 1.000 | 0.108 | 0.784 | 1.000 | 0.189 | 0.853 | 1.000 | 0.298 | 0.896 | 1.000 | 0.360 |
| 182 | 8 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 183 | 8 | `H` | 0.630 | 1.000 | 0.049 | 0.712 | 1.000 | 0.088 | 0.783 | 1.000 | 0.141 | 0.839 | 1.000 | 0.201 | 0.876 | 1.000 | 0.257 |
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
| 192 | 6 | `H` | 0.515 | 1.000 | 0.038 | 0.650 | 1.000 | 0.091 | 0.752 | 1.000 | 0.158 | 0.839 | 1.000 | 0.207 | 0.900 | 1.000 | 0.276 |
| 192 | 6 | `P` | 0.335 | 0.779 | 0.015 | 0.312 | 0.754 | 0.017 | 0.321 | 0.769 | 0.018 | 0.322 | 0.771 | 0.018 | 0.320 | 0.765 | 0.017 |
| 193 | 8 | `H` | 0.701 | 1.000 | 0.150 | 0.813 | 1.000 | 0.276 | 0.923 | 1.000 | 0.448 | 0.972 | 1.000 | 0.530 | 0.975 | 1.000 | 0.559 |
| 193 | 8 | `P` | 0.049 | 0.102 | 0.002 | 0.049 | 0.098 | 0.005 | 0.047 | 0.094 | 0.007 | 0.044 | 0.093 | 0.003 | 0.049 | 0.097 | 0.005 |
| 194 | 5 | `H` | 0.956 | 1.000 | 0.948 | 0.999 | 1.000 | 0.999 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 194 | 5 | `P` | 0.139 | 0.359 | 0.104 | 0.109 | 0.337 | 0.076 | 0.037 | 0.144 | 0.017 | 0.075 | 0.202 | 0.053 | 0.024 | 0.076 | 0.017 |
| 195 | 8 | `H` | 0.590 | 1.000 | 0.048 | 0.784 | 1.000 | 0.155 | 0.898 | 1.000 | 0.301 | 0.922 | 1.000 | 0.472 | 0.957 | 1.000 | 0.746 |
| 195 | 8 | `P` | 0.038 | 0.113 | 0.012 | 0.071 | 0.180 | 0.032 | 0.016 | 0.048 | 0.002 | 0.011 | 0.029 | 0.002 | 0.025 | 0.069 | 0.005 |
| 196 | 8 | `H` | 0.536 | 1.000 | 0.011 | 0.640 | 1.000 | 0.027 | 0.702 | 1.000 | 0.048 | 0.754 | 1.000 | 0.059 | 0.801 | 1.000 | 0.091 |
| 196 | 8 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 197 | 6 | `H` | 0.974 | 1.000 | 0.930 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 197 | 6 | `P` | 0.035 | 0.130 | 0.013 | 0.093 | 0.146 | 0.066 | 0.066 | 0.098 | 0.051 | 0.024 | 0.063 | 0.012 | 0.026 | 0.058 | 0.008 |
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
| 205 | 6 | `H` | 0.597 | 1.000 | 0.105 | 0.745 | 1.000 | 0.207 | 0.843 | 1.000 | 0.345 | 0.894 | 1.000 | 0.494 | 0.937 | 1.000 | 0.645 |
| 205 | 6 | `P` | 0.206 | 0.454 | 0.013 | 0.217 | 0.479 | 0.030 | 0.217 | 0.466 | 0.011 | 0.216 | 0.483 | 0.020 | 0.218 | 0.480 | 0.016 |
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
| 211 | 7 | `H` | 0.977 | 1.000 | 0.901 | 0.997 | 1.000 | 0.989 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 211 | 7 | `P` | 0.071 | 0.123 | 0.027 | 0.067 | 0.103 | 0.037 | 0.046 | 0.076 | 0.023 | 0.031 | 0.058 | 0.014 | 0.028 | 0.054 | 0.011 |
| 212 | 5 | `H` | 0.637 | 1.000 | 0.206 | 0.852 | 1.000 | 0.388 | 0.917 | 1.000 | 0.604 | 0.990 | 1.000 | 0.966 | 1.000 | 1.000 | 1.000 |
| 212 | 5 | `P` | 0.200 | 0.800 | 0.000 | 0.270 | 0.951 | 0.004 | 0.226 | 0.973 | 0.003 | 0.154 | 0.962 | 0.000 | 0.208 | 0.977 | 0.001 |
| 213 | 6 | `H` | 0.761 | 1.000 | 0.514 | 0.913 | 1.000 | 0.837 | 0.996 | 1.000 | 0.995 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 213 | 6 | `P` | 0.083 | 0.215 | 0.048 | 0.133 | 0.318 | 0.076 | 0.110 | 0.283 | 0.016 | 0.078 | 0.279 | 0.034 | 0.057 | 0.113 | 0.032 |
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
| 223 | 6 | `H` | 0.481 | 1.000 | 0.051 | 0.655 | 1.000 | 0.086 | 0.785 | 1.000 | 0.186 | 0.881 | 1.000 | 0.259 | 0.949 | 1.000 | 0.349 |
| 223 | 6 | `P` | 0.164 | 0.426 | 0.021 | 0.180 | 0.454 | 0.030 | 0.182 | 0.452 | 0.019 | 0.156 | 0.415 | 0.013 | 0.155 | 0.406 | 0.013 |
| 224 | 5 | `H` | 0.843 | 1.000 | 0.741 | 0.998 | 1.000 | 0.995 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 224 | 5 | `P` | 0.118 | 0.278 | 0.045 | 0.156 | 0.259 | 0.085 | 0.081 | 0.159 | 0.042 | 0.128 | 0.205 | 0.070 | 0.152 | 0.235 | 0.105 |
| 225 | 6 | `H` | 0.534 | 1.000 | 0.062 | 0.667 | 1.000 | 0.075 | 0.771 | 1.000 | 0.133 | 0.848 | 1.000 | 0.149 | 0.902 | 1.000 | 0.220 |
| 225 | 6 | `P` | 0.345 | 0.749 | 0.032 | 0.340 | 0.752 | 0.026 | 0.362 | 0.743 | 0.065 | 0.325 | 0.747 | 0.031 | 0.354 | 0.796 | 0.020 |
| 226 | 7 | `H` | 0.619 | 1.000 | 0.141 | 0.755 | 1.000 | 0.259 | 0.864 | 1.000 | 0.419 | 0.935 | 1.000 | 0.683 | 0.982 | 1.000 | 0.904 |
| 226 | 7 | `P` | 0.120 | 0.343 | 0.011 | 0.182 | 0.355 | 0.029 | 0.125 | 0.251 | 0.007 | 0.107 | 0.182 | 0.017 | 0.140 | 0.304 | 0.017 |
| 227 | 8 | `H` | 0.853 | 1.000 | 0.404 | 0.958 | 1.000 | 0.853 | 1.000 | 1.000 | 0.998 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 227 | 8 | `P` | 0.024 | 0.053 | 0.005 | 0.040 | 0.080 | 0.003 | 0.076 | 0.135 | 0.006 | 0.047 | 0.078 | 0.013 | 0.020 | 0.037 | 0.005 |
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
| 233 | 8 | `H` | 0.624 | 1.000 | 0.047 | 0.732 | 1.000 | 0.084 | 0.836 | 1.000 | 0.151 | 0.906 | 1.000 | 0.227 | 0.934 | 1.000 | 0.285 |
| 233 | 8 | `P` | 0.192 | 0.327 | 0.004 | 0.094 | 0.208 | 0.002 | 0.106 | 0.209 | 0.003 | 0.108 | 0.217 | 0.004 | 0.095 | 0.198 | 0.004 |
| 234 | 8 | `H` | 0.636 | 1.000 | 0.062 | 0.723 | 1.000 | 0.107 | 0.806 | 1.000 | 0.189 | 0.857 | 1.000 | 0.258 | 0.902 | 1.000 | 0.360 |
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
| 241 | 7 | `H` | 0.538 | 1.000 | 0.050 | 0.654 | 1.000 | 0.123 | 0.773 | 1.000 | 0.189 | 0.831 | 1.000 | 0.303 | 0.890 | 1.000 | 0.408 |
| 241 | 7 | `P` | 0.088 | 0.234 | 0.008 | 0.082 | 0.212 | 0.009 | 0.087 | 0.236 | 0.007 | 0.084 | 0.222 | 0.008 | 0.088 | 0.240 | 0.009 |
| 242 | 7 | `H` | 0.564 | 1.000 | 0.031 | 0.639 | 1.000 | 0.050 | 0.699 | 1.000 | 0.077 | 0.756 | 1.000 | 0.125 | 0.812 | 1.000 | 0.192 |
| 242 | 7 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 243 | 8 | `H` | 0.435 | 1.000 | 0.014 | 0.549 | 1.000 | 0.036 | 0.640 | 1.000 | 0.046 | 0.699 | 1.000 | 0.077 | 0.734 | 1.000 | 0.108 |
| 243 | 8 | `P` | 0.087 | 0.243 | 0.004 | 0.091 | 0.259 | 0.004 | 0.089 | 0.246 | 0.005 | 0.088 | 0.247 | 0.003 | 0.087 | 0.249 | 0.003 |
| 244 | 8 | `H` | 0.531 | 1.000 | 0.020 | 0.624 | 1.000 | 0.046 | 0.716 | 1.000 | 0.074 | 0.796 | 1.000 | 0.138 | 0.828 | 1.000 | 0.184 |
| 244 | 8 | `P` | 0.139 | 0.275 | 0.005 | 0.159 | 0.339 | 0.004 | 0.140 | 0.295 | 0.003 | 0.133 | 0.281 | 0.004 | 0.134 | 0.285 | 0.002 |
| 245 | 5 | `H` | 0.643 | 1.000 | 0.215 | 0.803 | 1.000 | 0.363 | 0.918 | 1.000 | 0.616 | 0.966 | 1.000 | 0.825 | 0.993 | 1.000 | 0.966 |
| 245 | 5 | `P` | 0.268 | 0.585 | 0.006 | 0.211 | 0.419 | 0.013 | 0.220 | 0.453 | 0.024 | 0.197 | 0.415 | 0.029 | 0.215 | 0.455 | 0.026 |
| 246 | 6 | `H` | 0.670 | 1.000 | 0.053 | 0.812 | 1.000 | 0.120 | 0.895 | 1.000 | 0.240 | 0.946 | 1.000 | 0.289 | 0.971 | 1.000 | 0.366 |
| 246 | 6 | `P` | 0.221 | 0.431 | 0.009 | 0.208 | 0.377 | 0.026 | 0.213 | 0.394 | 0.019 | 0.197 | 0.388 | 0.012 | 0.209 | 0.381 | 0.018 |
| 247 | 7 | `H` | 0.676 | 1.000 | 0.078 | 0.820 | 1.000 | 0.152 | 0.895 | 1.000 | 0.275 | 0.946 | 1.000 | 0.377 | 0.967 | 1.000 | 0.468 |
| 247 | 7 | `P` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 248 | 7 | `H` | 0.502 | 1.000 | 0.003 | 0.598 | 1.000 | 0.001 | 0.720 | 1.000 | 0.062 | 0.801 | 1.000 | 0.096 | 0.846 | 1.000 | 0.118 |
| 248 | 7 | `P` | 0.362 | 0.684 | 0.001 | 0.312 | 0.723 | 0.007 | 0.264 | 0.610 | 0.003 | 0.353 | 0.655 | 0.001 | 0.283 | 0.748 | 0.000 |
| 249 | 8 | `H` | 0.979 | 1.000 | 0.837 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 249 | 8 | `P` | 0.088 | 0.098 | 0.031 | 0.381 | 0.541 | 0.006 | 0.472 | 0.545 | 0.003 | 0.646 | 0.830 | 0.000 | 0.846 | 0.846 | 0.845 |

---

*Generated by `analyze_results/generate_results_markdown.py`.*
