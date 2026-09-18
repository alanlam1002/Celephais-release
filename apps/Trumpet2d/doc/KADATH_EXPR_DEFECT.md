# Defect report: `System_of_eqs` evaluates some sums differently depending on whether an operand is named

**Reproducer:** `apps/Trumpet2d/src/kadath_expr_repro_main.cpp` (build target
`kadath_expr_repro`). Stock `Space_polar`, two `add_cst` scalars, every
expression typed literally. No generated text and no application code.

## Summary

Two spellings of the same expression evaluate differently. With

```
  A1 = multr(dr(F))
  A2 = multr(dt(G) + divr(F))
```

registered as defs, and `F`, `G` two `add_cst` scalars:

```
  multr(dr(F)) + multr(dt(G) + divr(F))     2.0501336486770918
  A1 + A2                                   3.1191233812672889
                                            relative difference  0.343
```

Both are `add_def`s in the same `System_of_eqs`, read back with
`give_val_def_scalar_domain` at the same grid point.

## Scope, narrowed

Each line is one expression in two spellings; `AGREE` means the two evaluate
identically, to machine precision.

```
  op(inline sum)  vs  op(named sum),  for multr, divr, dr, dt      AGREE
  field + op(inline sum)                                           AGREE
  op(inline sum) + field                                           AGREE
  op(...) + op(...),   neither argument a sum                      AGREE
  op(...) + op(...),   one argument a sum of plain fields          AGREE  1.2e-16
  NAMED + op(inline sum)                                           AGREE
  op(...) + op(inline sum)                                         DIFFER 0.343
  op(...) + op(NAMED sum)                                          DIFFER 0.343
```

So it is **not** "a unary operator applied to a sum" — that is exact on its own
for all four operators tested. The failing shape is a **sum whose first operand
is an inline unary-operator application**, where the second operand is an
operator applied to a sum of operator applications.

**Naming the first operand makes it agree. Naming the second does not.**

## Reproducing

```
  kadath_add_exec(trumpet2d_expr_repro src/kadath_expr_repro_main.cpp)
  mpirun -n 1 kadath_expr_repro
```

The program prints one row per spelling pair with the relative difference and
exits after reporting how many pairs differ. It needs no input files.

## Environment

Celephais (`trumpet_bh` branch), GCC 13.1 / Intel MPI 2021.17, `-O3`,
`Space_polar` with `CHEB_TYPE`, 2 domains, `nr = 13`, `ntheta = 9`, values read
in the shell domain at an interior off-axis collocation point. The difference is
far larger than any rounding effect and is stable across resolutions.

## What we do not claim

We have not identified the mechanism and have not looked in the parser or in
`Ope_eq`. We do not know whether released applications are affected: the shape
does not arise in the row construction we depend on — we checked our own five
`O(j^2)` rows by registering each a second time with every term named and
comparing pointwise, and all five agree exactly — but that is a statement about
our expressions, not about Kadath's users generally.

A second anomaly we originally reported alongside this one did **not** reproduce
here and turned out to be in our own code generator, so it is excluded.
