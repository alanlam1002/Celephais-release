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

## Amendment: the variable is the operands' EVALUATION STATE, not the spelling

The scope table above was measured with a helper that registered each named def
and **read it back immediately**. Reading a def indexes its `Val_domain`, which
forces it into configuration space — so every "named" spelling in that table was
a named *and already-evaluated* def, and the variable being compared was never
isolated.

Varying it directly, with the operands' own read-back values as the reference:

```
  A = -1 * (-1 * (multr(dr(F))))
  B = -1 * (multr(dt(G) + divr(F)))

  a + b, the operands read back individually       -0.00404762175904   (reference)
  S = A + B, parsed while A and B are UNREAD        1.06494211083      DIFFERS
  the same S, re-read after A and B are read        1.06494211083      DIFFERS
  S = A + B, parsed after A and B are read         -0.00404762175904
  (-1 * (-1 * (multr(dr(F))))) + (-1 * (multr(dt(G) + divr(F))))
                                        inline      1.06494211083      DIFFERS
```

So a sum of two named definitions evaluates differently depending on whether
those definitions had been read before the sum was parsed, and a definition
parsed in the wrong state does not recover when its operands are read later.
Reading **either** operand is enough to change the result; we tested the
hypothesis that the first operand determines it, and it is wrong.

This also means the report above under-states the scope: the rows marked AGREE
were all measured in the read-first state, so they establish agreement in that
state only.

## What we do not claim

We have not identified the mechanism and have not looked in the parser or in
`Ope_eq`. We do not know whether released applications are affected. Our own five
`O(j^2)` rows are unaffected: each is registered a second time with every term
named **and read**, so the two spellings sit in the two different states, and
all five agree exactly on every layout we run. That is a statement about our
expressions, not about Kadath's users generally.

We have not identified which of the two values is the intended one beyond the
pointwise argument in the amendment (the value of a sum of two fields at a grid
point is the sum of their values there), and we have not determined whether the
angular spectral basis is what differs between the two states.

An earlier version of this report excluded a second anomaly as being in our own
code generator. That was wrong and is withdrawn: with the operand-read variable
varied, it reproduces here, and it is the amendment above.
