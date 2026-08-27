set(APP_NAME Trumpet1d)

# Phase 1 of the spinning-trumpet program: import the exact backbone onto a
# 1-D spectral grid and evaluate the T1.3 oracles.  No System_of_eqs yet.
kadath_add_exec(trumpet1d_backbone src/backbone_main.cpp)
set_target_properties(trumpet1d_backbone PROPERTIES OUTPUT_NAME backbone)

kadath_add_exec(trumpet1d_l0_system src/l0_system_main.cpp)
set_target_properties(trumpet1d_l0_system PROPERTIES OUTPUT_NAME l0_system)

# Phase 2 / T2.2: close the l=0 BVP with the six SPEC_T22_innerBC conditions
# and solve it.  Needs MPI (do_newton is collective), unlike the two
# evaluation-only apps above.
kadath_add_exec(trumpet1d_l0_solve src/l0_solve_main.cpp)
set_target_properties(trumpet1d_l0_solve PROPERTIES OUTPUT_NAME l0_solve)
