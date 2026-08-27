set(APP_NAME Trumpet1d)

# Phase 1 of the spinning-trumpet program: import the exact backbone onto a
# 1-D spectral grid and evaluate the T1.3 oracles.  No System_of_eqs yet.
kadath_add_exec(trumpet1d_backbone src/backbone_main.cpp)
set_target_properties(trumpet1d_backbone PROPERTIES OUTPUT_NAME backbone)

kadath_add_exec(trumpet1d_l0_system src/l0_system_main.cpp)
set_target_properties(trumpet1d_l0_system PROPERTIES OUTPUT_NAME l0_system)
