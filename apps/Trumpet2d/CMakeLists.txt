set(APP_NAME Trumpet2d)

# A0 (spinning_trumpet NOTES_global_solve.md section 9): reproduce Trumpet1d's
# O(j^2) numbers inside the 2-D machinery, BEFORE the finite-J equations exist.
# This separates "the 2-D machinery works" from "the equations are right".
kadath_add_exec(trumpet2d_backbone src/backbone2d_main.cpp)
set_target_properties(trumpet2d_backbone PROPERTIES OUTPUT_NAME backbone2d)
# table_io.hpp is shared with Trumpet1d and is not duplicated: the apps/ root
# goes on the include path so it resolves as "Trumpet1d/src/table_io.hpp".
target_include_directories(trumpet2d_backbone PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)

# A0 second rung: the ANGULAR direction, on a genuinely theta-dependent
# manufactured field.  backbone2d cannot test it -- the backbone is spherically
# symmetric, so theta-derivatives, the polar basis, the parity handling and the
# axis regularisation are all unexercised there.
kadath_add_exec(trumpet2d_theta src/theta2d_main.cpp)
set_target_properties(trumpet2d_theta PROPERTIES OUTPUT_NAME theta2d)
target_include_directories(trumpet2d_theta PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)

# A0 accuracy architecture (research round 279): does the Laplacian roundoff
# floor improve by subdivision at fixed total dof?  Pure discretisation -- no
# backbone table, because the question is about what a Chebyshev shell stack can
# represent, not about the backbone.
kadath_add_exec(trumpet2d_floor src/floor2d_main.cpp)
set_target_properties(trumpet2d_floor PROPERTIES OUTPUT_NAME floor2d)
target_include_directories(trumpet2d_floor PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)

# Research round 282: measure what a LOG radial domain buys on the FLOOR before
# building Domain_polar_shell_log.  Uses Kadath's own Domain_shell_log (3-D
# spheric -- there is no polar one), so what is measured is the implementation a
# port would carry over rather than a reimplementation of it.
kadath_add_exec(trumpet2d_logfloor src/logfloor_main.cpp)
set_target_properties(trumpet2d_logfloor PROPERTIES OUTPUT_NAME logfloor)
target_include_directories(trumpet2d_logfloor PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)

# A0 tau rung (research round 284): does the TAU machinery work on a log radial
# domain?  Domain_polar_shell_log inherits export_tau/affecte_tau/nbr_conditions
# unchanged on the argument that they are coefficient-space; three inherited-
# and-unexercised pieces have already been wrong, so it is measured.
kadath_add_exec(trumpet2d_tau src/tau2d_main.cpp)
set_target_properties(trumpet2d_tau PROPERTIES OUTPUT_NAME tau2d)
target_include_directories(trumpet2d_tau PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)

# A0c (research rounds 278/285): the O(j^2) solver on the 2-D polar layout.
# Row assembly is SHARED with Trumpet1d through l0_setup.hpp, so the T2.1 gate
# still certifies the operator this app solves.
kadath_add_exec(trumpet2d_l0_solve src/l0_solve2d_main.cpp)
set_target_properties(trumpet2d_l0_solve PROPERTIES OUTPUT_NAME l0_solve2d)
# Trumpet1d's own directory too: l0_setup.hpp includes "space/space_oned_trumpet.hpp"
# relative to ITS app root, and the shared header is the point.
target_include_directories(trumpet2d_l0_solve PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/../Trumpet1d)

# A1 rung 0 (research round 293): do the SECTION-4 equations mean, inside
# Kadath, what they mean in sympy?  The parser on 1.2k-3.6k character strings,
# dr/dt on a Domain_polar_shell, and add_cst fields under differentiation are
# four inherited-and-unexercised pieces, checked against the J = 0 seed whose
# answer round 102 established symbolically.
kadath_add_exec(trumpet2d_finiteJ_probe src/finiteJ_probe_main.cpp)
set_target_properties(trumpet2d_finiteJ_probe PROPERTIES OUTPUT_NAME finiteJ_probe)
target_include_directories(trumpet2d_finiteJ_probe PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/../Trumpet1d)
