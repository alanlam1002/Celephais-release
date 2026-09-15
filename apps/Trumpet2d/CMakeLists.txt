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
