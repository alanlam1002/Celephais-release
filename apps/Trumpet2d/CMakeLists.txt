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
