# aux func for setting include directories
# Use it to set the include directorys of the library lib_name.
# Please use it before the related bisheng_add_library.
function(set_bisheng_include_dirs lib_name)
    message(STATUS "Set include directories for lib${lib_name}")
    foreach(arg ${ARGN})
        set(include_dirs_tmp ${include_dirs_tmp};-I${arg})
    endforeach()
    set(${lib_name}_INCLUDE_DIRS ${include_dirs_tmp} PARENT_SCOPE)
endfunction()

# aux func for extracting library info
function(split_library_path library_path output_list)
    get_filename_component(library_dir ${library_path} DIRECTORY)
    get_filename_component(library_name ${library_path} NAME_WE)
    string(REGEX REPLACE "^lib" "" library_name ${library_name})
    set(link_options "-L${library_dir}" "-l${library_name}")
    set(${output_list} ${link_options} PARENT_SCOPE)
endfunction()

# aux func for setting link libraries
# Please use it before the related bisheng_add_library.
function(set_bisheng_link_libs lib_name)
    message(STATUS "Set link libraries for lib${lib_name}")
    foreach(arg ${ARGN})
        split_library_path(${arg}, split_result)
        set(all_link ${all_link};${split_result})
    endforeach()
    set(${lib_name}_LINK_LIBS ${all_link} PARENT_SCOPE)
endfunction()

function(bisheng_add_library LIB_NAME LIB_TYPE LIB_SRC)
    set(ASCEND_BISHENG_FLAGS_DEFAULT --std=c++17)
    set(BISHENG_COMPILER ${CMAKE_CCEC_COMPILER})

    if(${LIB_TYPE} STREQUAL "STATIC")
        set(BUILD_LIB_OPTION "--cce-build-static-lib")
        set(LIB_NAME_WITH_POSTFIX "lib${LIB_NAME}.a")
    elseif(${LIB_TYPE} STREQUAL "DYNAMIC")
        set(BUILD_LIB_OPTION "--shared")
        set(LIB_NAME_WITH_POSTFIX "lib${LIB_NAME}.so")
    else()
        set(BUILD_LIB_OPTION "")
    endif()

    add_custom_command(
        OUTPUT ${LIB_NAME_WITH_POSTFIX}
        COMMAND ${BISHENG_COMPILER} ${ASCEND_BISHENG_FLAGS_DEFAULT} ${ASCEND_BISHENG_FLAGS} ${LIB_SRC}
                ${BUILD_LIB_OPTION}
                ${${LIB_NAME}_INCLUDE_DIRS}
                ${${LIB_NAME}_LINK_LIBS}
                -o ${LIB_NAME_WITH_POSTFIX}
        DEPENDS ${LIB_SOURCES}
    )
endfunction()

