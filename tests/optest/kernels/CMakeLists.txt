set(CATLASS_KERNEL_SLICE_COUNT 4)

set(CATLASS_ALL_ARCHS "")

macro(add_kernel NAME ARCH)
    if(DEFINED CATLASS_KERNEL_LIST)
        list(FIND CATLASS_KERNEL_LIST ${NAME} _INDEX)
        if(_INDEX EQUAL -1)
            return()
        endif()
    endif()
    
    string(REPLACE ";" " " _ARCH_STR "${ARCH}")
    separate_arguments(_ARCH_LIST UNIX_COMMAND "${_ARCH_STR}")
    
    foreach(_ARCH ${_ARCH_LIST})
        list(FIND CATLASS_ARCH_LIST ${_ARCH} _ARCH_INDEX)
        if(_ARCH_INDEX EQUAL -1)
            continue()
        endif()
        
        list(APPEND CATLASS_ALL_ARCHS ${_ARCH})
        list(APPEND CATLASS_KERNELS_${_ARCH} ${NAME})
        set(CATLASS_KERNEL_${NAME}_${_ARCH}_SOURCES "${ARGN}")
    endforeach()
endmacro()

function(generate_kernel_libs)
    list(REMOVE_DUPLICATES CATLASS_ALL_ARCHS)
    
    foreach(_ARCH ${CATLASS_ALL_ARCHS})
        set(_KERNELS "${CATLASS_KERNELS_${_ARCH}}")
        list(REMOVE_DUPLICATES _KERNELS)
        list(LENGTH _KERNELS _KERNEL_COUNT)
        
        set(_SLICE_INDEX 0)
        set(_KERNEL_INDEX 0)
        set(_SUB_LIBS "")
        
        while(_KERNEL_INDEX LESS _KERNEL_COUNT)
            set(_SLICE_SOURCES "")
            
            foreach(_I RANGE 1 ${CATLASS_KERNEL_SLICE_COUNT})
                if(_KERNEL_INDEX LESS _KERNEL_COUNT)
                    list(GET _KERNELS ${_KERNEL_INDEX} _KERNEL_NAME)
                    foreach(_SRC ${CATLASS_KERNEL_${_KERNEL_NAME}_${_ARCH}_SOURCES})
                        get_filename_component(_SRC_ABS "${_SRC}" ABSOLUTE BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
                        list(APPEND _SLICE_SOURCES "${_SRC_ABS}")
                    endforeach()
                    math(EXPR _KERNEL_INDEX "${_KERNEL_INDEX} + 1")
                endif()
            endforeach()
            
            if(_SLICE_SOURCES)
                set(_SUB_LIB "catlass_kernel_${_ARCH}_slice_${_SLICE_INDEX}")
                add_library(${_SUB_LIB} SHARED ${_SLICE_SOURCES})
                
                target_include_directories(${_SUB_LIB} PRIVATE
                    ${ASCEND_HOME_PATH}/include
                    ${CATLASS_DIR}/include
                )
                
                install(TARGETS ${_SUB_LIB} 
                    DESTINATION ${PROJECT_SOURCE_DIR}/torch_catlass/lib/${_ARCH}
                )
                
                list(APPEND _SUB_LIBS ${_SUB_LIB})
                math(EXPR _SLICE_INDEX "${_SLICE_INDEX} + 1")
            endif()
        endwhile()
        
        if(_SUB_LIBS)
            add_library(catlass_kernel_${_ARCH} SHARED)
            target_link_libraries(catlass_kernel_${_ARCH} ${_SUB_LIBS})
            
            install(TARGETS catlass_kernel_${_ARCH}
                DESTINATION ${PROJECT_SOURCE_DIR}/torch_catlass/lib/${_ARCH}
            )
        endif()
    endforeach()
endfunction()

add_subdirectory(00_basic_matmul)

generate_kernel_libs()
