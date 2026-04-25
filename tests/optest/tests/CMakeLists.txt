function(generate_compile_definitions KERNEL_NAME LAYOUTA LAYOUTB LAYOUTC ELEMENT_A ELEMENT_B ELEMENT_C)
    set(COMPILE_DEFS
        "KERNEL_NAME=${KERNEL_NAME}"
        "ELEMENT_A=${ELEMENT_A}"
        "ELEMENT_B=${ELEMENT_B}"
        "ELEMENT_C=${ELEMENT_C}"
        "LAYOUT_A=${LAYOUTA}"
        "LAYOUT_B=${LAYOUTB}"
        "LAYOUT_C=${LAYOUTC}"
    )
endfunction()


macro(add_kernel_parallelized KERNEL_NAME ARCH LAYOUTA LAYOUTB LAYOUTC ELEMENT_A ELEMENT_B ELEMENT_C)
    
endmacro()
