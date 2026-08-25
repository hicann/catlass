// RUN: %tla_compile %s -o - | %filecheck %s

module {
  tla.func @extern_call_kernel() {
    %addr = arith.constant 0 : i64
    %gm = tla.inttoptr %addr : i64 -> !tla.ptr<f32, gm, 4>
    %ub = tla.alloc_ptr{size_bytes = 1024} -> !tla.ptr<f32, ub, 256>
    %count = arith.constant 256 : i32
    "tla.vector"() ({
      tla.call_extern @tla_user_gm_to_ub_f32(%gm, %ub, %count) :
        (!tla.ptr<f32, gm, 4>, !tla.ptr<f32, ub, 256>, i32) -> ()
    }) : () -> ()
    tla.return
  }

  tla.func @extern_call_cube_kernel() {
    %value = arith.constant 1 : i32
    "tla.cube"() ({
      tla.call_extern @tla_user_cube_only(%value) : (i32) -> ()
    }) : () -> ()
    tla.return
  }

  tla.func @extern_call_mix_kernel() {
    %value = arith.constant 1 : i32
    "tla.cube"() ({
      tla.call_extern @tla_user_shared(%value) : (i32) -> ()
    }) : () -> ()
    "tla.vector"() ({
      tla.call_extern @tla_user_shared(%value) : (i32) -> ()
    }) : () -> ()
    tla.return
  }
}

// CHECK-DAG: func.func private @tla_user_gm_to_ub_f32(i64, i64, i32) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>}
// CHECK-DAG: func.func private @tla_user_cube_only(i32) attributes {hivm.func_core_type = #hivm.func_core_type<AIC>}
// CHECK-DAG: func.func private @tla_user_shared(i32) attributes {hivm.func_core_type = #hivm.func_core_type<AIC_OR_AIV>}
// CHECK-LABEL: func.func @extern_call_kernel
// CHECK: call @tla_user_gm_to_ub_f32({{.*}}) : (i64, i64, i32) -> ()
// CHECK-LABEL: func.func @extern_call_cube_kernel
// CHECK: call @tla_user_cube_only({{.*}}) : (i32) -> ()
// CHECK-LABEL: func.func @extern_call_mix_kernel_mix_aic
// CHECK: call @tla_user_shared({{.*}}) : (i32) -> ()
// CHECK-LABEL: func.func @extern_call_mix_kernel_mix_aiv
// CHECK: call @tla_user_shared({{.*}}) : (i32) -> ()
// CHECK-NOT: tla.call_extern
