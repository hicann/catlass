// RUN: %tla_compile %s -o - | %filecheck %s

// Exhausts the legal cross flag id range (0-15) without erroring: 16 flags is
// the maximum the lowering supports, the 16th flag must land on sync_id 15.

module {
  tla.func @full_id_range() {
    "tla.vector"() ({
      %f00 = tla.cross_flag "flag_00" -> !tla.cross_flag<2>
      %f01 = tla.cross_flag "flag_01" -> !tla.cross_flag<2>
      %f02 = tla.cross_flag "flag_02" -> !tla.cross_flag<2>
      %f03 = tla.cross_flag "flag_03" -> !tla.cross_flag<2>
      %f04 = tla.cross_flag "flag_04" -> !tla.cross_flag<2>
      %f05 = tla.cross_flag "flag_05" -> !tla.cross_flag<2>
      %f06 = tla.cross_flag "flag_06" -> !tla.cross_flag<2>
      %f07 = tla.cross_flag "flag_07" -> !tla.cross_flag<2>
      %f08 = tla.cross_flag "flag_08" -> !tla.cross_flag<2>
      %f09 = tla.cross_flag "flag_09" -> !tla.cross_flag<2>
      %f10 = tla.cross_flag "flag_10" -> !tla.cross_flag<2>
      %f11 = tla.cross_flag "flag_11" -> !tla.cross_flag<2>
      %f12 = tla.cross_flag "flag_12" -> !tla.cross_flag<2>
      %f13 = tla.cross_flag "flag_13" -> !tla.cross_flag<2>
      %f14 = tla.cross_flag "flag_14" -> !tla.cross_flag<2>
      %f15 = tla.cross_flag "flag_15" -> !tla.cross_flag<2>
      tla.cross_core_set_flag %f00 {pipe = #tla.pipe<fix>} : !tla.cross_flag<2>
      tla.cross_core_set_flag %f01 {pipe = #tla.pipe<fix>} : !tla.cross_flag<2>
      tla.cross_core_set_flag %f02 {pipe = #tla.pipe<fix>} : !tla.cross_flag<2>
      tla.cross_core_set_flag %f03 {pipe = #tla.pipe<fix>} : !tla.cross_flag<2>
      tla.cross_core_set_flag %f04 {pipe = #tla.pipe<fix>} : !tla.cross_flag<2>
      tla.cross_core_set_flag %f05 {pipe = #tla.pipe<fix>} : !tla.cross_flag<2>
      tla.cross_core_set_flag %f06 {pipe = #tla.pipe<fix>} : !tla.cross_flag<2>
      tla.cross_core_set_flag %f07 {pipe = #tla.pipe<fix>} : !tla.cross_flag<2>
      tla.cross_core_set_flag %f08 {pipe = #tla.pipe<fix>} : !tla.cross_flag<2>
      tla.cross_core_set_flag %f09 {pipe = #tla.pipe<fix>} : !tla.cross_flag<2>
      tla.cross_core_set_flag %f10 {pipe = #tla.pipe<fix>} : !tla.cross_flag<2>
      tla.cross_core_set_flag %f11 {pipe = #tla.pipe<fix>} : !tla.cross_flag<2>
      tla.cross_core_set_flag %f12 {pipe = #tla.pipe<fix>} : !tla.cross_flag<2>
      tla.cross_core_set_flag %f13 {pipe = #tla.pipe<fix>} : !tla.cross_flag<2>
      tla.cross_core_set_flag %f14 {pipe = #tla.pipe<fix>} : !tla.cross_flag<2>
      tla.cross_core_set_flag %f15 {pipe = #tla.pipe<fix>} : !tla.cross_flag<2>
    }) : () -> ()
    "tla.cube"() ({
    }) : () -> ()
    tla.return
  }
}

// CHECK-LABEL: func.func @full_id_range_mix_aiv
// CHECK-COUNT-15: "hivm.intr.hivm.SET.INTRA.BLOCKI.mode"()
// CHECK: "hivm.intr.hivm.SET.INTRA.BLOCKI.mode"() <{pipe = 10 : i64, sync_id = 15 : i64}>
// CHECK-NOT: "hivm.intr.hivm.SET.INTRA.BLOCKI.mode"()
// CHECK-NOT: !tla.cross_flag
// CHECK-NOT: tla.cross_core
