# Vitis HLS C-synthesis report for the two load-bearing kernels, targeting the
# AWS F2 part (VU47P) at 250 MHz. No AWS platform needed -- part-level synth.
set PART   {xcvu47p-fsvh2892-2-e}
set PERIOD 4.0  ;# ns -> 250 MHz

foreach {proj top src} {
  p_gemv  synth_gemv_mlp   synth_gemv.cpp
  p_delta synth_delta_head synth_delta.cpp
} {
  open_project -reset $proj
  add_files $src -cflags "-I. -std=c++14"
  set_top $top
  open_solution -reset sol -flow_target vitis
  set_part $PART
  create_clock -period $PERIOD -name default
  csynth_design
  close_project
}
exit
