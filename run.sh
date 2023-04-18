#LD_PRELOAD=/lib/x86_64-linux-gnu/libudev.so.1 make check TARGET=sw_emu DEVICE=xilinx_u55c_gen3x16_xdma_3_202210_1
#LD_PRELOAD=/lib/x86_64-linux-gnu/libudev.so.1 make check TARGET=hw_emu DEVICE=xilinx_u55c_gen3x16_xdma_3_202210_1
LD_PRELOAD=/lib/x86_64-linux-gnu/libudev.so.1 make all TARGET=hw DEVICE=xilinx_u55c_gen3x16_xdma_3_202210_1
