# FPGA Implementation of the proposed model in master thesis using Vivado HLS.
### Components:
- Naive Convolution (without loop tiling)
- Convolution module using Line Buffer
- GRU (with loop tiling)
- Linear
- Softmax
- Some function from <hls_math> : exp(), tanh()

### Target hardware: ZCU 104 (xczu7ev-ffvc1156-2-e)
- Clock: 9.329 ns (Uncertainty 1.25 ns)
- Latency: 580661 
- Utilization (%): BRAM_18K (57), DSP48E (22), FF (28), LUT (55)

### Features:
- AXI4-Stream interface
- Store weights of model on-chip (without DDR)