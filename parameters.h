#ifndef _PARAMETERS_H_
#define _PARAMETERS_H_

#include <stdint.h>
// #include <ap_fixed.h>
#include <hls_stream.h>
#include <hls_math.h>


typedef float DTYPE;


struct axis_data {
    DTYPE data;
    bool last;
};


#define CONV_IN_SIZE_0      200
#define CONV_IN_SIZE_1      40
#define CONV_KERNEL_SIZE_0  3
#define CONV_KERNEL_SIZE_1  3
#define CONV_STRIDE         1
#define CONV_FILTERS        8

#define CONV_OFFSET_FILTERS (2 * CONV_KERNEL_SIZE_0 * CONV_KERNEL_SIZE_1) 
#define CONV_OUT_SIZE_0     (CONV_IN_SIZE_0 - CONV_KERNEL_SIZE_0 + 1)
#define CONV_OUT_SIZE_1     (CONV_IN_SIZE_1 - CONV_KERNEL_SIZE_1 + 1)

#define GRU_HIDDEN_SIZE     32
#define GRU_IN_SIZE         (CONV_OUT_SIZE_1 * CONV_FILTERS)
#define GRU_G_SIZE          (3 * GRU_HIDDEN_SIZE)
#define GRU_W_IH_SIZE       (3 * GRU_HIDDEN_SIZE * GRU_IN_SIZE)
#define GRU_B_IH_SIZE       (3 * GRU_HIDDEN_SIZE)
#define GRU_W_HH_SIZE       (3 * GRU_HIDDEN_SIZE * GRU_HIDDEN_SIZE)
#define GRU_B_HH_SIZE       (3 * GRU_HIDDEN_SIZE)


#define ATTENTION_SIZE      32
#define L1_ROWS             ATTENTION_SIZE
#define L1_COLS             GRU_HIDDEN_SIZE
#define L2_ROWS             GRU_HIDDEN_SIZE
#define L2_COLS             ATTENTION_SIZE
#define L3_ROWS             2
#define L3_COLS             GRU_HIDDEN_SIZE

#define OUTPUT_SIZE         2 //CONV_OUT_SIZE_0 * CONV_OUT_SIZE_1 * CONV_FILTERS

#endif
