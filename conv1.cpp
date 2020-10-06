#include "header.h"

#define BUFFER_SIZE (CONV_IN_SIZE_1 * (CONV_KERNEL_SIZE_0 -1) + CONV_KERNEL_SIZE_1)

// void cache_input(
//     hls::stream<DTYPE> &in,
//     DTYPE cache[CONV_IN_SIZE_0][CONV_IN_SIZE_1],
//     hls::stream<DTYPE> &out) {

//     DTYPE placeholder;

//     for (int i = 0; i < CONV_IN_SIZE_0; i++) {
//         for (int j = 0; j < CONV_IN_SIZE_1; j++) {
//             in >> placeholder;
//             cache[i][j] = placeholder;
//             out << placeholder;
//         }
//     }
// }

void conv_offset( 
    hls::stream<DTYPE> &in, 
    DTYPE w_offset[CONV_OFFSET_FILTERS][CONV_KERNEL_SIZE_0][CONV_KERNEL_SIZE_1], 
    DTYPE b_offset[CONV_OFFSET_FILTERS],
    hls::stream<DTYPE> &offsets) {
#pragma HLS RESOURCE variable=w_offset core=RAM_1P_LUTRAM
#pragma HLS RESOURCE variable=b_offset core=RAM_1P_LUTRAM
#pragma HLS ARRAY_PARTITION variable=w_offset cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=b_offset cyclic factor=8 dim=1
    DTYPE val1, val2;
    DTYPE sum, placeholder;
    Buffer<BUFFER_SIZE> conv_buff;
#pragma HLS ARRAY_PARTITION variable=conv_buff complete dim=0

    conv_init_buffer: for(int i = 0; i < BUFFER_SIZE; i++) {
        if(in.empty() == 0) {
            placeholder = in.read();
            conv_buff.insert_back(placeholder);
        }
    }

    conv_in_size_0: for (int i = 0; i < CONV_OUT_SIZE_0; i += CONV_STRIDE) {
        conv_in_size_1: for (int j = 0; j < CONV_OUT_SIZE_1; j += CONV_STRIDE) {
#pragma HLS PIPELINE II=32
            conv_filters: for (int filter = 0; filter < CONV_OFFSET_FILTERS; filter++) {
#pragma HLS UNROLL
                sum = 0;
                conv_kernel_size_0: for (int row_offset = 0; row_offset < CONV_KERNEL_SIZE_0; row_offset++) {
#pragma HLS UNROLL
                    conv_kernel_size_1: for (int col_offset = 0; col_offset < CONV_KERNEL_SIZE_1; col_offset++) {
#pragma HLS UNROLL
                        val1 = conv_buff.getval(row_offset * CONV_IN_SIZE_1 + col_offset);
                        val2 = w_offset[filter][row_offset][col_offset];
                        sum += val1 * val2;
                    }
                }
                offsets << sum + b_offset[filter];
            }

            if ((j + CONV_STRIDE < CONV_OUT_SIZE_1)) {
                if(in.empty() == 0) {
                    placeholder = in.read();
                    conv_buff.insert_back(placeholder);
                }
            } else if ((i + CONV_STRIDE < CONV_OUT_SIZE_0) && (j + CONV_STRIDE >= CONV_OUT_SIZE_1)) {
                conv_add_input: for(int p = 0 ; p < CONV_KERNEL_SIZE_1; p++) {
                    if(in.empty() == 0) {
                        placeholder = in.read();
                        conv_buff.insert_back(placeholder);
                    }
                }
            }
        }
    }
}

void conv_out(
    hls::stream<DTYPE> &offsets, 
    DTYPE in[CONV_IN_SIZE_0*CONV_IN_SIZE_1],
    DTYPE w_out[CONV_FILTERS][CONV_KERNEL_SIZE_0][CONV_KERNEL_SIZE_1],
    DTYPE b_out[CONV_FILTERS],
    hls::stream<DTYPE> &out) {

#pragma HLS RESOURCE variable=w_out core=RAM_1P_LUTRAM
#pragma HLS RESOURCE variable=b_out core=RAM_1P_LUTRAM
#pragma HLS ARRAY_PARTITION variable=w_out cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=b_out cyclic factor=8 dim=1
    // dcn
    int x_l, x_h, y_l, y_h;
    DTYPE offset_x, offset_y;
    DTYPE dy_h, dx_h, dy_l, dx_l;
    DTYPE p1=0, p2=0, p3=0, p4=0;
    DTYPE deformed;
    DTYPE out_tmp[CONV_FILTERS] = {0};
#pragma HLS ARRAY_PARTITION variable=out_tmp complete dim=0
    
    conv_out_size_0: for (int i = 0; i < CONV_OUT_SIZE_0; i += CONV_STRIDE) {
        conv_out_size_1: for (int j = 0; j < CONV_OUT_SIZE_1; j += CONV_STRIDE) {
#pragma HLS PIPELINE II=32
            conv_kernel_size_0: for (int row_offset = 0; row_offset < CONV_KERNEL_SIZE_0; row_offset++) {
#pragma HLS UNROLL
                conv_kernel_size_1: for (int col_offset = 0; col_offset < CONV_KERNEL_SIZE_1; col_offset++) {
#pragma HLS UNROLL
                    offsets >> offset_y;
                    offsets >> offset_x;

                    offset_x += j + col_offset;
                    offset_y += i + row_offset;

                    if ((offset_y <= -1) || (CONV_IN_SIZE_0 <= offset_y) || (offset_x <= -1) || (CONV_IN_SIZE_1 <= offset_x)) {
                        deformed = 0;
                    } else {
                        x_l = hls::floor(offset_x);
                        x_h = x_l + 1;
                        y_l = hls::floor(offset_y);
                        y_h = y_l + 1;
    
                        dy_h = offset_y - y_l;
                        dy_l = y_h - offset_y;
                        dx_h = offset_x - x_l;
                        dx_l = x_h - offset_x;
    
                        if (y_l >= 0 && x_l >= 0){
                            p1 = dy_l * dx_l * in[y_l * CONV_IN_SIZE_1 + x_l];
                            // p1 = dy_l * dx_l * in[y_l][x_l];
                        } else {
                            p1 = 0;
                        }

                        if (y_l >= 0 && x_h < CONV_IN_SIZE_1){
                            p2 = dy_l * dx_h * in[y_l * CONV_IN_SIZE_1 + x_h];
                            // p2 = dy_l * dx_h * in[y_l][x_h];
                        } else {
                            p2 = 0;
                        }

                        if (y_h < CONV_IN_SIZE_0 && x_l >= 0){
                            p3 = dy_h * dx_l * in[y_h * CONV_IN_SIZE_1 + x_l];
                            // p3 = dy_h * dx_l * in[y_h][x_l];
                        } else {
                            p3 = 0;
                        }

                        if (y_h < CONV_IN_SIZE_0 && x_h < CONV_IN_SIZE_1){
                            p4 = dy_h * dx_h * in[y_h * CONV_IN_SIZE_1 + x_h];
                            // p4 = dy_h * dx_h * in[y_h][x_h];
                        } else {
                            p4 = 0;
                        }
    
                        deformed = p1 + p2 + p3 + p4;

                    }

                    conv_filters: for (int filter = 0; filter < CONV_FILTERS; filter++) {
#pragma HLS UNROLL
                        if (row_offset == 0 && col_offset == 0) {
                            out_tmp[filter] = b_out[filter]; 
                        }
                        out_tmp[filter] += deformed * w_out[filter][row_offset][col_offset];
                    }
                }
            }
            conv_output: for (int filter = 0; filter < CONV_FILTERS; filter++) {
                out << out_tmp[filter];
            }

        }
    }
}
