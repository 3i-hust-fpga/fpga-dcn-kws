#include "header.h"

// void linear_1(
//     hls::stream<DTYPE> &in,
//     DTYPE w [L1_ROWS][L1_COLS],
//     DTYPE b [L1_ROWS],
//     hls::stream<DTYPE> &out,
//     hls::stream<DTYPE> &cache) {

// #pragma HLS ARRAY_PARTITION variable=w complete
// #pragma HLS ARRAY_PARTITION variable=b complete

//     DTYPE a_row[L1_COLS];
// #pragma HLS ARRAY_PARTITION variable=a_row complete
//     DTYPE placeholder;
//     DTYPE tmp;

//     loop_seq_len: for (int i = 0; i < CONV_OUT_SIZE_0; i++) {
//     	loop_rows: for (int j = 0; j < L1_ROWS; j++) {
// #pragma HLS PIPELINE II=32
//             tmp = 0;
//             if (j == 0) {
//             	loop_col_cache_a: for (int k = 0; k < L1_COLS; k++) {
//             		in >> placeholder;
//                     a_row[k] = placeholder;
//                     cache << placeholder;
//                 }
//             }
//             loop_col_mul: for (int k = 0; k < L1_COLS; k++) {
// #pragma HLS UNROLL
//                 tmp += (a_row[k] * w[j][k]);
//             }
//             out << hls::tanh(tmp + b[j]);
//         }
//     }
// }

void linear_1(
    hls::stream<DTYPE> &in,
    DTYPE w [L1_ROWS][L1_COLS],
    DTYPE b [L1_ROWS],
    hls::stream<DTYPE> &out,
    hls::stream<DTYPE> &cache) {

#pragma HLS ARRAY_PARTITION variable=w complete
#pragma HLS ARRAY_PARTITION variable=b complete

    DTYPE placeholder;
    DTYPE a_row[L1_COLS];
    DTYPE mul[L1_ROWS][L1_COLS];
    DTYPE acc[L1_ROWS];
    DTYPE activation[L1_ROWS];

#pragma HLS ARRAY_PARTITION variable=a_row complete
#pragma HLS ARRAY_PARTITION variable=mul complete
#pragma HLS ARRAY_PARTITION variable=acc complete
#pragma HLS ARRAY_PARTITION variable=activation complete

    loop_seq_len: for (int i = 0; i < CONV_OUT_SIZE_0; i++) {
#pragma HLS PIPELINE II=64
        loop_row_mul: for (int j = 0; j < L1_ROWS; j++) {
#pragma HLS UNROLL
            if (j == 0) {
                loop_col_cache_a: for (int k = 0; k < L1_COLS; k++) {
                    in >> placeholder;
                    a_row[k] = placeholder;
                    cache << placeholder;
                }
            }
            for (int k = 0; k < L1_COLS; k++) {
                mul[j][k] = (a_row[k] * w[j][k]);
            }
        }
        loop_row_bias: for (int j = 0; j < L1_ROWS; j++) {
#pragma HLS UNROLL
            acc[j] = b[j];
        }

        loop_row_acc: for (int j = 0; j < L1_ROWS; j++) {
#pragma HLS UNROLL
            for (int k = 0; k < L1_COLS; k++) {
                acc[j] += mul[j][k];
            }
        }

        loop_row_activation: for (int j = 0; j < L1_ROWS; j++) {
#pragma HLS UNROLL
            activation[j] = hls::tanh(acc[j]);
        }


        loop_row_out: for (int j = 0; j < L1_ROWS; j++) {
#pragma HLS UNROLL
            out << activation[j];
        }
        
    }
}

// void linear_2(
//     hls::stream<DTYPE> &in,
//     DTYPE w [L2_ROWS][L2_COLS],
//     DTYPE b [L2_ROWS],
//     hls::stream<DTYPE> &out) {

// #pragma HLS ARRAY_PARTITION variable=w complete
// #pragma HLS ARRAY_PARTITION variable=b complete

//     DTYPE a_row[L2_COLS];
// #pragma HLS ARRAY_PARTITION variable=a_row complete
//     DTYPE tmp;

//     loop_seq_len: for (int i = 0; i < CONV_OUT_SIZE_0; i++) {
//     	loop_rows: for (int j = 0; j < L2_ROWS; j++) {
// #pragma HLS PIPELINE II=32
//             tmp = 0;
//             if (j == 0) {
//             	loop_col_cache: for (int k = 0; k < L2_COLS; k++) {
//                     in >> a_row[k];
//                 }
//             }
//             loop_col_mul: for (int k = 0; k < L2_COLS; k++) {
// #pragma HLS UNROLL
//                 tmp += a_row[k] * w[j][k];
//             }
//             out << tmp + b[j];
//         }
//     }
// }

void linear_2(
    hls::stream<DTYPE> &in,
    DTYPE w [L2_ROWS][L2_COLS],
    DTYPE b [L2_ROWS],
    hls::stream<DTYPE> &out) {

#pragma HLS ARRAY_PARTITION variable=w complete
#pragma HLS ARRAY_PARTITION variable=b complete

    DTYPE a_row[L2_COLS];
    DTYPE mul[L2_ROWS][L2_COLS];
    DTYPE acc[L2_ROWS];

#pragma HLS ARRAY_PARTITION variable=a_row complete
#pragma HLS ARRAY_PARTITION variable=mul complete
#pragma HLS ARRAY_PARTITION variable=acc complete

    loop_seq_len: for (int i = 0; i < CONV_OUT_SIZE_0; i++) {
#pragma HLS PIPELINE II=64
        loop_row_mul: for (int j = 0; j < L2_ROWS; j++) {
#pragma HLS UNROLL
            if (j == 0) {
                loop_col_cache_a: for (int k = 0; k < L2_COLS; k++) {
                    in >> a_row[k];
                }
            }
            for (int k = 0; k < L2_COLS; k++) {
                mul[j][k] = (a_row[k] * w[j][k]);
            }
        }
        loop_row_bias: for (int j = 0; j < L2_ROWS; j++) {
#pragma HLS UNROLL
            acc[j] = b[j];
        }

        loop_row_acc: for (int j = 0; j < L2_ROWS; j++) {
#pragma HLS UNROLL
            for (int k = 0; k < L2_COLS; k++) {
                acc[j] += mul[j][k];
            }
        }

        loop_row_out: for (int j = 0; j < L2_ROWS; j++) {
#pragma HLS UNROLL
            out << acc[j];
        }
        
    }
}

void linear_3(
    hls::stream<DTYPE> &in,
    DTYPE w [L3_ROWS][L3_COLS],
    DTYPE b [L3_ROWS],
    hls::stream<DTYPE> &out) {
    
#pragma HLS ARRAY_PARTITION variable=w complete
#pragma HLS ARRAY_PARTITION variable=b complete

    DTYPE a_row[L3_COLS];
#pragma HLS ARRAY_PARTITION variable=a_row complete
    DTYPE tmp;

    loop_rows: for (int j = 0; j < L3_ROWS; j++) {
#pragma HLS PIPELINE II=32
        tmp = 0;
        if (j == 0) {
        	loop_col_cache: for (int k = 0; k < L3_COLS; k++) {
                in >> a_row[k];
            }
        }
        loop_col_mul: for (int k = 0; k < L3_COLS; k++) {
#pragma HLS UNROLL
            tmp += a_row[k] * w[j][k];
        }
        out << tmp + b[j];
    }
}

void softmax(
    hls::stream<DTYPE> &in,
    hls::stream<DTYPE> &cache,
    hls::stream<DTYPE> &out) {

    DTYPE e[CONV_OUT_SIZE_0][L2_ROWS];
    DTYPE denom[L2_ROWS];
#pragma HLS ARRAY_PARTITION variable=denom complete dim=1
    DTYPE tmp;
    DTYPE placeholder;

    for (int i = 0; i < CONV_OUT_SIZE_0; i++) {
#pragma HLS PIPELINE II=32
        for (int j = 0; j < L2_ROWS; j++) {
            if (i == 0) {
                denom[j] = 0;
            }
            in >> placeholder;
            tmp = hls::exp(placeholder);
            e[i][j] = tmp;
            denom[j] += tmp;
        }
    }

    for (int i = 0; i < CONV_OUT_SIZE_0; i++) {
#pragma HLS PIPELINE II=32
        for (int j = 0; j < L2_ROWS; j++) {
            cache >> placeholder;
            out << (e[i][j] / denom[j]) * placeholder;
        }
    }
}

void sum(
    hls::stream<DTYPE> &in,
    hls::stream<DTYPE> &out) {

    DTYPE tmp[L2_ROWS];
#pragma HLS ARRAY_PARTITION variable=tmp complete dim=1
    DTYPE placeholder;

    for (int i = 0; i < CONV_OUT_SIZE_0; i++) {
#pragma HLS PIPELINE II=32
        for (int j = 0; j < L2_ROWS; j++) {
            if (i == 0) {
                tmp[j] = 0;
            }
            in >> placeholder;
            tmp[j] += placeholder;
        }
    }

    for (int j = 0; j < L2_ROWS; j++) {
#pragma HLS PIPELINE
        out << tmp[j];
    }
}
