#include "header.h"

static DTYPE sigmoid_hls(DTYPE x){
    return 1./(1.+ hls::exp(-1* (float)(x)));
}

static DTYPE gru_core(
    DTYPE a0, DTYPE b0,
    DTYPE a1, DTYPE b1,
    DTYPE a2, DTYPE b2,
    DTYPE a3, DTYPE b3) {
#pragma HLS PIPELINE

    DTYPE mul0, mul1, mul2, mul3;
    DTYPE add00, add01, add10;

    mul0 = a0 * b0;
    mul1 = a1 * b1;
    mul2 = a2 * b2;
    mul3 = a3 * b3;

    add00 = mul0 + mul1;
    add01 = mul2 + mul3;

    add10 = add00 + add01;

    return add10;
}


void gru (
    hls::stream<DTYPE> &in,
    DTYPE w_ih [GRU_W_IH_SIZE],
    DTYPE b_ih [GRU_B_IH_SIZE],
    DTYPE w_hh [GRU_W_HH_SIZE],
    DTYPE b_hh [GRU_B_HH_SIZE],
    hls::stream<DTYPE> &out) {

#pragma HLS RESOURCE variable=w_ih core=RAM_1P_LUTRAM
#pragma HLS RESOURCE variable=b_ih core=RAM_1P_LUTRAM
#pragma HLS RESOURCE variable=w_hh core=RAM_1P_LUTRAM
#pragma HLS RESOURCE variable=b_hh core=RAM_1P_LUTRAM

#pragma HLS ARRAY_PARTITION variable=b_hh cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=w_hh cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=b_ih cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=w_ih cyclic factor=8 dim=1


    DTYPE placeholder0;
    DTYPE placeholder1;
    DTYPE placeholder2;
    DTYPE placeholder3;

    DTYPE mul1[GRU_G_SIZE], mul2[GRU_G_SIZE];
#pragma HLS ARRAY_PARTITION variable=mul1 cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=mul2 cyclic factor=8 dim=1

    DTYPE hidden[GRU_HIDDEN_SIZE]={0};
#pragma HLS ARRAY_PARTITION variable=hidden cyclic factor=8 dim=1

    DTYPE i_r[GRU_HIDDEN_SIZE], i_i[GRU_HIDDEN_SIZE], i_n[GRU_HIDDEN_SIZE];
    DTYPE h_r[GRU_HIDDEN_SIZE], h_i[GRU_HIDDEN_SIZE], h_n[GRU_HIDDEN_SIZE];
#pragma HLS ARRAY_PARTITION variable=i_r cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=i_i cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=i_n cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=h_r cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=h_i cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=h_n cyclic factor=8 dim=1

    DTYPE resetgate[GRU_HIDDEN_SIZE], inputgate[GRU_HIDDEN_SIZE], newgate[GRU_HIDDEN_SIZE];
#pragma HLS ARRAY_PARTITION variable=resetgate cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=inputgate cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=newgate cyclic factor=8 dim=1

    loop_seq_len: for (int t = 0; t < CONV_OUT_SIZE_0; t++){

        loop_g_in: for (int j = 0; j < GRU_IN_SIZE; j+=4) {
#pragma HLS PIPELINE II=16
            
            // read input
            in >> placeholder0;
            in >> placeholder1;
            in >> placeholder2;
            in >> placeholder3;

            loop_compute_g1: for (int i = 0; i < GRU_G_SIZE; i++) {
                if (j == 0) {
                    mul1[i+0] = 0;
                }
                // mul1[i+0] += w_ih[(j+0)*GRU_G_SIZE+i] * placeholder0;

                mul1[i+0] += gru_core(w_ih[(j+0)*GRU_G_SIZE+i], placeholder0,
                                      w_ih[(j+1)*GRU_G_SIZE+i], placeholder1,
                                      w_ih[(j+2)*GRU_G_SIZE+i], placeholder2,
                                      w_ih[(j+3)*GRU_G_SIZE+i], placeholder3);
            }
        }

        loop_g_hidden: for (int j = 0; j < GRU_HIDDEN_SIZE; j+=4) {
#pragma HLS PIPELINE II=16
            loop_compute_g2: for (int i = 0; i < GRU_G_SIZE; i++) {
                if (j == 0) {
                    mul2[i+0] = 0;
                }
                // mul2[i+0] += w_hh[(j+0)*GRU_G_SIZE+i] * hidden[j];

                mul2[i+0] += gru_core(w_hh[(j+0)*GRU_G_SIZE+i], hidden[j+0],
                                      w_hh[(j+1)*GRU_G_SIZE+i], hidden[j+1],
                                      w_hh[(j+2)*GRU_G_SIZE+i], hidden[j+2],
                                      w_hh[(j+3)*GRU_G_SIZE+i], hidden[j+3]);
            }
        }

        loop_chunk_g: for (int i = 0; i < GRU_HIDDEN_SIZE; i++) {
#pragma HLS UNROLL
            i_r[i] = mul1[i] + b_ih[i];
            h_r[i] = mul2[i] + b_hh[i];
            i_i[i] = mul1[GRU_HIDDEN_SIZE+i] + b_ih[GRU_HIDDEN_SIZE+i];
            h_i[i] = mul2[GRU_HIDDEN_SIZE+i] + b_hh[GRU_HIDDEN_SIZE+i];
            i_n[i] = mul1[2*GRU_HIDDEN_SIZE+i] + b_ih[2*GRU_HIDDEN_SIZE+i];
            h_n[i] = mul2[2*GRU_HIDDEN_SIZE+i] + b_hh[2*GRU_HIDDEN_SIZE+i];
        }

        loop_gru_layer: for (int i = 0; i < GRU_HIDDEN_SIZE; i++) {
#pragma HLS PIPELINE II=16
            resetgate[i] = sigmoid_hls(i_r[i] + h_r[i]);
            inputgate[i] = sigmoid_hls(i_i[i] + h_i[i]);
            newgate[i] = hls::tanh(resetgate[i] * h_n[i] + i_n[i]);
            hidden[i] = newgate[i] + (hidden[i] - newgate[i]) * inputgate[i];
            out << hidden[i];
        }
    }
}
