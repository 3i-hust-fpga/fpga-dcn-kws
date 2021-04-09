#include "header.h"

static DTYPE sigmoid_hls(DTYPE x){
    return 1./(1.+ hls::exp(-1* (float)(x)));
}

static DTYPE gru_4_cores(
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


static DTYPE gru_8_cores(
    DTYPE a0, DTYPE b0,
    DTYPE a1, DTYPE b1,
    DTYPE a2, DTYPE b2,
    DTYPE a3, DTYPE b3,
    DTYPE a4, DTYPE b4,
    DTYPE a5, DTYPE b5,
    DTYPE a6, DTYPE b6,
    DTYPE a7, DTYPE b7) {
#pragma HLS PIPELINE

    DTYPE mul0, mul1, mul2, mul3, mul4, mul5, mul6, mul7;
    DTYPE add00, add01, add10, add11;
    DTYPE add20, add21, add22;

    mul0 = a0 * b0;
    mul1 = a1 * b1;
    mul2 = a2 * b2;
    mul3 = a3 * b3;

    mul4 = a4 * b4;
    mul5 = a5 * b5;
    mul6 = a6 * b6;
    mul7 = a7 * b7;

    add00 = mul0 + mul1;
    add01 = mul2 + mul3;
    add10 = mul4 + mul5;
    add11 = mul6 + mul7;

    add20 = add00 + add01;
    add21 = add10 + add11;

    add22 = add20 + add21;
    return add22;
}

void gru (
    hls::stream<DTYPE> &in,
    DTYPE w_ih [GRU_W_IH_SIZE],
    DTYPE b_ih [GRU_B_IH_SIZE],
    DTYPE w_hh [GRU_W_HH_SIZE],
    DTYPE b_hh [GRU_B_HH_SIZE],
    hls::stream<DTYPE> &out) {

#pragma HLS RESOURCE variable=w_ih core=RAM_1P_LUTRAM
#pragma HLS RESOURCE variable=w_hh core=RAM_1P_LUTRAM
#pragma HLS RESOURCE variable=b_ih core=RAM_1P_LUTRAM
#pragma HLS RESOURCE variable=b_hh core=RAM_1P_LUTRAM

#pragma HLS ARRAY_PARTITION variable=w_hh cyclic factor=16 dim=1
#pragma HLS ARRAY_PARTITION variable=w_ih cyclic factor=16 dim=1
#pragma HLS ARRAY_PARTITION variable=b_hh complete
#pragma HLS ARRAY_PARTITION variable=b_ih complete


    DTYPE placeholder0;
    DTYPE placeholder1;
    DTYPE placeholder2;
    DTYPE placeholder3;
    DTYPE placeholder4;
    DTYPE placeholder5;
    DTYPE placeholder6;
    DTYPE placeholder7;

    DTYPE mul1[GRU_G_SIZE], mul2[GRU_G_SIZE];
#pragma HLS ARRAY_PARTITION variable=mul1 complete
#pragma HLS ARRAY_PARTITION variable=mul2 complete

    DTYPE hidden[GRU_HIDDEN_SIZE]={0};
#pragma HLS ARRAY_PARTITION variable=hidden complete

    DTYPE i_r[GRU_HIDDEN_SIZE], i_i[GRU_HIDDEN_SIZE], i_n[GRU_HIDDEN_SIZE];
    DTYPE h_r[GRU_HIDDEN_SIZE], h_i[GRU_HIDDEN_SIZE], h_n[GRU_HIDDEN_SIZE];
#pragma HLS ARRAY_PARTITION variable=i_r complete
#pragma HLS ARRAY_PARTITION variable=i_i complete
#pragma HLS ARRAY_PARTITION variable=i_n complete
#pragma HLS ARRAY_PARTITION variable=h_r complete
#pragma HLS ARRAY_PARTITION variable=h_i complete
#pragma HLS ARRAY_PARTITION variable=h_n complete

    DTYPE resetgate[GRU_HIDDEN_SIZE], inputgate[GRU_HIDDEN_SIZE], newgate[GRU_HIDDEN_SIZE];
#pragma HLS ARRAY_PARTITION variable=resetgate complete
#pragma HLS ARRAY_PARTITION variable=inputgate complete
#pragma HLS ARRAY_PARTITION variable=newgate   complete

    loop_seq_len: for (int t = 0; t < CONV_OUT_SIZE_0; t++){

        loop_g_in: for (int j = 0; j < GRU_IN_SIZE; j+=4) {
#pragma HLS PIPELINE II=16
            
            // read input
            in >> placeholder0;
            in >> placeholder1;
            in >> placeholder2;
            in >> placeholder3;
            // in >> placeholder4;
            // in >> placeholder5;
            // in >> placeholder6;
            // in >> placeholder7;

            loop_compute_g1: for (int i = 0; i < GRU_G_SIZE; i+=1) {
                if (j == 0) {
                    mul1[i+0] = 0;
                    // mul1[i+1] = 0;
                }
                // mul1[i+0] += w_ih[(j+0)*GRU_G_SIZE+i] * placeholder0;

                mul1[i+0] += gru_4_cores(w_ih[(j+0)*GRU_G_SIZE+i+0], placeholder0,
                                         w_ih[(j+1)*GRU_G_SIZE+i+0], placeholder1,
                                         w_ih[(j+2)*GRU_G_SIZE+i+0], placeholder2,
                                         w_ih[(j+3)*GRU_G_SIZE+i+0], placeholder3);

                // mul1[i+1] += gru_4_cores(w_ih[(j+0)*GRU_G_SIZE+i+1], placeholder0,
                //                          w_ih[(j+1)*GRU_G_SIZE+i+1], placeholder1,
                //                          w_ih[(j+2)*GRU_G_SIZE+i+1], placeholder2,
                //                          w_ih[(j+3)*GRU_G_SIZE+i+1], placeholder3);

                // mul1[i+0] += gru_8_cores(w_ih[(j+0)*GRU_G_SIZE+i], placeholder0,
                //                          w_ih[(j+1)*GRU_G_SIZE+i], placeholder1,
                //                          w_ih[(j+2)*GRU_G_SIZE+i], placeholder2,
                //                          w_ih[(j+3)*GRU_G_SIZE+i], placeholder3,
                //                          w_ih[(j+4)*GRU_G_SIZE+i], placeholder4,
                //                          w_ih[(j+5)*GRU_G_SIZE+i], placeholder5,
                //                          w_ih[(j+6)*GRU_G_SIZE+i], placeholder6,
                //                          w_ih[(j+7)*GRU_G_SIZE+i], placeholder7);
                
            }
        }

        loop_g_hidden: for (int j = 0; j < GRU_HIDDEN_SIZE; j+=4) {
#pragma HLS PIPELINE II=16
            loop_compute_g2: for (int i = 0; i < GRU_G_SIZE; i+=1) {
                if (j == 0) {
                    mul2[i+0] = 0;
                    // mul2[i+1] = 0;
                }
                // mul2[i+0] += w_hh[(j+0)*GRU_G_SIZE+i] * hidden[j];

                mul2[i+0] += gru_4_cores(w_hh[(j+0)*GRU_G_SIZE+i+0], hidden[j+0],
                                         w_hh[(j+1)*GRU_G_SIZE+i+0], hidden[j+1],
                                         w_hh[(j+2)*GRU_G_SIZE+i+0], hidden[j+2],
                                         w_hh[(j+3)*GRU_G_SIZE+i+0], hidden[j+3]);

                // mul2[i+1] += gru_4_cores(w_hh[(j+0)*GRU_G_SIZE+i+1], hidden[j+0],
                //                          w_hh[(j+1)*GRU_G_SIZE+i+1], hidden[j+1],
                //                          w_hh[(j+2)*GRU_G_SIZE+i+1], hidden[j+2],
                //                          w_hh[(j+3)*GRU_G_SIZE+i+1], hidden[j+3]);

                // mul2[i+0] += gru_8_cores(w_hh[(j+0)*GRU_G_SIZE+i], hidden[j+0],
                //                          w_hh[(j+1)*GRU_G_SIZE+i], hidden[j+1],
                //                          w_hh[(j+2)*GRU_G_SIZE+i], hidden[j+2],
                //                          w_hh[(j+3)*GRU_G_SIZE+i], hidden[j+3],
                //                          w_hh[(j+4)*GRU_G_SIZE+i], hidden[j+4],
                //                          w_hh[(j+5)*GRU_G_SIZE+i], hidden[j+5],
                //                          w_hh[(j+6)*GRU_G_SIZE+i], hidden[j+6],
                //                          w_hh[(j+7)*GRU_G_SIZE+i], hidden[j+7]);
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
