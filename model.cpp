#include "header.h"
#include "weight.h"
// #include "input.h"

// static void gen_input_1(DTYPE *in, hls::stream<DTYPE> &out, int size) {
//     for(int i=0 ; i < size; i++){
// #pragma HLS PIPELINE
//         out << in[i];
//     }
// }

// static void gen_output_1(hls::stream<DTYPE> &in, DTYPE *out, int size) {
//     for(int i=0 ; i < size; i++){
// #pragma HLS PIPELINE
//         out[i] = in.read();
//     } 
// }

// static void model_1(DTYPE *in, DTYPE *out) {

//     static hls::stream<DTYPE> stream_in("stream_in");
// #pragma HLS STREAM variable=stream_in depth=4096

//     static hls::stream<DTYPE> stream_out("stream_out");
// #pragma HLS STREAM variable=stream_out depth=2

//     static hls::stream<DTYPE> stream_cache_in("stream_cache_in");
// #pragma HLS STREAM variable=stream_cache_in depth=4096

//     static hls::stream<DTYPE> stream_offsets("stream_offsets");
// #pragma HLS STREAM variable=stream_offsets depth=32768

//     static hls::stream<DTYPE> stream_dcn("stream_dcn");
// #pragma HLS STREAM variable=stream_dcn depth=16384

//     static hls::stream<DTYPE> stream_gru("stream_gru");
// #pragma HLS STREAM variable=stream_gru depth=4096

//     static hls::stream<DTYPE> stream_gru_cache("stream_gru_cache");
// #pragma HLS STREAM variable=stream_gru_cache depth=4096

//     static hls::stream<DTYPE> stream_l1("stream_l1");
// #pragma HLS STREAM variable=stream_l1 depth=4096

//     static hls::stream<DTYPE> stream_l2("stream_l2");
// #pragma HLS STREAM variable=stream_l2 depth=4096

//     static hls::stream<DTYPE> stream_softmax("stream_softmax");
// #pragma HLS STREAM variable=stream_softmax depth=4096

//     static hls::stream<DTYPE> stream_sum("stream_sum");
// #pragma HLS STREAM variable=stream_sum depth=128

//     DTYPE array_cache_in[CONV_IN_SIZE_0][CONV_IN_SIZE_1];

// #pragma HLS DATAFLOW
//     gen_input_1(in, stream_in, CONV_IN_SIZE_0*CONV_IN_SIZE_1);

//     cache_input(stream_in, array_cache_in, stream_cache_in);
//     conv_offset(stream_cache_in, CONV_OFFSET_W, CONV_OFFSET_B, stream_offsets);
//     conv_out(stream_offsets, array_cache_in, CONV_OUT_W, CONV_OUT_B, stream_dcn);
//     gru(stream_dcn, GRU_W_IH, GRU_B_IH , GRU_W_HH, GRU_B_HH, stream_gru);
//     linear_1(stream_gru, L1_W, L1_B, stream_l1, stream_gru_cache);
//     linear_2(stream_l1, L2_W, L2_B, stream_l2);
//     softmax(stream_l2, stream_gru_cache, stream_softmax);
//     sum(stream_softmax, stream_sum);
//     linear_3(stream_sum, L3_W, L3_B, stream_out);

//     gen_output_1(stream_out, out, OUTPUT_SIZE);

// }



static void reg_input(DTYPE *in, hls::stream<DTYPE> &out) {
    for (int i = 0; i < CONV_IN_SIZE_0 * CONV_IN_SIZE_1; i++) {
#pragma HLS PIPELINE II=1
        out << in[i];
    }
}

static void gen_output_2(hls::stream<DTYPE> &in, hls::stream<axis_data> &out, int size) {
    axis_data tmp;
    DTYPE val;
    for(int i=0 ; i < size; i++){
#pragma HLS PIPELINE II=1
        val = in.read();
        tmp.data = val;
        tmp.last = (i == size-1 ? 1 : 0);
        out << tmp;
    } 
}

static void model_2(DTYPE *in1, DTYPE *in2, hls::stream<axis_data> &out) {

#pragma HLS STREAM variable=out depth=2

    static hls::stream<DTYPE> stream_in("stream_in");
#pragma HLS STREAM variable=stream_in depth=4096

    static hls::stream<DTYPE> stream_cache_in("stream_cache_in");
#pragma HLS STREAM variable=stream_cache_in depth=4096

    static hls::stream<DTYPE> stream_offsets("stream_offsets");
#pragma HLS STREAM variable=stream_offsets depth=32768

    static hls::stream<DTYPE> stream_dcn("stream_dcn");
#pragma HLS STREAM variable=stream_dcn depth=16384

    static hls::stream<DTYPE> stream_gru("stream_gru");
#pragma HLS STREAM variable=stream_gru depth=4096

    static hls::stream<DTYPE> stream_gru_cache("stream_gru_cache");
#pragma HLS STREAM variable=stream_gru_cache depth=4096

    static hls::stream<DTYPE> stream_l1("stream_l1");
#pragma HLS STREAM variable=stream_l1 depth=4096

    static hls::stream<DTYPE> stream_l2("stream_l2");
#pragma HLS STREAM variable=stream_l2 depth=4096

    static hls::stream<DTYPE> stream_softmax("stream_softmax");
#pragma HLS STREAM variable=stream_softmax depth=4096

    static hls::stream<DTYPE> stream_sum("stream_sum");
#pragma HLS STREAM variable=stream_sum depth=128

    static hls::stream<DTYPE> stream_out("stream_out");
#pragma HLS STREAM variable=stream_out depth=2



#pragma HLS DATAFLOW

    reg_input(in1, stream_in);

	conv_offset(stream_in, CONV_OFFSET_W, CONV_OFFSET_B, stream_offsets);
	conv_out(stream_offsets, in2, CONV_OUT_W, CONV_OUT_B, stream_dcn);
	gru(stream_dcn, GRU_W_IH, GRU_B_IH , GRU_W_HH, GRU_B_HH, stream_gru);
	linear_1(stream_gru, L1_W, L1_B, stream_l1, stream_gru_cache);
	linear_2(stream_l1, L2_W, L2_B, stream_l2);
	softmax(stream_l2, stream_gru_cache, stream_softmax);
	sum(stream_softmax, stream_sum);
	linear_3(stream_sum, L3_W, L3_B, stream_out);

	gen_output_2(stream_out, out, OUTPUT_SIZE);


}



void hls_model_wrapper(hls::stream<axis_data> &in, hls::stream<axis_data> &out) {
#pragma HLS INTERFACE axis port=in
#pragma HLS INTERFACE axis port=out
#pragma HLS INTERFACE ap_ctrl_none port=return

    DTYPE array_1[CONV_IN_SIZE_0 * CONV_IN_SIZE_1];
    DTYPE array_2[CONV_IN_SIZE_0 * CONV_IN_SIZE_1];
    axis_data tmp;
    for (int i = 0; i < CONV_IN_SIZE_0 * CONV_IN_SIZE_1; i++) {
#pragma HLS PIPELINE
        tmp = in.read();
        array_1[i] = tmp.data;
        array_2[i] = tmp.data;
    }

    model_2(array_1, array_2, out);

}

