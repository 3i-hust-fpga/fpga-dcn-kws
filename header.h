#ifndef _BUFFER_H_
#define _BUFFER_H_

#include "parameters.h"

template <int SIZE>
class Buffer {
public:
	Buffer();
	void setval(DTYPE val, int pos);
	void insert_front(DTYPE val);
	void insert_back(DTYPE val);
	DTYPE getval(int pos);
private:
	DTYPE array[SIZE];
};


template <int SIZE>
Buffer<SIZE>::Buffer(){
	for (int i = 0; i < SIZE; i++) {
		array[i] = 0;
	}
}

template <int SIZE>
void Buffer<SIZE>::setval(DTYPE val, int pos) {
	array[pos] = val;
}

template <int SIZE>
void Buffer<SIZE>::insert_front(DTYPE val) {
	for (int i = SIZE-1; i > 0; i--) {
		array[i] = array[i-1];
	}
	array[0] = val;
}

template <int SIZE>
void Buffer<SIZE>::insert_back(DTYPE val) {
	for(int i = 0; i < SIZE-1; i++) {
		array[i] = array[i + 1];
	}
	array[SIZE-1] = val;
}

template <int SIZE>
DTYPE Buffer<SIZE>::getval(int pos) {
	return array[pos];
}

void cache_input(
    hls::stream<DTYPE> &in,
    DTYPE cache[CONV_IN_SIZE_0][CONV_IN_SIZE_1],
    hls::stream<DTYPE> &out);

void conv_offset( 
    hls::stream<DTYPE> &in, 
    DTYPE w_offset[CONV_OFFSET_FILTERS][CONV_KERNEL_SIZE_0][CONV_KERNEL_SIZE_1], 
    DTYPE b_offset[CONV_OFFSET_FILTERS],
    hls::stream<DTYPE> &offsets);

void conv_out(
    hls::stream<DTYPE> &offsets, 
    DTYPE in[CONV_IN_SIZE_0*CONV_IN_SIZE_1],
    DTYPE w_out[CONV_FILTERS][CONV_KERNEL_SIZE_0][CONV_KERNEL_SIZE_1],
    DTYPE b_out[CONV_FILTERS],
    hls::stream<DTYPE> &out);

void gru (
    hls::stream<DTYPE> &in,
    DTYPE w_ih [GRU_W_IH_SIZE],
    DTYPE b_ih [GRU_B_IH_SIZE],
    DTYPE w_hh [GRU_W_HH_SIZE],
    DTYPE b_hh [GRU_B_HH_SIZE],
    hls::stream<DTYPE> &out);

void linear_1(
    hls::stream<DTYPE> &in,
    DTYPE w [L1_ROWS][L1_COLS],
    DTYPE b [L1_ROWS],
    hls::stream<DTYPE> &out,
    hls::stream<DTYPE> &cache);

void linear_2(
    hls::stream<DTYPE> &in,
    DTYPE w [L2_ROWS][L2_COLS],
    DTYPE b [L2_ROWS],
    hls::stream<DTYPE> &out);

void softmax(
    hls::stream<DTYPE> &in,
    hls::stream<DTYPE> &cache,
    hls::stream<DTYPE> &out);

void sum(
    hls::stream<DTYPE> &in,
    hls::stream<DTYPE> &out);

void linear_3(
    hls::stream<DTYPE> &in,
    DTYPE w [L3_ROWS][L3_COLS],
    DTYPE b [L3_ROWS],
    hls::stream<DTYPE> &out);

void hls_model_wrapper(hls::stream<axis_data> &in, hls::stream<axis_data> &out);


#endif

