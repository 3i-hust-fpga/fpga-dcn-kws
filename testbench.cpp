#include "header.h"
#include "input.h"
#include <stdio.h>

int main(int argc, char const *argv[]) {

	FILE *fp = fopen("out_sim.txt", "w");
    // DTYPE out[OUTPUT_SIZE];
    // hls_model_wrapper(out);
    axis_data tmp;

    hls::stream<axis_data> in, out;

    for (int i = 0; i < CONV_IN_SIZE_0 * CONV_IN_SIZE_1; ++i) {
        tmp.data = CONV_INPUT[i];
        tmp.last = (i == CONV_IN_SIZE_0 * CONV_IN_SIZE_1-1 ? 1 : 0);

        in << tmp;
    }

    hls_model_wrapper(in, out);

    for (int i = 0; i < OUTPUT_SIZE; ++i) {
        out >> tmp;
        printf("%d %.06f\n", i, tmp.data);
        fprintf(fp, "%d %.06f\n", i, tmp.data);
    }

    fclose(fp);

}
