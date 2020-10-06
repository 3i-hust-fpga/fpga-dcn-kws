#include <stdio.h>
#include "platform.h"
#include "xil_printf.h"
#include "xhls_model_wrapper.h"

XHls_model_wrapper HLS_wrapper;

int hls_init(XHls_model_wrapper *InstancePtr){
    XHls_model_wrapper_Config *cfgPtr;
    int status;

    cfgPtr = XHls_model_wrapper_LookupConfig(XPAR_XHLS_MODEL_WRAPPER_0_DEVICE_ID);
    if (!cfgPtr) {
        print("ERROR: Lookup of accelerator configuration failed.\n\r");
        return XST_FAILURE;
    }

    status = XHls_model_wrapper_CfgInitialize(InstancePtr, cfgPtr);
    if (status != XST_SUCCESS) {
        print("ERROR: Could not initialize accelerator.\n\r");
        return XST_FAILURE;
    }
    return status;
}

unsigned int float_to_u32(float val) {
    unsigned int result;
    union float_bytes {
        float v;
        unsigned char bytes[4];
    }data;
    data.v = val;

    result = (data.bytes[3] << 24) + (data.bytes[2] << 16) + (data.bytes[1] << 8) + (data.bytes[0]);
    return result;
}

float u32_to_float(unsigned int val) {
    union {
        float val_float;
        unsigned char bytes[4];
    } data;
    data.bytes[3] = (val >> (8*3)) & 0xff;
    data.bytes[2] = (val >> (8*2)) & 0xff;
    data.bytes[1] = (val >> (8*1)) & 0xff;
    data.bytes[0] = (val >> (8*0)) & 0xff;
    return data.val_float;
}

int main() {
    int i, len;
    int data[10];

    printf("Start\n");

    hls_init(&HLS_wrapper);

    if (XHls_model_wrapper_IsReady(&HLS_wrapper)){
        print("HLS peripheral is ready. Starting... ");
    } else {
        print("!!! HLS peripheral is not ready! Exiting...\n\r");
        return -1;
    }

    XHls_model_wrapper_Start(&HLS_wrapper);
    do {
        len = XHls_model_wrapper_Read_ditme_Words(&HLS_wrapper, 0, data, 64);
    } while (!XHls_model_wrapper_IsReady(&HLS_wrapper));

    print("Detected HLS peripheral complete. Result received.\n\r");
    printf("Result from HW:\n");

    printf("length = %d\n", len);
    for (i = 0; i < len; i++){
        printf("%d %f\n", i, u32_to_float(data[i]));
    }
    printf("Done!!!!\n");


    cleanup_platform();
    return 0;
}

