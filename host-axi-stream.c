#include <stdio.h>
#include "platform.h"
#include "xil_printf.h"
#include "xaxidma.h"

float CONV_INPUT[9 * 7] = {-1.5255959033966064, -0.7502318024635315, -0.6539809107780457, -1.6094847917556763, -0.1001671776175499, -0.6091889142990112, -0.9797722697257996, -1.6090962886810303, -0.7121446132659912, 0.30372199416160583, -0.777314305305481, -0.25145524740219116, -0.22227048873901367, 1.6871134042739868, 0.22842517495155334, 0.46763551235198975, -0.6969724297523499, -1.1607614755630493, 0.6995424032211304, 0.1990816295146942, 0.8656923770904541, 0.2444039285182953, -0.6629113554954529, 0.8073082566261292, 1.1016806364059448, -0.1759360432624817, -2.2455577850341797, -1.4464579820632935, 0.0611552819609642, -0.617744505405426, -0.7980698347091675, -0.13162320852279663, 1.8793457746505737, -0.07213178277015686, 0.15777060389518738, -0.7734549045562744, 0.1990565061569214, 0.04570277780294418, 0.15295691788196564, -0.47567880153656006, -0.11101982742547989, 0.2927352488040924, -0.1578451544046402, -0.028787139803171158, 2.3571109771728516, -1.0373387336730957, 1.5747981071472168, -1.7754007577896118, -0.025502461940050125, -1.023330569267273, -0.5961851477622986, -1.0055307149887085, -0.21060630679130554, -0.007547527551651001, 1.6734272241592407, 1.610317587852478, -0.703956663608551, -0.18526579439640045, -0.9962350726127625, -0.8312552571296692, -0.4610220193862915, -0.5600824356079102, 0.3955761790275574};

u32 float_to_u32(float val) {
    union {
        float v;
        u32 u;
    } data;
    data.v = val;
    return data.u;
}

float u32_to_float(u32 val) {
    union {
        float v;
        u32 u;
    } data;
    data.u = val;
    return data.v;
}

int init_dma(XAxiDma *axiDmaPtr);

#define IN_SIZE 9*7
#define OUT_SIZE 2

int main() {
    int status;
    int i;
    u32 data_in[IN_SIZE], data_out[OUT_SIZE];

    init_platform();
    XAxiDma axiDma;

    // Initialize the (simple) DMA engine
    status = init_dma(&axiDma);
    if (status != XST_SUCCESS) {
       exit(-1);
    }

    for (i = 0; i < IN_SIZE; i++){
        data_in[i] = float_to_u32(CONV_INPUT[i]);
    }

    Xil_DCacheFlushRange((u32)data_in, IN_SIZE * sizeof(u32));
    Xil_DCacheFlushRange((u32)data_out, OUT_SIZE * sizeof(u32));

//    XAxiDma_SimpleTransfer(&axiDma, (u32) data_out, OUT_SIZE * sizeof(u32), XAXIDMA_DEVICE_TO_DMA);

    printf("Simple transfer sending to device...\n");
    XAxiDma_SimpleTransfer(&axiDma, (u32) data_in, IN_SIZE * sizeof(u32), XAXIDMA_DMA_TO_DEVICE);

    printf("Simple transfer, receiving from device ...\n");
    XAxiDma_SimpleTransfer(&axiDma, (u32) data_out, OUT_SIZE * sizeof(u32), XAXIDMA_DEVICE_TO_DMA);

    while(XAxiDma_Busy(&axiDma, XAXIDMA_DEVICE_TO_DMA));

    Xil_DCacheInvalidateRange((u32) data_out, OUT_SIZE * sizeof(u32));

//    XAxiDma_SimpleTransfer(&axiDma, (u32) data_out, OUT_SIZE * sizeof(u32), XAXIDMA_DEVICE_TO_DMA);
//    XAxiDma_SimpleTransfer(&axiDma, (u32) data_in, IN_SIZE * sizeof(u32), XAXIDMA_DMA_TO_DEVICE);

    for (i = 0; i < OUT_SIZE; i++){
        printf("%d %f\n", i, u32_to_float(data_out[i]));
    }

    print("Done\n");

    cleanup_platform();
    return 0;
}

int init_dma(XAxiDma *axiDmaPtr) {
    XAxiDma_Config *cfgPtr;
    int status;

    // Get pointer to DMA configuration
    cfgPtr = XAxiDma_LookupConfig(XPAR_AXIDMA_0_DEVICE_ID);
    if(!cfgPtr){
        print("Error looking for AXI DMA config\n\r");
        return XST_FAILURE;
    }

    // Initialize the DMA handle
    status = XAxiDma_CfgInitialize(axiDmaPtr, cfgPtr);
    if(status != XST_SUCCESS){
        print("Error initializing DMA\n\r");
        return XST_FAILURE;
    }
    // Check for scatter gather mode - this example must have simple mode only
    if(XAxiDma_HasSg(axiDmaPtr)){
        print("Error DMA configured in SG mode\n\r");
        return XST_FAILURE;
    }
    // Disable the interrupts
    XAxiDma_IntrDisable(axiDmaPtr, XAXIDMA_IRQ_ALL_MASK, XAXIDMA_DEVICE_TO_DMA);
    XAxiDma_IntrDisable(axiDmaPtr, XAXIDMA_IRQ_ALL_MASK, XAXIDMA_DMA_TO_DEVICE);
    return XST_SUCCESS;

}
