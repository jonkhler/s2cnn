// Adapted from
// https://github.com/thuyen/signal/blob/master/torchsignal/src/fft_cuda.c

#include <THC/THC.h>
#include <cufft.h>

// this symbol will be resolved automatically from PyTorch libs
extern THCState *state;

int plan1d_c2c(int N, int istride, int idist, int ostride, int odist, int batch)
{
    int size[1] = { N };
    cufftHandle plan;
    cufftPlanMany(&plan, 1, size, size, istride, idist, size, ostride, odist, CUFFT_C2C, batch);
    return plan;
}

int plan2d_c2c(int N0, int N1, int istride, int idist, int ostride, int odist, int batch)
{
    int size[2] = { N0, N1 };
    cufftHandle plan;
    cufftPlanMany(&plan, 2, size, size, istride, idist, size, ostride, odist, CUFFT_C2C, batch);
    return plan;
}

void execute_c2c(int plan, THCudaTensor *input, THCudaTensor *output, int sign)
{
    THCudaTensor_resizeAs(state, output, input);

    THArgCheck(THCudaTensor_isContiguous(NULL, input), 2, "Input tensor must be contiguous");
    THArgCheck(THCudaTensor_isContiguous(NULL, output), 2, "Output tensor must be contiguous");

    cuComplex *input_data = (cuComplex*)THCudaTensor_data(NULL, input);
    cuComplex *output_data = (cuComplex*)THCudaTensor_data(NULL, output);

    cufftExecC2C(plan, (cufftComplex*)input_data, (cufftComplex*)output_data, sign);
}

int plan2d_r2c(int N0, int N1, int istride, int idist, int ostride, int odist, int batch)
{
    int size[2] = { N0, N1 };
    int inembed[2] = { N0, N1 };
	int onembed[2] = { N0, N1 / 2 + 1 };
    cufftHandle plan;
    cufftPlanMany(&plan, 2, size, inembed, istride, idist, onembed, ostride, odist, CUFFT_R2C, batch);
    return plan;
}

void execute_r2c(int plan, THCudaTensor *input, THCudaTensor *output)
{
    //THCudaTensor_resizeAs(state, output, input);

    THArgCheck(THCudaTensor_isContiguous(NULL, input), 2, "Input tensor must be contiguous");
    THArgCheck(THCudaTensor_isContiguous(NULL, output), 2, "Output tensor must be contiguous");

    float *input_data = (float*)THCudaTensor_data(NULL, input);
    cuComplex *output_data = (cuComplex*)THCudaTensor_data(NULL, output);

    // the sign is implicitly FORWARD, FORWARD = -1
    cufftExecR2C(plan, (cufftReal*)input_data, (cufftComplex*)output_data);
}

void destroy(int plan)
{
    cufftDestroy(plan);
}
