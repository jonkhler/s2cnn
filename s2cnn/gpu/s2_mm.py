# pylint: disable=R,C,E1101
from functools import lru_cache
import torch
from string import Template
import s2cnn.utils.cuda as cuda_utils


class S2_mm(torch.autograd.Function):
    def __init__(self):  # pylint: disable=W0235
        super(S2_mm, self).__init__()

    def forward(self, x, y):  # pylint: disable=W
        self.save_for_backward(x, y)
        return s2_mm(x, y)

    def backward(self, gradz):  # pylint: disable=W
        x, y = self.saved_tensors
        nl = round(x.size(0)**0.5)
        nbatch = x.size(1)
        nfeature_in = x.size(2)
        nfeature_out = y.size(2)
        nspec = (4 * nl**2 - 1) * nl // 3

        gradx_cuda_kernel = _setup_s2mm_gradx_cuda_kernel(nbatch=nbatch, nspec=nspec, nl=nl, nfeature_in=nfeature_in, nfeature_out=nfeature_out)
        grady_cuda_kernel = _setup_s2mm_grady_cuda_kernel(nbatch=nbatch, nspec=nspec, nl=nl, nfeature_in=nfeature_in, nfeature_out=nfeature_out)

        stream = cuda_utils.Stream(ptr=torch.cuda.current_stream().cuda_stream)

        gradx = grady = None

        if self.needs_input_grad[0]:
            gradx = gradz.new_empty((nl**2, nbatch, nfeature_in, 2))
            gradx_cuda_kernel(block=(cuda_utils.CUDA_NUM_THREADS, 1, 1),
                              grid=(cuda_utils.get_blocks(nl**2 * nbatch * nfeature_in, 1024), 1, 1),
                              args=[gradz.contiguous().data_ptr(), y.contiguous().data_ptr(), gradx.data_ptr()],
                              stream=stream)

        if self.needs_input_grad[1]:
            grady = gradz.new_empty((nl**2, nfeature_in, nfeature_out, 2))
            grady_cuda_kernel(block=(cuda_utils.CUDA_NUM_THREADS, 1, 1),
                              grid=(cuda_utils.get_blocks(nl**2 * nfeature_in * nfeature_out, 1024), 1, 1),
                              args=[gradz.contiguous().data_ptr(), x.contiguous().data_ptr(), grady.data_ptr()],
                              stream=stream)

        return gradx, grady


def s2_mm(x, y):
    '''
    :param x: [l * m,     batch,      feature_in,  complex]
    :param y: [l * m,     feature_in, feature_out, complex]
    :return:  [l * m * n, batch,      feature_out, complex]
    '''
    assert x.is_cuda and x.dtype == torch.float32
    assert y.is_cuda and y.dtype == torch.float32
    assert y.size(3) == 2
    assert x.size(3) == 2
    nbatch = x.size(1)
    nfeature_in = x.size(2)
    nfeature_out = y.size(2)
    assert y.size(1) == nfeature_in
    assert y.size(0) == x.size(0)
    nl = round(x.size(0)**0.5)
    nspec = (4 * nl**2 - 1) * nl // 3
    assert x.size(0) == nl ** 2
    assert y.size(0) == nl ** 2

    cuda_kernel = _setup_s2mm_cuda_kernel(nbatch=nbatch, nspec=nspec, nfeature_in=nfeature_in, nfeature_out=nfeature_out)

    stream = cuda_utils.Stream(ptr=torch.cuda.current_stream().cuda_stream)
    output = x.new_empty((nspec, nbatch, nfeature_out, 2))
    cuda_kernel(block=(cuda_utils.CUDA_NUM_THREADS, 1, 1),
                grid=(cuda_utils.get_blocks(nspec * nbatch * nfeature_out, 1024), 1, 1),
                args=[x.contiguous().data_ptr(), y.contiguous().data_ptr(), output.data_ptr()],
                stream=stream)
    # [l * m * n, batch, feature_out, complex]

    return output


@lru_cache(maxsize=32)
def _setup_s2mm_cuda_kernel(nbatch, nspec, nfeature_in, nfeature_out):
    kernel = Template('''
#define COMPUTE_LMN(s) \
    int l = powf(3.0/4.0 * s, 1.0/3.0) - 0.5; \
    int L = l * (4 * l * l - 1) / 3; \
    int rest = s - L; \
    if (rest >= (2 * l + 1) * (2 * l + 1)) { \
        ++l; \
        L = l * (4 * l * l - 1) / 3; \
        rest = s - L; \
    } \
    int m = rest / (2 * l + 1) - l; \
    int n = rest % (2 * l + 1) - l;

#define EXTRACT(i1, i2, n2, i3, n3) \
    int i1 = index; \
    int i3 = i1 % (n3);  i1 /= n3; \
    int i2 = i1 % (n2);  i1 /= n2;

#define CONTRACT1(s1, i2, n2, i3, n3) \
    (  ( (l * l + (l + (s1))) * (n2) + (i2) ) * (n3) + (i3)  )

#define CONTRACT2(s1, s2, i2, n2, i3, n3) \
    (  ( (L + (l + (s1)) * (2 * l + 1) + (l + (s2))) * (n2) + (i2) ) * (n3) + (i3)  )

extern "C"
__global__ void main_(const float* in_x, const float* in_y, float* out) {
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < ${nspec} * ${nbatch} * ${nfeature_out}; index += blockDim.x * gridDim.x) {
        EXTRACT(s, i, ${nbatch}, f_out, ${nfeature_out})

        // compute s -> (l,m,n)
        COMPUTE_LMN(s)

        float out_re = 0.0;
        float out_im = 0.0;

        for (int f_in = 0; f_in < ${nfeature_in}; ++f_in) {
            float x_re = in_x[CONTRACT1(m, i,    ${nbatch},      f_in,  ${nfeature_in} ) * 2 + 0];
            float x_im = in_x[CONTRACT1(m, i,    ${nbatch},      f_in,  ${nfeature_in} ) * 2 + 1];
            float y_re = in_y[CONTRACT1(n, f_in, ${nfeature_in}, f_out, ${nfeature_out}) * 2 + 0];
            float y_im = in_y[CONTRACT1(n, f_in, ${nfeature_in}, f_out, ${nfeature_out}) * 2 + 1];

            // x times y conjugate
            out_re += x_re * y_re + x_im * y_im;
            out_im += x_im * y_re - x_re * y_im;
        }

        out[index * 2 + 0] = out_re;
        out[index * 2 + 1] = out_im;
    }
}
''').substitute({'nbatch': nbatch,
                 'nspec': nspec,
                 'nfeature_in': nfeature_in,
                 'nfeature_out': nfeature_out})

    return cuda_utils.compile_kernel(kernel, b's2mm.cu', 'main_')


@lru_cache(maxsize=32)
def _setup_s2mm_gradx_cuda_kernel(nbatch, nspec, nl, nfeature_in, nfeature_out):
    kernel = Template('''
#define COMPUTE_LM(s) \
    int l = powf(s, 0.5); \
    int L = (4 * l * l - 1) * l / 3; \
    int m = s - l * l - l;

#define EXTRACT(i1, i2, n2, i3, n3) \
    int i1 = index; \
    int i3 = i1 % (n3);  i1 /= n3; \
    int i2 = i1 % (n2);  i1 /= n2;

#define CONTRACT1(s1, i2, n2, i3, n3) \
    (  ( (l * l + (l + (s1))) * (n2) + (i2) ) * (n3) + (i3)  )

#define CONTRACT2(s1, s2, i2, n2, i3, n3) \
    (  ( (L + (l + (s1)) * (2 * l + 1) + (l + (s2))) * (n2) + (i2) ) * (n3) + (i3)  )

extern "C"
__global__ void main_(const float* grad_z, const float* y, float* grad_x) {
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < (${nl} * ${nl}) * ${nbatch} * ${nfeature_in}; index += blockDim.x * gridDim.x) {
        EXTRACT(s, i, ${nbatch}, f_in, ${nfeature_in})

        // compute s -> (l,m)
        COMPUTE_LM(s)

        float out_re = 0.0;
        float out_im = 0.0;

        for (int f_out = 0; f_out < ${nfeature_out}; ++f_out) {
            for (int k = -l; k <= l; ++k) {
                float grad_z_re = grad_z[CONTRACT2(m, k, i,    ${nbatch},      f_out, ${nfeature_out}) * 2 + 0];
                float grad_z_im = grad_z[CONTRACT2(m, k, i,    ${nbatch},      f_out, ${nfeature_out}) * 2 + 1];
                float y_re =           y[CONTRACT1(k,    f_in, ${nfeature_in}, f_out, ${nfeature_out}) * 2 + 0];
                float y_im =           y[CONTRACT1(k,    f_in, ${nfeature_in}, f_out, ${nfeature_out}) * 2 + 1];

                // grad_z times y
                out_re += grad_z_re * y_re - grad_z_im * y_im;
                out_im += grad_z_re * y_im + grad_z_im * y_re;
            }
        }

        grad_x[index * 2 + 0] = out_re;
        grad_x[index * 2 + 1] = out_im;
    }
}
''').substitute({'nbatch': nbatch,
                 'nspec': nspec,
                 'nl': nl,
                 'nfeature_in': nfeature_in,
                 'nfeature_out': nfeature_out})

    return cuda_utils.compile_kernel(kernel, b's2mm_gradx.cu', 'main_')


@lru_cache(maxsize=32)
def _setup_s2mm_grady_cuda_kernel(nbatch, nspec, nl, nfeature_in, nfeature_out):
    kernel = Template('''
#define COMPUTE_LM(s) \
    int l = powf(s, 0.5); \
    int L = (4 * l * l - 1) * l / 3; \
    int m = s - l * l - l;

#define EXTRACT(i1, i2, n2, i3, n3) \
    int i1 = index; \
    int i3 = i1 % (n3);  i1 /= n3; \
    int i2 = i1 % (n2);  i1 /= n2;

#define CONTRACT1(s1, i2, n2, i3, n3) \
    (  ( (l * l + (l + (s1))) * (n2) + (i2) ) * (n3) + (i3)  )

#define CONTRACT2(s1, s2, i2, n2, i3, n3) \
    (  ( (L + (l + (s1)) * (2 * l + 1) + (l + (s2))) * (n2) + (i2) ) * (n3) + (i3)  )

extern "C"
__global__ void main_(const float* grad_z, const float* x, float* grad_y) {
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < (${nl} * ${nl}) * ${nfeature_in} * ${nfeature_out}; index += blockDim.x * gridDim.x) {
        EXTRACT(s, f_in, ${nfeature_in}, f_out, ${nfeature_out})

        // compute s -> (l,m)
        COMPUTE_LM(s)

        float out_re = 0.0;
        float out_im = 0.0;

        for (int i = 0; i < ${nbatch}; ++i) {
            for (int k = -l; k <= l; ++k) {
                float grad_z_re = grad_z[CONTRACT2(k, m, i, ${nbatch}, f_out, ${nfeature_out}) * 2 + 0];
                float grad_z_im = grad_z[CONTRACT2(k, m, i, ${nbatch}, f_out, ${nfeature_out}) * 2 + 1];
                float x_re =           x[CONTRACT1(k,    i, ${nbatch}, f_in,  ${nfeature_in} ) * 2 + 0];
                float x_im =           x[CONTRACT1(k,    i, ${nbatch}, f_in,  ${nfeature_in} ) * 2 + 1];

                // conjugate grad_z times x
                out_re += grad_z_re * x_re + grad_z_im * x_im;
                out_im += grad_z_re * x_im - grad_z_im * x_re;
            }
        }

        grad_y[index * 2 + 0] = out_re;
        grad_y[index * 2 + 1] = out_im;
    }
}
''').substitute({'nbatch': nbatch,
                 'nspec': nspec,
                 'nl': nl,
                 'nfeature_in': nfeature_in,
                 'nfeature_out': nfeature_out})

    return cuda_utils.compile_kernel(kernel, b's2mm_grady.cu', 'main_')
