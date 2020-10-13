# pylint: disable=R,C,E1101
from functools import lru_cache
import torch
import torch.cuda
from string import Template

# TODO simplify the cuda code like it was done in SO3_mm using only one code for the kernel


def s2_mm(x, y):
    '''
    :param x: [l * m,     batch,      feature_in,  complex]
    :param y: [l * m,     feature_in, feature_out, complex]
    :return:  [l * m * n, batch,      feature_out, complex]
    '''
    from s2cnn.utils.complex import complex_mm

    assert y.size(3) == 2
    assert x.size(3) == 2
    nbatch = x.size(1)
    nfeature_in = x.size(2)
    nfeature_out = y.size(2)
    assert y.size(1) == nfeature_in
    nspec = x.size(0)
    assert y.size(0) == nspec

    if x.is_cuda:
        return _cuda_S2_mm.apply(x, y)

    nl = round(nspec**0.5)

    Fz_list = []
    begin = 0
    for l in range(nl):
        L = 2 * l + 1
        size = L

        Fx = x[begin:begin+size]  # [m, batch,      feature_in,  complex]
        Fy = y[begin:begin+size]  # [m, feature_in, feature_out, complex]

        Fx = Fx.view(L * nbatch, nfeature_in, 2)  # [m * batch, feature_in, complex]

        Fy = Fy.transpose(0, 1)  # [feature_in, m, feature_out, complex]
        Fy = Fy.contiguous()
        Fy = Fy.view(nfeature_in, L * nfeature_out, 2)  # [feature_in, m * feature_out, complex]

        Fz = complex_mm(Fx, Fy, conj_y=True)  # [m_x * batch, m_y * feature_out, complex] m_x -> m, m_y -> n
        Fz = Fz.view(L, nbatch, L, nfeature_out, 2)  # [m, batch, n, feature_out, complex]
        Fz = Fz.transpose(1, 2)  # [m, n, batch, feature_out, complex]
        Fz = Fz.contiguous()
        Fz = Fz.view(L * L, nbatch, nfeature_out, 2)  # [m * n, batch, feature_out, complex]

        Fz_list.append(Fz)

        begin += size

    z = torch.cat(Fz_list, 0)  # [l * m * n, batch, feature_out, complex]
    return z


class _cuda_S2_mm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y):  # pylint: disable=W
        ctx.save_for_backward(x, y)
        return _cuda_s2_mm(x, y)

    @staticmethod
    def backward(ctx, gradz):  # pylint: disable=W
        import s2cnn.utils.cuda as cuda_utils
        x, y = ctx.saved_tensors
        nl = round(x.size(0) ** 0.5)
        nbatch = x.size(1)
        nfeature_in = x.size(2)
        nfeature_out = y.size(2)
        nspec = (4 * nl ** 2 - 1) * nl // 3
        device = torch.cuda.current_device()

        gradx_cuda_kernel = _setup_s2mm_gradx_cuda_kernel(nbatch=nbatch, nspec=nspec, nl=nl, nfeature_in=nfeature_in,
                                                          nfeature_out=nfeature_out, device=device)
        grady_cuda_kernel = _setup_s2mm_grady_cuda_kernel(nbatch=nbatch, nspec=nspec, nl=nl, nfeature_in=nfeature_in,
                                                          nfeature_out=nfeature_out, device=device)

        stream = cuda_utils.Stream(ptr=torch.cuda.current_stream().cuda_stream)

        gradx = grady = None

        if ctx.needs_input_grad[0]:
            gradx = gradz.new_empty((nl ** 2, nbatch, nfeature_in, 2))
            gradx_cuda_kernel(block=(cuda_utils.CUDA_NUM_THREADS, 1, 1),
                              grid=(cuda_utils.get_blocks(nl ** 2 * nbatch * nfeature_in, 1024), 1, 1),
                              args=[gradz.contiguous().data_ptr(), y.contiguous().data_ptr(), gradx.data_ptr()],
                              stream=stream)

        if ctx.needs_input_grad[1]:
            grady = gradz.new_empty((nl ** 2, nfeature_in, nfeature_out, 2))
            grady_cuda_kernel(block=(cuda_utils.CUDA_NUM_THREADS, 1, 1),
                              grid=(cuda_utils.get_blocks(nl ** 2 * nfeature_in * nfeature_out, 1024), 1, 1),
                              args=[gradz.contiguous().data_ptr(), x.contiguous().data_ptr(), grady.data_ptr()],
                              stream=stream)

        return gradx, grady


def _cuda_s2_mm(x, y):
    '''
    :param x: [l * m,     batch,      feature_in,  complex]
    :param y: [l * m,     feature_in, feature_out, complex]
    :return:  [l * m * n, batch,      feature_out, complex]
    '''
    import s2cnn.utils.cuda as cuda_utils
    assert x.is_cuda and x.dtype == torch.float32
    assert y.is_cuda and y.dtype == torch.float32
    assert y.size(3) == 2
    assert x.size(3) == 2
    nbatch = x.size(1)
    nfeature_in = x.size(2)
    nfeature_out = y.size(2)
    assert y.size(1) == nfeature_in
    assert y.size(0) == x.size(0)
    nl = round(x.size(0) ** 0.5)
    nspec = (4 * nl ** 2 - 1) * nl // 3
    assert x.size(0) == nl ** 2
    assert y.size(0) == nl ** 2

    device = torch.cuda.current_device()
    cuda_kernel = _setup_s2mm_cuda_kernel(nbatch=nbatch, nspec=nspec, nfeature_in=nfeature_in,
                                          nfeature_out=nfeature_out, device=device)

    stream = cuda_utils.Stream(ptr=torch.cuda.current_stream().cuda_stream)
    output = x.new_empty((nspec, nbatch, nfeature_out, 2))
    cuda_kernel(block=(cuda_utils.CUDA_NUM_THREADS, 1, 1),
                grid=(cuda_utils.get_blocks(nspec * nbatch * nfeature_out, 1024), 1, 1),
                args=[x.contiguous().data_ptr(), y.contiguous().data_ptr(), output.data_ptr()],
                stream=stream)
    # [l * m * n, batch, feature_out, complex]

    return output


@lru_cache(maxsize=32)
def _setup_s2mm_cuda_kernel(nbatch, nspec, nfeature_in, nfeature_out, device=0):
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

    import s2cnn.utils.cuda as cuda_utils
    return cuda_utils.compile_kernel(kernel, 's2mm.cu', 'main_')


@lru_cache(maxsize=32)
def _setup_s2mm_gradx_cuda_kernel(nbatch, nspec, nl, nfeature_in, nfeature_out, device=0):
    kernel = Template('''
#define COMPUTE_LM(s) \
    int l = sqrtf(s); \
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

    import s2cnn.utils.cuda as cuda_utils
    return cuda_utils.compile_kernel(kernel, 's2mm_gradx.cu', 'main_')


@lru_cache(maxsize=32)
def _setup_s2mm_grady_cuda_kernel(nbatch, nspec, nl, nfeature_in, nfeature_out, device=0):
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

    import s2cnn.utils.cuda as cuda_utils
    return cuda_utils.compile_kernel(kernel, 's2mm_grady.cu', 'main_')


def test_compare_cuda_cpu():
    x = torch.rand(1+3+5+7, 2, 3, 2)  # [l * m,     batch,      feature_in,  complex]
    y = torch.rand(1+3+5+7, 3, 5, 2)  # [l * m,     feature_in, feature_out, complex]
    z1 = s2_mm(x, y)
    z2 = s2_mm(x.cuda(), y.cuda()).cpu()
    q = (z1 - z2).abs().max().item() / z1.std().item()
    print(q)
    assert q < 1e-4


if __name__ == "__main__":
    test_compare_cuda_cpu()
