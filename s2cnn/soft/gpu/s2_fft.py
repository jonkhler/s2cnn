# pylint: disable=R,C,E1101
from functools import lru_cache
import torch
from string import Template
import s2cnn.utils.cuda as cuda_utils
from s2cnn.utils.decorator import cached_dirpklgz


# inspired by https://gist.github.com/szagoruyko/89f83b6f5f4833d3c8adf81ee49f22a8


def s2_fft(x, for_grad=False, b_out=None):
    '''
    :param x: [..., beta, alpha, complex]
    :return:  [l * m, ..., complex]
    '''
    assert x.is_cuda and x.dtype == torch.float32
    assert x.size(-1) == 2
    b_in = x.size(-2) // 2
    assert x.size(-2) == 2 * b_in
    assert x.size(-3) == 2 * b_in
    if b_out is None:
        b_out = b_in
    assert b_out <= b_in
    batch_size = x.size()[:-3]

    x = x.view(-1, 2 * b_in, 2 * b_in, 2)  # [batch, beta, alpha, complex]

    output = _s2_fft(x, for_grad=for_grad, b_in=b_in, b_out=b_out)  # [l * m, batch, complex]
    output = output.view(-1, *batch_size, 2)  # [l * m, ..., complex] (nspec, ..., 2)

    return output


def _s2_fft(x, for_grad, b_in, b_out):
    '''
    :param x: [batch, beta, alpha, complex] (nbatch, 2 * b_in, 2 * b_in, 2)
    :return: [l * m, batch, complex] (b_out**2, nbatch, 2)
    '''
    nspec = b_out**2
    nbatch = x.size(0)

    wigner = _setup_wigner(b_in, nl=b_out, weighted=not for_grad, device_type=x.device.type, device_index=x.device.index)
    cuda_kernel = _setup_s2fft_cuda_kernel(b=b_in, nspec=nspec, nbatch=nbatch)

    x = torch.fft(x, 1)  # [batch, beta, m, complex]

    stream = cuda_utils.Stream(ptr=torch.cuda.current_stream().cuda_stream)
    output = x.new_empty((nspec, nbatch, 2))
    cuda_kernel(block=(1024, 1, 1),
                grid=(cuda_utils.get_blocks(nspec * nbatch, 1024), 1, 1),
                args=[x.contiguous().data_ptr(), wigner.contiguous().data_ptr(), output.data_ptr()],
                stream=stream)
    # [l * m, batch, complex]

    return output


def s2_ifft(x, for_grad=False, b_out=None):
    '''
    :param x: [l * m, ..., complex]
    '''
    assert x.is_cuda and x.dtype == torch.float32
    assert x.size(-1) == 2
    nspec = x.size(0)
    b_in = round(nspec**0.5)
    assert nspec == b_in**2
    if b_out is None:
        b_out = b_in
    assert b_out >= b_in
    batch_size = x.size()[1:-1]

    x = x.view(nspec, -1, 2)  # [l * m, batch, complex] (nspec, nbatch, 2)

    output = _s2_ifft(x, for_grad=for_grad, b_in=b_in, b_out=b_out)  # [batch, beta, alpha, complex]

    output = output.view(*batch_size, 2 * b_out, 2 * b_out, 2)
    return output


def _s2_ifft(x, for_grad, b_in, b_out):
    '''
    :param x: [l * m, batch, complex] (b_in**2, nbatch, 2)
    :return: [batch, beta, alpha, complex] (nbatch, 2 b_out, 2 * b_out, 2)
    '''
    nbatch = x.size(1)

    wigner = _setup_wigner(b_out, nl=b_in, weighted=for_grad, device_type=x.device.type, device_index=x.device.index)  # [beta, l * m] (2 * b_out - 1, nspec)
    cuda_kernel = _setup_s2ifft_cuda_kernel(b=b_out, nl=b_in, nbatch=nbatch)

    stream = cuda_utils.Stream(ptr=torch.cuda.current_stream().cuda_stream)
    output = x.new_empty((nbatch, 2 * b_out, 2 * b_out, 2))
    cuda_kernel(block=(1024, 1, 1),
                grid=(cuda_utils.get_blocks(nbatch * (2 * b_out)**2, 1024), 1, 1),
                args=[x.data_ptr(), wigner.data_ptr(), output.data_ptr()],
                stream=stream)
    # [batch, beta, m, complex] (nbatch, 2 * b_out, 2 * b_out, 2)

    output = torch.ifft(output, 1) * output.size(-2)  # [batch, beta, alpha, complex]

    return output


@lru_cache(maxsize=32)
def _setup_wigner(b, nl, weighted, device_type, device_index):
    dss = _setup_s2_fft(b, nl, weighted)
    dss = torch.tensor(dss, dtype=torch.float32, device=torch.device(device_type, device_index))  # [beta, l * m] # pylint: disable=E1102
    return dss.contiguous()


@cached_dirpklgz("cache/setup_s2_fft")
def _setup_s2_fft(b, nl, weighted):
    from lie_learn.representations.SO3.wigner_d import wigner_d_matrix
    import lie_learn.spaces.S3 as S3
    import numpy as np
    import logging

    betas = (np.arange(2 * b) + 0.5) / (2 * b) * np.pi
    w = S3.quadrature_weights(b) * 2 * b
    assert len(w) == len(betas)

    logging.getLogger("trainer").info("Compute Wigner (only columns): b=%d nbeta=%d nl=%d nspec=%d", b, len(betas), nl, nl**2)

    dss = []
    for b, beta in enumerate(betas):
        ds = []
        for l in range(nl):
            d = wigner_d_matrix(l, beta,
                                field='complex', normalization='quantum', order='centered', condon_shortley='cs')
            d = d[:, l]  # d[m=:, n=0]

            if weighted:
                d *= w[b]
            else:
                d *= 2 * l + 1

            ds.append(d)  # [m]
        dss.append(np.concatenate(ds))  # [l * m]

    dss = np.concatenate(dss)  # [beta, l * m]
    return dss


@lru_cache(maxsize=32)
def _setup_s2fft_cuda_kernel(b, nspec, nbatch):
    kernel = Template('''
#define COMPUTE_LM(s) \
    int l = powf(s, 0.5); \
    int m = (s - l * l) - l;

#define MOD(i, n) (((i) + (n)) % (n))

extern "C"
__global__ void main_(const float* in, const float* wig, float* out) {
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < ${nspec} * ${nbatch}; index += blockDim.x * gridDim.x) {
        int i = index % ${nbatch}; // batch index
        int s = index / ${nbatch}; // spectral index

        // compute s -> (l,m)
        COMPUTE_LM(s)

        float out_re = 0.0;
        float out_im = 0.0;
        for (int beta = 0; beta < 2 * ${b}; ++beta) {
            float in_re = in[((i * 2 * ${b} + beta) * 2 * ${b} + MOD(m, 2 * ${b})) * 2 + 0];
            float in_im = in[((i * 2 * ${b} + beta) * 2 * ${b} + MOD(m, 2 * ${b})) * 2 + 1];
            float w = wig[beta * ${nspec} + s];

            out_re += w * in_re;
            out_im += w * in_im;
        }
        out[index * 2 + 0] = out_re;
        out[index * 2 + 1] = out_im;
    }
}
''').substitute({'b': b, 'nbatch': nbatch, 'nspec': nspec})

    return cuda_utils.compile_kernel(kernel, b's2fft.cu', 'main_')


@lru_cache(maxsize=32)
def _setup_s2ifft_cuda_kernel(b, nl, nbatch):
    kernel = Template('''
extern "C"
__global__ void main_(const float* in, const float* wig, float* out) {
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < ${nbatch} * 2 * ${b} * 2 * ${b}; index += blockDim.x * gridDim.x) {
        int i = index / (2 * ${b} * 2 * ${b}); // batch index
        int beta = (index / (2 * ${b})) % (2 * ${b});
        int m = index % (2 * ${b});

        // from 0,1,2, 3, 4   or  0,1,2, 3, 4, 5
        // to   0,1,2,-2,-1   or  0,1,2,-3,-2,-1
        int mm = m <= (2 * ${b} - 1) / 2 ? m : m - 2 * ${b};

        float out_re = 0.0;
        float out_im = 0.0;

        for (int l = abs(mm); l < ${nl}; ++l) {
            int s = l * l + (l + mm);

            float in_re = in[(s * ${nbatch} + i) * 2 + 0];
            float in_im = in[(s * ${nbatch} + i) * 2 + 1];
            float w = wig[beta * ${nspec} + s];

            out_re += in_re * w;
            out_im += in_im * w;
        }

        out[index * 2 + 0] = out_re;
        out[index * 2 + 1] = out_im;
    }
}
''').substitute({'b': b, 'nbatch': nbatch, 'nl': nl, 'nspec': nl**2})

    return cuda_utils.compile_kernel(kernel, b's2ifft.cu', 'main_')


class S2_fft_real(torch.autograd.Function):
    def __init__(self, b_out=None):
        super(S2_fft_real, self).__init__()
        self.b_in = None
        self.b_out = b_out

    def forward(self, x):  # pylint: disable=W
        from s2cnn.utils.complex import as_complex
        self.b_in = x.size(-1) // 2
        return s2_fft(as_complex(x), b_out=self.b_out)

    def backward(self, grad_output):  # pylint: disable=W
        return s2_ifft(grad_output, for_grad=True, b_out=self.b_in)[..., 0]


class S2_ifft_real(torch.autograd.Function):
    def __init__(self, b_out=None):
        super(S2_ifft_real, self).__init__()
        self.b_in = None
        self.b_out = b_out

    def forward(self, x):  # pylint: disable=W
        nspec = x.size(0)
        self.b_in = round(nspec**0.5)
        return s2_ifft(x, b_out=self.b_out)[..., 0]

    def backward(self, grad_output):  # pylint: disable=W
        from s2cnn.utils.complex import as_complex
        return s2_fft(as_complex(grad_output), for_grad=True, b_out=self.b_in)
