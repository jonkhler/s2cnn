# pylint: disable=R,C,E1101,E1102
from functools import lru_cache
import torch
import torch.cuda
from string import Template
from s2cnn.utils.decorator import cached_dirpklgz


# inspired by https://gist.github.com/szagoruyko/89f83b6f5f4833d3c8adf81ee49f22a8


def s2_fft(x, for_grad=False, b_out=None):
    '''
    :param x: [..., beta, alpha, complex]
    :return:  [l * m, ..., complex]
    '''
    assert x.size(-1) == 2
    b_in = x.size(-2) // 2
    assert x.size(-2) == 2 * b_in
    assert x.size(-3) == 2 * b_in
    if b_out is None:
        b_out = b_in
    assert b_out <= b_in
    batch_size = x.size()[:-3]

    x = x.view(-1, 2 * b_in, 2 * b_in, 2)  # [batch, beta, alpha, complex]

    '''
    :param x: [batch, beta, alpha, complex] (nbatch, 2 * b_in, 2 * b_in, 2)
    :return: [l * m, batch, complex] (b_out**2, nbatch, 2)
    '''
    nspec = b_out ** 2
    nbatch = x.size(0)

    wigner = _setup_wigner(b_in, nl=b_out, weighted=not for_grad, device=x.device)
    wigner = wigner.view(2 * b_in, -1)  # [beta, l * m] (2 * b_in, nspec)

    x = torch.view_as_real(torch.fft.fft(torch.view_as_complex(x)))  # [batch, beta, m, complex]

    output = x.new_empty((nspec, nbatch, 2))
    if x.is_cuda and x.dtype == torch.float32:
        import s2cnn.utils.cuda as cuda_utils
        cuda_kernel = _setup_s2fft_cuda_kernel(b=b_in, nspec=nspec, nbatch=nbatch, device=x.device.index)
        stream = cuda_utils.Stream(ptr=torch.cuda.current_stream().cuda_stream)
        cuda_kernel(block=(1024, 1, 1),
                    grid=(cuda_utils.get_blocks(nspec * nbatch, 1024), 1, 1),
                    args=[x.contiguous().data_ptr(), wigner.contiguous().data_ptr(), output.data_ptr()],
                    stream=stream)
        # [l * m, batch, complex]
    else:
        for l in range(b_out):
            s = slice(l ** 2, l ** 2 + 2 * l + 1)
            xx = torch.cat((x[:, :, -l:], x[:, :, :l + 1]), dim=2) if l > 0 else x[:, :, :1]
            output[s] = torch.einsum("bm,zbmc->mzc", (wigner[:, s], xx))

    output = output.view(-1, *batch_size, 2)  # [l * m, ..., complex] (nspec, ..., 2)
    return output


def s2_ifft(x, for_grad=False, b_out=None):
    '''
    :param x: [l * m, ..., complex]
    '''
    assert x.size(-1) == 2
    nspec = x.size(0)
    b_in = round(nspec ** 0.5)
    assert nspec == b_in ** 2
    if b_out is None:
        b_out = b_in
    assert b_out >= b_in
    batch_size = x.size()[1:-1]

    x = x.view(nspec, -1, 2)  # [l * m, batch, complex] (nspec, nbatch, 2)

    '''
    :param x: [l * m, batch, complex] (b_in**2, nbatch, 2)
    :return: [batch, beta, alpha, complex] (nbatch, 2 b_out, 2 * b_out, 2)
    '''
    nbatch = x.size(1)

    wigner = _setup_wigner(b_out, nl=b_in, weighted=for_grad, device=x.device)
    wigner = wigner.view(2 * b_out, -1)  # [beta, l * m] (2 * b_out, nspec)

    if x.is_cuda and x.dtype == torch.float32:
        import s2cnn.utils.cuda as cuda_utils
        cuda_kernel = _setup_s2ifft_cuda_kernel(b=b_out, nl=b_in, nbatch=nbatch, device=x.device.index)
        stream = cuda_utils.Stream(ptr=torch.cuda.current_stream().cuda_stream)
        output = x.new_empty((nbatch, 2 * b_out, 2 * b_out, 2))
        cuda_kernel(block=(1024, 1, 1),
                    grid=(cuda_utils.get_blocks(nbatch * (2 * b_out) ** 2, 1024), 1, 1),
                    args=[x.data_ptr(), wigner.data_ptr(), output.data_ptr()],
                    stream=stream)
        # [batch, beta, m, complex] (nbatch, 2 * b_out, 2 * b_out, 2)
    else:
        output = x.new_zeros((nbatch, 2 * b_out, 2 * b_out, 2))
        for l in range(b_in):
            s = slice(l ** 2, l ** 2 + 2 * l + 1)
            out = torch.einsum("mzc,bm->zbmc", (x[s], wigner[:, s]))
            output[:, :, :l + 1] += out[:, :, -l - 1:]
            if l > 0:
                output[:, :, -l:] += out[:, :, :l]

    output = torch.view_as_real(torch.fft.ifft(torch.view_as_complex(output))) * output.size(-2)  # [batch, beta, alpha, complex]
    output = output.view(*batch_size, 2 * b_out, 2 * b_out, 2)
    return output


@lru_cache(maxsize=32)
def _setup_wigner(b, nl, weighted, device):
    dss = _setup_s2_fft(b, nl, weighted)
    dss = torch.tensor(dss, dtype=torch.float32, device=device)  # [beta, l * m] # pylint: disable=E1102
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

    logging.getLogger("trainer").info("Compute Wigner (only columns): b=%d nbeta=%d nl=%d nspec=%d", b, len(betas), nl,
                                      nl ** 2)

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

    dss = np.stack(dss)  # [beta, l * m]
    return dss


@lru_cache(maxsize=32)
def _setup_s2fft_cuda_kernel(b, nspec, nbatch, device=0):
    kernel = Template('''
#define COMPUTE_LM(s) \
    int l = sqrtf(s); \
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

    import s2cnn.utils.cuda as cuda_utils
    return cuda_utils.compile_kernel(kernel, 's2fft.cu', 'main_')


@lru_cache(maxsize=32)
def _setup_s2ifft_cuda_kernel(b, nl, nbatch, device=0):
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
''').substitute({'b': b, 'nbatch': nbatch, 'nl': nl, 'nspec': nl ** 2})

    import s2cnn.utils.cuda as cuda_utils
    return cuda_utils.compile_kernel(kernel, 's2ifft.cu', 'main_')


class S2_fft_real(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, b_out=None):  # pylint: disable=W
        from s2cnn.utils.complex import as_complex
        ctx.b_out = b_out
        ctx.b_in = x.size(-1) // 2
        return s2_fft(as_complex(x), b_out=ctx.b_out)

    @staticmethod
    def backward(ctx, grad_output):  # pylint: disable=W
        return s2_ifft(grad_output, for_grad=True, b_out=ctx.b_in)[..., 0], None


class S2_ifft_real(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, b_out=None):  # pylint: disable=W
        nspec = x.size(0)
        ctx.b_out = b_out
        ctx.b_in = round(nspec ** 0.5)
        return s2_ifft(x, b_out=ctx.b_out)[..., 0]

    @staticmethod
    def backward(ctx, grad_output):  # pylint: disable=W
        from s2cnn.utils.complex import as_complex
        return s2_fft(as_complex(grad_output), for_grad=True, b_out=ctx.b_in), None


def test_s2fft_cuda_cpu():
    x = torch.rand(1, 2, 12, 12, 2)  # [..., beta, alpha, complex]
    z1 = s2_fft(x, b_out=5)
    z2 = s2_fft(x.cuda(), b_out=5).cpu()
    q = (z1 - z2).abs().max().item() / z1.std().item()
    print(q)
    assert q < 1e-4


def test_s2ifft_cuda_cpu():
    x = torch.rand(12 ** 2, 10, 2)  # [l * m, ..., complex]
    z1 = s2_ifft(x, b_out=13)
    z2 = s2_ifft(x.cuda(), b_out=13).cpu()
    q = (z1 - z2).abs().max().item() / z1.std().item()
    print(q)
    assert q < 1e-4


if __name__ == "__main__":
    test_s2fft_cuda_cpu()
    test_s2ifft_cuda_cpu()
