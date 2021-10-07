# pylint: disable=R,C,E1101,E1102
import math
from functools import lru_cache
import torch
import torch.cuda
from s2cnn.utils.decorator import cached_dirpklgz


# inspired by https://gist.github.com/szagoruyko/89f83b6f5f4833d3c8adf81ee49f22a8


def so3_fft(x, for_grad=False, b_out=None):
    '''
    :param x: [..., beta, alpha, gamma, complex]
    :return: [l * m * n, ..., complex]
    '''
    assert x.size(-1) == 2, x.size()
    b_in = x.size(-2) // 2
    assert x.size(-2) == 2 * b_in
    assert x.size(-3) == 2 * b_in
    assert x.size(-4) == 2 * b_in
    if b_out is None:
        b_out = b_in
    batch_size = x.size()[:-4]

    x = x.view(-1, 2 * b_in, 2 * b_in, 2 * b_in, 2)  # [batch, beta, alpha, gamma, complex]

    '''
    :param x: [batch, beta, alpha, gamma, complex] (nbatch, 2 b_in, 2 b_in, 2 b_in, 2)
    :return: [l * m * n, batch, complex] (b_out (4 b_out**2 - 1) // 3, nbatch, 2)
    '''
    nspec = b_out * (4 * b_out ** 2 - 1) // 3
    nbatch = x.size(0)

    wigner = _setup_wigner(b_in, nl=b_out, weighted=not for_grad, device=x.device)  # [beta, l * m * n]

    x = torch.view_as_real(torch.fft.fftn(torch.view_as_complex(x),dim=[2,3]))  # [batch, beta, m, n, complex]

    output = x.new_empty((nspec, nbatch, 2))
    if x.is_cuda and x.dtype == torch.float32:
        cuda_kernel = _setup_so3fft_cuda_kernel(b_in=b_in, b_out=b_out, nbatch=nbatch, real_input=False, device=x.device.index)
        cuda_kernel(x, wigner, output)  # [l * m * n, batch, complex]
    else:
        if b_in < b_out:
            output.fill_(0)
        for l in range(b_out):
            s = slice(l * (4 * l ** 2 - 1) // 3, l * (4 * l ** 2 - 1) // 3 + (2 * l + 1) ** 2)
            l1 = min(l, b_in - 1)  # if b_out > b_in, consider high frequencies as null

            xx = x.new_zeros((x.size(0), x.size(1), 2 * l + 1, 2 * l + 1, 2))
            xx[:, :, l: l + l1 + 1, l: l + l1 + 1] = x[:, :, :l1 + 1, :l1 + 1]
            if l1 > 0:
                xx[:, :, l - l1:l, l: l + l1 + 1] = x[:, :, -l1:, :l1 + 1]
                xx[:, :, l: l + l1 + 1, l - l1:l] = x[:, :, :l1 + 1, -l1:]
                xx[:, :, l - l1:l, l - l1:l] = x[:, :, -l1:, -l1:]

            out = torch.einsum("bmn,zbmnc->mnzc", (wigner[:, s].view(-1, 2 * l + 1, 2 * l + 1), xx))
            output[s] = out.view((2 * l + 1) ** 2, -1, 2)

    output = output.view(-1, *batch_size, 2)  # [l * m * n, ..., complex]
    return output


def so3_rfft(x, for_grad=False, b_out=None):
    '''
    :param x: [..., beta, alpha, gamma]
    :return: [l * m * n, ..., complex]
    '''
    b_in = x.size(-1) // 2
    assert x.size(-1) == 2 * b_in
    assert x.size(-2) == 2 * b_in
    assert x.size(-3) == 2 * b_in
    if b_out is None:
        b_out = b_in
    batch_size = x.size()[:-3]

    x = x.contiguous().view(-1, 2 * b_in, 2 * b_in, 2 * b_in)  # [batch, beta, alpha, gamma]

    '''
    :param x: [batch, beta, alpha, gamma] (nbatch, 2 b_in, 2 b_in, 2 b_in)
    :return: [l * m * n, batch, complex] (b_out (4 b_out**2 - 1) // 3, nbatch, 2)
    '''
    nspec = b_out * (4 * b_out ** 2 - 1) // 3
    nbatch = x.size(0)

    wigner = _setup_wigner(b_in, nl=b_out, weighted=not for_grad, device=x.device)

    output = x.new_empty((nspec, nbatch, 2))
    if x.is_cuda and x.dtype == torch.float32:
        x = torch.view_as_real(torch.fft.rfftn(x, dim=[2,3]))  # [batch, beta, m, n, complex]
        cuda_kernel = _setup_so3fft_cuda_kernel(b_in=b_in, b_out=b_out, nbatch=nbatch, real_input=True, device=x.device.index)
        cuda_kernel(x, wigner, output)
    else:
        x = torch.view_as_real(torch.fft.rfftn(torch.view_as_complex(torch.stack((x, torch.zeros_like(x)), dim=-1)), dim=[2,3]))
        if b_in < b_out:
            output.fill_(0)
        for l in range(b_out):
            s = slice(l * (4 * l**2 - 1) // 3, l * (4 * l**2 - 1) // 3 + (2 * l + 1) ** 2)
            l1 = min(l, b_in - 1)  # if b_out > b_in, consider high frequencies as null

            xx = x.new_zeros((x.size(0), x.size(1), 2 * l + 1, 2 * l + 1, 2))
            xx[:, :, l: l + l1 + 1, l: l + l1 + 1] = x[:, :, :l1 + 1, :l1 + 1]
            if l1 > 0:
                xx[:, :, l - l1:l, l: l + l1 + 1] = x[:, :, -l1:, :l1 + 1]
                xx[:, :, l: l + l1 + 1, l - l1:l] = x[:, :, :l1 + 1, -l1:]
                xx[:, :, l - l1:l, l - l1:l] = x[:, :, -l1:, -l1:]

            out = torch.einsum("bmn,zbmnc->mnzc", (wigner[:, s].view(-1, 2 * l + 1, 2 * l + 1), xx))
            output[s] = out.view((2 * l + 1) ** 2, -1, 2)
    
    output = output.view(-1, *batch_size, 2)  # [l * m * n, ..., complex]
    return output


def so3_ifft(x, for_grad=False, b_out=None):
    '''
    :param x: [l * m * n, ..., complex]
    '''
    assert x.size(-1) == 2
    nspec = x.size(0)
    b_in = round((3 / 4 * nspec) ** (1 / 3))
    assert nspec == b_in * (4 * b_in ** 2 - 1) // 3
    if b_out is None:
        b_out = b_in
    batch_size = x.size()[1:-1]

    x = x.view(nspec, -1, 2)  # [l * m * n, batch, complex] (nspec, nbatch, 2)

    '''
    :param x: [l * m * n, batch, complex] (b_in (4 b_in**2 - 1) // 3, nbatch, 2)
    :return: [batch, beta, alpha, gamma, complex] (nbatch, 2 b_out, 2 b_out, 2 b_out, 2)
    '''
    nbatch = x.size(1)

    wigner = _setup_wigner(b_out, nl=b_in, weighted=for_grad, device=x.device)  # [beta, l * m * n] (2 * b_out, nspec)

    output = x.new_empty((nbatch, 2 * b_out, 2 * b_out, 2 * b_out, 2))
    if x.is_cuda and x.dtype == torch.float32:
        cuda_kernel = _setup_so3ifft_cuda_kernel(b_in=b_in, b_out=b_out, nbatch=nbatch, real_output=False, device=x.device.index)
        cuda_kernel(x, wigner, output)  # [batch, beta, m, n, complex]
    else:
        output.fill_(0)
        for l in range(min(b_in, b_out)):
            s = slice(l * (4 * l**2 - 1) // 3, l * (4 * l**2 - 1) // 3 + (2 * l + 1) ** 2)
            out = torch.einsum("mnzc,bmn->zbmnc", (x[s].view(2 * l + 1, 2 * l + 1, -1, 2), wigner[:, s].view(-1, 2 * l + 1, 2 * l + 1)))
            l1 = min(l, b_out - 1)  # if b_out < b_in
            output[:, :, :l1 + 1, :l1 + 1] += out[:, :, l: l + l1 + 1, l: l + l1 + 1]
            if l > 0:
                output[:, :, -l1:, :l1 + 1] += out[:, :, l - l1: l, l: l + l1 + 1]
                output[:, :, :l1 + 1, -l1:] += out[:, :, l: l + l1 + 1, l - l1: l]
                output[:, :, -l1:, -l1:] += out[:, :, l - l1: l, l - l1: l]

    output = torch.view_as_real(torch.fft.ifftn(torch.view_as_complex(output), dim=[2,3])) * output.size(-2) ** 2  # [batch, beta, alpha, gamma, complex]    
    output = output.view(*batch_size, 2 * b_out, 2 * b_out, 2 * b_out, 2)
    return output


def so3_rifft(x, for_grad=False, b_out=None):
    '''
    :param x: [l * m * n, ..., complex]
    '''
    assert x.size(-1) == 2
    nspec = x.size(0)
    b_in = round((3 / 4 * nspec) ** (1 / 3))
    assert nspec == b_in * (4 * b_in ** 2 - 1) // 3
    if b_out is None:
        b_out = b_in
    batch_size = x.size()[1:-1]

    x = x.view(nspec, -1, 2)  # [l * m * n, batch, complex] (nspec, nbatch, 2)

    '''
    :param x: [l * m * n, batch, complex] (b_in (4 b_in**2 - 1) // 3, nbatch, 2)
    :return: [batch, beta, alpha, gamma] (nbatch, 2 b_out, 2 b_out, 2 b_out)
    '''
    nbatch = x.size(1)

    wigner = _setup_wigner(b_out, nl=b_in, weighted=for_grad, device=x.device)  # [beta, l * m * n] (2 * b_out, nspec)

    output = x.new_empty((nbatch, 2 * b_out, 2 * b_out, 2 * b_out, 2))
    if x.is_cuda and x.dtype == torch.float32:
        cuda_kernel = _setup_so3ifft_cuda_kernel(b_in=b_in, b_out=b_out, nbatch=nbatch, real_output=True, device=x.device.index)
        cuda_kernel(x, wigner, output)  # [batch, beta, m, n, complex]
    else:
        # TODO can be optimized knowing that the output is real, like in _setup_so3ifft_cuda_kernel(real_output=True)
        output.fill_(0)
        for l in range(min(b_in, b_out)):
            s = slice(l * (4 * l**2 - 1) // 3, l * (4 * l**2 - 1) // 3 + (2 * l + 1) ** 2)
            out = torch.einsum("mnzc,bmn->zbmnc", (x[s].view(2 * l + 1, 2 * l + 1, -1, 2), wigner[:, s].view(-1, 2 * l + 1, 2 * l + 1)))
            l1 = min(l, b_out - 1)  # if b_out < b_in
            output[:, :, :l1 + 1, :l1 + 1] += out[:, :, l: l + l1 + 1, l: l + l1 + 1]
            if l > 0:
                output[:, :, -l1:, :l1 + 1] += out[:, :, l - l1: l, l: l + l1 + 1]
                output[:, :, :l1 + 1, -l1:] += out[:, :, l: l + l1 + 1, l - l1: l]
                output[:, :, -l1:, -l1:] += out[:, :, l - l1: l, l - l1: l]

    output = torch.view_as_real(torch.fft.ifftn(torch.view_as_complex(output), dim=[2,3])) * output.size(-2) ** 2  # [batch, beta, alpha, gamma, complex]
    output = output[..., 0]  # [batch, beta, alpha, gamma]
    output = output.contiguous()

    output = output.view(*batch_size, 2 * b_out, 2 * b_out, 2 * b_out)
    return output


@lru_cache(maxsize=32)
def _setup_wigner(b, nl, weighted, device):
    dss = _setup_so3_fft(b, nl, weighted)
    dss = torch.tensor(dss, dtype=torch.float32, device=device)  # [beta, l * m * n] # pylint: disable=E1102
    return dss.contiguous()


@cached_dirpklgz("cache/setup_so3_fft")
def _setup_so3_fft(b, nl, weighted):
    from lie_learn.representations.SO3.wigner_d import wigner_d_matrix
    import lie_learn.spaces.S3 as S3
    import numpy as np
    import logging

    betas = (np.arange(2 * b) + 0.5) / (2 * b) * np.pi
    w = S3.quadrature_weights(b)
    assert len(w) == len(betas)

    logging.getLogger("trainer").info("Compute Wigner: b=%d nbeta=%d nl=%d nspec=%d", b, len(betas), nl, nl ** 2)

    dss = []
    for b, beta in enumerate(betas):
        ds = []
        for l in range(nl):
            d = wigner_d_matrix(l, beta,
                                field='complex', normalization='quantum', order='centered', condon_shortley='cs')
            d = d.reshape(((2 * l + 1) ** 2,))

            if weighted:
                d *= w[b]
            else:
                d *= 2 * l + 1

            # d # [m * n]
            ds.append(d)
        ds = np.concatenate(ds)  # [l * m * n]
        dss.append(ds)
    dss = np.stack(dss)  # [beta, l * m * n]
    return dss


@lru_cache(maxsize=32)
def _setup_so3fft_cuda_kernel(b_in, b_out, nbatch, real_input, device=0):
    kernel = '''
#define B_IN {}
#define B_OUT {}
#define NSPEC {}
#define NBATCH {}
'''.format(b_in, b_out, b_out * (4 * b_out ** 2 - 1) // 3, nbatch)

    if real_input:
        kernel += '''
#define REAL_IN
'''

    kernel += '''
#define MOD(i, n) (((i) + (n)) % (n))
#define MAX(x, y) ((x) < (y) ? (y) : (x))
#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

extern "C"
__global__ void main_(const float* in, const float* wig, float* out)
{
    // blockIdx = (l, batch, mn)
    // blockDim = (32, 32, 1)
    // threadIdx = (sub l, sub batch, 0)
    // gridDim = (b / 32, nbatch / 32, (2b-1)**2)
    int m = (blockIdx.z / (2 * B_OUT - 1)) - (B_OUT - 1);
    int n = (blockIdx.z % (2 * B_OUT - 1)) - (B_OUT - 1);

    int l_min = MAX(abs(m), abs(n));

    if (blockIdx.x * 32 + 31 < l_min) {
        // for blocks fully out of l-range
        return; // note: this return does not depend on threadIdx
    }

#ifdef REAL_IN
    if (n < 0 || (n == 0 && m < 0)) {
        return; // note: this return does not depend on threadIdx
    }
#endif

    int batch = blockIdx.y * 32 + threadIdx.y;
    int l = blockIdx.x * 32 + threadIdx.x;

    int lmn = (4 * l*l - 1) * l / 3 + (l+m) * (2 * l + 1) + (l+n);

    float sum_re = 0.0;
    float sum_im = 0.0;

    for (int tile = 0; tile < CEIL_DIV(2 * B_IN, 32); ++tile) {
        __shared__ float tileA[32][32][2];
        __shared__ float tileB[32][32];

        int beta = tile * 32 + threadIdx.x;
#ifdef REAL_IN
        // `in` shape is (NBATCH, 2 * B_IN, 2 * B_IN, B_IN + 1, 2)
        // http://www.fftw.org/fftw3_doc/Multi_002dDimensional-DFTs-of-Real-Data.html
        int i = (((batch * 2*B_IN + beta) * 2*B_IN + MOD(m, 2*B_IN)) * (B_IN + 1) + n) * 2;
#else
        int i = (((batch * 2*B_IN + beta) * 2*B_IN + MOD(m, 2*B_IN)) * 2*B_IN + MOD(n, 2*B_IN)) * 2;
#endif
        tileA[threadIdx.y][threadIdx.x][0] = beta < 2*B_IN && batch < NBATCH && m < B_IN && n < B_IN && m > -B_IN && n > -B_IN ? in[i + 0] : 0.0;
        tileA[threadIdx.y][threadIdx.x][1] = beta < 2*B_IN && batch < NBATCH && m < B_IN && n < B_IN && m > -B_IN && n > -B_IN ? in[i + 1] : 0.0;
        // add constraints to m and n to remove aliasing (when b_out > b_in)

        beta = tile * 32 + threadIdx.y;
        tileB[threadIdx.y][threadIdx.x] = beta < 2*B_IN && l_min <= l && l < B_OUT ? wig[beta * NSPEC + lmn] : 0.0;

        __syncthreads();

        for (int beta = 0; beta < 32; ++beta) {
            sum_re += tileA[threadIdx.y][beta][0] * tileB[beta][threadIdx.x];
            sum_im += tileA[threadIdx.y][beta][1] * tileB[beta][threadIdx.x];
        }

        __syncthreads();
    }

    // About this if: some blocks are used to compute but not to save the results
    if (l_min <= l && l < B_OUT && batch < NBATCH) {
        out[(lmn * NBATCH + batch) * 2 + 0] = sum_re;
        out[(lmn * NBATCH + batch) * 2 + 1] = sum_im;

#ifdef REAL_IN
        lmn = (4 * l*l - 1) * l / 3 + (l-m) * (2 * l + 1) + (l-n);
        float fudge = (m - n) % 2 == 0 ? 1.0 : -1.0;
        out[(lmn * NBATCH + batch) * 2 + 0] = fudge * sum_re;
        out[(lmn * NBATCH + batch) * 2 + 1] = -fudge * sum_im;
#endif
    }
}
'''
    import s2cnn.utils.cuda as cuda_utils
    kernel = cuda_utils.compile_kernel(kernel, 'so3fft.cu', 'main_')
    stream = cuda_utils.Stream(ptr=torch.cuda.current_stream().cuda_stream)

    def fun(x, wigner, output):
        assert output.is_contiguous()
        kernel(block=(32, 32, 1),
               grid=(math.ceil(b_out / 32), math.ceil(nbatch / 32), (2 * b_out - 1) ** 2),
               args=[x.contiguous().data_ptr(), wigner.contiguous().data_ptr(), output.data_ptr()],
               stream=stream)

    return fun


@lru_cache(maxsize=32)
def _setup_so3ifft_cuda_kernel(b_in, b_out, nbatch, real_output, device=0):
    kernel = '''
#define B_IN {}
#define B_OUT {}
#define NSPEC {}
#define NBATCH {}
'''.format(b_in, b_out, b_in * (4 * b_in ** 2 - 1) // 3, nbatch)

    if real_output:
        kernel += '''
#define REAL_OUT
'''

    kernel += '''
#define MOD(i, n) (((i) + (n)) % (n))
#define MAX(x, y) ((x) < (y) ? (y) : (x))
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

extern "C"
__global__ void main_(const float* in, const float* wig, float* out)
{
    int m = (blockIdx.z / (2 * B_OUT - 1)) - (B_OUT - 1);
    int n = (blockIdx.z % (2 * B_OUT - 1)) - (B_OUT - 1);

#ifdef REAL_OUT
    if (n < 0 || (n == 0 && m < 0)) {
        return; // note: this return does not depend on threadIdx
    }
#endif

    int l_min = MAX(abs(m), abs(n));

    int batch = blockIdx.y * 32 + threadIdx.y;

    float sum_re = 0.0;
    float sum_im = 0.0;

    // will not calculate when l > min(b_in, b_out)-1
    for (int tile = 0; tile < CEIL_DIV(MIN(B_IN, B_OUT) - l_min, 32); ++tile) {
        __shared__ float tileA[2][32][32];
        __shared__ float tileB[32][32+1];

        int l = l_min + tile * 32 + threadIdx.x;
        int lmn = (4 * l*l - 1) * l / 3 + (l+m) * (2 * l + 1) + (l+n);
        int i = (lmn * NBATCH + batch) * 2;
        tileA[0][threadIdx.y][threadIdx.x] = l < MIN(B_IN, B_OUT) && batch < NBATCH && m < B_OUT && n < B_OUT && m > -B_OUT && n > -B_OUT ? in[i + 0] : 0.0;
        tileA[1][threadIdx.y][threadIdx.x] = l < MIN(B_IN, B_OUT) && batch < NBATCH && m < B_OUT && n < B_OUT && m > -B_OUT && n > -B_OUT ? in[i + 1] : 0.0;
        // add constraints to m and n to remove aliasing (when b_out < b_in)

        int beta = blockIdx.x * 32 + threadIdx.y;
        tileB[threadIdx.x][threadIdx.y] = l < MIN(B_IN, B_OUT) && beta < 2*B_OUT ? wig[beta * NSPEC + lmn] : 0.0;

        __syncthreads();

        for (int l = 0; l < 32; ++l) {
            sum_re += tileA[0][threadIdx.y][l] * tileB[l][threadIdx.x];
            sum_im += tileA[1][threadIdx.y][l] * tileB[l][threadIdx.x];
        }

        __syncthreads();
    }

    int beta = blockIdx.x * 32 + threadIdx.x;

    if (beta < 2*B_OUT && batch < NBATCH) {
        int i = (((batch * 2*B_OUT + beta) * 2*B_OUT + MOD(m, 2*B_OUT)) * 2*B_OUT + MOD(n, 2*B_OUT)) * 2;
        out[i + 0] = sum_re;
        out[i + 1] = sum_im;

#ifdef REAL_OUT
        i = (((batch * 2*B_OUT + beta) * 2*B_OUT + MOD(-m, 2*B_OUT)) * 2*B_OUT + MOD(-n, 2*B_OUT)) * 2;
        out[i + 0] = sum_re;
        out[i + 1] = -sum_im;
#endif
    }
}
'''
    import s2cnn.utils.cuda as cuda_utils
    kernel = cuda_utils.compile_kernel(kernel, 'so3ifft.cu', 'main_')
    stream = cuda_utils.Stream(ptr=torch.cuda.current_stream().cuda_stream)

    def fun(x, wigner, output):
        output[:] = 0
        kernel(block=(32, 32, 1),
               grid=(math.ceil(2 * b_out / 32), math.ceil(nbatch / 32), (2 * b_out - 1) ** 2),
               args=[x.data_ptr(), wigner.data_ptr(), output.data_ptr()],
               stream=stream)

    return fun


class SO3_fft_real(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, b_out=None):  # pylint: disable=W
        ctx.b_out = b_out
        ctx.b_in = x.size(-1) // 2
        return so3_rfft(x, b_out=ctx.b_out)

    @staticmethod
    def backward(self, grad_output):  # pylint: disable=W
        # ifft of grad_output is not necessarily real, therefore we cannot use rifft
        return so3_ifft(grad_output, for_grad=True, b_out=self.b_in)[..., 0], None


class SO3_ifft_real(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, b_out=None):  # pylint: disable=W
        nspec = x.size(0)
        ctx.b_out = b_out
        ctx.b_in = round((3 / 4 * nspec) ** (1 / 3))
        return so3_rifft(x, b_out=ctx.b_out)

    @staticmethod
    def backward(ctx, grad_output):  # pylint: disable=W
        return so3_rfft(grad_output, for_grad=True, b_out=ctx.b_in), None
