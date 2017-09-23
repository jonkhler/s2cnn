import s2cnn.ops.gpu.lib_cufft as cufft

class Plan1d_c2c:
    def __init__(self, N, batch=1, istride=1, idist=None, ostride=None, odist=None):
        if idist is None:
            idist = N
        if ostride is None:
            ostride = istride
        if odist is None:
            odist = idist

        self.handler = cufft.plan1d_c2c(N, istride, idist, ostride, odist, batch)

    def __call__(self, in_, out_, sign):
        cufft.execute_c2c(self.handler, in_, out_, sign)

    def __del__(self):
        cufft.destroy(self.handler)

class Plan2d_c2c:
    def __init__(self, N0, N1, batch=1, istride=1, idist=None, ostride=None, odist=None):
        if idist is None:
            idist = N0 * N1
        if ostride is None:
            ostride = istride
        if odist is None:
            odist = idist

        self.handler = cufft.plan2d_c2c(N0, N1, istride, idist, ostride, odist, batch)

    def __call__(self, in_, out_, sign):
        cufft.execute_c2c(self.handler, in_, out_, sign)

    def __del__(self):
        cufft.destroy(self.handler)

class Plan2d_r2c:
    def __init__(self, N0, N1, batch=1, istride=1, idist=None, ostride=None, odist=None):
        if idist is None:
            idist = N0 * N1
        if ostride is None:
            ostride = istride
        if odist is None:
            odist = N0 * (N1 // 2 + 1)

        self.handler = cufft.plan2d_r2c(N0, N1, istride, idist, ostride, odist, batch)

    def __call__(self, in_, out_):
        cufft.execute_r2c(self.handler, in_, out_)

    def __del__(self):
        cufft.destroy(self.handler)
