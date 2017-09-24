from torch.utils.ffi import create_extension

ffi_plan_cufft = create_extension(
    's2cnn.ops.gpu.lib_cufft',
    headers=['s2cnn/ops/gpu/plan_cufft.h'],
    package=True,
    sources=['s2cnn/ops/gpu/plan_cufft.c'],
    define_macros=[('WITH_CUDA', None)],
    relative_to=__file__,
    libraries=['cufft'],
    with_cuda=True
)

if __name__ == '__main__':
    print("build CUDA dependencies")
    ffi_plan_cufft.build()
