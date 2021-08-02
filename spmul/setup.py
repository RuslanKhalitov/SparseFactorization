from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='spmul_cuda',
    ext_modules=[
        CUDAExtension('spmul_cuda', [
            'spmul_cuda.cu'
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
