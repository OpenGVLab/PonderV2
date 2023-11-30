from setuptools import setup
from torch.utils import cpp_extension

setup(
    name="smooth_sampler",
    version="0.0.1",
    author="Tymoteusz Bleja, Jingwen Wang",
    author_email="tymoteusz.bleja@gmail.com, jingwen.wang.17@ucl.ac.uk",
    description=(
        "Trilinear sampler with smoothstep and double backpropagation - Pytorch extension."
    ),
    ext_modules=[
        cpp_extension.CUDAExtension(
            "smooth_sampler._C",
            [
                "smooth_sampler/csrc/smooth_sampler.cpp",
                "smooth_sampler/csrc/smooth_sampler_kernel.cu",
            ],
        )
    ],
    packages=["smooth_sampler"],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: GPU :: NVIDIA CUDA",
        "Topic :: Utilities",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: C++",
        "Programming Language :: CUDA",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
