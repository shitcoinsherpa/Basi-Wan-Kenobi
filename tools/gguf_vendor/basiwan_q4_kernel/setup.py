"""Build script for the native Marlin Q4_K + Q6_K CUDA extension.

Run manually:
    cd tools/gguf_vendor/q4_marlin
    python setup.py build_ext --inplace

Or call `_load_extension()` from __init__.py to build via torch.utils.cpp_extension.load.
"""
from __future__ import annotations

from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

HERE = Path(__file__).resolve().parent

setup(
    name="wan_q4k_q6k_marlin",
    ext_modules=[
        CUDAExtension(
            name="wan_q4k_q6k_marlin_ext",
            sources=[str(HERE / "q4k_q6k_marlin.cu")],
            include_dirs=[str(HERE)],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "-std=c++17",
                    "--use_fast_math",
                    "-gencode=arch=compute_89,code=sm_89",
                    "--maxrregcount=128",
                    "--ptxas-options=-v",  # print register/SMEM usage at compile
                ],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
