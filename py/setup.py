from setuptools import Extension
from setuptools import setup
import pybind11

ext_modules = [
    Extension(
        "philentropy.philentropy_cpp",
        sources=[
            "../src/python_bindings.cpp",
            "../src/dist_interface.cpp",
        ],
        include_dirs=[
            pybind11.get_include(),
            "../inst/include",
            "../src",
        ],
        language="c++",
        extra_compile_args=["-std=c++11", "-O3"],
    ),
]

setup(ext_modules=ext_modules)
