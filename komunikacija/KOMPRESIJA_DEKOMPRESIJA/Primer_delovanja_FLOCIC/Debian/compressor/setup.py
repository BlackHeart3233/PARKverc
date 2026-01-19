from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "compressor",
        ["pycompress.cpp", "kompresija.cpp", "decompress.cpp"],
        cxx_std=17,
    ),
]

setup(
    name="compressor",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
