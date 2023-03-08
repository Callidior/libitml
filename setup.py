from setuptools import setup, Extension

def main():
    setup(
        name="itml",
        version="1.0.0",
        description="Efficient C/C++ implementation of Information-Theoretic Metric Learning (ITML)",
        author="Björn Barz",
        url="https://github.com/Callidior/libitml",
        py_modules=["itml"],
        ext_modules=[Extension("itml.lib", ["libitml.cc"], extra_compile_args=["-march=native", "-fopenmp"], extra_link_args=["-fopenmp"])],
    )

if __name__ == '__main__':
    main()