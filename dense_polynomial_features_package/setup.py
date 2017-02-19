from Cython.Build import cythonize
from distutils.core import setup
from os.path import realpath

setup(
    name="dense_polynomial_features",
    version="1.0.0",
    packages=['dense_polynomial_features/'],
    description="Baseline.",
    author="Andrew Nystrom",
    author_email="AWNystrom@gmail.com",
    url="https://github.com/AWNystrom/SparsePolynomialFeatures",
    keywords=["sparse", "feature", "features", "polynomial", "interaction", "combination", "combinations"],
    license="Apache 2.0",
    long_description=open(realpath('README.md')).read(),
    classifiers=["Programming Language :: Python",
                 "Programming Language :: Python :: 2.7",
                 "Programming Language :: Python :: 2",
                 "License :: OSI Approved :: Apache Software License",
                 "Operating System :: OS Independent",
                 "Development Status :: 4 - Beta",
                 "Intended Audience :: Developers"
                 ],
    install_requires=['numpy'],
    ext_modules=cythonize("./dense_polynomial_features/dense_polynomial_features.pyx")
    )
