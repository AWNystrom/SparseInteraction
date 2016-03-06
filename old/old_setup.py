from Cython.Build import cythonize
from distutils.core import setup
from os.path import realpath

setup(
    name="sparse_interaction",
    version="0.0.0",
    packages=['sparse_interaction/'],
    description="Compute second degree interaction features on a sparse matrix.",
    author="Andrew Nystrom",
    author_email="AWNystrom@gmail.com",
    url="https://github.com/AWNystrom/SparseInteraction",
    keywords=["sparse", "feature", "interaction", "combination", "combinations"],
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
    install_requires=['numpy', 'sklearn', 'scipy'],
    ext_modules=cythonize("./sparse_interaction/sparse_interaction.pyx")
    )