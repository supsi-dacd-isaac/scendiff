import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="scendiff",
    version="0.1",
    author="Lorenzo Nespoli",
    author_email="lorenzo.nespoli@supsi.ch",
    description="Obtain stochastic trees from scenarios by gradient descent",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/supsi-dacd-isaac/scendiff",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNUv3",
        "Operating System :: OS Independent",
    ],
    install_requires=["jax>=0.4.10",
                      "jaxlib>=0.4.1",
                      "matplotlib>=3.7.1",
                      "networkx>=3.1",
                      "numpy>=1.24.3",
                      "pandas>=2.0.1",
                      "scipy>=1.10.1",
                      "setuptools>=65.5.1",
                      "seaborn>=0.12.2",
                      ],
    python_requires='>=3.8',
)