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
    install_requires=[
                      ],
    python_requires='>=3.8',
)