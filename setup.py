from setuptools import setup

setup(
    name='torch-dct',
    version='0.1.2',
    packages=['torch_dct'],
    install_requires=['torch>=0.4.1'],
    url='https://github.com/zh217/torch-dct',
    license='MIT',
    author='Ziyang Hu',
    author_email='hu.ziyang@cantab.net',
    description='Discrete Cosine Transform (DCT) for pytorch'
)
