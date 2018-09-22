from setuptools import setup

setup(
    name='torch-dct',
    version='0.1.5',
    packages=['torch_dct'],
    platforms='any',
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3'
    ],
    install_requires=['torch>=0.4.1'],
    url='https://github.com/zh217/torch-dct',
    license='MIT',
    author='Ziyang Hu',
    author_email='hu.ziyang@cantab.net',
    description='Discrete Cosine Transform (DCT) for pytorch',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown'
)
