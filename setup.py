from setuptools import setup, find_packages

setup(
    name='sketch-to-artwork',
    version='0.0.1',
    description='Artwork Generation based on Sketches',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'tqdm',
    ],
)
