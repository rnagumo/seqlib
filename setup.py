
from setuptools import setup, find_packages


install_requires = [
    "torch==1.5.1",
    "torchvision==0.6.1",
]


setup(
    name="seqlib",
    version="0.1",
    description="2D sequential models by PyTorch",
    packages=find_packages(),
    install_requires=install_requires,
)
