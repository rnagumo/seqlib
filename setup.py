
from setuptools import setup, find_packages


install_requires = [
    "torch>=1.6",
    "torchvision>=0.7",
    "scikit-learn>=0.23",
]


extras_require = {
    "testing": ["pytest"],
    "example": [
        "matplotlib>=3.2",
        "tqdm>=4.47",
        "tensorboardX>=2.1",
    ],
}


setup(
    name="seqlib",
    version="0.1",
    description="2D sequential models in PyTorch",
    packages=find_packages(),
    install_requires=install_requires,
    extras_require=extras_require,
)
