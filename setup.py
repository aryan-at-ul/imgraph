from setuptools import find_packages, setup

__version__ = '0.1.2'

install_requires = [
    'tqdm',
    'numpy',
    'scipy',
    'jinja2',
    'requests',
    'pyparsing',
    'scikit-learn',
    'psutil',
]

graphgym_requires = [
    'yacs',
    'hydra-core',
    'protobuf<4.21',
    'pytorch-lightning',
]

modelhub_requires = [
    'huggingface_hub',
]

full_requires = graphgym_requires + modelhub_requires + [
    'ase',
    'h5py',
    'numba',
    'sympy',
    'pandas',
    'captum',
    'rdflib',
    'trimesh',
    'networkx',
    'graphviz',
    'tabulate',
    'matplotlib',
    'torchmetrics',
    'scikit-image',
    'pytorch-memlab',
    'pgmpy',
    'opt_einsum',  # required for pgmpy
    'statsmodels',
    "scikit-image",
    "opencv-python",
    "timm",
    "torchvision",
    "torchmetrics",
    "torchvision",
    "seaborn",
    # "torch-scatter",
    # "torch-sparse",
    # "torch-cluster",
    # "torch-spline-conv",
    # "torch-geometric-temporal",
    "torch",
    "torch-geometric",
]

benchmark_requires = [
    'protobuf<4.21',
    'wandb',
    'pandas',
    'networkx',
    'matplotlib',
]

test_requires = [
    'pytest',
    'pytest-cov',
    'onnx',
    'onnxruntime',
]

dev_requires = test_requires + [
    'pre-commit',
]

setup(
    name='imgraph',
    version=__version__,
    install_requires=full_requires,
    extras_require={
        'graphgym': graphgym_requires,
        'modelhub': modelhub_requires,
        'full': full_requires,
        'benchmark': benchmark_requires,
        'test': test_requires,
        'dev': dev_requires,
    },
    packages=find_packages(),
    # include_package_data=True,  # Ensure that `*.jinja` files are found.
)