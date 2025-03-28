
from setuptools import setup

setup(
    name                 = 'scalign',
    version              = '0.2.1',
    description          = 'align query dataset to reference atlases.',
    author               = 'Zheng Yang',
    author_email         = 'xornent@outlook.com',
    license              = 'MIT',
    packages             = ['scalign'],
    install_requires     = [
        'umap-learn >= 0.5.0',
        'scvi-tools >= 1.3.0',
        'scanpy',
        'pandas',
        'numpy',
        'matplotlib',
        'seaborn',
        'dill',
        'torch >= 2.0',
        'pytorch_lightning >= 2.0'
    ],
    include_package_data = True,
    zip_safe             = False
)