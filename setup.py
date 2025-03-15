
from setuptools import setup

setup(
    name                 = 'scalign',
    version              = '0.1.0',
    description          = 'align query dataset to reference atlases.',
    author               = 'Zheng Yang',
    author_email         = 'xornent@outlook.com',
    license              = 'MIT',
    packages             = ['scalign'],
    install_requires     = [
        'scalign-umap >= 0.1.0, < 0.2.0',
        'scvi-tools >= 1.3.0',
        'scanpy',
        'pandas',
        'numpy',
        'matplotlib',
        'seaborn'
    ],
    extras_require       = {
        "parametric": [
            "tensorflow >= 2.1",
            "keras >= 3.0"
        ]
    },
    include_package_data = True,
    zip_safe             = False
)