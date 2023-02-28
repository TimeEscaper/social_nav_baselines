from setuptools import setup


setup(
    name='planners',
    version='0.1.1',
    packages=[
        'core',
        'core.controllers',
        'core.predictors',
        'core.utils',
        'core.visualizer'
        'examples'
    ],
    install_requires=[
        'pygame',
        'numpy',
        'pillow',
        'numba',
        'scipy',
        'do_mpc',
        'casadi',
        'pytorch_mppi',
        'fire',
        'pyyaml'
    ],
    extras_require={
        'test': [
            'pylint',
            'pytest',
        ],
    },
)
