from setuptools import setup


setup(
    name='planners',
    version='0.1.0',
    packages=[
        'planners',
        'planners.core',
        'planners.core.configurator',
        'planners.core.controllers',
        'planners.core.predictors',
        'planners.core.utils',
        'planners.core.visualizer',
        'planners.configs',
    ],
    install_requires=[
        'pygame',
        'numpy',
        'pillow',
        'numba',
        'scipy',
        'do_mpc',
        'casadi'
    ],
    extras_require={
        'test': [
            'pylint',
            'pytest',
        ],
    },
)
