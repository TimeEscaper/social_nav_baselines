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
