from setuptools import setup, find_packages

setup(
    name='gsvp-py',
    version='0.1.0',
    packages=find_packages(),
    install_requires = [
    'numpy',
    'pandas',
    'matplotlib',
    'scipy',
    'scikit-learn',
    'kneefinder',
]
)


