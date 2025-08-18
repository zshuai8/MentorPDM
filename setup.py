from setuptools import setup, find_packages

setup(
    name='mentorpdm',
    version='1.0.0',
    description='Multi-modal Graph Neural Network for Predictive Maintenance',
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.21.0',
        'pandas>=1.3.0',
        'scipy>=1.7.0',
        'matplotlib>=3.4.0',
        'scikit-learn>=1.0.0',
        'torch>=1.12.0',
        'torch-geometric>=2.1.0',
        'tqdm>=4.62.0',
        'PyYAML>=6.0',
    ],
)