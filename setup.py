# Noventis-Data/setup.py

from setuptools import setup, find_packages

with open('requirement.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='noventis',
    version='0.1.0',
    description='A data cleaning and analysis toolkit.',
    author='Your Name', # Ganti dengan nama Anda
    author_email='your.email@example.com', # Ganti dengan email Anda
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)