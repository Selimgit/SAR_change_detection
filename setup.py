# setup.py

from setuptools import setup, find_packages

setup(
    name='SAR_change_detector',
    version='0.1.3',
    author='Selim Behloul',
    author_email='selim.behloul@gmail.com',
    description='A package for detecting changes between two images using Isolation Forest',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Selimgit/SAR_change_detection',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scikit-learn',
        'scipy'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU Affero General Public License v3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)

