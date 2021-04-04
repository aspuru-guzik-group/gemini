#!/usr/bin/env python

# import versioneer
from setuptools import setup

#==============================================================================

def readme():
    with open('README.md', encoding='utf-8') as content:
        return content.read()

def requirements():
    with open('requirements.txt', 'r') as content:
        return content.readlines()

#==============================================================================

setup(
    name='gemini',
    version='0.1.0',
    description='Gemini: Dynamic bias correction for autonomous experimentation and molecular simulation',
    long_description=readme(),
    long_description_content_type='text/markdown',
    classifiers=[
                'Intended Audience :: Science/Research',
        		'Operating System :: Unix',
        		'Programming Language :: Python',
        		'Topic :: Scientific/Engineering',
	],
    package_dir={'': 'src'},
    zip_safe=False,
    tests_require=['pytest'],
    install_requires=requirements(),
    python_requires='>=3.6.7',
)
