"""Some tools I use to train and test neural networks.

Kyle Roth. 2019-07-06.
"""


from os import path
from setuptools import setup

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='training_tools',

    version='0.1a2',

    description=__doc__.split('\n')[0],
    long_description=long_description,
    long_description_content_type='text/markdown',

    # url

    author='Kyle Roth',
    author_email='kylrth@gmail.com',

    classifiers=[
        'Development Status :: 3 - Alpha',

        'Intended Audience :: Science/Research',

        'License :: OSI Approved :: MIT License',

        'Programming Language :: Python :: 3 :: Only'
    ],

    keywords='machine-learning training',

    py_modules=['components', 'architectures'],

    install_requires=['Pillow'],

    zip_safe=True
)

