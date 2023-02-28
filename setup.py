from setuptools import setup, find_packages

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

with open('requirements.txt') as f:
    required_packages = f.read().splitlines()

setup(
    name='imgraph',
    version='0.0.4',
    description='Converts an image to a graph and apply GNNs for various tasks.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/aryan-at-ul/imgraph',
    author='Aryan Singh',
    author_email='aryan.singh@ul.ie',
    license='MIT',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    keywords='image to graph',
    # packages=find_packages(where="src"),
    py_modules = ["imgraph"],
    package_dir={'': 'src'},
    install_requires=required_packages,
    #entry_points={
    #    'console_scripts': [
    #        'imgraph=imgraph.imgraph:main',
    #    ],
    #},
)
