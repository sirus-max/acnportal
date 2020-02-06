import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()
setuptools.setup(
    name='acnportal',
    version='0.1.3',
    author='Zachary Lee',
    author_email="zlee@caltech.edu",
    url='https://github.com/zach401/acnportal',
    description="A package of tools for large-scale EV charging research.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    package_data={'': ['LICENSE.txt', 'THANKS.txt', 'acnportal/signals/tariffs/tariff_schedules/*']},
    include_package_data=True,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python',
        'License :: OSI Approved :: BSD License',
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'requests',
        'pytz'
    ],
    extras_require={
        "gym": ['gym<=0.15.3']
    }
)
