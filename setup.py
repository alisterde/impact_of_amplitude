import setuptools

# with open("README.md", "r") as fh:
#     long_description = fh.read()


setuptools.setup(
    name='electrochemistry_modelling',
    version='0.0.0.1',
    author='Alister Dale-Evans',
    author_email="",
    description="A package to model capacitive current in voltammetric experiments",
    # long_description=long_description,  # README_1.md file as description
    # long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    install_requires=['cma==3.3.0',
                    'contourpy==1.2.0',
                    'cycler==0.12.1',
                    'fonttools==4.44.3',
                    'importlib-resources==6.1.1',
                    'kiwisolver==1.4.5',
                    'llvmlite==0.41.1',
                    'matplotlib==3.8.1',
                    'numba==0.58.1',
                    'numpy==1.26.2',
                    'packaging==23.2',
                    'Pillow==10.1.0',
                    'pints==0.5.0',
                    'pyparsing==3.1.1',
                    'python-dateutil==2.8.2',
                    'scipy==1.11.3',
                    'six==1.16.0',
                    'tabulate==0.9.0',
                    'threadpoolctl==3.2.0',
                    'zipp==3.17.0'],  # Install requirements extracted from requirements.txt
    include_package_data=True,  # Allow to include other files than .py in package
    classifiers=[
        'Development Status :: 4 - Beta',
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
)