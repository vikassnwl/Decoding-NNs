from setuptools import setup, find_packages

setup(
    name='my_project',  # Choose a name for your project
    version='0.1',
    packages=find_packages(include=['utils', 'utils.*']),  # Include the 'utils' package
    install_requires=[  # Add any dependencies you need
        # Example: 'numpy', 'tensorflow'
    ],
    package_data={  # If you have non-Python files, include them
        'utils': ['data/*.txt'],  # Example of including non-Python files (if needed)
    },
    include_package_data=True,  # To include package_data
)
