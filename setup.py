from setuptools import setup, find_packages

setup(
    name="my_package",                # Your package name
    version="0.1.0",                  # Initial version
    description="A simple Python package",  # Short description
    author="Your Name",              # Author's name
    author_email="you@example.com",  # Author's email
    packages=find_packages(),        # Automatically find all packages
    install_requires=[               # Dependencies
        "requests>=2.25.0",
    ],
    entry_points={                   # Optional: CLI tool entry point
        "console_scripts": [
            "my-cli=my_package.cli:main",  # command=module:function
        ],
    },
    classifiers=[                    # Optional: Metadata for PyPI
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",         # Minimum Python version requirement
)
