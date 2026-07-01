from setuptools import setup, find_packages

setup(
    name="dq_evaluator",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "mistralai>=1.0.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "openpyxl>=3.1.0",
        "pyarrow>=14.0.0",
    ],
    entry_points={
        "console_scripts": [
            "dq-evaluate=dq_evaluator.main:main",
        ]
    },
)
