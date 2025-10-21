from setuptools import setup, find_packages

setup(
    name="training",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "train_noble = training.train_noble:main"
        ]
    },
    author="Luca Ghafourpour",
    description="Training package for NOBLE.",
    python_requires=">=3.10",
) 