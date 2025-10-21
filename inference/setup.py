from setuptools import setup, find_packages
import sys
from pathlib import Path

# Add the src directory to Python path so we can import from training
noble_root = Path(__file__).parent.parent
src_path = noble_root / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

setup(
    name="noble-inference",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
    ],
    author="Luca Ghafourpour",
    description="Inference scripts for NOBLE.",
    python_requires=">=3.10",
)
