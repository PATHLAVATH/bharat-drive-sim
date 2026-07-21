from setuptools import setup, find_packages

setup(
    name="bharatsim",
    version="0.2.0",
    description="bharat-drive-sim: a full-scale simulator for Indian roads and ADAS/AD evaluation",
    packages=find_packages(),
    install_requires=["numpy>=1.24", "opencv-python-headless>=4.8", "pyyaml>=6.0"],
    python_requires=">=3.9",
)
