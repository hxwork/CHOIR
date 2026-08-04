from setuptools import setup, find_packages

setup(
    name="utils_simba",
    version="0.1.0",
    description="Utility functions for computer vision and 3D processing",
    author="",
    author_email="",
    packages=find_packages(include=["."]),
    python_requires=">=3.7",
    install_requires=[
        "numpy",
        "pillow",
        "opencv-python==4.8.0.74",
        "omegaconf",
        "rich",
        "rerun-sdk",
        "smplx",
        "trimesh",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
) 