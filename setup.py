from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="surveillance-enhancement",
    version="1.0.0",
    author="Simone Mele",
    author_email="melesimone@gmail.com",
    description="AI-powered image enhancement for surveillance footage",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pandoradiscoverer/surveillance-enhancement",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Legal Industry",
        "Topic :: Scientific/Engineering :: Image Processing",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "opencv-python>=4.8.0",
        "flask>=3.0.0",
        "gfpgan>=1.3.8",
        "realesrgan>=0.3.0",
    ],
    entry_points={
        "console_scripts": [
            "surveillance-enhance=app:main",
        ],
    },
)