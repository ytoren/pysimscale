import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="SimScale",
    version="0.0.1",
    author="Yizhar (Izzy) Toren",
    author_email="ytoren+pysimscale@gmail.com",
    description="Large scale similarity matrix calculus (parallelisation & chage of scale)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="github.com/ytoren/pysimscale",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6'
)
