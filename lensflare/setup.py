import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="LensFlare",
    version="0.0.1",
    author="Gordon MacMillan",
    author_email="gmacilla@ymail.com",
    description="A small library of hand-rolled deep learning models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/GdMacmillan/deep_learning_library/tree/master/lensflare",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
