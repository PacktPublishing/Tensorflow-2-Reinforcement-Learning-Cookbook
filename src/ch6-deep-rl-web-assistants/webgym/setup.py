#!/usr/bin/env/ python
# WebGym Visual MiniWoB environment registration script
# TensorFlow 2 Reinforcement Learning Cookbook | Praveen Palanisamy

from setuptools import setup, find_packages
import pathlib

parent_dir = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (parent_dir / "README.md").read_text(encoding="utf-8")

setup(
    name="webgym",  # Required
    version="1.0.0",  # Required
    description="Reinforcement Learning Environments for 50+ web-based tasks",  # Optional
    long_description=long_description,  # Optional
    long_description_content_type="text/markdown",  # Optional (see note above)
    url="https://github.com/praveen-palanisamy/Tensorflow-2-Reinforcement-Learning-Cookbook",  # Optional
    author="Praveen Palanisamy",  # Optional
    author_email="praveen.palanisamy@outlook.com",
    classifiers=[  # Optional
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3 :: Only",
    ],
    keywords="webgym, rl web tasks, rl in browser, Gym environments",  # Optional
    # `src/`, it is necessary to specify the `package_dir` argument.
    package_dir={"": "."},  # Optional
    packages=find_packages(where="."),  # Required
    # https://packaging.python.org/guides/distributing-packages-using-setuptools/#python-requires
    python_requires=">=3.6, <4",
    # List additional URLs that are relevant to your project as a dict.
    #
    # This field corresponds to the "Project-URL" metadata fields:
    # https://packaging.python.org/specifications/core-metadata/#project-url-multiple-use
    #
    # Examples listed include a pattern for specifying where the package tracks
    # issues, where the source is hosted, where to say thanks to the package
    # maintainers, and where to support the project financially. The key is
    # what's used to render the link text on PyPI.
    project_urls={  # Optional
        "Source": "https://github.com/praveen-palanisamy/Tensorflow-2-Reinforcement-Learning-Cookbook",
        "Author website": "https://praveenp.com",
    },
)