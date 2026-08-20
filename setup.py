from setuptools import find_packages, setup

setup(
    name="soccermaster-research",
    version="0.2.0",
    packages=find_packages(where="research/src"),
    package_dir={"": "research/src"},
)
