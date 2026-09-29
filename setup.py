"""Package VisionHOPE under the visionhope namespace."""

from setuptools import find_packages, setup

setup(
    packages=["visionhope"] + ["visionhope." + name for name in find_packages(".")],
    package_dir={"visionhope": "."},
)
