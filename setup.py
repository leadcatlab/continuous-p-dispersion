# setup.py
from setuptools import setup, find_packages

setup(
    name="pdispersion",              # the name you’ll import
    version="0.1",
    package_dir={"": "src"},          # look for packages under src/
    packages=find_packages("src"),    # find all packages there
)
