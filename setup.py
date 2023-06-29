from setuptools import find_packages, setup
from typing import List


MINUSEDOT = '-e .'

def get_requirements(file_path)->List[str]:
    requirements = []
    with open(file_path) as f:
        requirements = f.readlines()
        requirements = [r.replace("\n","") for r in requirements]
        requirements = [r for r in requirements if r!=MINUSEDOT]
    return requirements

setup(
    name='ML_Project',
    version='0.0.1',
    author='Dhruv',
    author_email='dhruvnp48@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)