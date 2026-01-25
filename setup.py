from setuptools import setup, find_packages
from typing import List

HYPEN_DOT_E = '-e .'

def get_requirements(file_path: str) -> List[str]:
    """Reads the requirements from a file and returns them as a list."""
    with open(file_path, 'r') as file:
        requirements = file.readlines()
        requirements = [req.replace('\n', '') for req in requirements]
        
        if HYPEN_DOT_E in requirements:
            requirements.remove(HYPEN_DOT_E)
    return requirements

setup(
    name='almond-gpt',
    version='0.0.1',
    author='Adin Ramdan Farelino',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    install_requires=get_requirements('requirements.txt'),
)