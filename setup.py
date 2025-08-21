from setuptools import find_packages,setup
from typing import List 

HYPHEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    """
    This function reads a requirements file and returns a list of packages.
    It removes any version specifiers and comments.
    """
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements=[req.replace('\n', '') for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
        return requirements

setup(
    name='Student-Performance-Prediction',
    version='0.1.0',
    author='Shreeharsh Shivpuje',
    author_email='info.shreeharshshivpuje@gmail.com',
    description='A machine learning project to predict student performance based on various features.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
)