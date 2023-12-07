from typing import List

from setuptools import find_packages, setup


HYPHEN_E_DOT = '-e .'
def get_requirements(file_path: str) -> List[str]:
    """

    :param file_path: give path for installing required files
    :return: this function will return a list of requirements
    """
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [module.replace("\n", '') for module in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

        if ""  in requirements:
            requirements.remove("")




setup(
    name='mlproject',
    version='0.0.1',
    author_email='sumanthomgowda@gmailcom',
    author='Sumanth',
    packages=find_packages(),
    # install_requires=['pandas','numpy','seaborn','scikit-learn','scipy','matplotlib']
    install_requires=get_requirements('requirements.txt')
)
