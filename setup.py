from setuptools import setup, find_packages

setup(
    name='SmartPathClustering',
    version='1.0.0',
    description='Career recommendation system using Machine learning Algorithim',
    author='Eugene Maina, Beryl Okelo, Beth Nyambura, Allan Ofula, Rachael Nyawira ',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.23',
        'pandas>=1.5',
        'matplotlib>=3.6',
        'scikit-learn>=1.2',
        'hdbscan>=0.8.29'
        'streamlit>=1.27'
    ],
    python_requires='>=3.8',
)
