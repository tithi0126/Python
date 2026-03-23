from setuptools import setup, find_packages

setup(
    name='arti26',      
    version='1.0.0',                    
    packages=find_packages(),           
    description='A complete automated ML pipeline',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='arti',
    author_email='arti@example.com',
    install_requires=[                  
        'pandas',
        'numpy',
        'scikit-learn',
        'matplotlib',
        'seaborn'
    ],
)
