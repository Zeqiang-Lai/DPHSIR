from setuptools import setup, find_packages

setup(
    name='dphsir',
    packages=find_packages(),
    version='0.1',
    include_package_data=True,
    entry_points='''
        [console_scripts]
        dphsir=cli.main:main
    '''
)