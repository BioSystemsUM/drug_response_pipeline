from setuptools import setup, find_packages

setup(name='src', version='1.0', packages=find_packages(),
      package_data={"src": ["preprocessing/filtering_files/*"]})
# include_package_data=True