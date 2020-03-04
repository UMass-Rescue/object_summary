from setuptools import setup, find_packages

setup(name='object_summary',
      version='0.1.6',
      description='A library that makes use of object detection to provide insights into image data.',
      url='https://github.com/UMass-Rescue/object_summary',
      author='Prasanna Lakkur Subramanyam',
      author_email='prasanna.lakkur@gmail.com',
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False)