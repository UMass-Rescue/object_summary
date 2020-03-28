from setuptools import setup, find_packages

setup(name='object_summary',
      version='0.2.0',
      description='A library that makes use of object detection to provide insights into image data.',
      url='https://github.com/UMass-Rescue/object_summary',
      author='Prasanna Lakkur Subramanyam',
      author_email='prasanna.lakkur@gmail.com',
      packages=find_packages(),
      include_package_data=True,
      install_requires=[
            'tinydb',
            'tqdm',
            'numpy',
            'pandas',
            'pydotplus',
            'six',
            'opencv_python>=4.1.2.30',
            'ipython',
            'Pillow>=7.0.0',
            'scikit_learn',
            'tf_object_detection_util'
      ],
      zip_safe=False)