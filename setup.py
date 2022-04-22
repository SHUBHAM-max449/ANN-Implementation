from setuptools import setup

with open("README.md","r",encoding="utf-8") as f:
    long_description=f.read()

setup(
    name="src",
    version="0.0.1",
    author="Shubham",
    description="A small package for ANN Implementation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SHUBHAM-max449/ANN-Implementation.git",
    author_email="chitaguppeshubham@gmail.com",
    packages=["src"],
    python_requires=[
       "tensorflow"
"seaborn",
"matplotlib",
"numpy",
"pandas"
    ]
    
)