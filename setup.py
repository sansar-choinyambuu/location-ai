import os

from setuptools import setup, find_packages

# Setup the project
setup(
    name="location-ai",
    version="0.1",
    packages=find_packages(exclude=["test"]),
    install_requires=[
        "aniso8601==8.0.0",
        "attrs==19.3.0",
        "certifi==2019.11.28",
        "chardet==3.0.4",
        "click==7.1.1",
        "cycler==0.10.0",
        "Flask==1.1.1",
        "flask-restplus==0.13.0",
        "geographiclib==1.50",
        "geopy==1.21.0",
        "idna==2.9",
        "importlib-metadata==1.5.0",
        "itsdangerous==1.1.0",
        "Jinja2==2.11.1",
        "joblib==0.14.1",
        "jsonschema==3.2.0",
        "kiwisolver==1.1.0",
        "MarkupSafe==1.1.1",
        "numpy==1.18.1",
        "pandas==1.0.1",
        "patsy==0.5.1",
        "pyparsing==2.4.6",
        "pyrsistent==0.15.7",
        "python-dateutil==2.8.1",
        "pytz==2019.3",
        "requests==2.23.0",
        "scikit-learn==0.22.2.post1",
        "scipy==1.4.1",
        "Shapely==1.7.0",
        "six==1.14.0",
        "urllib3==1.25.8",
        "Werkzeug==0.16.1",
        "zipp==3.1.0",
        ],
    # other arguments here...
    entry_points={
        "console_scripts": [
            "location-scouter = scouter:scout",
        ],
    },
    # metadata
    author="Sansar Choinyambuu",
    author_email="sansar.choinyambuu@gmail.com",
    description="Application providing location intelligence",
    include_package_data=True,
    license="Apache 2.0",
    url="https://github.com/sansar-choinyambuu/location-ai",
)