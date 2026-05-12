from setuptools import setup, find_packages

setup(
    name="stock-manipulation-detector",
    version="1.0.0",
    description="AI-powered stock manipulation detection system",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        line.strip()
        for line in open('requirements.txt').readlines()
        if line.strip() and not line.startswith('#')
    ],
    python_requires='>=3.10',
)
