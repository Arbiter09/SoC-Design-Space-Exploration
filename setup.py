from setuptools import setup, find_packages

setup(
    name="soc-dse",
    version="0.1.0",
    description="AI-Driven SoC Design Space Exploration Platform",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "PyYAML>=6.0",
        "Jinja2>=3.1",
        "networkx>=3.2",
        "torch>=2.2",
        "torch-geometric>=2.5",
        "scikit-learn>=1.4",
        "matplotlib>=3.8",
        "plotly>=5.20",
        "pandas>=2.2",
        "numpy>=1.26",
    ],
)
