from setuptools import setup, find_packages

setup(
    name="mitrailleuse",
    version="0.1.0",
    packages=find_packages(include=["mitrailleuse", "mitrailleuse.*"]),
    package_data={
        "mitrailleuse": ["proto/*.proto"],
    },
    install_requires=[
        "grpcio",
        "grpcio-tools",
        "grpcio-health-checking",
        "grpcio-reflection",
        "pydantic",
        "httpx",
        "openai",
        "deepl",
    ],
    python_requires=">=3.12",
) 