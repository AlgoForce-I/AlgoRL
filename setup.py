from pathlib import Path

from setuptools import find_packages, setup


def read_requirements() -> list[str]:
    requirements_path = Path(__file__).parent / "requirements.txt"
    with requirements_path.open(encoding="utf-8") as requirements_file:
        return [
            line.strip()
            for line in requirements_file
            if line.strip() and not line.startswith("#")
        ]


setup(
    name="algorl",
    version="0.0.1",
    description="Implementation of Model-Based RL Algorithms",
    long_description=Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["tests", "tests.*"]),
    install_requires=read_requirements(),
    python_requires=">=3.10",
)
