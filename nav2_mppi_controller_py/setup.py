from setuptools import setup

package_name = "nav2_mppi_controller_py"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name, f"{package_name}.critics", f"{package_name}.models", f"{package_name}.tools"],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=[
        "setuptools",
        "numpy",
    ],
    zip_safe=True,
    author="OpenAI Codex",
    author_email="codex@example.com",
    maintainer="OpenAI Codex",
    maintainer_email="codex@example.com",
    description="Python port of the Navigation2 MPPI controller.",
    license="Apache License 2.0",
    tests_require=["pytest"],
    entry_points={},
)
