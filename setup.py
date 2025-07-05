from setuptools import setup

setup(
    name="soccerbackbone",  # pip安装时的名称
    version="0.1",
    packages=[""],          # 空字符串表示当前目录是包根目录
    package_dir={"": "."},  # 将空包名映射到当前目录
)