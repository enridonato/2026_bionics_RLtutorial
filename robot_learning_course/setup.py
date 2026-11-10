from setuptools import find_packages, setup

setup(
    name="robot_learning_course",
    version="1.0.0",
    packages=find_packages(exclude=["test"]),
    python_requires=">=3.10",
    install_requires=["numpy", "mujoco==3.3.7", "torch", "matplotlib", "jupyter", "pytest"],
    maintainer="Enrico Donato",
    maintainer_email="enrico.donato.santannapisa@gmail.com",
    description="Robot Learning MSc course: FR3 + MuJoCo + PyTorch",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            # the simulated experiments
            "run_sim = robot_learning_course.run_sim:main",
            "run_collect = robot_learning_course.run_collect:main",
            "run_control = robot_learning_course.run_control:main",
            # learning scripts
            "train_fk = robot_learning_course.learning.train_fk:main",
            "train_ik = robot_learning_course.learning.train_ik:main",
            "make_dataset_fast = robot_learning_course.learning.make_dataset_fast:main",
        ],
    },
)
