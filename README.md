# DLO_following

[[Project Website](https://mingrui-yu.github.io/DLO_following/)]

Repository for the IROS 2024 paper "In-Hand Following of Deformable Linear Objects Using Dexterous Fingers with Tactile Sensing".

In this repo, we provide the code for

- The optimization-based inverse kinematics (IK) solver.
- The tactile-based in-hand 3-D DLO pose estimation approach.

## Installation

```
git clone https://github.com/Mingrui-Yu/DLO_following.git
```

In your python3 env:

```
pip install matplotlib scikit-learn open3d pybullet imageio opencv-python
```

## Usage

### Test the IK solver

```
cd DLO_following

python test_ik.py global_rotation
# or
python test_ik.py v_shape_rotation
```

The `global_rotation` and `v_shape_rotation` are corresponding to the experiments in the paper.

### Test the in-hand 3-D DLO pose estimation

We provide the data of two sequences for a quick test run. The two sequences are the same as those shown in the supplementary video.

```
cd DLO_following

python test_cable_sensing.py seq_0
# or
python test_cable_sensing.py seq_1
```
