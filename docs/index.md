# In-Hand Following of Deformable Linear Objects Using Dexterous Fingers with Tactile Sensing

The paper has been accepted to IROS 2024 (oral presentation, top 12%).

The source code will be released soon.

[[arXiv](https://arxiv.org/abs/2403.12676)]

## Video

<p align="center">
<iframe width="800" height="450" src="./main_1080p.mp4" title="24_DLO_Following" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen> </iframe>
</p>

## Abstract

Most research on deformable linear object (DLO) manipulation assumes rigid grasping. However, beyond rigid grasping and re-grasping, in-hand following is also an essential skill that humans use to dexterously manipulate DLOs, which requires continuously changing the grasp point by in-hand sliding while holding the DLO to prevent it from falling. Achieving such a skill is very challenging for robots without using specially designed but not versatile end-effectors.
Previous works have attempted using generic parallel grippers, but their robustness is unsatisfactory owing to the conflict between following and holding, which is hard to balance with a one-degree-of-freedom gripper.
In this work, inspired by how humans use fingers to follow DLOs, we explore the usage of a generic dexterous hand with tactile sensing to imitate human skills and achieve robust in-hand DLO following.
To enable the hardware system to function in the real world, we develop a framework that includes Cartesian-space arm-hand control, tactile-based in-hand 3-D DLO pose estimation, and task-specific motion design.
Experimental results demonstrate the significant superiority of our method over using parallel grippers, as well as its great robustness, generalizability, and efficiency.

## Something not Included in the Paper

Due to the page limit, we cannot include everything in the conference paper. Here we share more details and discussions.

### More details

- The weights for IK problem was manually defined for different tasks.
- The image morphological processing for in-hand DLO pose estimation includes:
  - Connected component detection: removing the connected components whose area is less than a threhold.
  - Close operation.
- The contact region segmentation for the V-shape grasping: the size of the bottom region was manually defined.
- The requirements of stable sensing in Section III-C are defined as: the contact area on both tactile sensors are larger than a threshold.
- All programs are run on a laptop with Ubuntu 20.04 and an Intel i7-9750H CPU (2.6Hz).
- The optimization problems for IK and in-hand DLO pose estimation can be solved within 50 ms, respectively. Note that the implementation had not been highly optimized and the solving speed could be further improved.

### Limitations

In this work, we pioneeringly explore the usage of dexterous hands to enhance the DLO following. We acknowledge that the current version contains some limitations and there is a lot of room for improvement.

- During V-shape DLO following, the contact region segmentation was not very stable, since the two Gelsight contact with each other with a large contact force. The contact region segmentation approach in such situations should be investigated further.
- The current motion design for DLO following is really simple, which is just based on some simple human-designed feedback control laws. In our experiments, we found that online adjusting the gripping angle and gripping force did not contribute much to the performance of DLO following (That is why we did not introduce the related details much in the paper). A learning-based policy may be better.

To be honest, we do not recommend developing your project directly based on our code as it has not been optimized. The code is mainly released for reference or a quick test run.

### Potential future work

- More complicated robot arm motion, beyond just moving parallel to the table surface.
- Utilizing more fingers and more contact points.

## Contact

If you have any question, feel free to contact the authors: Mingrui Yu, <mingruiyu98@gmail.com> .

Mingrui Yu's Homepage is at [mingrui-yu.github.io](https://mingrui-yu.github.io).
