# URDF


### Generate .urdf from .xacro

1. Change the 'package_dir' in fanuc_leaphand.xacro to the absolute path.

1. Auto generation:
    ```
    cd catkin_ws
    source devel/setup.bash

    cd your/path/to/urdf
    rosrun xacro xacro -o fanuc_leaphand.urdf fanuc_leaphand.xacro
    ```

1. Change the 'package_dir' back.


### Generate .xml from .urdf

1. Create a new folder called 'fanuc_leaphand_xml'
1. Copy the fanuc_leaphand.urdf to this folder.
1. Copy all the mesh file to this folder (no subfolder).
4. Modify and run the ```urdf2xml.py```. 