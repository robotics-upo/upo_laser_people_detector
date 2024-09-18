# 2D LiDAR People Detection for ROS 2

This ROS package contains nodes for detecting people using 2D LiDAR.

## System requirements

- Ubuntu 22.04 Jammy
- ROS 2 Humble
- CUDA Toolkit 12.1 (tested)
- ONNX Runtime 1.16.3 (tested):

```bash
wget https://robotics.upo.es/~famozur/onnx/onnxruntime-gpu_1.16.3_amd64.deb
sudo apt install ./onnxruntime-gpu_1.16.3_amd64.deb
```

## Nodes

This package contains two nodes: `lasermodelhost` (implementing LFE-PPN) and `lasermodelhost_peaks` (implementing LFE-Peaks).
Each node can be launched using the following command:

```bash
ros2 run upo_laser_people_detector lasermodelhost{_peaks} --ros-args -p model_file:=some_model.onnx -p other_param:=value ...
```

List of parameters:

- `model_file` (string): Path to the ONNX file containing model weights. **This parameter must be explicitly provided**. Please look at the pre-trained models section below.
- `laser_topic` (string): Name of the input ROS topic containing `sensor_msgs/LaserScan` messages from the 2D LiDAR. Defaults to `/scanfront`.
- `output_topic` (string): Name of the output ROS topic for `upo_laser_people_msgs/PersonDetectionList` messages. Defaults to `detected_people` (namespace relative).
- `marker_topic` (string): Name of the output ROS topic for `visualization_msgs/MarkerArray` messages for use with RViz. Defaults to `detected_people_markers` (namespace relative).
- `scan_near` (float): Minimum distance between the 2D LiDAR and the person, in meters. Defaults to 0.02 m.
- `scan_far` (float): Maximum distance between the 2D LiDAR and the person, in meters. Defaults to 10 m.
- `score_threshold` (float): Score threshold for considering a person detection. Defaults to an appropriate value for each model.
- `person_radius` (float): Radius of the person bounding circles in meters (only for LFE-Peaks). Defaults to 0.4 m.

## Pre-trained models

- [LFE-PPN](https://robotics.upo.es/~famozur/onnx/LFE-PPN.onnx) (for use with `lasermodelhost`)
- [LFE-Peaks](https://robotics.upo.es/~famozur/onnx/LFE-Peaks.onnx) (for use with `lasermodelhost_peaks`)
