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

## Pre-trained models

- [LFE-PPN](https://robotics.upo.es/~famozur/onnx/LFE-PPN.onnx) (for use with `lasermodelhost`)
- [LFE-Peaks](https://robotics.upo.es/~famozur/onnx/LFE-Peaks.onnx) (for use with `lasermodelhost_peaks`)
