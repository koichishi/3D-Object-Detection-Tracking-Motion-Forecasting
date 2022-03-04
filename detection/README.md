# Module 1: Object Detection

This repository contains the starter code for Assignment 1 of CSC490H1.
In this assignment, you will implement a basic object detector for self-driving.

## How to run

### Overfitting

To overfit the detector to a single frame of PandaSet, run the following command
from the root directory of this repository:

```bash
python -m detection.main overfit --data_root=/u/csc490h/dataset --output_root=/h/u9/c7/00/shigongy/Desktop/csc490/3D-Object-Detection/output/overfit/

srun -p csc490-compute -c 6 -N 1 --gres gpu -o ~/Desktop/csc490/3D-Object-Detection/output/overfit/outfile python3 -m detection.main overfit --data_root=/u/csc490h/dataset --output_root=/h/u9/c7/00/shigongy/Desktop/csc490/3D-Object-Detection/output/overfit/
```

This command will write model checkpoints and visualizations to `<your_path_to_outputs>`.

### Training

To train the detector on the training split, run the following command
from the root directory of this repository:

```bash
python -m detection.main train --data_root=/u/csc490h/dataset --output_root=/h/u9/c7/00/shigongy/Desktop/csc490/3D-Object-Detection/output/train/

srun -p csc490-compute -c 6 -N 1 --gres gpu -o ~/Desktop/csc490/3D-Object-Detection/output/outfile python3 -m detection.main train --data_root=/u/csc490h/dataset --output_root=/h/u9/c7/00/shigongy/Desktop/csc490/3D-Object-Detection/output/train/ --checkpoint_path=/h/u9/c7/00/shigongy/Desktop/csc490/3D-Object-Detection/output/train/
```

This command will write model checkpoints and visualizations to `<your_path_to_outputs>`.

### Visualization

To visualize the detections of the detector, run the following command
from the root directory of this repository:

```bash
python -m detection.main test --data_root=/u/csc490h/dataset --output_root=/h/u9/c7/00/shigongy/Desktop/csc490/3D-Object-Detection/output/visualization

srun -p csc490-compute -c 6 -N 1 --gres gpu -o ~/Desktop/csc490/3D-Object-Detection/output/visualization/outfile python3 -m detection.main test --data_root=/u/csc490h/dataset --output_root=/h/u9/c7/00/shigongy/Desktop/csc490/3D-Object-Detection/output/visualization --checkpoint_path=/h/u9/c7/00/shigongy/Desktop/csc490/3D-Object-Detection/output/train/pa_009.pth
```

This command will save detection visualizations to `<your_path_to_outputs>`.

### Evaluation

To evaluate the detections of the detector, run the following command
from the root directory of this repository:

```bash
python -m detection.main evaluate --data_root=/u/csc490h/dataset --output_root=~/Desktop/csc490/3D-Object-Detection/output --checkpoint_path<your_path_to_checkpoint>
```

This command will save detection visualizations and metrics to `<your_path_to_outputs>`.
