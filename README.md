### Update 21/12/24
- Scripts for pre training the VO and IMU models are done.
- The final script which combines VO , IMU and PVGO is in progress.

### VO Pre-Training
- Train Commmand - ./tools/dist_train.sh ./projects/configs/PanoOcc/Occupancy/Occ3d-nuScenes/VOTrain.py 1   (Same as mentioned in PanoOcc Docs)
- Training code - projects/mmdet3d_plugin/bevformer/detectors/VOTrain.py
- Config File - projects/configs/PanoOcc/Occupancy/Occ3d-nuScenes/VOTrain.py

### IMU Pre-Training
- Train Command -  usage: train.py [-h] [--checkpoint_path CHECKPOINT_PATH] [--experiment EXPERIMENT] mode dataset_folder
