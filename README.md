### Update 21/12/24
- Scripts for pre training the VO and IMU models are done.
- The final script which combines VO , IMU and PVGO is in progress.

### VO Pre-Training
- Install the environment - https://github.com/Robertwyq/PanoOcc/blob/main/docs/install.md
- In case of dependency issues, run these:

```bash
!pip install "matplotlib<3.6.0"
!pip install pandas==1.4.4
!pip install IPython ipdb
!pip install yapf==0.40.1
!pip install simplejson

- Train Commmand - ./tools/dist_train.sh ./projects/configs/PanoOcc/Occupancy/Occ3d-nuScenes/VOTrain.py   (Same as mentioned in PanoOcc Docs)
- Training code - projects/mmdet3d_plugin/bevformer/detectors/VOTrain.py
- Config File - projects/configs/PanoOcc/Occupancy/Occ3d-nuScenes/VOTrain.py

### IMU Pre-Training
- Train Command -  usage: train.py [-h] [--checkpoint_path CHECKPOINT_PATH] [--experiment EXPERIMENT] mode dataset_folder
