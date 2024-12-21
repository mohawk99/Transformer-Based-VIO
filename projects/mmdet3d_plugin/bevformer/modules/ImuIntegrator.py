from nuscenes.can_bus.can_bus_api import NuScenesCanBus
import numpy as np
from scipy.spatial.transform import Rotation as R
import os
import argparse
import pypose as pp
from datetime import datetime
import torch.utils.data as Data
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.collections import PatchCollection
import torch

#nusc_can_bus = NuScenesCanBus(dataroot='/media/mohak/DiskG')

def quaternion_to_SO3(quaternion):

        x, y, z, w = quaternion
        R = torch.tensor([
            [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)]
        ])
        return R 


class NS_IMU(Data.Dataset):
    def __init__(self, nusc_can_bus, scene_name = 'scene-0001', duration=10, step_size=1, mode='train'):
        super().__init__()
        self.duration = duration
        self.data = NS_IMU.load_imu_data(nusc_can_bus,scene_name)

        if not self.data:
            self.seq_len = 0
            self.index_map = []
            print(f"No data available for scene {scene_name}")
            return 
    
        self.seq_len = len(self.data)
        assert mode in ['evaluate', 'train',
                        'test'], "{} mode is not supported.".format(mode)

        self.dt = torch.tensor(0.02)
        self.acc = torch.tensor([pose['accel'] for pose in self.data], dtype=torch.float32)
        self.gyro = torch.tensor([pose['rotation_rate'] for pose in self.data], dtype=torch.float32)
        self.gt_pos = torch.tensor([pose['pos'] for pose in self.data], dtype=torch.float32)
        self.gt_rot = torch.stack([pp.mat2SO3(quaternion_to_SO3(pose['orientation'])) for pose in self.data]).to(dtype=torch.float32)
        self.gt_vel = torch.stack([self.gt_rot[i] @ torch.tensor(pose['vel'], dtype=torch.float32) for i, pose in enumerate(self.data)])

        start_frame = 0
        end_frame = self.seq_len
        if mode == 'train':
            end_frame = np.floor(self.seq_len * 0.5).astype(int)
        elif mode == 'test':
            start_frame = np.floor(self.seq_len * 0.5).astype(int)

        self.index_map = [i for i in range(
            0, end_frame - start_frame - self.duration, step_size)]

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, i):
        frame_id = self.index_map[i]
        end_frame_id = frame_id + self.duration

        if end_frame_id > self.seq_len:
            raise IndexError("End frame exceeds the sequence length.")
        
        return {
            'dt': self.dt,
            'acc': self.acc[frame_id: end_frame_id],
            'gyro': self.gyro[frame_id: end_frame_id],
            'gyro': self.gyro[frame_id: end_frame_id],
            'gt_pos': self.gt_pos[frame_id+1: end_frame_id+1],
            'gt_rot': self.gt_rot[frame_id+1: end_frame_id+1],
            'gt_vel': self.gt_vel[frame_id+1: end_frame_id+1],
            'init_pos': self.gt_pos[frame_id][None, ...],
            'init_rot': self.gt_rot[frame_id: end_frame_id],
            'init_vel': self.gt_vel[frame_id][None, ...],
        }

    def get_init_value(self):
        return {'pos': self.gt_pos[:1],
                'rot': self.gt_rot[:1],
                'vel': self.gt_vel[:1]}
    
    def load_imu_data(nusc_can_bus, scene_name):
        try:
            data = nusc_can_bus.get_messages(scene_name, 'pose')
            if data is None:
                # Debug print to see if no data is being loaded
                print(f"No data found for scene {scene_name}")
                return []
            # Debug print to inspect the structure of the loaded data
            print(f"Loaded IMU data for scene {scene_name}")
            return data
        except Exception as e:
            print(f"Error loading IMU data: {e}")
            return []

    
def imu_collate(data):
    acc = torch.stack([d['acc'] for d in data])
    gyro = torch.stack([d['gyro'] for d in data])

    gt_pos = torch.stack([d['gt_pos'] for d in data])
    gt_rot = torch.stack([d['gt_rot'] for d in data])
    gt_vel = torch.stack([d['gt_vel'] for d in data])

    init_pos = torch.stack([d['init_pos'] for d in data])
    init_rot = torch.stack([d['init_rot'] for d in data])
    init_vel = torch.stack([d['init_vel'] for d in data])

    dt = torch.stack([d['dt'] for d in data]).unsqueeze(-1)

    return {
        'dt': dt,
        'acc': acc,
        'gyro': gyro,

        'gt_pos': gt_pos,
        'gt_vel': gt_vel,
        'gt_rot': gt_rot,

        'init_pos': init_pos,
        'init_vel': init_vel,
        'init_rot': init_rot,
    }


def move_to(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            res[k] = move_to(v, device)
        return res
    elif isinstance(obj, list):
        res = []
        for v in obj:
            res.append(move_to(v, device))
        return res
    else:
        raise TypeError("Invalid type for move_to", obj)
    

def plot_gaussian(ax, means, covs, color=None, sigma=3):
    ellipses = []
    means_2d = means[:, :2] 
    for i in range(len(means_2d)):
        cov_2d = covs[i][:2, :2] 
        eigvals, eigvecs = np.linalg.eig(cov_2d)
        axis = np.sqrt(eigvals) * sigma
        slope = eigvecs[1][0] / eigvecs[1][1]
        angle = 180.0 * np.arctan(slope) / np.pi
        ellipses.append(Ellipse(means_2d[i], axis[0], axis[1], angle=angle))
    ax.add_collection(PatchCollection(ellipses, edgecolors=color, linewidth=1))

        
def main():
    nusc_can_bus = NuScenesCanBus(dataroot='/media/mohak/DiskG')
    parser = argparse.ArgumentParser(description='IMU Preintegration')
    parser.add_argument("--device",
                        type=str,
                        default='cpu',
                        help="cuda or cpu")
    parser.add_argument("--batch-size",
                        type=int,
                        default=1,
                        help="batch size, only support 1 now") #why?
    parser.add_argument("--step-size",
                        type=int,
                        default=2,
                        help="the size of the integration for one interval")
    parser.add_argument("--save",
                        type=str,
                        default='/home/mohak/Desktop/',
                        help="location of png files to save")
    parser.add_argument("--dataroot",
                        type=str,
                        default= nusc_can_bus,
                        help="dataset location downloaded")
    parser.add_argument("--dataname",
                        type=str,
                        default='plots',
                        help="dataset name")
    parser.add_argument("--datadrive",
                        nargs='+',
                        type=str,
                        default=["scene-0001","scene-0002","scene-0003","scene-0004","scene-0005" ],
                        help="data sequences")
    parser.add_argument('--plot3d',
                        dest='plot3d',
                        action='store_true',
                        help="plot in 3D space, default: False")
    parser.set_defaults(plot3d=False)
    args, unknown = parser.parse_known_args()
    print(args)
    os.makedirs(os.path.join(args.save), exist_ok=True)
    torch.set_default_tensor_type(torch.FloatTensor)


    for drive in args.datadrive:

        # Step 1: Define dataloader using the ``KITTI_IMU`` class we defined above
        dataset = NS_IMU(args.dataroot,
                            drive,
                            duration=args.step_size,
                            step_size=args.step_size,
                            mode='evaluate')
        loader = Data.DataLoader(dataset=dataset,
                                batch_size=args.batch_size,
                                collate_fn=imu_collate,
                                shuffle=False)

        # Step 2: Get the initial position, rotation and velocity, all 0 here
        init = dataset.get_init_value()

        # Step 3: Define the IMUPreintegrator.
        integrator = pp.module.IMUPreintegrator(init['pos'].float(),
                                                init['rot'].float(),
                                                init['vel'].float(),
                                                reset=False).to(args.device)

        # Step 4: Perform integration
        poses, poses_gt = [init['pos']], [init['pos']]
        covs = [torch.zeros(9, 9)]

        for idx, data in enumerate(loader):
            data = move_to(data, args.device)

            if len(data['dt'].shape) < len(data['gyro'].shape):
                data['dt'] = data['dt'].unsqueeze(-1) 


            #print("dt shape:", data['dt'].shape)
            #print("gyro shape:", data['gyro'].shape)
            #print("acc shape:", data['acc'].shape)
            #print("init rot shape:", data['init_rot'].shape)
            #print(f"init_rot shape : {data['init_rot'].shape}")
            #print(f"init_rot data : {data['init_rot']}")
            data['dt'] = data['dt'].repeat(1, args.step_size, 1)
            #print("dt shape after:", data['dt'].shape) 

            state = integrator(dt=data['dt'],
                            gyro=data['gyro'],
                            acc=data['acc'],
                            rot=data['init_rot'])
            poses_gt.append(data['gt_pos'][..., -1, :].cpu())
            poses.append(state['pos'][..., -1, :].cpu())
            covs.append(state['cov'][..., -1, :, :].cpu())

        poses = torch.cat(poses).numpy()
        poses_gt = torch.cat(poses_gt).numpy()
        covs = torch.stack(covs, dim=0).numpy()

        # Step 5: Visualization
        plt.figure(figsize=(5, 5))
        if args.plot3d:
            ax = plt.axes(projection='3d')
            ax.plot3D(poses[:, 0], poses[:, 1], poses[:, 2], 'b')
            ax.plot3D(poses_gt[:, 0], poses_gt[:, 1], poses_gt[:, 2], 'r')
        else:
            ax = plt.axes()
            ax.plot(poses[:, 0], poses[:, 1], 'b')
            ax.plot(poses_gt[:, 0], poses_gt[:, 1], 'r')
            plot_gaussian(ax, poses[:, 0:2], covs[:, 6:8, 6:8])
        plt.title("PyPose IMU Integrator")
        plt.legend(["PyPose", "Ground Truth"])
        figure = os.path.join(args.save, args.dataname+'_'+drive+'.png')
        plt.savefig(figure)
        print("Saved to", figure)

if __name__ == "__main__":
    main()

    
""" def main():
    # Argument parser for custom inputs
    parser = argparse.ArgumentParser(description="IMU Data Loader")
    parser.add_argument('--scene_name', type=str, default='scene-0001', help='Scene name for the CAN bus data.')
    parser.add_argument('--duration', type=int, default=10, help='Duration of the sequence.')
    parser.add_argument('--step_size', type=int, default=1, help='Step size between sequences.')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'evaluate'], help='Mode of the dataset.')

    args = parser.parse_args()

    # Initialize the CAN bus dataset
    nusc_can_bus = NuScenesCanBus(dataroot='/media/mohak/DiskG')

    # Create the dataset instance
    imu_dataset = NS_IMU(nusc_can_bus, scene_name=args.scene_name, duration=args.duration, step_size=args.step_size, mode=args.mode)

    # DataLoader for batching and shuffling (useful during training)
    data_loader = Data.DataLoader(dataset=imu_dataset, batch_size=2, shuffle=True)

    # Fetch one batch of data to test
    for batch_idx, data in enumerate(data_loader):
        print(f"Batch {batch_idx + 1}")
        print("dt:", data['dt'])
        print("acc:", data['acc'])
        print("gyro:", data['gyro'])
        print("gt_pos:", data['gt_pos'])
        print("gt_rot:", data['gt_rot'])
        print("gt_vel:", data['gt_vel'])
        print("init_pos:", data['init_pos'])
        print("init_rot:", data['init_rot'])
        print("init_vel:", data['init_vel'])
        break  # Only test the first batch

if __name__ == "__main__":
    main() """
