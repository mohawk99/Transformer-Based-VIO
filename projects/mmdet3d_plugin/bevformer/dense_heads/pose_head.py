import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
from mmdet.models import HEADS

@HEADS.register_module()
class FuseModule(nn.Module):
    def __init__(self, channels, reduction):
        super(FuseModule, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


@HEADS.register_module()
class PoseNet(nn.Module):
    '''
    Fuse both features and output the 6 DOF camera pose
    '''
    def __init__(self, input_size=1024):
        super(PoseNet, self).__init__()

        self.se = FuseModule(input_size, 16)

        # Assuming we know the dimensions of visual_fea after flattening
        visual_input_size = 96000  # Replace this with the actual size after flattening if different
        imu_input_size = 512       # This should match the size of the imu_fea input
        
        # Projection layer to match visual and IMU feature sizes for concatenation
        self.projection_layer = nn.Linear(visual_input_size, imu_input_size)

        # Adjust the input size of the RNN to be the sum of visual_input_size and imu_input_size
        self.rnn = nn.LSTM(input_size=input_size,  # After projection, sizes match
                           hidden_size=1024,
                           num_layers=2,
                           batch_first=True)

        self.fc1 = nn.Sequential(nn.Linear(1024, 7))

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.LSTM):
                init.xavier_normal_(m.all_weights[0][0], gain=np.sqrt(1))
                init.xavier_normal_(m.all_weights[0][1], gain=np.sqrt(1))
                init.xavier_normal_(m.all_weights[1][0], gain=np.sqrt(1))
                init.xavier_normal_(m.all_weights[1][1], gain=np.sqrt(1))
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight.data, gain=np.sqrt(1))
                if m.bias is not None:
                    init.zeros_(m.bias)

    def forward(self, visual_fea, imu_fea):
        self.rnn.flatten_parameters()
        if imu_fea is not None:
            B, t, _ = imu_fea.shape
            imu_input = imu_fea.view(B, t, -1)

            if isinstance(visual_fea, list) and len(visual_fea) == 1:
                visual_fea = visual_fea[0]  # Extract the tensor
            
            print(f"visual fea d type: {visual_fea.dtype}")
            # Step 2: Reshape visual_fea and imu_fea for concatenation
            B, t, C, H, W = visual_fea.shape
            visual_input = visual_fea.view(B, t, -1)

            visual_input = visual_input[:, :1, :] 
            
            # Apply the projection layer to match dimensions
            visual_input = self.projection_layer(visual_input)
            
            print(f"visual shape pn: {visual_input.shape}")
            print(f"visual shape d type: {visual_input.dtype}")
            print(f"imu shape pn: {imu_input.shape}")

            inpt = torch.cat((visual_input, imu_input), dim=2)
        else:
            inpt = visual_fea

        inpt = self.se(inpt)
        out, (h, c) = self.rnn(inpt)
        out = 0.01 * self.fc1(out)
        return out
