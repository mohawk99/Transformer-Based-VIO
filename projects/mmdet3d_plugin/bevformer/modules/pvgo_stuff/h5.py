import h5py
import plotly.graph_objects as go
import numpy as np
# Open the .h5 file
with h5py.File("/home/mohak/Thesis/PanoOcc/projects/mmdet3d_plugin/bevformer/modules/pvgo_stuff/voxel_det.h5", "r") as f:
    voxel_data = f["Voxel"][:]  # Load the data
    voxel_array = np.squeeze(voxel_data)  # Remove unnecessary dimensions
x, y, z = np.where(voxel_array > 0)

# Create 3D scatter plot
fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='markers',
                                   marker=dict(size=3, opacity=0.5))])

fig.update_layout(title="3D Voxel Grid", scene=dict(xaxis_title="X",
                                                    yaxis_title="Y",
                                                    zaxis_title="Z"))
fig.show()
