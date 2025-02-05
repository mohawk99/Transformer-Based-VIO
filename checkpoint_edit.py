import torch
import torch.nn as nn


def add_pose_head_to_checkpoint(checkpoint_path, output_path, pose_head_config):
    """
    Adds missing `pose_head` weights to a checkpoint file.

    Args:
        checkpoint_path (str): Path to the existing checkpoint file.
        output_path (str): Path to save the updated checkpoint.
        pose_head_config (dict): Configuration for the pose_head structure.
    """
    # Load the existing checkpoint
    checkpoint = torch.load(checkpoint_path)
    state_dict = checkpoint.get('state_dict', {})

    # Define the pose_head structure
    pose_head = nn.Sequential(
        nn.LayerNorm(pose_head_config["input_dim"]),
        nn.Linear(pose_head_config["input_dim"], pose_head_config["hidden_dim1"]),
        nn.GELU(),
        nn.Linear(pose_head_config["hidden_dim1"], pose_head_config["hidden_dim2"]),
        nn.GELU(),
        nn.Dropout(pose_head_config["dropout"]),
        nn.Linear(pose_head_config["hidden_dim2"], pose_head_config["output_dim"])  # 7 outputs (3 for translation, 4 for rotation)
    )

    # Initialize the pose_head weights
    for m in pose_head.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    # Add initialized pose_head weights to the state_dict
    for name, param in pose_head.named_parameters():
        state_dict[f"pose_head.{name}"] = param.data

    # Update the checkpoint
    checkpoint['state_dict'] = state_dict
    torch.save(checkpoint, output_path)
    print(f"Updated checkpoint saved at {output_path}")


# Example usage
if __name__ == "__main__":
    checkpoint_path = "occupancy_panoocc_small.pth"  # Existing checkpoint
    output_path = "occupancy_panoocc_small_updated.pth"  # Path to save updated checkpoint
    pose_head_config = {
        "input_dim": 1280,       # Dimension of the transformer output
        "hidden_dim1": 640,      # First hidden layer size
        "hidden_dim2": 320,      # Second hidden layer size
        "output_dim": 7,         # 7 outputs (3 for translation, 4 for rotation)
        "dropout": 0.1           # Dropout rate
    }

    add_pose_head_to_checkpoint(checkpoint_path, output_path, pose_head_config)
