import torch
import torch.nn as nn


class TrajectoryHeadingSimpleMLP(nn.Module):
    def __init__(self, hidden_layer_dim=512, num_poses=8):
        super().__init__()
        self.num_poses = num_poses
        self.mlp = nn.Sequential(
            nn.Linear(8 * 2, hidden_layer_dim),
            nn.ReLU(),
            nn.Linear(hidden_layer_dim, hidden_layer_dim),
            nn.ReLU(),
            nn.Linear(hidden_layer_dim, hidden_layer_dim),
            nn.ReLU(),
            nn.Linear(hidden_layer_dim, num_poses * 3),
        )

    def forward(self, traj_xy):
        x = traj_xy.reshape(traj_xy.shape[0], -1)
        out = self.mlp(x)
        out = out.view(-1, self.num_poses, 3)
        heading_pred = torch.tanh(out[:, :, 2:]) * 3.14159
        return heading_pred


def load_heading_model(
    ckpt_path: str, device: str = "cpu", hidden_layer_dim=512, num_poses=8
) -> nn.Module:
    model = TrajectoryHeadingSimpleMLP(
        hidden_layer_dim=hidden_layer_dim, num_poses=num_poses
    )
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
