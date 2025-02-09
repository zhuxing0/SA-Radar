import torch
import torch.nn as nn
import torch.nn.functional as F
from core.submodule import BasicConv, hourglass_v2
import pdb

try:
    autocast = torch.cuda.amp.autocast
except:
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass

class ICFARNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        hidden_dims = args.hidden_dims # default 32
        output_dims = args.output_dims # default 1

        self.cube_stem = BasicConv(2, hidden_dims, is_3d=True, kernel_size=3, stride=1, padding=1)
        self.cube_agg = hourglass_v2(hidden_dims)
        self.output_layer = nn.Conv3d(8, output_dims, 3, 1, 1, bias=False)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, radar_point, segment_mask):
        """
            input shape: (B, 1, R, D, A), (B, 1, R, D, A)
            output shape: (B, output_dims, R, D, A)
        """
        radar_cube = torch.cat((radar_point, segment_mask), dim=1)

        radar_cube = self.cube_stem(radar_cube) # (B, 1, R, D, A) -> (B, hidden_dims, R, D, A)
        radar_cube = self.cube_agg(radar_cube)  # (B, hidden_dims, R, D, A) -> (B, 8, R, D, A)

        radar_cube = nn.ReLU()(self.output_layer(radar_cube)) # (B, 8, R, D, A) -> (B, output_dims, R, D, A)
        if torch.isnan(radar_cube).any():
            pdb.set_trace()
        return radar_cube