from neuralop.models import FNO
import torch
import os
from pathlib import Path

def load_model(model_path: str, device: str):

    model_path = os.path.expanduser(model_path)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    filename = Path(model_path).stem

    params = dict(token.split("-", 1) for token in filename.split("_") if "-" in token)
    
    model = FNO(n_modes                  = (int(params["nmodes"]),),
                in_channels              = int(params["in"]),
                out_channels             = int(params["out"]),
                n_layers                 = int(params["nlayers"]),
                projection_channel_ratio = int(params["projectionratio"]),
                hidden_channels          = int(params["hc"]))

    print(
        f"Loading FNO model with "
        f"modes={int(params['nmodes'])}, "
        f"in_channels={int(params['in'])}, "
        f"out_channels={int(params['out'])}, "
        f"nlayers={int(params['nlayers'])}, "
        f"projection_ratio={int(params['projectionratio'])}, ",
        f"hidden_channels={int(params['hc'])}, "
        f"device={device}.\n"
    )
                
    model.to(device)

    model.load_state_dict(checkpoint)

    model.eval()

    return model