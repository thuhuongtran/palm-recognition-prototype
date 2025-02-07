import torch
import netron

from self_supervised.model import PalmprintEncoder

model = PalmprintEncoder()
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy_input, "palmprint_encoder.onnx")

netron.start("palmprint_encoder.onnx")