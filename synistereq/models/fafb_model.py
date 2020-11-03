import torch
import os
from funlib.learn.torch.models import Vgg3D

from .model import Model

class FafbModel(Model):
    def __init__(self):
        dataset = "FAFB"
        checkpoint = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                  "../checkpoints/fafb_checkpoint")
        input_shape = (16,160,160)
        neurotransmitter_list = ["gaba", "acetylcholine", "glutamate",
                                 "serotonin", "octopamine", "dopamine"]
        super().__init__(dataset, checkpoint, input_shape, neurotransmitter_list)

    def init_model(self):
        input_shape = self.input_shape
        fmaps = 12
        downsample_factors = [(1, 2, 2), (1, 2, 2), (1, 2, 2), (2, 2, 2)]
        neurotransmitter_list = self.neurotransmitter_list
        fmap_inc = (2,2,2,2)
        n_convolutions = (2,2,2,2)
        output_classes = len(neurotransmitter_list)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = Vgg3D(input_size=input_shape, 
                      fmaps=fmaps,
                      downsample_factors=downsample_factors,
                      fmap_inc=fmap_inc,
                      n_convolutions=n_convolutions,
                      output_classes=output_classes)
        model.to(device)
        checkpoint = torch.load(self.checkpoint, 
                                map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model

