import torch
import os
from funlib.learn.torch.models import Vgg3D

from .model import Model

class HemiModel(Model):
    def __init__(self):
        dataset = "HEMI"
        input_shape = (80, 80, 80)
        checkpoint = None
        neurotransmitter_list = ["gaba", "acetylcholine", "glutamate",
                                 "serotonin", "octopamine", "dopamine"]
        super().__init__(dataset, checkpoint, input_shape, neurotransmitter_list)
        self.checkpoint = self.get_checkpoint_path()

    def init_model(self):
        input_shape = self.input_shape
        fmaps = 24
        downsample_factors = [(2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)]
        fmap_inc = (2, 2, 2, 2)
        n_convolutions = (4, 2, 2, 2)
        neurotransmitter_list = self.neurotransmitter_list
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

