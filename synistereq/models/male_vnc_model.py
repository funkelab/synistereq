import torch
from funlib.learn.torch.models import Vgg3D

from .model import Model

class MaleVncModel(Model):
    def __init__(self):
        dataset = "MALE_VNC"
        input_shape = (80, 80, 80)
        checkpoint = None
        neurotransmitter_list = ["gaba", "acetylcholine", "glutamate"]
        super().__init__(dataset, checkpoint, input_shape, neurotransmitter_list)
        self.checkpoint = self.get_checkpoint_path()

    def init_model(self):
        input_shape = self.input_shape
        fmaps = 16
        downsample_factors = [(2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)]
        fmap_inc = (2, 2, 2, 2)
        n_convolutions = (2, 2, 2, 2)
        neurotransmitter_list = self.neurotransmitter_list
        output_classes = len(neurotransmitter_list)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(device)
        print(self.checkpoint)
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

