from abc import ABC, abstractmethod
import os
import torch
import torch.nn.functional as F
import configparser


class Model(ABC):
    def __init__(self, dataset, checkpoint, input_shape, neurotransmitter_list):
        self.dataset = dataset
        self.checkpoint = checkpoint
        self.input_shape = input_shape
        self.neurotransmitter_list = neurotransmitter_list
        super().__init__()

    @abstractmethod
    def init_model(self):
        """
        Initializes a trained model with the given 
        checkpoint and returns it.
            
            Returns:
                model (object): The initialized pytorch model
        """
        pass

    def prepare_batch(self, batch):
        """
        Prepare data batch to be input into the model.

        Args:
            batch (np.4darray): raw data with 0 dim = batch dim

        Returns:
            A 5d pytorch tensor with a channel dimension.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        batch = torch.tensor(batch, device=device)
        batch = batch.unsqueeze(1)
        return batch

    def softmax(self, prediction):
        return F.softmax(prediction, dim=1)

    def get_checkpoint_path(self):
        with open(os.path.join(os.path.dirname(__file__), 
                               "../checkpoint_paths.ini"), "r") as fp:
            config = configparser.ConfigParser()
            config.readfp(fp)
            checkpoint = os.path.abspath(config.get("Models", self.dataset))
        return checkpoint

