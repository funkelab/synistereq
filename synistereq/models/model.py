from abc import ABC, abstractmethod
import os
import torch
import torch.nn.functional as F


class Model(ABC):
    def __init__(self, dataset, checkpoint, input_shape, neurotransmitter_list):
        if not os.path.exists(checkpoint):
            raise ValueError(f"Checkpoint {checkpoint} does not exist")

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
