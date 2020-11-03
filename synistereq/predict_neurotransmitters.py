import numpy as np

def predict_neurotransmitters(positions,
                              dataset,
                              model,
                              batch_size=24):
    """
    positions `list of array-like of ints`:
        Synaptic postions in the given dataset [(z0,y0,x0), (z1, y1, x1), ...]

    returns:
        list of dictionaries of neurotransmitter probabilities
    """
    if dataset.name != model.dataset:
        raise ValueError(f"Dataset ({dataset.name}) model ({model.dataset}) missmatch")

    torch_model = model.init()
    torch_model.eval()
    
    nt_probabilities = []
    for i in tqdm(range(0, len(positions), batch_size)):
        batched_positions = positions[i:i+batch_size]
        crops = dataset.get_crops(positions,
                                  model.input_shape)
        crops = dataset.normalize(crops)
        crops = model.prepare_batch(crops)
        prediction = torch_model(crops)
        prediction = model.softmax(prediction)
        
        # Iterate over batch and grab predictions
        for k in range(np.shape(prediction)[0]):
            out_k = output[k,:].tolist()
            nt_probability = {model.neurotransmitter_list[i]: 
                              out_k[i] for i in range(len(model.neurotransmitter_list))}
            nt_probabilities.append(nt_probability)

    return nt_probabilities

