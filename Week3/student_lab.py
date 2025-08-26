# lab.py
import torch

def get_entropy_of_dataset(tensor: torch.Tensor):
    """
    Calculate the entropy of the dataset.
    """
    target = tensor[:, -1]  # last column is target
    classes, counts = torch.unique(target, return_counts=True)
    probs = counts.float() / target.size(0)
    entropy = -torch.sum(probs * torch.log2(probs))
    return entropy.item()


def get_avg_info_of_attribute(tensor: torch.Tensor, attribute: int):
    """
    Calculate the average information (weighted entropy) for an attribute.
    """
    attr_values = tensor[:, attribute]
    total = tensor.size(0)
    avg_info = 0.0

    for val in torch.unique(attr_values):
        subset = tensor[attr_values == val]
        subset_entropy = get_entropy_of_dataset(subset)
        weight = subset.size(0) / total
        avg_info += weight * subset_entropy

    return avg_info


def get_information_gain(tensor: torch.Tensor, attribute: int):
    """
    Information Gain = Entropy(S) - Avg_Info(attribute)
    """
    dataset_entropy = get_entropy_of_dataset(tensor)
    avg_info = get_avg_info_of_attribute(tensor, attribute)
    info_gain = dataset_entropy - avg_info
    return round(info_gain, 4)


def get_selected_attribute(tensor: torch.Tensor):
    """
    Returns a dict of attribute: info_gain and the best attribute index
    """
    num_attributes = tensor.size(1) - 1  # exclude target
    info_gains = {}

    for attr in range(num_attributes):
        info_gains[attr] = get_information_gain(tensor, attr)

    best_attr = max(info_gains, key=info_gains.get)
    return info_gains, best_attr
