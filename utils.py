import torch

def calc_acc(outputs, labels):
    _, predict = torch.max(outputs, 1)
    correct = (predict == labels).sum().item()
    accurancay = correct / labels.size(0)

    return accurancay