from http import client
import math
import numpy as np
import torch
from torch.autograd import Function

def categorical_accuracy(preds, y):
    """
    Returns accuracy per batch
    """
    top_pred = preds.argmax(1, keepdim = True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc

def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    
    with torch.no_grad():
        for batch in iterator:
            x, class_label = batch
            predictions = model(x)
            loss = criterion(predictions, class_label)
            acc = categorical_accuracy(predictions, class_label)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def make2d(in_data):
    new_shape = int(math.sqrt(in_data.shape[1]))
    out_data = np.zeros((in_data.shape[0], new_shape, new_shape))
    for i in range(in_data.shape[0]):
        for j in range(new_shape):
            out_data[i][j] = in_data[i][j*new_shape:(j+1)*new_shape]
    return torch.tensor(out_data)
