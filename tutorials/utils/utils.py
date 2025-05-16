import numpy as np
import torch
from tqdm import tqdm


# def check_accuracy(model, testloader, device, topk=(1, 5)):  
#     correct = [0] * len(topk)  
#     total = 0  
#     model.eval()  
#     with torch.no_grad():  
#         for batch in testloader:  
#             if isinstance(batch, dict):  # ImageNet format  
#                 inputs = batch['pixel_values'].to(device)  
#                 targets = batch['labels'].to(device)  
#             else:  # CIFAR10 format  
#                 inputs, targets = batch  
#                 inputs, targets = inputs.to(device), targets.to(device)  
  
#             outputs = model(inputs)  
#             maxk = max(topk)  
#             batch_size = targets.size(0)  
  
#             _, pred = outputs.topk(maxk, 1, True, True)  
#             pred = pred.t()  
#             correct_k = pred.eq(targets.view(1, -1).expand_as(pred))  
  
#             for i, k in enumerate(topk):  
#                 correct[i] += correct_k[:k].reshape(-1).float().sum(0, keepdim=True).item()  
  
#             total += batch_size  
  
#     accuracies = [c / total * 100.0 for c in correct]  
#     return accuracies[0], accuracies[1]  # Top-1 and Top-5 accuracies  

def accuracy_topk(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def check_accuracy(model, testloader, two_input=False):
    correct1 = 0
    correct5 = 0
    total = 0
    model = model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        for batch in testloader:
            if isinstance(batch, dict):  # ImageNet format
                X = batch['pixel_values'].to(device)
                y = batch['labels'].to(device)
            else:  # CIFAR10 format
                X, y = batch
                X = X.to(device)
                y = y.to(device)
            
            if two_input:
                y_pred = model(X, X)
            else:
                y_pred = model(X)
            
            total += y.size(0)

            prec1, prec5 = accuracy_topk(y_pred.data, y, topk=(1, 5))
            
            correct1 += prec1.item() * y.size(0)
            correct5 += prec5.item() * y.size(0)

    model = model.train()
    accuracy1 = correct1 / total
    accuracy5 = correct5 / total
    return accuracy1, accuracy5

# def accuracy_topk(output, target, topk=(1,)):
#     """Computes the precision@k for the specified values of k"""
#     maxk = max(topk)
#     batch_size = target.size(0)

#     _, pred = output.topk(maxk, 1, True, True)
#     pred = pred.t()
#     correct = pred.eq(target.view(1, -1).expand_as(pred))

#     res = []
#     for k in topk:
#         correct_k = correct[:k].reshape(-1).view(-1).float().sum(0, keepdim=True)
#         res.append(correct_k)
#     return res


# def check_accuracy(model, testloader, two_input=False):
#     correct1 = 0
#     correct5 = 0
#     total = 0
#     model = model.eval()
#     device = next(model.parameters()).device
#     with torch.no_grad():
#         for X, y in testloader:
#             X = X.to(device)
#             y = y.to(device)
#             if two_input:
#                 y_pred = model.forward(X, X)
#             else:
#                 y_pred = model.forward(X)
#             total += y.size(0)

#             prec1, prec5 = accuracy_topk(y_pred.data, y, topk=(1, 5))
            
#             correct1 += prec1.item()
#             correct5 += prec5.item()

#     model = model.train()
#     accuracy1 = correct1 / total
#     accuracy5 = correct5 / total
#     return accuracy1, accuracy5


def check_accuracy_onnx(model_path, testloader, two_input=False):
    import onnxruntime as ort
    sess_options = ort.SessionOptions()
    ort_sess = ort.InferenceSession(model_path, sess_options)   
    correct1 = 0
    correct5 = 0
    total = 0

    for X, y in testloader:
        try:
            if not two_input:
                outputs = ort_sess.run(None, {'input.1': X.numpy()})[0]
            else:
                outputs = ort_sess.run(None, {'input.1': X.numpy(), 'input.2': X.numpy()})[0]
        except:
            continue
        prec1, prec5 = accuracy_topk(torch.tensor(outputs), y.data, topk=(1, 5))
        correct1 += prec1.item()
        correct5 += prec5.item()
        total += y.size(0)

    accuracy1 = correct1 / total
    accuracy5 = correct5 / total
    return accuracy1, accuracy5

def compute_output_onnx_given_input(model_path, input_tensor):
    import onnxruntime as ort
    ort_sess = ort.InferenceSession(model_path)
    output = ort_sess.run(None, {'input.1': input_tensor.numpy()})[0]
    return output