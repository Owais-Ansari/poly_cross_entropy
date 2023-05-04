
import torch
import torch.nn as nn

def poly_cross_entropy1(logits,labels, epsilon = 1.0, num_classes=2, ignore_label=-1, gpu = 0):
    '''
    logits: Prediction after linear layer    
    labels: GT mask
    num_classes:num of classes
    ignore_label: label of ignore class
    epsilon : Parameter to tune the polyloss
    gpu : gpu_id
    '''
    ce = nn.CrossEntropyLoss(weight=None,label_smoothing=0,reduction ='none',ignore_index=ignore_label)(logits,labels)
    probs = torch.nn.functional.softmax(logits,dim=1)
    #one_hot_encodig
    out = torch.argmax(probs,1)
    Y = torch.zeros(list(probs.shape), dtype = torch.uint8).cuda(torch.device('cuda:'+ str(gpu)))
    for i in range(list(probs.shape)[0]):
        for j in range(list(probs.shape)[1]):
            Y[i][j][out[i]==j]=1 
    pt = torch.sum(Y*probs,dim = 1)
    poly_loss =  ce + epsilon * (1-pt)
    return torch.mean(poly_loss)