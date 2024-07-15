import torch
import numpy as np
from torch.nn import functional as F
from sklearn.metrics import accuracy_score, precision_recall_curve
import cv2
from torch import nn
import matplotlib.pyplot as plt
import pandas as pd
from statistics import mean
import os
from tqdm import tqdm
from functools import partial
from sklearn.manifold import TSNE
import copy
import csv
from models.recttt import DynamicMutualLoss, DynamicFlipLoss


def modify_grad(x, inds, factor=0.):
    inds = inds.expand_as(x)
    x[inds] *= factor
    return x


def global_cosine(a, b, stop_grad=True):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    weight = [1, 1, 1]
    for item in range(len(a)):
        if stop_grad:
            loss += torch.mean(1 - cos_loss(a[item].view(a[item].shape[0], -1).detach(),
                                            b[item].view(b[item].shape[0], -1))) * weight[item]
        else:
            loss += torch.mean(1 - cos_loss(a[item].view(a[item].shape[0], -1),
                                            b[item].view(b[item].shape[0], -1))) * weight[item]
    return loss


def global_cosine_hm(a, b, alpha=1., factor=0., weight=[1, 1, 1]):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        a_ = a[item].detach()
        b_ = b[item]
        with torch.no_grad():
            point_dist = 1 - cos_loss(a_, b_).unsqueeze(1)
        mean_dist = point_dist.mean()
        std_dist = point_dist.reshape(-1).std()

        loss += torch.mean(1 - cos_loss(a_.view(a_.shape[0], -1),
                                        b_.view(b_.shape[0], -1))) * weight[item]
        thresh = mean_dist + alpha * std_dist
        partial_func = partial(modify_grad, inds=point_dist < thresh, factor=factor)
        b_.register_hook(partial_func)

    return loss

def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image - a_min) / (a_max - a_min)

def cvt2heatmap(gray):
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap

def return_best_thr(y_true, y_score):
    precs, recs, thrs = precision_recall_curve(y_true, y_score)

    f1s = 2 * precs * recs / (precs + recs + 1e-7)
    f1s = f1s[:-1]
    thrs = thrs[~np.isnan(f1s)]
    f1s = f1s[~np.isnan(f1s)]
    best_thr = thrs[np.argmax(f1s)]
    return best_thr

def specificity_score(y_true, y_score):
    y_true = np.array(y_true)
    y_score = np.array(y_score)

    TN = (y_true[y_score == 0] == 0).sum()
    N = (y_true == 0).sum()
    return TN / N

def evaluation_ttt2_semi(model, dataloader, optimizer2, alpha, device, n_iter, checkpoint, type_loss='none', hm=True, per= '', multi=False):
    preds = []
    preds2 = []
    confs = []
    confs2 = []
    labels = []
    final_preds = []
    loss_list = []
    plt_feat = False
    all_features = []
    all_labels = []
    layers_train= [True, True, True, False]
    if multi:
        model.module.test_train(layers_train)
    else:
        model.test_train(layers_train)
    ckt_ = copy.deepcopy(checkpoint)
    kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)
    if type_loss == 'ce':
        ce_loss = nn.CrossEntropyLoss(reduction="none")
    elif 'dmt' in type_loss:
        ce_loss = DynamicMutualLoss()
    elif 'flip' in type_loss:
        ce_loss = DynamicFlipLoss()
    else:
        ce_loss = None

    for t_img, t_label in tqdm(dataloader):
        t_img = t_img.to(device, non_blocking=True)
        #print('new batch')
        for i in range(n_iter):
            en, de, cls_p, cls2_p = model(t_img)
            if hm:
                aux_loss = global_cosine_hm(en[:3], de[:3], alpha=alpha, factor=0., weight=[1,1,1]) / 4 + \
                        global_cosine_hm(en[3:6], de[3:6], alpha=alpha, factor=0., weight=[1,1,1]) / 4 + \
                        global_cosine_hm(en[6:9], de[6:9], alpha=alpha, factor=0., weight=[1,1,1]) / 4 + \
                        global_cosine_hm(en[9:], de[9:], alpha=alpha, factor=0., weight=[1,1,1]) / 4
            else:
                aux_loss = global_cosine(en[:3], de[:3]) / 4 + \
                        global_cosine(en[3:6], de[3:6]) / 4 + \
                        global_cosine(en[6:9], de[6:9]) / 4 + \
                        global_cosine(en[9:], de[9:]) / 4
            
            if type_loss == 'dmt':
                ''' CE DMT'''
                cls_loss, _ = ce_loss(inputs=cls_p, targets=cls2_p, gamma1=1, gamma2=1)
            elif type_loss == 'dmt+':
                cls_loss, _ = ce_loss(inputs=cls_p, targets=(cls_p+cls2_p)/2, gamma1=1, gamma2=1, log=(i==n_iter-1))
            elif type_loss == 'dmt2':
                cls_loss1, _ = ce_loss(inputs=cls_p, targets=(cls_p+cls2_p)/2, gamma1=1, gamma2=1)
                cls_loss2, _ = ce_loss(inputs=cls2_p, targets=(cls_p+cls2_p)/2, gamma1=1, gamma2=1)
                cls_loss = cls_loss1 + cls_loss2
            elif type_loss == 'ce':
                ''' CE ON CHANGE '''
                pred_prob = np.array(F.softmax(cls_p, dim=1).argmax(dim=1).detach().cpu())
                pred_prob2 = np.array(F.softmax(cls2_p, dim=1).argmax(dim=1).detach().cpu())
                pseudo_lab = np.array((cls_p + cls2_p).argmax(dim=1).detach().cpu())
                
                mask1 = torch.ne(F.softmax(cls_p, dim=1).argmax(dim=1), (cls_p + cls2_p).argmax(dim=1))
                mask2 = torch.ne(F.softmax(cls2_p, dim=1).argmax(dim=1), (cls_p + cls2_p).argmax(dim=1))
                
                ce_loss_1 = ce_loss(cls_p, (cls_p + cls2_p).argmax(dim=1).detach()) * mask1
                ce_loss_2 = ce_loss(cls2_p, (cls_p + cls2_p).argmax(dim=1).detach()) * mask2
                
                cls_loss = ce_loss_1.sum()/sum(mask1) + ce_loss_2.sum()/sum(mask2)
            elif type_loss == 'flip':
                cls_loss, _ = ce_loss(inputs=cls_p, targets=(cls_p+cls2_p)/2, gamma1=1, gamma2=1, log=(i==n_iter-1))
            else:
                ''' NO PSEUDO '''
                cls_loss = 0
            aux_w = 1
            final_loss = aux_loss * aux_w + cls_loss
            #print(aux_loss.item(), cls_loss.item(), final_loss.item())
            loss_list.append(final_loss.item())
            final_loss.backward()
            optimizer2.step()
            optimizer2.zero_grad(set_to_none=True)
            
        model.train(False)
        ''' BATCH EVAL '''
        with torch.no_grad():
            _, _, pred, pred2 = model(t_img)
            
            if plt_feat:
                features =  torch.flatten(model.get_features(t_img), start_dim=1)
                all_features.append(features.cpu().numpy())
                all_labels.append(t_label.numpy())
            
            final_preds.append(np.array((pred + pred2).argmax(dim=1).cpu()))
            preds.append(np.array(pred.argmax(dim=1).cpu()))
            preds2.append(np.array(pred2.argmax(dim=1).cpu()))
            labels.append(np.array(t_label.cpu()))
            confs.append(np.array(pred.max(dim=1)[0].cpu()))
            confs2.append(np.array(pred2.max(dim=1)[0].cpu()))
        
        #Reload weights
        model.load_state_dict(ckt_['model_state_dict'])
        if multi:
            model.module.test_train(layers_train)
        else:
            model.test_train(layers_train)

    #TSNE
    if plt_feat:
        print('Generating T-SNE')
        features = np.concatenate(all_features)
        labels = np.concatenate(all_labels)
            
        tsne = TSNE(n_components=2, random_state=42)
        embedded_features = tsne.fit_transform(features)

        plt.figure(figsize=(10, 8))
        plt.scatter(embedded_features[:, 0], embedded_features[:, 1], c=labels, cmap='viridis')
        plt.colorbar()
        plt.title('t-SNE - {} - {} iterations'.format(per , n_iter))
        plt.savefig('tsne/tsne_{}_{}.png'.format(per, n_iter))

    confs = np.concatenate(confs, axis=None)
    confs2 = np.concatenate(confs2, axis=None)
    preds = np.concatenate(preds, axis=None)
    preds2 = np.concatenate(preds2, axis=None)
    final_preds = np.concatenate(final_preds, axis=None)
    labels = np.concatenate(labels, axis=None)
    
    acc = accuracy_score(labels, preds)
    print('{}: acc1 {}'.format(n_iter, acc))
    acc = accuracy_score(labels, preds2)
    print('{}: acc2 {}'.format(n_iter, acc))
    
    count_diff = 0
    for fp, p1 in zip(final_preds, preds):
        if fp != p1:
            count_diff +=1
    acc = accuracy_score(final_preds, preds)
    print('Acc on {} iter, {} diff: concordance {}'.format(n_iter, count_diff,acc))

    ''' FINAL SCORES '''
    acc = accuracy_score(labels, final_preds)
    return np.mean(loss_list), 0, acc

def evaluation_ttt(model, dataloader, optimizer2, alpha, device, n_iter, checkpoint, type_loss='none', hm=True, per= '', multi=False):
    preds = []
    preds2 = []
    confs = []
    confs2 = []
    labels = []
    final_preds = []
    loss_list = []
    plt_feat = False
    all_features = []
    all_labels = []
    layers_train= [True, True, True, False]
    if multi:
        model.module.test_train(layers_train)
    else:
        model.test_train(layers_train)
        
    ckt_ = copy.deepcopy(checkpoint)

    for t_img, t_label in tqdm(dataloader):
        t_img = t_img.to(device, non_blocking=True)
        #print('new batch')
        for i in range(n_iter):
            en, de, cls_p = model(t_img)
            if hm:
                aux_loss = global_cosine_hm(en[:3], de[:3], alpha=alpha, factor=0., weight=[1,1,1]) / 4 + \
                        global_cosine_hm(en[3:6], de[3:6], alpha=alpha, factor=0., weight=[1,1,1]) / 4
            else:
                aux_loss = global_cosine(en[:3], de[:3]) / 4 + \
                        global_cosine(en[3:6], de[3:6]) / 4

            final_loss = aux_loss
            loss_list.append(final_loss.item())
            final_loss.backward()
            optimizer2.step()
            optimizer2.zero_grad(set_to_none=True)
            
        model.train(False)
        ''' BATCH EVAL '''
        with torch.no_grad():
            _, _, pred = model(t_img)
            
            if plt_feat:
                features =  torch.flatten(model.get_features(t_img), start_dim=1)
                all_features.append(features.cpu().numpy())
                all_labels.append(t_label.numpy())
            
            preds.append(np.array(pred.argmax(dim=1).cpu()))
            labels.append(np.array(t_label.cpu()))
            confs.append(np.array(pred.max(dim=1)[0].cpu()))
        
        #Reload weights
        model.load_state_dict(ckt_['model_state_dict'])
        if multi:
            model.module.test_train(layers_train)
        else:
            model.test_train(layers_train)

    #TSNE
    if plt_feat:
        features = np.concatenate(all_features)
        labels = np.concatenate(all_labels)
            
        tsne = TSNE(n_components=2, random_state=42)
        embedded_features = tsne.fit_transform(features)

        plt.figure(figsize=(10, 8))
        plt.scatter(embedded_features[:, 0], embedded_features[:, 1], c=labels, cmap='viridis')
        plt.colorbar()
        plt.title('t-SNE - {} - {} iterations'.format(per , n_iter))
        plt.savefig('tsne_{}.png'.format(n_iter))

    confs = np.concatenate(confs, axis=None)
    preds = np.concatenate(preds, axis=None)
    labels = np.concatenate(labels, axis=None)
    
    ''' FINAL SCORES '''
    acc = accuracy_score(labels, preds)
    return np.mean(loss_list), 0, acc

def save_images(digits, labels, title):
    n = 10
    indexes = np.random.choice(len(labels), size=n)
    n_digits = digits[indexes]

    fig = plt.figure(figsize=(20, 4))
    plt.title(title)
    plt.yticks([])
    plt.xticks([])

    for i in range(10):
        ax = fig.add_subplot(1, 10, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow((n_digits[i].permute(1, 2, 0).numpy()* 255).astype(np.uint8))
    
    plt.savefig('{}.png'.format(title))

def write_res_csv(save_dir, results, results_best):
    csv_file_path = os.path.join(save_dir, 'results.csv')

    # Define the CSV header
    header = ["Corruption Type", "Level", "Epochs", "TTT iterations", "Final Loss", "Aux Loss", "Cls Loss", "Test Loss",
            "Mean F1", "Mean Accuracy", "Best F1", "Best Accuracy"]
    data = []
    for (corruptions, level, epochs, ttt_it, final_loss, aux_loss, cls_loss, test_loss,
            mean_f1, mean_acc), (c, best_f1, best_acc) in zip(results,results_best):
    
        data.append([corruptions, level, epochs, ttt_it, final_loss, aux_loss, cls_loss, test_loss,
            mean_f1, mean_acc, best_f1, best_acc])

    # Write to the CSV file
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Write the header
        writer.writerow(header)
        
        # Write the data
        for result in data:
            writer.writerow(result)
