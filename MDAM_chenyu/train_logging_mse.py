import numpy as np
from sklearn.metrics import roc_auc_score
import torch
import csv
import pandas as pd
import re
from torch import nn

SAVEDMODELS_DIR = './Data/shao_data/脂溶性/checkpoints/'
class Globals: # container for all objects getting passed between log calls
    evaluate_called = False

g = Globals()
mse = nn.MSELoss()


def all_evaluate(logits,target,mse):
    #print(target,output)
    lengh=len(logits)
    scores = torch.sigmoid(logits)
    # auroc = roc_auc_score(target, scores)
    auroc = roc_auc_score(target, scores, multi_fclass='ovr')
    scores = np.array(scores).astype(float)
    sum_scores = np.sum(scores)
    ave_scores = sum_scores / lengh
    target = np.array(target).astype(int)

    Confusion_M = np.zeros((2, 2), dtype=float)  # (TN FP),(FN,TP)
    for i in range(lengh):
        if (scores[i] < ave_scores):
            scores[i] = 0
        else:
            scores[i] = 1
    scores = np.array(scores).astype(int)

    for i in range(lengh):
        if(target[i]==scores[i]):
            if(target[i]==1):
                Confusion_M[0][0] += 1#TP
            else:
                Confusion_M[1][1] += 1#TN
        else:
            if(target[i]==1):
                Confusion_M[0][1] += 1#FP
            else:
                Confusion_M[1][0]  +=1#FN

    Confusion_M = np.array(Confusion_M, dtype=float)
    print('Confusion_M:', Confusion_M)
    accuracy = (Confusion_M[1][1] + Confusion_M[0][0]) / (
            Confusion_M[0][0] + Confusion_M[1][1] + Confusion_M[0][1] + Confusion_M[1][0])

    recall = Confusion_M[0][0] / (Confusion_M[0][0] + Confusion_M[0][1])
    precision = Confusion_M[0][0] / (Confusion_M[0][0] + Confusion_M[1][0])
    F1=2*precision*recall
    h=precision+recall
    F1=F1/h
    sum = 0.0
    for i in range(lengh):
        sum = sum + (target[i] - scores[i]) * (target[i] - scores[i])

    return F1,accuracy,recall,precision,auroc


SCORE_FUNCTIONS = {
    'All':all_evaluate}


def feed_train_net(net,dataloader,predictor,bcewithlogitsloss, mse, criterion):

    batch_outputs = []
    batch_losses = []
    batch_targets = []
    i = 0
    for i_batch, batch in enumerate(dataloader):
        adj_1, nd_1, ed_1,target,d1,mask_1 = batch
        output = net(adj_1, nd_1, ed_1, d1,mask_1)
        # data1 = pd.read_csv("/home/ntu/T-KG/AMDE/Data/FreeSolv/KGemb-FreeSolv-train().csv")
        # data1 = np.array(data1)
        # data3 = np.array(output)
        # data3 = np.asmatrix(data3)
        # for j in range(0, 64):
        #     x = i + j
        #     demo1 = data1[x][0][8:]
        #     demo1 = demo1.split(',')
        #     result_demo1 = []
        #     for item in demo1:
        #         result_demo1.append(float(''.join(re.sub(r'(?=[-+][.]*$)', 'e', item))))
        #     t_emb = np.matrix(result_demo1)
        #     # final_emb = np.outer(data3[j][0], t_emb)
        #     # final_emb = torch.from_numpy(final_emb)
        #     # final_emb = final_emb.unsqueeze(0).unsqueeze(0)
        #     # final_emb = final_emb.to(torch.float32)
        #     # convo = nn.Conv2d(1,16,3,2,1)
        #     # final_emb = convo(final_emb)
        #     # final_emb = con(final_emb)
        #     data3[j][0] = data3[j][0] + t_emb
        # final_emb = torch.from_numpy(data3)
        # final_emb = final_emb.to(torch.float32)
        # i += 64
        logits = predictor(output)
        # loss = criterion(logits, target)
        # loss = bcewithlogitsloss(logits, target)
        mse_loss = mse(logits, target)
        loss = torch.sqrt(mse_loss)
        batch_outputs.append(output)
        batch_losses.append(loss.item())
        batch_targets.append(target)

    outputs = torch.cat(batch_outputs)

    loss = np.mean(batch_losses)#average loss
    targets = torch.cat(batch_targets)
    # return outputs, loss, targets
    return loss

def feed_val_net(net,dataloader,predictor,mse, bcewithlogitsloss,criterion):

    batch_outputs = []
    batch_losses = []
    batch_targets = []
    i = 0
    for i_batch, batch in enumerate(dataloader):
        adj_1, nd_1, ed_1,target,d1,mask_1 = batch
        output = net(adj_1, nd_1, ed_1, d1,mask_1)
        logits = predictor(output)
        # loss = criterion(logits, target)
        # loss = bcewithlogitsloss(logits, target)
        mse_loss = mse(logits, target)
        loss = torch.sqrt(mse_loss)
        batch_outputs.append(output)
        batch_losses.append(loss.item())
        batch_targets.append(target)

    outputs = torch.cat(batch_outputs)

    loss = np.mean(batch_losses)#average loss
    targets = torch.cat(batch_targets)
    # return outputs, loss, targets
    return loss

def feed_test_net(net,dataloader,predictor,bcewithlogitsloss, mse,criterion):

    batch_outputs = []
    batch_losses = []
    batch_targets = []
    for i_batch, batch in enumerate(dataloader):
        adj_1, nd_1, ed_1,target,d1,mask_1 = batch
        output = net(adj_1, nd_1, ed_1, d1,mask_1)
        logits = predictor(output)
        # loss = criterion(logits, target)
        # loss = bcewithlogitsloss(logits, target)
        mse_loss = mse(logits, target)
        loss = torch.sqrt(mse_loss)
        batch_outputs.append(output)
        batch_losses.append(loss.item())
        batch_targets.append(target)

    outputs = torch.cat(batch_outputs)

    loss = np.mean(batch_losses)#average loss
    targets = torch.cat(batch_targets)
    # return outputs, loss, targets
    return loss

def evaluate_net(net,train_dataloader, validation_dataloader, test_dataloader,predictor,bcewithlogitsloss, mse,criterion, args):
    global g
    mse = nn.MSELoss()
    if not g.evaluate_called:
        g.evaluate_called = True
        g.best_mean_train_score, g.best_mean_validation_score, g.best_mean_test_score = 0, 0, 0
        g.train_subset_loader = train_dataloader


    train_output, train_loss, train_target = feed_train_net(net,g.train_subset_loader,predictor,bcewithlogitsloss,mse, criterion)
    validation_output, validation_loss, validation_target = feed_val_net(net,validation_dataloader, predictor,bcewithlogitsloss, mse,criterion)
    test_output, test_loss, test_target = feed_test_net(net,test_dataloader,predictor,bcewithlogitsloss, criterion,mse)

    train_scores = SCORE_FUNCTIONS[args.score](train_output, train_target)
    validation_scores = SCORE_FUNCTIONS[args.score](validation_output, validation_target)
    test_scores = SCORE_FUNCTIONS[args.score](test_output, test_target)
    new_best_model_found = validation_scores[1] > g.best_mean_validation_score


    if new_best_model_found:
        g.best_mean_train_score = train_scores[4]
        g.best_mean_validation_score = validation_scores[4]
        g.best_mean_test_score = test_scores[4]

        if args.savemodel:
            path = SAVEDMODELS_DIR + type(net).__name__
            torch.save(net, path)

    if(args.score=='All'):
     return{
         'loss':{'train': train_loss},
         'F1 score':{'train': train_scores[0], 'validation': validation_scores[0], 'test': test_scores[0]},
         'Accuracy':{'train': train_scores[1], 'validation': validation_scores[1], 'test': test_scores[1]},
         'Recall':{'train': train_scores[2], 'validation': validation_scores[2], 'test': test_scores[2]},
         'Precision':{'train': train_scores[3], 'validation': validation_scores[3], 'test': test_scores[3]},
         'auroc':{'train': train_scores[4], 'validation': validation_scores[4], 'test': test_scores[4]},
        'best mean':{'train': g.best_mean_train_score, 'validation': g.best_mean_validation_score, 'test': g.best_mean_test_score}
        }


def get_run_info(net, args):
    return {
        'net': type(net).__name__,
        'args': ', '.join([str(k) + ': ' + str(v) for k, v in vars(args).items()]),
        'modules': {name: str(module) for name, module in  net._modules.items()}
    }


def less_log(net,train_dataloader, validation_dataloader, test_dataloader,predictor,bcewithlogitsloss,criterion, epoch, args):

    scalars = evaluate_net(net,train_dataloader, validation_dataloader, test_dataloader,predictor,bcewithlogitsloss,criterion, args)
    global g
    if not g.evaluate_called:
        run_info = get_run_info(net, args)
        print('net: ' + run_info['net'])
        print('args: {' + run_info['args'] + '}')
        print('****** MODULES: ******')
        for name, description in run_info['modules'].items():
            print(name + ': ' + description)
        print('**********************')

    if(args.score=='All'):
        print('epoch {}, F1 score :training mean: {}, validation mean: {}, testing mean: {}'.format(
         epoch+1,
         scalars['F1 score']['train'],
         scalars['F1 score']['validation'],
            scalars['F1 score']['test'])
            )
        print('          ACC:training mean: {}, validation mean: {}, testing mean: {}'.format(
        scalars['Accuracy']['train'],
        scalars['Accuracy']['validation'],
        scalars['Accuracy']['test']))

        print('          Precision:training mean: {}, validation mean: {}, testing mean: {}'.format(
            scalars['Precision']['train'],
            scalars['Precision']['validation'],
            scalars['Precision']['test']))
        print('          Recall:training mean: {}, validation mean: {}, testing mean: {}'.format(
            scalars['Recall']['train'],
            scalars['Recall']['validation'],
            scalars['Recall']['test']))
        print('          AUROC:training mean: {}, validation mean: {}, testing mean: {}'.format(
            scalars['auroc']['train'],
            scalars['auroc']['validation'],
            scalars['auroc']['test']))

        print('          best auroc:training mean: {}, validation mean: {}, testing mean: {}'.format(
             scalars['best mean']['train'],
             scalars['best mean']['validation'],
                scalars['best mean']['test']))

        print('          loss:training mean: {}'.format(
            scalars['loss']['train'],))

    else:
         mean_score_key = 'mean {}'.format(args.score)
         print('epoch {}, training mean {}: {}, validation mean {}: {}, testing mean {}:{}'.format(
                epoch + 1,
                args.score, scalars[mean_score_key]['train'],
                args.score, scalars[mean_score_key]['validation'],
                args.score, scalars[mean_score_key]['test']),
         )


def run(net, train_dataloader, validation_dataloader, test_dataloader, predictor, bcewithlogitsloss, criterion, mse,epoch, args):
    train_loss = feed_train_net(net,train_dataloader,predictor,bcewithlogitsloss, mse, criterion)
    validation_loss = feed_val_net(net,validation_dataloader, predictor,bcewithlogitsloss,mse, criterion)
    test_loss = feed_test_net(net,test_dataloader,predictor,bcewithlogitsloss, mse, criterion)
    print('epoch {}, training mean {}'.format(
        epoch + 1,
        train_loss),
    )
    print('epoch {}, val mean {}'.format(
        epoch + 1,
        validation_loss),
    )
    print('epoch {}, test mean {}'.format(
        epoch + 1,
        test_loss),
    )

LOG = {
    'less':run}