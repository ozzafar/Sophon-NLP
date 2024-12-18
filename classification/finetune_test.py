import random
import numpy as np
from torch.utils.data import DataLoader, Dataset
import os
from datetime import datetime
import argparse
import sys
sys.path.append('../')
def args_parser():
    parser = argparse.ArgumentParser(description='train N shadow models')
    parser.add_argument('--bs', default=200, type=int)
    parser.add_argument('--arch', default='', type=str)
    parser.add_argument('--gpus', default='0,1', type=str)
    parser.add_argument('--truly_finetune_epochs', default=20, type=int)
    parser.add_argument('--finetune_lr', default=0.0001, type=float)
    parser.add_argument('--notes', default=None, type=str)
    parser.add_argument('--path', default=None, type=str) 
    parser.add_argument('--dataset', default='CIFAR10', type=str) 
    parser.add_argument('--seed', default=99, type=int)
    parser.add_argument('--start', default='', type=str)
    
    
    args = parser.parse_args()
    return args
args = args_parser()
if args.gpus:
    gpu_list = args.gpus.split(',')
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(gpu_list)
    # os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [4,2]))
    devices_id = [id for id in range(len(gpu_list))]
from utils import test, process, resume_dict, initialize00, set_seed, get_finetuned_model, get_pretrained_model, get_init_model
from timm.models import create_model
from tqdm import tqdm
import torch
from torch import nn, optim
from utils import get_dataset, test_accuracy
import wandb
import timm

def test_finetune_final(args, mode, model, trainset, testset, epochs, lr):
    model = nn.DataParallel(model)
    trainloader = DataLoader(trainset, batch_size=args.bs, shuffle=True, num_workers=4,drop_last=True)
    testloader = DataLoader(testset, batch_size=args.bs, shuffle=False, num_workers=4,drop_last=True)
    # optimizer = optim.SGD(model.module.score.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    optimizer = optim.SGD(model.module.score.parameters(), lr=lr, weight_decay=0.01)


    criterion = nn.CrossEntropyLoss()
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    model.train()
    accs = []
    losses = []
    for ep in tqdm(range(epochs), desc="epoch"):
        model.train()
        train_loss = 0
        for batch in tqdm(trainloader, desc="batch"):
            inputs, targets, attention_mask = torch.stack(batch["input_ids"], dim=1).cuda(), batch["label"].cuda(), torch.stack(batch["attention_mask"], dim=1).cuda()
            outputs = model(inputs, attention_mask=attention_mask)
            loss = criterion(outputs.logits, targets)
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        wandb.log({f'{mode}: train loss': train_loss / len(trainloader.dataset)})
        test_acc, test_loss = test(model, testloader, torch.device('cuda'))
        accs.append(test_acc)
        losses.append(test_loss)
        wandb.log({f'{mode}: test accuracy':test_acc, f'{mode}: test loss':test_loss,})
    print(f'test accuracy is {accs}, test loss is {losses}')
    return round(test_acc,2), round(test_loss,2)

if __name__ == '__main__':
    args = args_parser()
    import wandb
    wandb.init(
        project="sophon classification finetune test nlp",
        config = args,
        name = f"{args.arch}_{args.dataset}" ,
        notes = args.notes)
    seed = args.seed
    set_seed(seed)
    trainset_tar, testset_tar = get_dataset(args.dataset, '../../../datasets', args=args)
    test_model_path = args.path
    ####  finetuned ckpt
    if args.start == 'sophon':
        print('========test finetuned: direct all=========')
        model = get_finetuned_model(args, test_model_path)
        acc, test_loss = test_finetune_final(args, 'finetuned/direct all', model.cuda(), trainset_tar, testset_tar, args.truly_finetune_epochs, args.finetune_lr)

    ### normal pretrained
    elif args.start == 'normal':
        print('========test normal pretrained: direct all=========')
        model = get_pretrained_model(args)
        acc, test_loss = test_finetune_final(args, 'normal pretrained/direct all', model.cuda(), trainset_tar, testset_tar, args.truly_finetune_epochs, args.finetune_lr)

    # ### train from scratch
    elif args.start == 'scratch':
        print('========test train from scratch=========')
        acc, test_loss = test_finetune_final(args, 'train from scratch/', model.cuda(), trainset_tar, testset_tar, args.truly_finetune_epochs, args.finetune_lr)

    else:
        assert(0)

    print(f'test accuracy is {acc}, test loss is {test_loss}')