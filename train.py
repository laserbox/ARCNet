import os
import argparse

import numpy as np
from tqdm import tqdm
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torchvision import transforms

from lib.dataset import Dataset
from lib.utils.utils import AverageMeter, str2bool, RandomErase
from lib.metrics import compute_accuracy
from lib.losses import FocalLoss, LabelSmoothingLoss
from lib.optimizers import RAdam
from lib.datapath import img_path_generator
from lib.models.RA import RA
from lib.models.model_factory import get_model
from lib.models.gcn import SoftLabelGCN


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                        help='model architecture: ' + ' (default: resnet34)')
    parser.add_argument('--freeze_bn', default=True, type=str2bool)
    parser.add_argument('--dropout_p', default=0, type=float)
    parser.add_argument('--loss', default='CrossEntropyLoss',
                        choices=['CrossEntropyLoss', 'FocalLoss', 'MSELoss', 'LabelSmoothingLoss'])
    parser.add_argument('--reg_coef', default=1.0, type=float)
    parser.add_argument('--cls_coef', default=0.1, type=float)
    parser.add_argument('--epochs', default=30, type=int,
                        metavar='N', help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=32, type=int,
                        metavar='N', help='mini-batch size (default: 32)')
    parser.add_argument('--img_size', default=256, type=int,
                        help='input image size (default: 256)')
    parser.add_argument('--input_size', default=224, type=int,
                        help='input image size (default: 224)')
    parser.add_argument('--optimizer', default='SGD')
    parser.add_argument('--pred_type', default='classification',
                        choices=['classification', 'regression'])
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau'])
    parser.add_argument('--lr', '--learning_rate', default=1e-3,
                        type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--min_lr', default=1e-5,
                        type=float, help='minimum learning rate')
    parser.add_argument('--factor', default=0.5, type=float)
    parser.add_argument('--patience', default=5, type=int)
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=1e-4,
                        type=float, help='weight decay')
    parser.add_argument('--nesterov', default=False,
                        type=str2bool, help='nesterov')
    parser.add_argument('--gpus', default='2', type=str)
    parser.add_argument('--mode', default='baseline', choices=['baseline', 'arcnet', 'gcn'])

    # lstm
    parser.add_argument('--lstm_layers', default=3, type=int)
    parser.add_argument('--lstm_hidden', default=256, type=int)
    parser.add_argument('--lstm_recurrence', default=10, type=int)

    # preprocessing
    parser.add_argument('--scale_radius', default=True, type=str2bool)
    parser.add_argument('--normalize', default=False, type=str2bool)
    parser.add_argument('--padding', default=False, type=str2bool)

    # data augmentation
    parser.add_argument('--rotate', default=True, type=str2bool)
    parser.add_argument('--rotate_min', default=-180, type=int)
    parser.add_argument('--rotate_max', default=180, type=int)
    parser.add_argument('--rescale', default=True, type=str2bool)
    parser.add_argument('--rescale_min', default=0.8889, type=float)
    parser.add_argument('--rescale_max', default=1.0, type=float)
    parser.add_argument('--shear', default=True, type=str2bool)
    parser.add_argument('--shear_min', default=-36, type=int)
    parser.add_argument('--shear_max', default=36, type=int)
    parser.add_argument('--translate', default=False, type=str2bool)
    parser.add_argument('--translate_min', default=0, type=float)
    parser.add_argument('--translate_max', default=0, type=float)
    parser.add_argument('--flip', default=True, type=str2bool)
    parser.add_argument('--contrast', default=True, type=str2bool)
    parser.add_argument('--contrast_min', default=0.9, type=float)
    parser.add_argument('--contrast_max', default=1.1, type=float)
    parser.add_argument('--random_erase', default=True, type=str2bool)
    parser.add_argument('--random_erase_prob', default=0.5, type=float)
    parser.add_argument('--random_erase_sl', default=0.02, type=float)
    parser.add_argument('--random_erase_sh', default=0.4, type=float)
    parser.add_argument('--random_erase_r', default=0.3, type=float)

    # dataset
    parser.add_argument('--train_dataset', default='ucm',
                        choices=['ucm', 'whu', 'opt', 'nwpu', 'aid'])
    parser.add_argument('--cv', default=True, type=str2bool)
    parser.add_argument('--n_splits', default=5, type=int)

    parser.add_argument('--pretrained_model')

    args = parser.parse_args()

    return args


def train(args, train_loader, model, criterion, optimizer, epoch):
    losses = AverageMeter()
    ac_scores = AverageMeter()

    model.train()

    for i, (input, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        input = input.cuda()
        target = target.cuda()

        output = model(input)
        if args.mode == 'baseline':
            output = output
        elif args.mode == 'gcn':
            output, adj = output
        else:
            output = output

        if args.pred_type == 'classification':
            loss = criterion(output, target)
        elif args.pred_type == 'regression':
            loss = criterion(output.view(-1), target.float())
        elif args.pred_type == 'multitask':
            loss = args.reg_coef * criterion['regression'](output[:, 0], target.float()) + \
                args.cls_coef * \
                criterion['classification'](output[:, 1:], target)
            output = output[:, 0].unsqueeze(1)

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ac_score = compute_accuracy(output, target)

        losses.update(loss.item(), input.size(0))
        ac_scores.update(ac_score, input.size(0))
    if args.mode == 'gcn':
        print(torch.max(adj))
    return losses.avg, ac_scores.avg


def validate(args, val_loader, model, criterion):
    losses = AverageMeter()
    ac_scores = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (input, target) in tqdm(enumerate(val_loader), total=len(val_loader)):
            input = input.cuda()
            target = target.cuda()

            output = model(input)
            if args.mode == 'baseline':
                output = output
            elif args.mode == 'gcn':
                output, adj = output
            else:
                output = output

            if args.pred_type == 'classification':
                loss = criterion(output, target)
            elif args.pred_type == 'regression':
                loss = criterion(output.view(-1), target.float())
            elif args.pred_type == 'multitask':
                loss = args.reg_coef * criterion['regression'](output[:, 0], target.float()) + \
                    args.cls_coef * \
                    criterion['classification'](output[:, 1:], target)
                output = output[:, 0].unsqueeze(1)

            ac_score = compute_accuracy(output, target)

            losses.update(loss.item(), input.size(0))
            ac_scores.update(ac_score, input.size(0))

    return losses.avg, ac_scores.avg


def main():
    args = parse_args()

    if args.name is None:
        args.name = '%s_%s' % (args.mode, args.arch)
    if not os.path.exists('models/%s' % args.name):
        os.makedirs('models/%s' % args.name)

    print('Config -----')
    for arg in vars(args):
        print('- %s: %s' % (arg, getattr(args, arg)))
    print('------------')

    with open('models/%s/args.txt' % args.name, 'w') as f:
        for arg in vars(args):
            print('- %s: %s' % (arg, getattr(args, arg)), file=f)

    joblib.dump(args, 'models/%s/args.pkl' % args.name)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    # switch to benchmark model, a little forward results fluctuation, a little fast training
    cudnn.benchmark = True
    # switch to deterministic model, more stable
    # cudnn.deterministic = True

    img_path, img_labels, num_outputs = img_path_generator(
        dataset=args.train_dataset)
    if args.pred_type == 'regression':
        num_outputs = 1

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=0)
    img_paths = []
    labels = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(img_path, img_labels)):
        img_paths.append((img_path[train_idx], img_path[val_idx]))
        labels.append((img_labels[train_idx], img_labels[val_idx]))

    train_transform = []
    train_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        # transforms.RandomAffine(
        #     degrees=(args.rotate_min, args.rotate_max) if args.rotate else 0,
        #     translate=(args.translate_min, args.translate_max) if args.translate else None,
        #     scale=(args.rescale_min, args.rescale_max) if args.rescale else None,
        #     shear=(args.shear_min, args.shear_max) if args.shear else None,
        # ),
        transforms.RandomCrop(args.input_size, padding=4),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        # transforms.ColorJitter(
        #     brightness=0,
        #     contrast=args.contrast,
        #     saturation=0,
        #     hue=0),
        RandomErase(
            prob=args.random_erase_prob if args.random_erase else 0,
            sl=args.random_erase_sl,
            sh=args.random_erase_sh,
            r=args.random_erase_r),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.CenterCrop(args.input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    if args.loss == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss().cuda()
    elif args.loss == 'FocalLoss':
        criterion = FocalLoss().cuda()
    elif args.loss == 'MSELoss':
        criterion = nn.MSELoss().cuda()
    elif args.loss == 'LabelSmoothingLoss':
        criterion = LabelSmoothingLoss(classes=num_outputs, smoothing=0.8).cuda()
    else:
        raise NotImplementedError

    folds = []
    best_losses = []
    best_ac_scores = []
    best_epochs = []

    for fold, ((train_img_paths, val_img_paths), (train_labels, val_labels)) in enumerate(zip(img_paths, labels)):
        print('Fold [%d/%d]' % (fold+1, len(img_paths)))

        # if os.path.exists('models/%s/model_%d.pth' % (args.name, fold+1)):
        #     log = pd.read_csv('models/%s/log_%d.csv' % (args.name, fold+1))
        #     best_loss, best_ac_score = log.loc[log['val_loss'].values.argmin(
        #     ), ['val_loss', 'val_score', 'val_ac_score']].values
        #     folds.append(str(fold + 1))
        #     best_losses.append(best_loss)
        #     best_ac_scores.append(best_ac_score)
        #     continue

        # train
        train_set = Dataset(
            train_img_paths,
            train_labels,
            transform=train_transform)

        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            sampler=None)

        val_set = Dataset(
            val_img_paths,
            val_labels,
            transform=val_transform)
        val_loader = torch.utils.data.DataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4)

        # create model
        if args.mode == 'baseline':
            model = get_model(model_name=args.arch, num_outputs=num_outputs,
                              freeze_bn=args.freeze_bn, dropout_p=args.dropout_p)
        elif args.mode == 'gcn':
            model_path = 'models/%s/model_%d.pth' % (
                'baseline_' + args.arch, fold + 1)
            if not os.path.exists(model_path):
                print('%s is not exists' % model_path)
                continue
            model = SoftLabelGCN(cnn_model_name=args.arch, cnn_pretrained=False, num_outputs=num_outputs)
            pretrained_dict = torch.load(model_path)
            model_dict = model.cnn.state_dict()
            pretrained_dict = {k: v for k,
                               v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            model.cnn.load_state_dict(model_dict)
            for p in model.cnn.parameters():
                p.requires_grad = False
        else:
            model = RA(cnn_model_name=args.arch, input_size=args.input_size, hidden_size=args.lstm_hidden,
                       layer_num=args.lstm_layers, recurrent_num=args.lstm_recurrence, class_num=num_outputs, pretrain=True)
            # model_path = 'models/%s/model_%d.pth' % (
            #     'baseline_' + args.arch, fold + 1)
            # if not os.path.exists(model_path):
            #     print('%s is not exists' % model_path)
            #     continue
            # model = RA(cnn_model_name=args.arch, input_size=args.input_size, hidden_size=args.lstm_hidden,
            #            layer_num=args.lstm_layers, recurrent_num=args.lstm_recurrence, class_num=num_outputs)
            # pretrained_dict = torch.load(model_path)
            # model_dict = model.cnn.state_dict()
            # pretrained_dict = {k: v for k,
            #                    v in pretrained_dict.items() if k in model_dict}
            # model_dict.update(pretrained_dict)
            # model.cnn.load_state_dict(model_dict)
            # for p in model.cnn.parameters():
            #     p.requires_grad = False

        device = torch.device('cuda')
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)
        # model = model.cuda()
        if args.pretrained_model is not None:
            model.load_state_dict(torch.load(
                'models/%s/model_%d.pth' % (args.pretrained_model, fold+1)))

        # print(model)

        if args.optimizer == 'Adam':
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        elif args.optimizer == 'AdamW':
            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        elif args.optimizer == 'RAdam':
            optimizer = RAdam(
                filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        elif args.optimizer == 'SGD':
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                                  momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
            # optimizer = optim.SGD(model.get_config_optim(args.lr, args.lr * 10, args.lr * 10), lr=args.lr,
            #                       momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)

        if args.scheduler == 'CosineAnnealingLR':
            scheduler = lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs, eta_min=args.min_lr)
        elif args.scheduler == 'ReduceLROnPlateau':
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.factor, patience=args.patience,
                                                       verbose=1, min_lr=args.min_lr)

        log = pd.DataFrame(index=[], columns=[
            'epoch', 'loss', 'ac_score', 'val_loss', 'val_ac_score',
        ])
        log = {
            'epoch': [],
            'loss': [],
            'ac_score': [],
            'val_loss': [],
            'val_ac_score': [],
        }

        best_loss = float('inf')
        best_ac_score = 0
        best_epoch = 0

        for epoch in range(args.epochs):
            print('Epoch [%d/%d]' % (epoch + 1, args.epochs))

            # train for one epoch
            train_loss, train_ac_score = train(
                args, train_loader, model, criterion, optimizer, epoch)

            # evaluate on validation set
            val_loss, val_ac_score = validate(
                args, val_loader, model, criterion)

            if args.scheduler == 'CosineAnnealingLR':
                scheduler.step()
            elif args.scheduler == 'ReduceLROnPlateau':
                scheduler.step(val_loss)

            print('loss %.4f - ac_score %.4f - val_loss %.4f - val_ac_score %.4f'
                  % (train_loss, train_ac_score, val_loss, val_ac_score))

            log['epoch'].append(epoch)
            log['loss'].append(train_loss)
            log['ac_score'].append(train_ac_score)
            log['val_loss'].append(val_loss)
            log['val_ac_score'].append(val_ac_score)

            pd.DataFrame(log).to_csv('models/%s/log_%d.csv' %
                                     (args.name, fold+1), index=False)

            if val_ac_score > best_ac_score:
                if args.mode == 'baseline':
                    torch.save(model.state_dict(), 'models/%s/model_%d.pth' %
                               (args.name, fold+1))
                best_loss = val_loss
                best_ac_score = val_ac_score
                best_epoch = epoch
                print("=> saved best model")

        print('val_loss:  %f' % best_loss)
        print('val_ac_score: %f' % best_ac_score)

        folds.append(str(fold + 1))
        best_losses.append(best_loss)
        best_ac_scores.append(best_ac_score)
        best_epochs.append(best_epoch)

        results = pd.DataFrame({
            'fold': folds + ['mean'],
            'best_loss': best_losses + [np.mean(best_losses)],
            'best_ac_score': best_ac_scores + [np.mean(best_ac_scores)],
            'best_epoch': best_epochs + [''],
        })

        print(results)
        results.to_csv('models/%s/results.csv' % args.name, index=False)
        torch.cuda.empty_cache()

        if not args.cv:
            break


if __name__ == '__main__':
    main()
