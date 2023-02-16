import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from typing import Optional
from tqdm import tqdm
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader, dataset
from torch.nn import functional as F
from models import LoadModel, Classifier, Discriminator, CalculateOutSize
from utils.pytorch_utils import init_weights, weight_for_balanced_classes


def adjust_learning_rate(optimizer: nn.Module, epoch: int, args):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 50:
        lr = args.lr * 0.1
    if epoch >= 100:
        lr = args.lr * 0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def PGD(feature_ext: nn.Module, classifier: nn.Module,
        discriminator: nn.Module, x: torch.Tensor, y: torch.Tensor, eps: float,
        alpha: float, steps: int, args):
    """ PGD attack """
    device = next(feature_ext.parameters()).device
    criterion_cal = nn.CrossEntropyLoss().to(device)
    criterion_prob = nn.MSELoss().to(device)

    data_loader = DataLoader(dataset=TensorDataset(x, y),
                             batch_size=128,
                             shuffle=False,
                             drop_last=False)

    feature_ext.eval()
    classifier.eval()
    discriminator.eval()
    for step, (batch_x, batch_y) in enumerate(data_loader):
        batch_x = batch_x.clone().detach().to(device)
        batch_y = batch_y.clone().detach().to(device)

        # craft adversarial examples
        batch_adv_x = batch_x.clone().detach() + torch.empty_like(
            batch_x).uniform_(-eps, eps)
        for _ in range(steps):
            batch_adv_x.requires_grad = True
            with torch.enable_grad():
                # loss = criterion(discriminator(feature_ext(batch_adv_x)),
                #                  batch_y)
                loss1 = criterion_cal(discriminator(feature_ext(batch_adv_x)),
                                      batch_y)
                loss2 = criterion_prob(classifier(feature_ext(batch_adv_x)),
                                       classifier(feature_ext(batch_x)))
                loss = args.alpha * loss1 + loss2
            grad = torch.autograd.grad(loss,
                                       batch_adv_x,
                                       retain_graph=False,
                                       create_graph=False)[0]

            batch_adv_x = batch_adv_x.detach() - alpha * grad.detach().sign()

            # projection
            delta = torch.clamp(batch_adv_x - batch_x, min=-eps, max=eps)

            batch_adv_x = (batch_x + delta).detach()

        if step == 0: adv_x = batch_adv_x
        else: adv_x = torch.cat([adv_x, batch_adv_x], dim=0)

    return adv_x.cpu()


def unlearnable(x, y_label, s_label, args):
    chans, samples = x.shape[2], x.shape[3]
    feature_ext = LoadModel(model_name=args.feature_d,
                            Chans=chans,
                            Samples=samples)
    feature_ext.to(args.device)
    classifier = Classifier(
        input_dim=CalculateOutSize(feature_ext, chans, samples),
        n_classes=len(np.unique(y_label.numpy()))).to(args.device)
    discriminator = Discriminator(
        input_dim=CalculateOutSize(feature_ext, chans, samples),
        n_subjects=len(np.unique(s_label.numpy()))).to(args.device)

    feature_ext.apply(init_weights)
    classifier.apply(init_weights)
    discriminator.apply(init_weights)

    sample_weights = weight_for_balanced_classes(y_label)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(
        sample_weights, len(sample_weights))
    data_loader = DataLoader(dataset=TensorDataset(x, y_label, s_label),
                             batch_size=args.batch_size,
                             sampler=sampler,
                             drop_last=False)

    params = []
    for _, v in feature_ext.named_parameters():
        params += [{'params': v, 'lr': args.lr}]
    for _, v in classifier.named_parameters():
        params += [{'params': v, 'lr': args.lr}]
    for _, v in discriminator.named_parameters():
        params += [{'params': v, 'lr': args.lr}]
    optimizer = optim.Adam(params, weight_decay=5e-4)
    # optimizer = optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss().to(args.device)

    for epoch in tqdm(range(150)):
        adjust_learning_rate(optimizer=optimizer, epoch=epoch + 1, args=args)
        feature_ext.train()
        classifier.train()
        discriminator.train()
        for step, (batch_x, batch_y, batch_s) in enumerate(data_loader):
            batch_x, batch_y, batch_s = batch_x.to(args.device), batch_y.to(
                args.device), batch_s.to(args.device)
            optimizer.zero_grad()
            feature = feature_ext(batch_x)
            y_pred = classifier(feature)
            s_pred = discriminator(feature)

            loss1 = criterion(y_pred, batch_y)
            loss2 = criterion(s_pred, batch_s)
            loss = loss1 + 0.1 * loss2
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 5 == 0:
            x = PGD(feature_ext,
                    classifier,
                    discriminator,
                    x,
                    s_label,
                    eps=1e-2,
                    alpha=2e-3,
                    steps=5,
                    args=args)
            # x = gen_optim(feature_ext, classifier, discriminator, x, y_label, s_label, args)
            data_loader = DataLoader(dataset=TensorDataset(
                x, y_label, s_label),
                                     batch_size=args.batch_size,
                                     shuffle=True,
                                     drop_last=False)

    return x


def unlearnable_one(x, y_label, s_label, uid_list, args):
    chans, samples = x.shape[2], x.shape[3]
    feature_ext = LoadModel(model_name=args.feature_d,
                            Chans=chans,
                            Samples=samples)
    feature_ext.to(args.device)
    classifier = Classifier(
        input_dim=CalculateOutSize(feature_ext, chans, samples),
        n_classes=len(np.unique(y_label.numpy()))).to(args.device)
    discriminator = Discriminator(
        input_dim=CalculateOutSize(feature_ext, chans, samples),
        n_subjects=len(np.unique(s_label.numpy()))).to(args.device)

    feature_ext.apply(init_weights)
    classifier.apply(init_weights)
    discriminator.apply(init_weights)

    sample_weights = weight_for_balanced_classes(y_label)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(
        sample_weights, len(sample_weights))
    data_loader = DataLoader(dataset=TensorDataset(x, y_label, s_label),
                             batch_size=args.batch_size,
                             sampler=sampler,
                             drop_last=False)

    params = []
    for _, v in feature_ext.named_parameters():
        params += [{'params': v, 'lr': args.lr}]
    for _, v in classifier.named_parameters():
        params += [{'params': v, 'lr': args.lr}]
    for _, v in discriminator.named_parameters():
        params += [{'params': v, 'lr': args.lr}]
    optimizer = optim.Adam(params, weight_decay=5e-4)
    # optimizer = optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss().to(args.device)

    for epoch in tqdm(range(150)):
        adjust_learning_rate(optimizer=optimizer, epoch=epoch + 1, args=args)
        feature_ext.train()
        classifier.train()
        discriminator.train()
        for step, (batch_x, batch_y, batch_s) in enumerate(data_loader):
            batch_x, batch_y, batch_s = batch_x.to(args.device), batch_y.to(
                args.device), batch_s.to(args.device)
            optimizer.zero_grad()
            feature = feature_ext(batch_x)
            y_pred = classifier(feature)
            s_pred = discriminator(feature)

            loss1 = criterion(y_pred, batch_y)
            loss2 = criterion(s_pred, batch_s)
            loss = loss1 + 0.1 * loss2
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 5 == 0:
            u_x = PGD(feature_ext,
                      classifier,
                      discriminator,
                      x,
                      s_label,
                      eps=1e-2,
                      alpha=2e-3,
                      steps=5,
                      args=args)

            for uid in uid_list:
                x[s_label == uid] = u_x[s_label == uid]
            data_loader = DataLoader(dataset=TensorDataset(
                x, y_label, s_label),
                                     batch_size=args.batch_size,
                                     shuffle=True,
                                     drop_last=False)

    return x


def unlearnable_optim(x, y_label, s_label, args):
    chans, samples = x.shape[2], x.shape[3]
    feature_ext = LoadModel(model_name=args.feature_d,
                            Chans=chans,
                            Samples=samples)
    feature_ext.to(args.device)
    classifier = Classifier(
        input_dim=CalculateOutSize(feature_ext, chans, samples),
        n_classes=len(np.unique(y_label.numpy()))).to(args.device)
    discriminator = Discriminator(
        input_dim=CalculateOutSize(feature_ext, chans, samples),
        n_subjects=len(np.unique(s_label.numpy()))).to(args.device)

    feature_ext.apply(init_weights)
    classifier.apply(init_weights)
    discriminator.apply(init_weights)

    sample_weights = weight_for_balanced_classes(y_label)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(
        sample_weights, len(sample_weights))
    data_loader = DataLoader(dataset=TensorDataset(x, y_label, s_label),
                             batch_size=args.batch_size,
                             sampler=sampler,
                             drop_last=False)

    params = []
    for _, v in feature_ext.named_parameters():
        params += [{'params': v, 'lr': args.lr}]
    for _, v in classifier.named_parameters():
        params += [{'params': v, 'lr': args.lr}]
    for _, v in discriminator.named_parameters():
        params += [{'params': v, 'lr': args.lr}]
    optimizer = optim.Adam(params, weight_decay=5e-4)
    # optimizer = optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss().to(args.device)

    print('train sub model')
    for epoch in tqdm(range(150)):
        adjust_learning_rate(optimizer=optimizer, epoch=epoch + 1, args=args)
        feature_ext.train()
        classifier.train()
        discriminator.train()
        for step, (batch_x, batch_y, batch_s) in enumerate(data_loader):
            batch_x, batch_y, batch_s = batch_x.to(args.device), batch_y.to(
                args.device), batch_s.to(args.device)
            optimizer.zero_grad()
            feature = feature_ext(batch_x)
            y_pred = classifier(feature)
            s_pred = discriminator(feature)

            loss1 = criterion(y_pred, batch_y)
            loss2 = criterion(s_pred, batch_s)
            loss = loss1 + 0.1 * loss2
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            feature_ext.eval()
            classifier.eval()
            discriminator.eval()
            train_loss, train_acc = eval(feature_ext, classifier, criterion, x,
                                         y_label)
            test_loss, test_acc = eval(feature_ext, discriminator, criterion,
                                       x, s_label)

            print(
                'Epoch {}/{}: task loss: {:.4f} task acc: {:.2f} | subject loss: {:.4f} subject acc: {:.2f}'
                .format(epoch + 1, args.epochs, train_loss, train_acc,
                        test_loss, test_acc))

    print('gen unlearnable examples')
    # gen unlearnable examples
    feature_ext.eval()
    classifier.eval()
    discriminator.eval()

    perturbation = torch.zeros(
        size=[len(torch.unique(s_label)), 1, chans, samples]).to(args.device)
    perturbation = Variable(perturbation, requires_grad=True)
    nn.init.normal_(perturbation, mean=0, std=1e-3)
    optimizer = optim.Adam([perturbation], lr=1e-3)

    criterion_acc = nn.MSELoss().to(args.device)

    for epoch in tqdm(range(100)):
        for step, (batch_x, batch_y, batch_s) in enumerate(data_loader):
            batch_x, batch_y, batch_s = batch_x.to(args.device), batch_y.to(
                args.device), batch_s.to(args.device)
            optimizer.zero_grad()

            perb_batch_x = batch_x + perturbation[batch_s.squeeze()]
            feature = feature_ext(perb_batch_x)
            y_pred = classifier(feature)
            s_pred = discriminator(feature)

            loss1 = criterion_acc(y_pred, classifier(feature_ext(batch_x)))
            loss2 = criterion(s_pred, batch_s)
            loss3 = torch.sum(torch.pow(perturbation, 2))
            loss = loss1 + args.alpha * loss2 + (1e-6 / args.alpha) * loss3
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            feature_ext.eval()
            classifier.eval()
            discriminator.eval()
            train_loss, train_acc = eval(
                feature_ext, classifier, criterion,
                (x + perturbation[s_label.squeeze()].cpu()).detach(), y_label)
            test_loss, test_acc = eval(
                feature_ext, discriminator, criterion,
                (x + perturbation[s_label.squeeze()].cpu()).detach(), s_label)

            print(
                'Epoch {}/{}: task loss: {:.4f} task acc: {:.2f} | subject loss: {:.4f} subject acc: {:.2f} | L2 loss: {:.2f}'
                .format(epoch + 1, args.epochs, train_loss, train_acc,
                        test_loss, test_acc, loss3.item()))

    return (x + perturbation[s_label.squeeze()].cpu()).detach()


def unlearnable_optim_one(x, y_label, s_label, uid_list, args):
    chans, samples = x.shape[2], x.shape[3]
    feature_ext = LoadModel(model_name=args.feature_d,
                            Chans=chans,
                            Samples=samples)
    feature_ext.to(args.device)
    classifier = Classifier(
        input_dim=CalculateOutSize(feature_ext, chans, samples),
        n_classes=len(np.unique(y_label.numpy()))).to(args.device)
    discriminator = Discriminator(
        input_dim=CalculateOutSize(feature_ext, chans, samples),
        n_subjects=len(np.unique(s_label.numpy()))).to(args.device)

    feature_ext.apply(init_weights)
    classifier.apply(init_weights)
    discriminator.apply(init_weights)

    sample_weights = weight_for_balanced_classes(y_label)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(
        sample_weights, len(sample_weights))
    data_loader = DataLoader(dataset=TensorDataset(x, y_label, s_label),
                             batch_size=args.batch_size,
                             sampler=sampler,
                             drop_last=False)

    params = []
    for _, v in feature_ext.named_parameters():
        params += [{'params': v, 'lr': args.lr}]
    for _, v in classifier.named_parameters():
        params += [{'params': v, 'lr': args.lr}]
    for _, v in discriminator.named_parameters():
        params += [{'params': v, 'lr': args.lr}]
    optimizer = optim.Adam(params, weight_decay=5e-4)
    # optimizer = optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss().to(args.device)

    print('train sub model')
    for epoch in tqdm(range(150)):
        adjust_learning_rate(optimizer=optimizer, epoch=epoch + 1, args=args)
        feature_ext.train()
        classifier.train()
        discriminator.train()
        for step, (batch_x, batch_y, batch_s) in enumerate(data_loader):
            batch_x, batch_y, batch_s = batch_x.to(args.device), batch_y.to(
                args.device), batch_s.to(args.device)
            optimizer.zero_grad()
            feature = feature_ext(batch_x)
            y_pred = classifier(feature)
            s_pred = discriminator(feature)

            loss1 = criterion(y_pred, batch_y)
            loss2 = criterion(s_pred, batch_s)
            loss = loss1 + 0.1 * loss2
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            feature_ext.eval()
            classifier.eval()
            discriminator.eval()
            train_loss, train_acc = eval(feature_ext, classifier, criterion, x,
                                         y_label)
            test_loss, test_acc = eval(feature_ext, discriminator, criterion,
                                       x, s_label)

            print(
                'Epoch {}/{}: task loss: {:.4f} task acc: {:.2f} | subject loss: {:.4f} subject acc: {:.2f}'
                .format(epoch + 1, args.epochs, train_loss, train_acc,
                        test_loss, test_acc))

    print('gen unlearnable examples')
    # gen unlearnable examples
    feature_ext.eval()
    classifier.eval()
    discriminator.eval()

    perturbation = torch.zeros(size=[len(torch.unique(s_label)), 1, chans, samples]).to(args.device)
    perturbation = Variable(perturbation, requires_grad=True)
    nn.init.normal_(perturbation, mean=0, std=1e-3)
    optimizer = optim.Adam([perturbation], lr=5e-3)

    criterion_acc = nn.MSELoss().to(args.device)

    uid_idx = torch.zeros(size=[len(s_label)]).bool()
    for uid in uid_list:
        idx = (s_label == uid)
        uid_idx = uid_idx | idx

    sample_weights = weight_for_balanced_classes(y_label[uid_idx])
    sampler = torch.utils.data.sampler.WeightedRandomSampler(
        sample_weights, len(sample_weights))
    data_loader = DataLoader(dataset=TensorDataset(x[uid_idx], y_label[uid_idx],s_label[uid_idx]),
                             batch_size=args.batch_size,
                             sampler=sampler,
                             drop_last=False)

    for epoch in tqdm(range(100)):
        for step, (batch_x, batch_y, batch_s) in enumerate(data_loader):
            batch_x, batch_y, batch_s = batch_x.to(args.device), batch_y.to(
                args.device), batch_s.to(args.device)
            optimizer.zero_grad()

            perb_batch_x = batch_x + perturbation[batch_s.squeeze()]
            feature = feature_ext(perb_batch_x)
            y_pred = classifier(feature)
            s_pred = discriminator(feature)

            loss1 = criterion_acc(y_pred, classifier(feature_ext(batch_x)))
            loss2 = criterion(s_pred, batch_s)
            loss3 = torch.sum(torch.pow(perturbation, 2))
            loss = loss1 + args.alpha * loss2 + (1e-6 / args.alpha) * loss3
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            feature_ext.eval()
            classifier.eval()
            discriminator.eval()
            train_loss, train_acc = eval(
                feature_ext, classifier, criterion,
                (x + perturbation[s_label.squeeze()].cpu()).detach(), y_label)
            test_loss, test_acc = eval(
                feature_ext, discriminator, criterion,
                (x + perturbation[s_label.squeeze()].cpu()).detach(), s_label)

            print(
                'Epoch {}/{}: task loss: {:.4f} task acc: {:.2f} | subject loss: {:.4f} subject acc: {:.2f} | L2 loss: {:.2f}'
                .format(epoch + 1, args.epochs, train_loss, train_acc,
                        test_loss, test_acc, loss3.item()))

    return (x + perturbation[s_label.squeeze()].cpu()).detach()


def gen_optim(feature_ext, classifier, discriminator, x, y_label, s_label,
              args):
    feature_ext.eval()
    classifier.eval()
    discriminator.eval()

    data_loader = DataLoader(dataset=TensorDataset(x, y_label, s_label),
                             batch_size=args.batch_size,
                             shuffle=True,
                             drop_last=False)

    perturbation = torch.zeros(
        size=[len(torch.unique(s_label)), 1, x.shape[-2], x.shape[-1]]).to(
            args.device)
    perturbation = Variable(perturbation, requires_grad=True)
    nn.init.normal_(perturbation, mean=0, std=1e-3)
    optimizer = optim.Adam([perturbation], lr=1e-4)
    criterion = nn.CrossEntropyLoss().to(args.device)
    criterion_acc = nn.MSELoss().to(args.device)

    for epoch in tqdm(range(100)):
        for step, (batch_x, batch_y, batch_s) in enumerate(data_loader):
            batch_x, batch_y, batch_s = batch_x.to(args.device), batch_y.to(
                args.device), batch_s.to(args.device)
            optimizer.zero_grad()

            perb_batch_x = batch_x + perturbation[batch_s.squeeze()]
            feature = feature_ext(perb_batch_x)
            y_pred = classifier(feature)
            s_pred = discriminator(feature)

            loss1 = criterion_acc(y_pred, classifier(feature_ext(batch_x)))
            loss2 = criterion(s_pred, batch_s)
            loss3 = torch.sum(torch.pow(perturbation, 2))
            loss = loss1 + 0.5 * loss2 + 1e-3 * loss3
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            feature_ext.eval()
            classifier.eval()
            discriminator.eval()
            train_loss, train_acc = eval(
                feature_ext, classifier, criterion,
                (x + perturbation[s_label.squeeze()].cpu()).detach(), y_label)
            test_loss, test_acc = eval(
                feature_ext, discriminator, criterion,
                (x + perturbation[s_label.squeeze()].cpu()).detach(), s_label)

            print(
                'Epoch {}/{}: task loss: {:.4f} task acc: {:.2f} | subject loss: {:.4f} subject acc: {:.2f} | L2 loss: {:.2f}'
                .format(epoch + 1, args.epochs, train_loss, train_acc,
                        test_loss, test_acc, loss3.item()))

    return (x + perturbation[s_label.squeeze()].cpu()).detach()


def eval(model1: nn.Module, model2: nn.Module, criterion: nn.Module,
         x_test: torch.Tensor, y_test: torch.Tensor):
    device = next(model1.parameters()).device
    data_loader = DataLoader(dataset=TensorDataset(x_test, y_test),
                             batch_size=128,
                             shuffle=False,
                             drop_last=False)

    loss, correct = 0., 0
    labels, preds = [], []
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            out = model2(model1(x))
            pred = nn.Softmax(dim=1)(out).cpu().argmax(dim=1)
            loss += criterion(out, y).item()
            correct += pred.eq(y.cpu().view_as(pred)).sum().item()
            labels.extend(y.cpu().tolist())
            preds.extend(pred.tolist())
    loss /= len(data_loader.dataset)
    acc = correct / len(data_loader.dataset)

    return loss, acc