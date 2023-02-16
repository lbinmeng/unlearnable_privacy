import os
import logging
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
from models import LoadModel, Classifier, Discriminator, CalculateOutSize
from utils.data_loader import MI109Load
from utils.pytorch_utils import init_weights, print_args, seed, weight_for_balanced_classes
from unlearnable_gen import unlearnable, unlearnable_optim


def trainer(feature_ext, model, x_train, y_train, x_test_list, y_test_list, phase, args):
    feature_ext.apply(init_weights)
    model.apply(init_weights)
    params = []
    for _, v in feature_ext.named_parameters():
            params += [{'params': v, 'lr': args.lr}]
    for _, v in model.named_parameters():
        params += [{'params': v, 'lr': args.lr}]
    optimizer = optim.Adam(params, weight_decay=5e-4)
    # optimizer = optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss().to(args.device)

    # data loader
    # for unbalanced dataset, create a weighted sampler
    sample_weights = weight_for_balanced_classes(y_train)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(
        sample_weights, len(sample_weights))
    train_loader = DataLoader(dataset=TensorDataset(x_train, y_train),
                              batch_size=args.batch_size,
                              sampler=sampler,
                              drop_last=False)

    train_recorder, test_recorder = [], []
    for epoch in range(args.epochs):
        # model training
        adjust_learning_rate(optimizer=optimizer, epoch=epoch + 1, args=args)
        feature_ext.train()
        model.train()
        for step, (batch_x, batch_y) in enumerate(train_loader):
            batch_x, batch_y = batch_x.to(args.device), batch_y.to(args.device)
            optimizer.zero_grad()
            out = model(feature_ext(batch_x))
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            feature_ext.MaxNormConstraint()
            if phase == 'c': model.MaxNormConstraint()

        if (epoch + 1) % 1 == 0:
            feature_ext.eval()
            model.eval()
            train_loss, train_acc, train_bca = eval(feature_ext, model,
                                                    criterion, train_loader)
            test_loss_list, test_acc_list, test_bca_list = [], [], []
            for session in range(len(x_test_list)):
                test_loader = DataLoader(dataset=TensorDataset(x_test_list[session], y_test_list[session]),
                             batch_size=args.batch_size,
                             shuffle=True,
                             drop_last=False)
                test_loss, test_acc, test_bca = eval(feature_ext, model, criterion,
                                                     test_loader)
                test_loss_list.append(test_loss)
                test_acc_list.append(test_acc)
                test_bca_list.append(test_bca)

            logging.info(
                'Epoch {}/{}: train loss: {:.4f} train acc: {:.2f} | test loss: {:.4f} test acc: {:.2f}'
                .format(epoch + 1, args.epochs, train_loss, train_acc,
                        np.mean(test_loss_list), np.mean(test_acc_list)))

            if phase == 'c':
                train_recorder.append(train_bca)
                test_recorder.append(test_bca_list)
            else:
                train_recorder.append(train_acc)
                test_recorder.append(test_acc_list)
    return feature_ext, model, train_recorder, test_recorder


def train(x_session_old, y_session_old, s_session_old, x_session_new, y_session_new, s_session_new,
          u_x_session, session, args):
    # initialize the model
    chans, samples = x_session_new[session].shape[2], x_session_new[session].shape[3]
    if x_session_old == None:
        n_subjects = len(np.unique(s_session_new[session].numpy()))
    else:
        n_subjects = len(np.unique(s_session_old[session].numpy())) + len(
            np.unique(s_session_new[session].numpy()))

    feature_ext = LoadModel(model_name=args.feature_c,
                            Chans=chans,
                            Samples=samples)

    discriminator = Discriminator(input_dim=CalculateOutSize(
        feature_ext, chans, samples),n_subjects=n_subjects)

    feature_ext.to(args.device)
    discriminator.to(args.device)

    criterion = nn.CrossEntropyLoss().to(args.device)

    # test loader
    x_test_list, s_test_list = [], []
    for i in x_session_new.keys():
        if x_session_old == None:
            if i == session:
                x_train = x_session_new[i]
                s_train = s_session_new[i]
            else:
                x_test_list.append(x_session_new[i])
                s_test_list.append(s_session_new[i])
        else:
            if i == session:
                x_train = torch.cat([x_session_old[i], x_session_new[i]])
                s_train = torch.cat([s_session_old[i], s_session_new[i]])
            else:
                x_test_list.append(torch.cat([x_session_old[i], x_session_new[i]]))
                s_test_list.append(torch.cat([s_session_old[i], s_session_new[i]]))


    logging.info('*' * 25 + ' train discriminator ' + '*' * 25)
    feature_ext, discriminator, _, _ = trainer(feature_ext,
                                         discriminator,
                                         x_train,
                                         s_train,
                                         x_test_list,
                                         s_test_list,
                                         phase='d',
                                         args=args)
    feature_ext.eval()
    discriminator.eval()

    sda_list, new_sda_list = [], []
    for i in range(len(x_test_list)):
        dis_loader = DataLoader(dataset=TensorDataset(x_test_list[i], s_test_list[i]),
                        batch_size=args.batch_size,
                        shuffle=False,
                        drop_last=False)
        _, s_acc, new_s_acc = eval(feature_ext, discriminator, criterion, dis_loader)
        sda_list.append(s_acc)
        new_sda_list.append(new_s_acc)

    logging.info(f'Session: {session} | sda: {sda_list}, new sda: {new_sda_list}')
    logging.info(f'Mean sda: {np.mean(sda_list)}, mean new sda: {np.mean(new_sda_list)}')

    # unlearnable
    u_x_test_list = []
    for i in x_session_new.keys():
        if x_session_old == None:
            x_temp, y_temp, s_temp = x_session_new[i], y_session_new[i], s_session_new[i]
        else:
            x_temp = torch.cat([u_x_session[i], x_session_new[i]])
            y_temp = torch.cat([y_session_old[i], y_session_new[i]])
            s_temp = torch.cat([s_session_old[i], s_session_new[i]])
        
        if args.subject_wise:
            u_x_new = unlearnable_optim(x_temp, y_temp, s_temp, args)
        else:
            u_x_new = unlearnable(x_temp, y_temp, s_temp, args)

        if x_session_old == None:
            u_x_session[i] = u_x_new
        else:
            u_x_session[i] = torch.cat([u_x_session[i], u_x_new[len(u_x_session[i]):]])

        if i == session:
            u_x_train = u_x_session[i]
        else:
            u_x_test_list.append(u_x_session[i])

    logging.info('*' * 25 + ' train discriminator (unlearnable) ' + '*' * 25)
    feature_ext, discriminator, _, _ = trainer(feature_ext,
                                         discriminator,
                                         u_x_train,
                                         s_train,
                                         u_x_test_list,
                                         s_test_list,
                                         phase='d',
                                         args=args)
    feature_ext.eval()
    discriminator.eval()

    u_sda_list, new_u_sda_list = [], []
    for i in range(len(x_test_list)):
        dis_loader = DataLoader(dataset=TensorDataset(u_x_test_list[i], s_test_list[i]),
                        batch_size=args.batch_size,
                        shuffle=False,
                        drop_last=False)
        _, s_acc, new_s_acc = eval(feature_ext, discriminator, criterion, dis_loader)
        u_sda_list.append(s_acc)
        new_u_sda_list.append(new_s_acc)

    logging.info(f'Session: {session} | unlearnalbe sda: {u_sda_list}, new unlearnalbe sda: {new_u_sda_list}')
    logging.info(f'Mean unlearnalbe sda: {np.mean(u_sda_list)}, mean new unlearnalbe sda: {np.mean(new_u_sda_list)}')


    if x_session_old == None:
        x_session_old, y_session_old, s_session_old = x_session_new, y_session_new, s_session_new
    else:
        for i in x_session_new.keys():
            x_session_old[i] = torch.cat([x_session_old[i], x_session_new[i]])
            y_session_old[i] = torch.cat([y_session_old[i], y_session_new[i]])
            s_session_old[i] = torch.cat([s_session_old[i], s_session_new[i]])

    return x_session_old, y_session_old, s_session_old, u_x_session, sda_list, new_sda_list, u_sda_list, new_u_sda_list



def eval(model1: nn.Module, model2: nn.Module, criterion: nn.Module,
         data_loader: DataLoader):
    loss, correct = 0., 0
    labels, preds = [], []
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(args.device), y.to(args.device)
            out = model2(model1(x))
            pred = nn.Softmax(dim=1)(out).cpu().argmax(dim=1)
            loss += criterion(out, y).item()
            correct += pred.eq(y.cpu().view_as(pred)).sum().item()
            labels.extend(y.cpu().tolist())
            preds.extend(pred.tolist())
    loss /= len(data_loader.dataset)
    acc = correct / len(data_loader.dataset)

    max_label = max(labels)
    new_idx = [x for x in range(len(labels)) if labels[x] > max_label - 10]
    new_acc = len([x for x in new_idx if labels[x] == preds[x]]) / len(new_idx)

    return loss, acc, new_acc


def adjust_learning_rate(optimizer: nn.Module, epoch: int, args):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 50:
        lr = args.lr * 0.1
    if epoch >= 100:
        lr = args.lr * 0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model train')
    parser.add_argument('--gpu_id', type=str, default='4')

    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--optim', type=str, default='Adam')
    parser.add_argument('--lr', type=float, default=0.01)

    parser.add_argument('--feature_c', type=str, default='EEGNet')
    parser.add_argument('--feature_d', type=str, default='EEGNet')
    parser.add_argument('--subject_wise', type=bool, default=False)
    parser.add_argument('--alpha', type=float,
                        default=0.03)  # alpha MI: 0.1 ERN: 1 EPFL: 1.5
    parser.add_argument('--log', type=str, default='')

    args = parser.parse_args()

    args.device = f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu'

    subject_num_dict = {
        'MI4C': 9,
        'ERN': 16,
        'EPFL': 8,
        'NICU': 14,
        'MI109': 109
    }

    log_path = f'results_continue_new/log/'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_name = os.path.join(log_path, f'{args.feature_c}_{args.feature_d}.log')

    npz_path = f'results_continue_new/npz/{args.feature_c}_{args.feature_d}'

    if args.subject_wise:
        log_name = log_name.replace('.log', '_subject.log')
        npz_path = npz_path + '/subject'

    if not os.path.exists(npz_path):
        os.makedirs(npz_path)

    if len(args.log): log_name = log_name.replace('.log', f'_{args.log}.log')

    # logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # print logging
    print_log = logging.StreamHandler()
    logger.addHandler(print_log)
    # save logging
    save_log = logging.FileHandler(log_name, mode='w', encoding='utf8')
    logger.addHandler(save_log)

    logging.info(print_args(args) + '\n')

    x_session, y_session, s_session = MI109Load(isEA=False)

    # model train
    for repeat in range(10):
        seed(repeat)
        r_s_acc, r_new_s_acc, r_u_s_acc, r_new_u_s_acc = [], [], [], []
        for session in x_session.keys():
            x_session_old, y_session_old, s_session_old = None, None, None
            u_x_session = {}
            s_acc_list, new_s_acc_list, u_s_acc_list, new_u_s_acc_list = [], [], [], []
            for id in range(11):
                x_session_new, y_session_new, s_session_new = {}, {}, {}
                for s in x_session.keys():
                    idx = (s_session[s] >= (id*10)) & (s_session[s] < (id*10 + 10))
                    x_session_new[s] = Variable(torch.from_numpy(x_session[s][idx]).type(torch.FloatTensor))
                    y_session_new[s] = Variable(torch.from_numpy(y_session[s][idx]).type(torch.LongTensor))
                    s_session_new[s] = Variable(torch.from_numpy(s_session[s][idx]).type(torch.LongTensor))
                
                x_session_old, y_session_old, s_session_old, u_x_session, s_acc, new_s_acc, u_s_acc, new_u_s_acc = train(
                    x_session_old, y_session_old, s_session_old, x_session_new, y_session_new, s_session_new, u_x_session, session, args)

                s_acc_list.append(s_acc)
                new_s_acc_list.append(new_s_acc)
                u_s_acc_list.append(u_s_acc)
                new_u_s_acc_list.append(new_u_s_acc)

            r_s_acc.append(s_acc_list)
            r_new_s_acc.append(new_s_acc_list)
            r_u_s_acc.append(u_s_acc_list)
            r_new_u_s_acc.append(new_u_s_acc_list)

        logging.info(f'Mean dis acc: {np.mean(r_s_acc, axis=(0, 2))}')
        logging.info(f'Mean new dis acc: {np.mean(r_new_s_acc, axis=(0, 2))}')
        logging.info(f'Mean unlearnable dis acc: {np.mean(r_u_s_acc, axis=(0, 2))}')
        logging.info(f'Mean new unlearnable dis acc: {np.mean(r_new_u_s_acc, axis=(0, 2))}')
        np.savez(npz_path + f'/result_{repeat}.npz',
                s_acc=r_s_acc,
                new_s_acc=r_new_s_acc,
                u_s_acc=r_u_s_acc,
                new_u_s_acc=r_new_u_s_acc)