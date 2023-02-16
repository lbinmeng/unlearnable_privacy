import os
import logging
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader, dataset
from models import LoadModel, Classifier, Discriminator, CalculateOutSize
from utils.data_loader import EPFLLoad, MI4CLoad, ERNLoad, NICULoad, MI109Load
from utils.pytorch_utils import init_weights, print_args, seed, weight_for_balanced_classes, bca_score
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
                             shuffle=False,
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


def train(x_session, y_session, s_session, npz_path: str, repeat: int, session: int, args):
    x_test_list, y_test_list, s_test_list = [], [], []
    for i in x_session.keys():
        if i == session:
            x_train = Variable(torch.from_numpy(x_session[i]).type(torch.FloatTensor))
            y_train = Variable(torch.from_numpy(y_session[i]).type(torch.LongTensor))
            s_train = Variable(torch.from_numpy(s_session[i]).type(torch.LongTensor))
        else:
            x_test_list.append(Variable(torch.from_numpy(x_session[i]).type(torch.FloatTensor)))
            y_test_list.append(Variable(torch.from_numpy(y_session[i]).type(torch.LongTensor)))
            s_test_list.append(Variable(torch.from_numpy(s_session[i]).type(torch.LongTensor)))

    # initialize the model
    chans, samples = x_train.shape[2], x_train.shape[3]
    feature_ext = LoadModel(model_name=args.feature_c,
                            Chans=chans,
                            Samples=samples)

    classifier = Classifier(input_dim=CalculateOutSize(feature_ext, chans,
                                                       samples),
                            n_classes=len(np.unique(y_train.numpy())))
    discriminator = Discriminator(input_dim=CalculateOutSize(
        feature_ext, chans, samples),
                                  n_subjects=len(np.unique(s_train.numpy())))

    feature_ext.to(args.device)
    classifier.to(args.device)
    discriminator.to(args.device)

    criterion = nn.CrossEntropyLoss().to(args.device)

    # classification
    logging.info('*' * 25 + ' train classifier ' + '*' * 25)
    feature_ext, classifier, c_train_bca, c_test_bca = trainer(
        feature_ext,
        classifier,
        x_train,
        y_train,
        x_test_list,
        y_test_list,
        phase='c',
        args=args)

    feature_ext.eval()
    classifier.eval()

    bca_list = []
    for i in range(len(x_test_list)):
        cal_loader = DataLoader(dataset=TensorDataset(x_test_list[i], y_test_list[i]),
                        batch_size=args.batch_size,
                        shuffle=False,
                        drop_last=False)

        _, cal_acc, cal_bca = eval(feature_ext, classifier, criterion, cal_loader)
        bca_list.append(cal_bca)

    # subject discriminate
    logging.info('*' * 25 + ' train discriminator ' + '*' * 25)
    feature_ext, discriminator, d_train_acc, d_test_acc = trainer(
        feature_ext,
        discriminator,
        x_train,
        s_train,
        x_test_list,
        s_test_list,
        phase='d',
        args=args)

    # baseline test 
    feature_ext.eval()
    discriminator.eval()

    sda_list = []
    for i in range(len(x_test_list)):
        dis_loader = DataLoader(dataset=TensorDataset(x_test_list[i], s_test_list[i]),
                        batch_size=args.batch_size,
                        shuffle=False,
                        drop_last=False)
        _, s_acc, s_bca = eval(feature_ext, discriminator, criterion, dis_loader)
        sda_list.append(s_acc)

    logging.info(f'Session: {session} | bca: {bca_list}, sda: {sda_list}')
    logging.info(f'Mean bca: {np.mean(bca_list)}, sda: {np.mean(sda_list)}')

    # unlearnable
    if args.subject_wise:
        u_x_train = unlearnable_optim(x_train, y_train, s_train, args)
    else:
        u_x_train = unlearnable(x_train, y_train, s_train, args)

    # session unlearnable
    u_x_test_list = []
    for i in range(len(x_test_list)):
        if args.subject_wise:
            u_x_test = unlearnable_optim(x_test_list[i], y_test_list[i], s_test_list[i], args)
        else:
            u_x_test = unlearnable(x_test_list[i], y_test_list[i], s_test_list[i], args)
        u_x_test_list.append(u_x_test)

    # classification
    logging.info('*' * 25 + ' train classifier (unlearnable) ' + '*' * 25)
    feature_ext, classifier, u_c_train_bca, u_c_test_bca = trainer(
        feature_ext,
        classifier,
        u_x_train,
        y_train,
        u_x_test_list,
        y_test_list,
        phase='c',
        args=args)

    feature_ext.eval()
    classifier.eval()

    u_bca_list, u_u_bca_list = [], []
    for i in range(len(x_test_list)):
        cal_loader = DataLoader(dataset=TensorDataset(x_test_list[i], y_test_list[i]),
                        batch_size=args.batch_size,
                        shuffle=False,
                        drop_last=False)

        _, cal_acc, cal_bca = eval(feature_ext, classifier, criterion, cal_loader)
        u_bca_list.append(cal_bca)

        cal_loader = DataLoader(dataset=TensorDataset(u_x_test_list[i], y_test_list[i]),
                        batch_size=args.batch_size,
                        shuffle=False,
                        drop_last=False)

        _, cal_acc, cal_bca = eval(feature_ext, classifier, criterion, cal_loader)
        u_u_bca_list.append(cal_bca)

    # discriminator
    logging.info('*' * 25 + ' train discriminator (unlearnable) ' + '*' * 25)
    feature_ext, discriminator, u_d_train_acc, u_d_test_acc = trainer(
        feature_ext,
        discriminator,
        u_x_train,
        s_train,
        u_x_test_list,
        s_test_list,
        phase='d',
        args=args)

    feature_ext.eval()
    discriminator.eval()

    u_sda_list, u_u_sda_list = [] ,[]
    for i in range(len(x_test_list)):
        dis_loader = DataLoader(dataset=TensorDataset(x_test_list[i], s_test_list[i]),
                        batch_size=args.batch_size,
                        shuffle=False,
                        drop_last=False)
        _, s_acc, s_bca = eval(feature_ext, discriminator, criterion, dis_loader)
        u_sda_list.append(s_acc)

        # session unlearnable
        dis_loader = DataLoader(dataset=TensorDataset(u_x_test_list[i], s_test_list[i]),
                        batch_size=args.batch_size,
                        shuffle=False,
                        drop_last=False)
        _, s_acc, s_bca = eval(feature_ext, discriminator, criterion, dis_loader)
        u_u_sda_list.append(s_acc)

    logging.info(f'Session: {session} | unlearnalbe bca: {u_bca_list}, sda: {u_sda_list}')
    logging.info(f'Mean unlearnalbe bca: {np.mean(u_bca_list)}, sda: {np.mean(u_sda_list)}')

    logging.info(f'Session: {session} | test unlearnalbe bca: {u_u_bca_list}, sda: {u_u_sda_list}')
    logging.info(f'Mean test unlearnalbe bca: {np.mean(u_u_bca_list)}, sda: {np.mean(u_u_sda_list)}')

    np.savez(npz_path + f'/repeat{repeat}_session{session}.npz',
             x_train=x_train.numpy(),
             y_train=y_train.numpy(),
             s_train=s_train.numpy(),
             x_test=[x.numpy() for x in x_test_list],
             y_test=[x.numpy() for x in y_test_list],
             s_test=[x.numpy() for x in s_test_list],
             u_x_train=u_x_train.numpy(),
             c_train_bca=c_train_bca,
             c_test_bca=c_test_bca,
             d_train_acc=d_train_acc,
             d_test_acc=d_test_acc,
             u_c_train_bca=u_c_train_bca,
             u_c_test_bca=u_c_test_bca,
             u_d_train_acc=u_d_train_acc,
             u_d_test_acc=u_d_test_acc)

    return bca_list, sda_list, u_bca_list, u_sda_list, u_u_bca_list, u_u_sda_list


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
    bca = bca_score(labels, preds)

    return loss, acc, bca


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
    parser.add_argument('--gpu_id', type=str, default='1')
    parser.add_argument('--dataset', type=str, default='EPFLnoClip')

    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--optim', type=str, default='Adam')
    parser.add_argument('--lr', type=float, default=0.01)

    parser.add_argument('--feature_c', type=str, default='EEGNet')
    parser.add_argument('--feature_d', type=str, default='EEGNet')
    parser.add_argument('--subject_wise', type=bool, default=False)
    parser.add_argument('--alpha', type=float, default=1.0) 
    parser.add_argument('--log', type=str, default='')

    args = parser.parse_args()

    args.device = f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu'

    subject_num_dict = {
        'MI4C': 9,
        'ERN': 16,
        'RSVP': 11,
        'EPFL': 8,
        'EPFLnoClip': 8,
        'NICU': 14,
        'MI109': 109
    }

    model_path = f'model/{args.dataset}/{args.feature_c}_{args.feature_d}/'

    log_path = f'results_0129/log/'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_name = os.path.join(
        log_path, f'{args.dataset}_{args.feature_c}_{args.feature_d}.log')

    npz_path = f'results_0129/npz/{args.dataset}_{args.feature_c}_{args.feature_d}'

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

    # load data
    if args.dataset == 'MI4C':
        x_session, y_session, s_session = MI4CLoad(isEA=False)
    elif args.dataset == 'ERN':
        x_session, y_session, s_session = ERNLoad(isEA=False)
    elif args.dataset == 'EPFL':
        x_session, y_session, s_session = EPFLLoad(isEA=False)
    elif args.dataset == 'NICU':
        x_session, y_session, s_session = NICULoad(isEA=False)
    elif args.dataset == 'MI109':
        x_session, y_session, s_session = MI109Load(isEA=False)

    # model train
    r_bca, r_sda, r_u_bca, r_u_sda, r_u_u_bca, r_u_u_sda = [], [], [], [], [], []
    for repeat in range(5):
        seed(repeat)
        s_bca, s_sda, s_u_bca, s_u_sda, s_u_u_bca, s_u_u_sda = [], [], [], [], [], []
        for session in x_session.keys():
            # model train
            model_save_path = os.path.join(model_path, f'{session}')
            if not os.path.exists(model_save_path):
                os.makedirs(model_save_path)
            
            bca, sda, u_bca, u_sda, u_u_bca, u_u_sda = train(x_session, y_session, s_session, 
                                                            npz_path, repeat, session, args)
            s_bca.append(bca)
            s_sda.append(sda)
            s_u_bca.append(u_bca)
            s_u_sda.append(u_sda)
            s_u_u_bca.append(u_u_bca)
            s_u_u_sda.append(u_u_sda)

            logging.info(f'session {session + 1}')
            logging.info(f'Mean bca: {np.mean(bca)}')
            logging.info(f'Mean sda: {np.mean(sda)}')
            logging.info(f'Mean unlearnable bca: {np.mean(u_bca)}')
            logging.info(f'Mean unlearnable sda: {np.mean(u_sda)}')
            logging.info(f'Mean test unlearnable bca: {np.mean(u_u_bca)}')
            logging.info(f'Mean test unlearnable sda: {np.mean(u_u_sda)}')

        r_bca.append(s_bca)
        r_sda.append(s_sda)
        r_u_bca.append(s_u_bca)
        r_u_sda.append(s_u_sda)
        r_u_u_bca.append(s_u_u_bca)
        r_u_u_sda.append(s_u_u_sda)
        logging.info(f'repeat {repeat + 1}')
        logging.info(f'Mean bca: {np.mean(s_bca)}')
        logging.info(f'Mean sda: {np.mean(s_sda)}')
        logging.info(f'Mean unlearnable bca: {np.mean(s_u_bca)}')
        logging.info(f'Mean unlearnable sda: {np.mean(s_u_sda)}')
        logging.info(f'Mean test unlearnable bca: {np.mean(s_u_u_bca)}')
        logging.info(f'Mean test unlearnable sda: {np.mean(s_u_u_sda)}')

    logging.info(f'Avg results')
    logging.info(f'Mean bca: {np.mean(r_bca)}')
    logging.info(f'Mean sda: {np.mean(r_sda)}')
    logging.info(f'Mean unlearnable bca: {np.mean(r_u_bca)}')
    logging.info(f'Mean unlearnable sda: {np.mean(r_u_sda)}')
    logging.info(f'Mean test unlearnable bca: {np.mean(r_u_u_bca)}')
    logging.info(f'Mean test unlearnable sda: {np.mean(r_u_u_sda)}')
    np.savez(npz_path + '/result.npz',
            bca=r_bca,
            sda=r_sda,
            u_bca=r_u_bca,
            u_sda=r_u_sda,
            u_u_bca=r_u_u_bca,
            u_u_sda=r_u_u_sda)
