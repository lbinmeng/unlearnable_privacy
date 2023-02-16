from re import T
from typing import Optional
from scipy import signal
from scipy.signal import resample
import scipy.linalg as la
import scipy.io as scio
import numpy as np
import pickle
import os
import math


def split(x, y, ratio=0.8, shuffle=True):
    idx = np.arange(len(x))
    if shuffle:
        idx = np.random.permutation(idx)
    train_size = int(len(x) * ratio)

    return x[idx[:train_size]], y[idx[:train_size]], x[idx[train_size:]], y[
        idx[train_size:]]


def balance_split(x, y, num_class, ratio):
    lb_idx = []
    for c in range(num_class):
        idx = np.where(y == c)[0]
        idx = np.random.choice(idx, int(np.ceil(len(idx) * ratio)), False)
        lb_idx.extend(idx)
    ulb_idx = np.array(sorted(list(set(range(len(x))) - set(lb_idx))))

    return x[lb_idx], y[lb_idx], x[ulb_idx], y[ulb_idx]


def align(data):
    """data alignment"""
    data_align = []
    length = len(data)
    rf_matrix = np.dot(data[0], np.transpose(data[0]))
    for i in range(1, length):
        rf_matrix += np.dot(data[i], np.transpose(data[i]))
    rf_matrix /= length

    rf = la.inv(la.sqrtm(rf_matrix))
    if rf.dtype == complex:
        rf = rf.astype(np.float64)

    for i in range(length):
        data_align.append(np.dot(rf, data[i]))

    return np.asarray(data_align).squeeze(), rf


def MI4CLoad(isEA=False):
    data_path = '../EEG_data/MI4C/processed_session/'
    x_session, y_session, s_session = {}, {}, {}
    for i in range(9):
        data = scio.loadmat(data_path + f's{i}.mat')
        x, y = data['x'], data['y']
        for session in range(len(x)):
            x_temp, y_temp = x[session], y[session].squeeze()
            if isEA:
                x_temp, _ = align(x_temp.squeeze())
                x_temp = x_temp[:, np.newaxis, :, :]

            if session in x_session.keys():
                x_session[session].append(x_temp)
                y_session[session].append(y_temp)
                s_session[session].extend([i] * len(x_temp))
            else:
                x_session[session] = [x_temp]
                y_session[session] = [y_temp]
                s_session[session] = [i] * len(x_temp)

    for session in x_session.keys():
        x_session[session] = np.concatenate(x_session[session], axis=0)
        y_session[session] = np.concatenate(y_session[session], axis=0).squeeze()
        s_session[session] = np.array(s_session[session])

    return x_session, y_session, s_session


def RSVPLoad(isEA=False):
    data_path = '../EEG_data/RSVP/'

    x_train, y_train, s_train, x_test, y_test, s_test = [], [], [], [], [], []
    for i in range(11):
        data = scio.loadmat(data_path + f'A{i+1}.mat')
        if isEA:
            x, _ = align(data['x'].squeeze())
            x = x[:, np.newaxis, :, :]
        else:
            x = data['x']
        x_train_sub, y_train_sub, x_test_sub, y_test_sub = split(x,
                                                                 data['y'],
                                                                 ratio=0.5,
                                                                 shuffle=True)
        x_train.extend(x_train_sub)
        y_train.extend(y_train_sub)
        s_train.extend([i] * len(y_train_sub))
        x_test.extend(x_test_sub)
        y_test.extend(y_test_sub)
        s_test.extend([i] * len(y_test_sub))
    x_train, x_test = np.array(x_train), np.array(x_test)
    s_train, s_test = np.array(s_train), np.array(s_test)
    y_train, y_test = np.concatenate(y_train, axis=0), np.concatenate(y_test,
                                                                      axis=0)

    return x_train, y_train.squeeze(), s_train, x_test, y_test.squeeze(
    ), s_test


def MI109Load(isEA=False):
    data_path = '../EEG_data/MI109/processed_session/'
    x_session, y_session, s_session = {}, {}, {}
    for i in range(109):
        data = scio.loadmat(data_path + f's{i}.mat')
        x, y = data['x'], data['y']
        if i == 103: x, y = x[0], y[0]
        for session in range(len(x)):
            x_temp, y_temp = x[session], y[session]
            if isEA:
                x_temp, _ = align(x_temp.squeeze())
                x_temp = x_temp[:, np.newaxis, :, :]

            if session in x_session.keys():
                x_session[session].append(x_temp)
                y_session[session].append(y_temp)
                s_session[session].extend([i] * len(x_temp))
            else:
                x_session[session] = [x_temp]
                y_session[session] = [y_temp]
                s_session[session] = [i] * len(x_temp)

    for session in x_session.keys():
        x_session[session] = np.concatenate(x_session[session], axis=0)
        y_session[session] = np.concatenate(y_session[session], axis=0).squeeze()
        s_session[session] = np.array(s_session[session])

    return x_session, y_session, s_session


def ERNLoad(isEA=False):
    data_path = '../EEG_data/ERN/processed_session/'
    x_session, y_session, s_session = {}, {}, {}
    for i in range(16):
        data = scio.loadmat(data_path + f's{i}.mat')
        x, y = data['x'][0], data['y'][0]
        for session in range(len(x)):
            x_temp, y_temp = x[session], y[session].squeeze()
            if isEA:
                x_temp, _ = align(x_temp.squeeze())
                x_temp = x_temp[:, np.newaxis, :, :]

            if session in x_session.keys():
                x_session[session].append(x_temp)
                y_session[session].append(y_temp)
                s_session[session].extend([i] * len(x_temp))
            else:
                x_session[session] = [x_temp]
                y_session[session] = [y_temp]
                s_session[session] = [i] * len(x_temp)

    for session in x_session.keys():
        x_session[session] = np.concatenate(x_session[session], axis=0)
        y_session[session] = np.concatenate(y_session[session], axis=0).squeeze()
        s_session[session] = np.array(s_session[session])

    return x_session, y_session, s_session


def EPFLLoad(isEA=False):
    data_path = '../EEG_data/EPFL/processed_session/'
    x_session, y_session, s_session = {}, {}, {}
    for i in range(8):
        data = scio.loadmat(data_path + f's{i}.mat')
        x, y = data['x'][0], data['y'][0]
        for session in range(len(x)):
            x_temp, y_temp = x[session], y[session].squeeze()
            if isEA:
                x_temp, _ = align(x_temp.squeeze())
                x_temp = x_temp[:, np.newaxis, :, :]

            if session in x_session.keys():
                x_session[session].append(x_temp)
                y_session[session].append(y_temp)
                s_session[session].extend([i] * len(x_temp))
            else:
                x_session[session] = [x_temp]
                y_session[session] = [y_temp]
                s_session[session] = [i] * len(x_temp)

    for session in x_session.keys():
        x_session[session] = np.concatenate(x_session[session], axis=0)
        y_session[session] = np.concatenate(y_session[session], axis=0).squeeze()
        s_session[session] = np.array(s_session[session])

    return x_session, y_session, s_session


def NICULoad(isEA=False):
    data_path = '../EEG_data/NICU/processed/'
    uid_list = [10, 12, 20, 30, 33, 35, 43, 46, 49, 51, 61, 65, 67, 74]
    x_session, y_session, s_session = {}, {}, {}
    for i in range(len(uid_list)):
        data = scio.loadmat(data_path + f's{uid_list[i]}.mat')
        x, y = data['x'], data['y'].squeeze()
        length = math.ceil(len(x)/3)
        for session in range(3):
            x_temp, y_temp = x[session*length:(session+1)*length], y[session*length:(session+1)*length]
            if isEA:
                x_temp, _ = align(x_temp.squeeze())
                x_temp = x_temp[:, np.newaxis, :, :]

            if session in x_session.keys():
                x_session[session].append(x_temp)
                y_session[session].append(y_temp)
                s_session[session].extend([i] * len(x_temp))
            else:
                x_session[session] = [x_temp]
                y_session[session] = [y_temp]
                s_session[session] = [i] * len(x_temp)

    for session in x_session.keys():
        x_session[session] = np.concatenate(x_session[session], axis=0)
        y_session[session] = np.concatenate(y_session[session], axis=0).squeeze()
        s_session[session] = np.array(s_session[session])

    return x_session, y_session, s_session


def MI109Load_dc(isEA=False, offline_num=10):
    data_path = '../EEG_data/MI109/processed/'
    x_train, y_train, s_train, x_test, y_test, s_test = [], [], [], [], [], []
    t_x_train, t_y_train, t_x_test, t_y_test = [], [], [], []
    h_x_train, h_y_train, h_s_train, h_x_test, h_y_test, h_s_test = [], [], [], [], [], []
    offline_idx = np.random.permutation(np.arange(109))
    offline_idx = offline_idx[-offline_num:]
    oid, hid = 0, 0
    for i in range(109):
        data = scio.loadmat(data_path + f's{i}.mat')
        if isEA:
            x, _ = align(data['x'].squeeze())
            x = x[:, np.newaxis, :, :]
        else:
            x = data['x']
        x_train_sub, y_train_sub, x_test_sub, y_test_sub = split(x,
                                                                 data['y'],
                                                                 ratio=0.5,
                                                                 shuffle=True)
        if i == offline_idx[0]:
            t_x_train = x_train_sub
            t_y_train = y_train_sub
            t_x_test = x_test_sub
            t_y_test = y_test_sub
        elif i in offline_idx[1:]:
            h_x_train.extend(x_train_sub)
            h_y_train.extend(y_train_sub)
            h_s_train.extend([hid] * len(y_train_sub))
            h_x_test.extend(x_test_sub)
            h_y_test.extend(y_test_sub)
            h_s_test.extend([hid] * len(y_test_sub))
            hid += 1
        else:
            x_train.extend(x_train_sub)
            y_train.extend(y_train_sub)
            s_train.extend([oid] * len(y_train_sub))
            x_test.extend(x_test_sub)
            y_test.extend(y_test_sub)
            s_test.extend([oid] * len(y_test_sub))
            oid += 1
    x_train, x_test = np.array(x_train), np.array(x_test)
    s_train, s_test = np.array(s_train), np.array(s_test)
    y_train, y_test = np.concatenate(y_train, axis=0), np.concatenate(y_test,
                                                                      axis=0)

    h_x_train, h_x_test = np.array(h_x_train), np.array(h_x_test)
    h_s_train, h_s_test = np.array(h_s_train), np.array(h_s_test)
    h_y_train, h_y_test = np.concatenate(h_y_train,
                                         axis=0), np.concatenate(h_y_test,
                                                                 axis=0)

    return x_train, y_train.squeeze(
    ), s_train, x_test, y_test.squeeze(), s_test, h_x_train, h_y_train.squeeze(
    ), h_s_train, h_x_test, h_y_test.squeeze(), h_s_test, t_x_train, t_y_train.squeeze(), t_x_test, t_y_test.squeeze()
