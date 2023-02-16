import os
import argparse
import logging
import numpy as np
from mne.decoding import CSP as mne_CSP
from pyriemann.tangentspace import TangentSpace
from pyriemann.estimation import XdawnCovariances
from pyriemann.channelselection import ElectrodeSelection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from utils.pytorch_utils import print_args, seed, bca_score


def balanced_sample(x, u_x, y, s):
    balanced_x, balanced_u_x, balanced_y = [], [], []
    for i in range(len(np.unique(s))):
        uid_idx = (s == i)
        x_temp, u_x_temp, y_temp = x[uid_idx], u_x[uid_idx], y[uid_idx]
        x_temp_class0 = x_temp[y_temp == 0]
        x_temp_class1 = x_temp[y_temp == 1]
        u_x_temp_class0 = u_x_temp[y_temp == 0]
        u_x_temp_class1 = u_x_temp[y_temp == 1]
        y_temp_class0 = y_temp[y_temp == 0]
        y_temp_class1 = y_temp[y_temp == 1]
        # resample
        idx = np.random.permutation(np.arange(len(x_temp_class0)))
        x_temp_class0 = x_temp_class0[idx[:len(x_temp_class1)]]
        u_x_temp_class0 = u_x_temp_class0[idx[:len(x_temp_class1)]]
        y_temp_class0 = y_temp_class0[idx[:len(x_temp_class1)]]
        balanced_x.extend([x_temp_class0, x_temp_class1])
        balanced_u_x.extend([u_x_temp_class0, u_x_temp_class1])
        balanced_y.extend([y_temp_class0, y_temp_class1])
    # shuffle
    balanced_x = np.concatenate(balanced_x)
    balanced_u_x = np.concatenate(balanced_u_x)
    balanced_y = np.concatenate(balanced_y)
    idx = np.random.permutation(np.arange(len(balanced_x)))
    return balanced_x[idx], balanced_u_x[idx], balanced_y[idx]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='traditional model')
    parser.add_argument('--dataset', type=str, default='MI109')

    parser.add_argument('--feature_c', type=str, default='EEGNet')
    parser.add_argument('--feature_d', type=str, default='EEGNet')
    parser.add_argument('--subject_wise', type=bool, default=False)
    parser.add_argument('--log', type=str, default='')

    args = parser.parse_args()

    subject_num_dict = {
        'MI4C': 9,
        'ERN': 16,
        'EPFL': 8,
        'NICU': 14,
        'MI109': 109
    }

    data_path = f'results_0129/npz/{args.dataset}_{args.feature_c}_{args.feature_d}'

    log_path = f'results_0129/log/'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_name = os.path.join(log_path, f'{args.dataset}_traditional.log')

    npz_name = f'results_0129/npz/traditional/{args.dataset}.npz'

    if args.subject_wise:
        data_path += '/subject'
        log_name = log_name.replace('.log', '_subject.log')
        npz_name = npz_name.replace('.npz', '_subject.npz')

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

    r_bca, r_u_bca = [], []
    for r in range(5):
        seed(r)
        # MI1: 2 MI2: 3
        s_bca, s_u_bca = [], []
        for session in range(3):
            data = np.load(data_path + f'/repeat{r}_session{session}.npz', allow_pickle=True)
            x_train = np.squeeze(data['x_train']).astype(np.double)
            y_train = data['y_train']
            s_train = data['s_train']
            x_test_list = data['x_test']
            y_test_list = data['y_test']
            u_x_train = np.squeeze(data['u_x_train']).astype(np.double)

            if args.dataset == 'EPFL':
                x_train, u_x_train, y_train = balanced_sample(
                    x_train, u_x_train, y_train, s_train)

            # benign data
            feature_ext = mne_CSP(n_components=6,
                                transform_into='average_power',
                                log=False,
                                cov_est='epoch')
            # if args.dataset == 'MI4C' or args.dataset == 'MI109':
            #     feature_ext = mne_CSP(n_components=8, transform_into='average_power', log=False, cov_est='epoch')
            # elif args.dataset == 'EPFL' or args.dataset == 'ERN':
            #     xd = XdawnCovariances(nfilter=5, applyfilters=True, estimator='lwf')
            #     ts = TangentSpace(metric='logeuclid')
            #     feature_ext = Pipeline([('xDAWN', xd), ('TangentSpace', ts),])

            lr = LogisticRegression(solver='sag',
                                    max_iter=200,
                                    C=0.01,
                                    class_weight='balanced')

            model = Pipeline([('feature', feature_ext), ('LR', lr)])

            model.fit(x_train, y_train)

            if r == 0: feature = feature_ext.transform(x_train)

            bca_list = []
            for i in range(len(x_test_list)):
                x_test, y_test = x_test_list[i].astype(np.double), y_test_list[i]
                y_pred = np.argmax(model.predict_proba(x_test.squeeze()), axis=1)
                bca = bca_score(y_test, y_pred)
                bca_list.append(bca)

            # unlearnable data
            feature_ext = mne_CSP(n_components=6,
                                transform_into='average_power',
                                log=False,
                                cov_est='epoch')
            # if args.dataset == 'MI4C' or args.dataset == 'MI109':
            #     feature_ext = mne_CSP(n_components=8, transform_into='average_power', log=False, cov_est='epoch')
            # elif args.dataset == 'EPFL' or args.dataset == 'ERN':
            #     xd = XdawnCovariances(nfilter=5, applyfilters=True, estimator='lwf')
            #     ts = TangentSpace(metric='logeuclid')
            #     feature_ext = Pipeline([('xDAWN', xd), ('TangentSpace', ts),])

            lr = LogisticRegression(solver='sag',
                                    max_iter=200,
                                    C=0.01,
                                    class_weight='balanced')

            model = Pipeline([('feature', feature_ext), ('LR', lr)])

            model.fit(u_x_train, y_train)

            if r == 0: u_feature = feature_ext.transform(u_x_train)

            u_bca_list = []
            for i in range(len(x_test_list)):
                x_test, y_test = x_test_list[i].astype(np.double), y_test_list[i]
                y_pred = np.argmax(model.predict_proba(x_test.squeeze()), axis=1)
                u_bca = bca_score(y_test, y_pred)
                u_bca_list.append(u_bca)

            s_bca.append(bca_list)
            s_u_bca.append(u_bca_list)
            logging.info(f'Repeat {r} | session {session}: bca={np.mean(bca_list)}, u_bca={np.mean(u_bca_list)}')

        r_bca.append(s_bca)
        r_u_bca.append(s_u_bca)
        logging.info(f'Repeat {r} result')
        logging.info(f'Mean cal bca: {np.mean(s_bca)}')
        logging.info(f'Mean unlearnable cal bca: {np.mean(s_u_bca)}')

    logging.info(f'Avg result')
    logging.info(f'Mean cal bca: {np.mean(r_bca)}')
    logging.info(f'Mean unlearnable cal bca: {np.mean(r_u_bca)}')
    np.savez(npz_name,
             feature=feature,
             u_feature=u_feature,
             bca=np.array(r_bca).reshape(-1,1),
             u_bca=np.array(r_u_bca).reshape(-1,1))
