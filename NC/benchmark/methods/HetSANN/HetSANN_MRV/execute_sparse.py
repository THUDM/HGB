import os
import time
import random
import scipy.sparse as sp
import numpy as np
import tensorflow as tf
import argparse

from models.sp_hgat import SpHGAT
from utils import process
from scripts.data_loader import data_loader

import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', help='Dataset.', default='imdb', type=str)
parser.add_argument('--epochs', help='Epochs.', default=100000, type=int)
parser.add_argument('--patience', help='Patience for early stopping.', default=100, type=int)
parser.add_argument('--lr', help='Learning rate.', default=0.005, type=float)
parser.add_argument('--l2_coef', help='Weight decay.', default=0.0005, type=float)
parser.add_argument('--dropout', help='Dropout.', default=0.6, type=float)
parser.add_argument('--train_rate', help='Label rate for training.', default=0.1, type=float)
parser.add_argument('--seed', help='Random seed for data splitting.', default=None, type=int)
parser.add_argument('--layers', help='Number of layers.', default=2, type=int)
parser.add_argument('--hid', help='Number of hidden units per head in each layer.',
                    nargs='*', default=[8, 8], type=int)
parser.add_argument('--heads', help='Number of attention heads in each layer.',
                    nargs='*', default=[8, 1], type=int)
parser.add_argument('--residue', help='Using residue.', action='store_true')
parser.add_argument('--repeat', help='Repeat.', default=1, type=int)
parser.add_argument('--random_feature', help='Random features', action='store_true')
parser.add_argument('--target_node', help='index of target nodes for classification.',
                    nargs='*', default=[0, 1], type=int)
parser.add_argument('--target_is_multilabels', help='each type of target node for classification is multi-labels or not.(0 means not else means yes)',
                    nargs='*', default=[0, 1], type=int)
parser.add_argument('--saved_model_suffix', help='to splite checkpoint by suffix', default="", type=str)
parser.add_argument('--no_attn_reg', help='Do not use edge direction regularization', action='store_true')
parser.add_argument('--simple_inner', help='Use original inner product', action='store_true')
parser.add_argument('--loop_coef', help='Coefficient for regularization.', default= 1e-3, type=float)
parser.add_argument('--inv_coef', help='Coefficient for regularization.', default=1e-3, type=float)
parser.add_argument('--feats-type', type=int, default=0,
                    help='Type of the node features used. ' +
                         '0 - loaded features; ' +
                         '1 - only target node features (zero vec for others); ' +
                         '2 - only target node features (id vec for others); ' +
                         '3 - all id vec. ;' +
                        '4 - only term features (id vec for others);' + 
                        '5 - only term features (zero vec for others).')

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True

args= parser.parse_args()
dataset = args.dataset
checkpt_file = 'pre_trained/{}/{}/{}.ckpt'.format(dataset, args.saved_model_suffix, dataset)
checkpt_file = checkpt_file.replace('//', '/')
process.mkdir(os.path.split(checkpt_file)[0])
# training params
batch_size = 1
train_rate = args.train_rate
seed = args.seed
nb_epochs = args.epochs
patience = args.patience
lr = args.lr  # learning rate
l2_coef = args.l2_coef  # weight decay
dropout = args.dropout
repeat = args.repeat
random_feature = args.random_feature
target_node = args.target_node
is_multilabel = [False if t==0 else True for t in args.target_is_multilabels]
loop_coef = args.loop_coef
inv_coef = args.inv_coef
feats_type = args.feats_type

layers = args.layers
hid = args.hid
if len(hid) == 1:
    hid_units = hid * layers
elif len(hid) == layers:
    hid_units = hid
heads = args.heads
if len(heads) == 1:
    n_heads = heads * layers
elif len(heads) == 2:
    n_heads = [heads[0]] * (layers - 1) + [heads[1]]
elif len(heads) == layers:
    n_heads = heads

residual = args.residue # False
nonlinearity = tf.nn.elu
model = SpHGAT

no_attn_reg = args.no_attn_reg
simple_inner = args.simple_inner

random.seed(seed) # random seed for random data split only

print('Dataset: ' + dataset)
print('Train rate: ' + str(train_rate))
print('----- Opt. hyperparams -----')
print('lr: ' + str(lr))
print('l2_coef: ' + str(l2_coef))
print('----- Archi. hyperparams -----')
print('nb. layers: ' + str(len(hid_units)))
print('nb. units per layer: ' + str(hid_units))
print('nb. attention heads: ' + str(n_heads))
print('residual: ' + str(residual))
print('nonlinearity: ' + str(nonlinearity))
print('model: ' + str(model))
print('target nodes: ', target_node)
print('is_multilabel: ', is_multilabel)
print('loop_coef:', loop_coef)
print('inv_coef:', inv_coef)

sparse = True
metr_num = 2
total_vl_acc = np.array([0.]*(len(target_node)*metr_num))  # should be array
total_ts_acc = np.array([0.]*(len(target_node)*metr_num))  # should be array

def get_loss_acc(logits, labels, msk, is_multilabel=False):
    global model
    class_num = labels.shape[-1]
    log_resh = tf.reshape(logits, [-1, class_num])
    lab_resh = tf.reshape(labels, [-1, class_num])
    msk_resh = tf.reshape(msk, [-1])
    
    if is_multilabel:
        loss = model.masked_sigmoid_cross_entropy(log_resh, lab_resh, msk_resh)
        accuracy = [model.micro_f1(log_resh, lab_resh, msk_resh), model.macro_f1(log_resh, lab_resh, msk_resh)]
        acc_name = ['if1', 'af1']
        acc_full_name = ['micro f1', 'macro f1']

        #predicted = tf.round(tf.nn.sigmoid(logits))
        #predicted = tf.cast(predicted, dtype=tf.int32)

    else:
        loss = model.masked_softmax_cross_entropy(log_resh, lab_resh, msk_resh)
        accuracy = [model.micro_f1_onelabel(log_resh, lab_resh, msk_resh), model.macro_f1_onelabel(log_resh, lab_resh, msk_resh)]
        acc_name = ['if1', 'af1']
        acc_full_name = ['micro f1', 'macro f1']

    return loss, accuracy, acc_name, acc_full_name


def print_eachclass_info(train_loss_each, train_acc_each, val_loss_each, val_acc_each, acc_name):
    tl_average = np.mean(np.array(train_loss_each), axis=0)
    ta_average = np.mean(np.array(train_acc_each), axis=0)
    vl_average = np.mean(np.array(val_loss_each), axis=0)
    va_average = np.mean(np.array(val_acc_each), axis=0)
    metric_num = int(len(ta_average)/len(tl_average))
    for i in range(len(tl_average)):
        line = '\t\t target %s: loss = %.3f, ' % (i, tl_average[i])
        for j in range(metric_num):
            line += '%s = %.5f, ' % (acc_name[i*metric_num+j], ta_average[i*metric_num+j])
        line += '| Val: loss = %.3f, ' % (vl_average[i])
        for j in range(metric_num):
            line += '%s = %.5f, ' % (acc_name[i*metric_num+j], va_average[i*metric_num+j])
        print(line)

path = '../../../data/' + dataset
loader = data_loader(path)
result_list = []

for repeat_i in range(repeat):
    print('Run #' + str(repeat_i) + ':')
    adj, adj_type, edge_list, features, y_train, y_val, y_test,\
        train_mask, val_mask, test_mask = process.load_heterogeneous_data(dataset, train_rate=train_rate, target_node=target_node, feats_type=feats_type)
    features = [process.preprocess_features(feature)[0] for feature in features]
    
    nb_nodes = [feature.shape[0] for feature in features]
    ft_size = [feature.shape[1] for feature in features]

    nb_classes = [y.shape[1] for y in y_train]
    
    features = [feature[np.newaxis] for feature in features]
    y_train = [y[np.newaxis] for y in y_train]
    y_val = [y[np.newaxis] for y in y_val]
    y_test = [y[np.newaxis] for y in y_test]
    train_mask = [m[np.newaxis] for m in train_mask]
    val_mask = [m[np.newaxis] for m in val_mask]
    test_mask = [m[np.newaxis] for m in test_mask]
    
    if random_feature:
        features[0] = np.random.standard_normal(features[0].shape)
    
    if sparse:
        biases = [process.preprocess_adj_hete(a) for a in adj] # transposed here
    else:
        biases = []
        for a in adj:
            a = a.todense()
            a = a[np.newaxis]
            
    if no_attn_reg:
        edge_list = [(i,) for i in range(len(adj_type))]
    if simple_inner:
        edge_list = []
    
    with tf.Graph().as_default():
        with tf.name_scope('input'):
            ftr_in = [tf.placeholder(dtype=tf.float32,
                shape=(batch_size, nb, ft)) for nb, ft in zip(nb_nodes, ft_size)]
            if sparse:
                bias_in = [tf.sparse_placeholder(dtype=tf.float32) for _ in biases]
            else:
                bias_in = None
            lbl_in = [tf.placeholder(dtype=tf.int32, shape=(batch_size, nb_nodes[target_node[i]], nb_classes[i])) for i in range(len(nb_classes))]
            msk_in = [tf.placeholder(dtype=tf.int32, shape=(batch_size, nb_nodes[target_node[i]])) for i in range(len(nb_classes))]
            attn_drop = tf.placeholder(dtype=tf.float32, shape=())
            ffd_drop = tf.placeholder(dtype=tf.float32, shape=())
            is_train = tf.placeholder(dtype=tf.bool, shape=())

        print("residual: ")
        print(residual)
        logits = model.inference(ftr_in, nb_classes, nb_nodes, is_train,
                                    attn_drop, ffd_drop, target_nodes=target_node,
                                    bias_mat=bias_in, adj_type=adj_type,
                                    edge_list=edge_list,
                                    hid_units=hid_units, n_heads=n_heads,
                                    residual=residual, activation=nonlinearity)
        with tf.name_scope('loss_acc'):
            loss, accuracy, acc_name, acc_full_name = [], [], [], []
            all_class_loss = 0.0
            for tn in range(len(target_node)):
                tn_logits = logits[tn]
                tn_labels = lbl_in[tn]
                tn_masks = msk_in[tn]
                tn_is_multilabel = is_multilabel[tn]
                tn_loss, tn_accuracy, tn_acc_name, tn_acc_full_name = get_loss_acc(tn_logits, tn_labels, tn_masks, is_multilabel=tn_is_multilabel)
                loss.append(tn_loss)
                accuracy.extend(tn_accuracy)
                acc_name.extend(tn_acc_name)
                acc_full_name.extend(tn_acc_full_name)
                all_class_loss += tn_loss
            loss_loop = tf.add_n(tf.get_collection('loss_loop')) * loop_coef
            loss_inv= tf.add_n(tf.get_collection('loss_inv')) * inv_coef
        train_op = model.training(all_class_loss + loss_loop + loss_inv, lr, l2_coef)
        saver = tf.train.Saver()
    
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    
        vlss_mn = np.inf
        vacc_mx = 0.0
        curr_step = 0
    
        with tf.Session(config=config) as sess:
            sess.run(init_op)
            vacc_early_model = 0.0
            vlss_early_model = 0.0
            vacc_each_early_model = np.array([0.]*(len(target_node)*metr_num))
            for epoch in range(nb_epochs):
                # summary information
                train_loss_avg = 0
                train_acc_avg = 0
                val_loss_avg = 0
                val_acc_avg = 0
                # for each class information
                train_loss_each = []
                train_acc_each = []
                val_loss_each = []
                val_acc_each = []
                tr_step = 0
                tr_size = features[0].shape[0]
                while tr_step * batch_size < tr_size:
                    if sparse:
                        fd = {i: d for i, d in zip(ftr_in, features)}
                        fd.update({i: d for i, d in zip(bias_in, biases)})
                    else:
                        fd = {i: d[tr_step * batch_size:(tr_step + 1) * batch_size]
                           for i, d in zip(ftr_in, features)}
                        fd.update({i: d[tr_step * batch_size:(tr_step + 1) * batch_size]
                           for i, d in zip(bias_in, biases)})
                    fd.update({i:d[tr_step*batch_size:(tr_step+1)*batch_size] for i, d in zip(lbl_in, y_train)})
                    fd.update({i:d[tr_step*batch_size:(tr_step+1)*batch_size] for i, d in zip(msk_in, train_mask)})
                    fd.update({is_train: True})
                    fd.update({attn_drop: dropout, ffd_drop:dropout})
                    _, loss_list_tr, acc_list_tr, loss_loop_tr, loss_inv_tr = sess.run([train_op, loss, accuracy, loss_loop, loss_inv], feed_dict=fd)
                    train_loss_each.append(np.array(loss_list_tr))
                    train_acc_each.append(np.array(acc_list_tr))
                    train_loss_avg += np.sum(np.array(loss_list_tr))
                    train_acc_avg += np.sum(np.array(acc_list_tr))
                    tr_step += 1
    
                vl_step = 0
                vl_size = features[0].shape[0]
                while vl_step * batch_size < vl_size:
                    if sparse:
                        fd = {i: d for i, d in zip(ftr_in, features)}
                        fd.update({i: d for i, d in zip(bias_in, biases)})
                    else:
                        fd = {i: d[vl_step * batch_size:(vl_step + 1) * batch_size]
                           for i, d in zip(ftr_in, features)}
                        fd.update({i: d[vl_step * batch_size:(vl_step + 1) * batch_size]
                           for i, d in zip(bias_in, biases)})
                    fd.update({i:d[vl_step*batch_size:(vl_step+1)*batch_size] for i, d in zip(lbl_in, y_val)})
                    fd.update({i:d[vl_step*batch_size:(vl_step+1)*batch_size] for i, d in zip(msk_in, val_mask)})
                    fd.update({is_train: False})
                    fd.update({attn_drop: 0.0, ffd_drop:0.0})
                    pre, loss_list_vl, acc_list_vl = sess.run([logits, loss, accuracy], feed_dict=fd)
                    acc_list_vl = [0. if np.isnan(acc_vl) else acc_vl for acc_vl in acc_list_vl]
                    val_loss_each.append(np.array(loss_list_vl))
                    val_acc_each.append(np.array(acc_list_vl))
                    val_loss_avg += np.sum(np.array(loss_list_vl))
                    val_acc_avg += np.sum(np.array(acc_list_vl))
                    vl_step += 1

                if is_multilabel[0]:
                    print('is multilabel')
                else:
                    onehot = np.eye(loader.labels_test['num_classes'], dtype=np.int32)
                    prediction = onehot[pre[0][test_mask[0][0]].argmax(axis=1)]
                    result = loader.evaluate(prediction)
                    print("micro_f1 and macro_f1:")
                    print(result)

                print('Training %s: loss = %.5f, %s = %.5f, loss_loop = %.5f, loss_inv = %.5f | Val: loss = %.5f, %s = %.5f' %
                        (epoch, train_loss_avg/tr_step, 'acc/F1', train_acc_avg/tr_step,
                         loss_loop_tr, loss_inv_tr,
                        val_loss_avg/vl_step, 'acc/F1', val_acc_avg/vl_step))
                print_eachclass_info(train_loss_each, train_acc_each, val_loss_each, val_acc_each, acc_name)
    
                if val_acc_avg/vl_step > vacc_mx or val_loss_avg/vl_step < vlss_mn:
                    if val_acc_avg/vl_step > vacc_mx and val_loss_avg/vl_step < vlss_mn:
                        vacc_early_model = val_acc_avg/vl_step
                        vlss_early_model = val_loss_avg/vl_step
                        vacc_each_early_model = np.mean(np.array(val_acc_each), axis=0)
                        saver.save(sess, checkpt_file)
                        print("saved model as %s"%checkpt_file)
                    vacc_mx = np.max((val_acc_avg/vl_step, vacc_mx))
                    vlss_mn = np.min((val_loss_avg/vl_step, vlss_mn))
                    curr_step = 0
                else:
                    curr_step += 1
                    if curr_step == patience:
                        print('Early stop! Min loss: ', vlss_mn,
                              ', Max', 'acc/F1', ': ', vacc_mx)
                        print('Early stop model validation loss: ', vlss_early_model,
                              ', ', 'acc/F1', ': ', vacc_early_model)
                        total_vl_acc += vacc_each_early_model
                        break

            if curr_step < patience:
                print('Min loss: ', vlss_mn, ', Max', 'acc/F1', ': ', vacc_mx)
                print('model validation loss: ', vlss_early_model, ', ', 'acc/F1', ': ', vacc_early_model)
                total_vl_acc += vacc_each_early_model
    
            saver.restore(sess, checkpt_file)
    
            ts_size = features[0].shape[0]
            ts_step = 0
            test_loss_each = []
            test_acc_each = []
    
            while ts_step * batch_size < ts_size:
                if sparse:
                    fd = {i: d for i, d in zip(ftr_in, features)}
                    fd.update({i: d for i, d in zip(bias_in, biases)})
                else:
                    fd = {i: d[ts_step * batch_size:(ts_step + 1) * batch_size]
                       for i, d in zip(ftr_in, features)}
                    fd.update({i: d[ts_step * batch_size:(ts_step + 1) * batch_size]
                       for i, d in zip(bias_in, biases)})
                fd.update({i:d[ts_step*batch_size:(ts_step+1)*batch_size] for i, d in zip(lbl_in, y_test)})
                fd.update({i:d[ts_step*batch_size:(ts_step+1)*batch_size] for i, d in zip(msk_in, test_mask)})
                fd.update({is_train: False})
                fd.update({attn_drop: 0.0, ffd_drop:0.0})
                pre, loss_list_ts, acc_list_ts = sess.run([logits, loss, accuracy], feed_dict=fd)
                test_loss_each.append(np.array(loss_list_ts))
                test_acc_each.append(np.array(acc_list_ts))
                ts_step += 1
                if is_multilabel[0]:
                    print('is multilabel')
                else:
                    onehot = np.eye(loader.labels_test['num_classes'], dtype=np.int32)
                    prediction = onehot[pre[0][test_mask[0][0]].argmax(axis=1)]
                    result = loader.evaluate(prediction)
                    print("micro_f1 and macro_f1:")
                    print(result)
                    result_list.append(result)

            test_loss_each = np.mean(np.array(test_loss_each), axis=0)
            test_acc_each = np.mean(np.array(test_acc_each), axis=0)
            print('*'*10,'Test information:', '*'*10)
            for e in range(len(target_node)):
                print('target %s: loss: %.3f, %s:%.5f, %s:%.5f' % (e, test_loss_each[e], acc_full_name[e*metr_num], test_acc_each[e*metr_num], acc_full_name[e*metr_num+1], test_acc_each[e*metr_num+1]))
            total_ts_acc += test_acc_each
    
            sess.close()



# result = loader.evaluate(prediction)


print('Validation:', total_vl_acc/repeat, 'Test:', total_ts_acc/repeat)
print('all results:')
print(result_list)
