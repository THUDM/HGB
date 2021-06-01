import torch as th
import numpy as np
from torch.utils.data import DataLoader
from torch import nn
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
import pickle
import sys
from collections import defaultdict
from utils import *
from GNN import GCN, GAT

sys.path.append('../../')
from scripts.data_loader import data_loader

device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
print(f'use device: {device}')


def train_gnn(edge_list, feat_list, train_data, args, model, dl, eval_type):
    for i in range(len(feat_list)):
        feat_list[i] = feat_list[i].to(device)
    edge_list = edge_list.to(device)
    train_data_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False, num_workers=0)
    early_stopping = EarlyStop(save_path=args.model_path, patience=args.patience)
    edge_num = len(edge_list)
    lossFun = nn.BCELoss()
    optimizer = th.optim.Adam([{'params': model.parameters()}], lr=args.lr, weight_decay=args.weight_decay)
    for epoch in range(args.epochs):
        """Refresh neg sample for train and valid"""
        # train_data.refresh_train_neg(eval_type, dl)
        # train_data.refresh_valid_neg(eval_type, dl)
        for index, train_data_bach in enumerate(train_data_loader):
            # train_data_bach = train_data_bach.to(device)
            model.train()
            hid_feat = model.encode([feat_list, edge_list])
            r_list = [0] * train_data_bach[0].shape[0]
            out_feat = model.decode(r_list, hid_feat[train_data_bach[0]], hid_feat[train_data_bach[1]])
            out_feat = th.sigmoid(out_feat)
            target = train_data_bach[2].float().to(device)
            loss = lossFun(out_feat, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f'epoch {epoch} | batch {index} |train loss {loss.item()}')

            if index % 10 == 0:
                # valid
                valid_list, valid_label = train_data.valid_data, train_data.valid_label
                model.eval()
                with th.no_grad():
                    hid_feat = model.encode([feat_list, edge_list])
                    r_list = [0] * len(valid_label)
                    out_feat = model.decode(r_list, hid_feat[valid_list[0]], hid_feat[valid_list[1]]).to(device)
                    out_feat = th.sigmoid(out_feat)
                    target = th.FloatTensor(valid_label).to(device)
                    valid_loss = lossFun(out_feat, target)

                    valid_score = dl.evaluate(
                        edge_list=valid_list,
                        confidence=out_feat.flatten().detach().cpu().numpy(),
                        labels=target.flatten().detach().cpu().numpy())
                    print(
                        f"Valid: loss {valid_loss}, score:{valid_score}")
                    early_stopping.step(valid_loss, model)
                    if early_stopping.stop:
                        break
        if early_stopping.stop:
            break

    # 2 hop test
    test_list, test_label = dl.get_test_neigh()
    test_list, test_label = test_list[eval_type], test_label[eval_type]
    model.load_state_dict(th.load(args.model_path))
    model.eval()
    with th.no_grad():
        target = th.FloatTensor(test_label).view(-1, 1).to(device)
        hid_feat = model.encode([feat_list, edge_list])
        r_list = [0] * len(test_label)
        out_feat = model.decode(r_list, hid_feat[test_list[0]], hid_feat[test_list[1]])
        out_feat = th.sigmoid(out_feat)
        confidence = out_feat.flatten().detach().cpu().numpy()
        dl.gen_file_for_evaluate(test_list, confidence, eval_type, file_path="./preds/amazon_pred_0")
        test_score = dl.evaluate(
            edge_list=test_list,
            confidence=confidence,
            labels=target.flatten().detach().cpu().numpy())

    # full random test
    with th.no_grad():
        test_list, test_label = dl.get_test_neigh_full_random()
        test_list, test_label = test_list[eval_type], test_label[eval_type]
        model.load_state_dict(th.load(args.model_path))
        model.eval()
        target = th.FloatTensor(test_label).to(device)
        hid_feat = model.encode([feat_list, edge_list])
        r_list = [0] * len(test_label)
        out_feat = model.decode(r_list, hid_feat[test_list[0]], hid_feat[test_list[1]])
        out_feat = th.sigmoid(out_feat)
        random_test_score = dl.evaluate(
            edge_list=test_list,
            confidence=out_feat.flatten().detach().cpu().numpy(),
            labels=target.flatten().detach().cpu().numpy())

    return random_test_score, test_score


if __name__ == '__main__':
    args = read_args()
    print(args)
    data_name = args.data
    data_dir = os.path.join('../../data', data_name)
    node_type_file = os.path.join('../../data', data_name, 'node.dat')
    dl_pickle_f = os.path.join(data_dir, 'dl_pickle')
    if os.path.exists(dl_pickle_f):
        dl = pickle.load(open(dl_pickle_f, 'rb'))
        print(f'Info: load {data_name} from {dl_pickle_f}')
    else:
        dl = data_loader(data_dir)
        pickle.dump(dl, open(dl_pickle_f, 'wb'))
        print(f'Info: load {data_name} from original data and generate {dl_pickle_f} ')
    model = None

    # feat = gen_feat(dl.nodes['total']).to(device)
    dim_list = list()
    for node_type in sorted(dl.nodes['count'].keys()):
        dim_list.append(dl.nodes['count'][node_type])
    feat_list = gen_feat_list(dim_list=dim_list)
    feat_list = [f.to(device) for f in feat_list]
    edge_list = gen_edge_list(dl, reverse=True).to(device)
    if args.model == 'GCN':
        model = GCN(in_feats=dim_list, hid_feats=64, n_layers=args.n_layers, dropout=args.dropout,
                    decoder=args.decoder)
    elif args.model == "GAT":
        heads = args.n_heads * args.n_layers + [1]
        model = GAT(in_feats=dim_list, hid_feats=64, n_layers=args.n_layers, heads=heads,
                    dropout=args.dropout, decoder=args.decoder)
    else:
        exit('please input true model_type within [GCN, GAT]')
    test_score, random_test_score = defaultdict(list), defaultdict(list)
    eval_types = list(dl.links_test['data'].keys())
    for eval_type in eval_types:
        train_data = hom_data(dl, eval_type)
        print(f'dataset: {data_name}, eval_type: {eval_type}')
        model = model.to(device)
        random_score, score = train_gnn(edge_list=edge_list, feat_list=feat_list, train_data=train_data, args=args,
                                   model=model,
                                   dl=dl,
                                        eval_type=eval_type)
        test_score['roc_auc'].append(score['roc_auc'])
        test_score['MRR'].append(score['MRR'])
        random_test_score['roc_auc'].append(random_score['roc_auc'])
        random_test_score['MRR'].append(random_score['MRR'])

    print(f'Sec neigh test: score list: {test_score}')
    print(f'Random test: score list: {random_test_score}')

    for s in test_score.keys():
        test_score[s] = np.round(np.mean(test_score[s]),4)
    print(f'Sec neigh test:average score: {test_score}')

    for s in random_test_score.keys():
        random_test_score[s] = np.round(np.mean(random_test_score[s]),4)
    print(f'Random test: average score: {random_test_score}')
