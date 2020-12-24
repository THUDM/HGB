import torch
from sklearn.metrics import f1_score
import dgl
from utils import load_data, EarlyStopping
import torch.nn.functional as F

def score(logits, labels):
    _, indices = torch.max(logits, dim=1)
    prediction = indices.long().cpu().numpy()
    labels = labels.cpu().numpy()

    accuracy = (prediction == labels).sum() / len(prediction)
    micro_f1 = f1_score(labels, prediction, average='micro')
    macro_f1 = f1_score(labels, prediction, average='macro')

    return accuracy, micro_f1, macro_f1

def evaluate(model, g, features, labels, mask, loss_func):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
    loss = loss_func(logits[mask], labels[mask])
    accuracy, micro_f1, macro_f1 = score(logits[mask], labels[mask])

    return loss, accuracy, micro_f1, macro_f1

def main(args):
    # If args['hetero'] is True, g would be a heterogeneous graph.
    # Otherwise, it will be a list of homogeneous graphs.
    g, features, labels, num_classes, train_idx, val_idx, test_idx, train_mask, \
    val_mask, test_mask, meta_paths = load_data(args['dataset'], feat_type=0)

    if args['model'] != 'han':
        g = dgl.to_homogeneous(g)
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)
        N = g.number_of_nodes()
        N_p = features.size()[0]
        D = features.size()[1]
        new_labels = torch.zeros(N-N_p, dtype=torch.long)
        labels = torch.cat((new_labels, labels))
        new_features = torch.zeros(N-N_p, D)
        features = torch.cat((new_features, features), 0)
        new_mask = torch.zeros(N-N_p, dtype=torch.uint8)
        train_mask = torch.cat((new_mask, train_mask))
        val_mask = torch.cat((new_mask, val_mask))
        test_mask = torch.cat((new_mask, test_mask))

    if hasattr(torch, 'BoolTensor'):
        train_mask = train_mask.bool()
        val_mask = val_mask.bool()
        test_mask = test_mask.bool()

    features = features.to(args['device'])
    labels = labels.to(args['device'])
    train_mask = train_mask.to(args['device'])
    val_mask = val_mask.to(args['device'])
    test_mask = test_mask.to(args['device'])

    if args['hetero']:
        if args['model'] == 'han':
            from model_hetero import HAN
            if args['dataset']=='DBLP':
                model = HAN(
                    meta_paths=meta_paths,
                            in_size=features.shape[1],
                            hidden_size=args['hidden_units'],
                            out_size=num_classes,
                            num_heads=args['num_heads'],
                            dropout=args['dropout']).to(args['device'])
        elif args['model'] == 'gcn':
            from GNN import GCN
            model = GCN(D, args['hidden_units'], num_classes, args['num_layers'], F.relu, args['dropout']).to(args['device'])
        elif args['model'] == 'gat':
            from GNN import GAT
            heads = args['num_heads']*args['num_layers'] + [1]
            slope = 0.1
            model = GAT(D, args['hidden_units'], num_classes, args['num_layers'], F.elu, args['dropout'], args['dropout'], heads, slope).to(args['device'])
        g = g.to(args['device'])
        
    else:
        from model import HAN
        model = HAN(num_meta_paths=len(g),
                    in_size=features.shape[1],
                    hidden_size=args['hidden_units'],
                    out_size=num_classes,
                    num_heads=args['num_heads'],
                    dropout=args['dropout']).to(args['device'])
        g = [graph.to(args['device']) for graph in g]

    stopper = EarlyStopping(patience=args['patience'])
    loss_fcn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'],
                                 weight_decay=args['weight_decay'])

    for epoch in range(args['num_epochs']):
        model.train()
        logits = model(g, features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc, train_micro_f1, train_macro_f1 = score(logits[train_mask], labels[train_mask])
        val_loss, val_acc, val_micro_f1, val_macro_f1 = evaluate(model, g, features, labels, val_mask, loss_fcn)
        early_stop = stopper.step(val_loss.data.item(), val_acc, model)

        print('Epoch {:d} | Train Loss {:.4f} | Train Micro f1 {:.4f} | Train Macro f1 {:.4f} | '
              'Val Loss {:.4f} | Val Micro f1 {:.4f} | Val Macro f1 {:.4f}'.format(
            epoch + 1, loss.item(), train_micro_f1, train_macro_f1, val_loss.item(), val_micro_f1, val_macro_f1))

        if early_stop:
            break

    stopper.load_checkpoint(model)
    test_loss, test_acc, test_micro_f1, test_macro_f1 = evaluate(model, g, features, labels, test_mask, loss_fcn)
    print('Test loss {:.4f} | Test Micro f1 {:.4f} | Test Macro f1 {:.4f}'.format(
        test_loss.item(), test_micro_f1, test_macro_f1))

if __name__ == '__main__':
    import argparse

    from utils import setup

    parser = argparse.ArgumentParser('HAN')
    parser.add_argument('-s', '--seed', type=int, default=1,
                        help='Random seed')
    parser.add_argument('-ld', '--log-dir', type=str, default='results',
                        help='Dir for saving training results')
    parser.add_argument('--hetero', action='store_true',
                        help='Use metapath coalescing with DGL\'s own dataset')
    parser.add_argument('--model', type=str)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dataset', type=str, default='ACMRaw')
    args = parser.parse_args().__dict__

    args = setup(args)

    main(args)
