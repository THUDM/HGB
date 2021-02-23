import torch
from sklearn.metrics import f1_score
import dgl
from utils import load_data, EarlyStopping
import torch.nn.functional as F
from model_hetero_multi import HAN
import numpy as np

def evaluate(model, g, features, labels, mask, loss_func, dl):
    return loss


def main(args):
    g, features, labels, num_classes, train_idx, val_idx, test_idx, train_mask, \
    val_mask, test_mask, meta_paths, dl = load_data(args['dataset'], feat_type=0)

    if hasattr(torch, 'BoolTensor'):
        train_mask = train_mask.bool()
        val_mask = val_mask.bool()
        test_mask = test_mask.bool()

    features = features.to(args['device'])
    labels = labels.to(args['device'])
    train_mask = train_mask.to(args['device'])
    val_mask = val_mask.to(args['device'])
    test_mask = test_mask.to(args['device'])

    model = HAN(
        meta_paths=meta_paths,
        in_size=features.shape[1],
        hidden_size=args['hidden_units'],
        out_size=num_classes,
        num_heads=args['num_heads'],
        dropout=args['dropout']).to(args['device'])

    g = g.to(args['device'])

    stopper = EarlyStopping(patience=args['patience'])
    loss_fcn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'],
                                 weight_decay=args['weight_decay'])

    for epoch in range(args['num_epochs']):
        model.train()
        logits = model(g, features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(g, features)
            val_loss = loss_fcn(logits[val_mask], labels[val_mask])
            early_stop = stopper.step(val_loss.data.item(), 1, model)
            print('Epoch {:d} | Train Loss {:.4f} | Val Loss {:.4f} |'.format(
                epoch + 1, loss.item(), val_loss.item()))
        if early_stop:
            break

    stopper.load_checkpoint(model)
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
        test_loss = loss_fcn(logits[test_mask], labels[test_mask])
        test_pred = (logits[test_mask].cpu().numpy() > 0).astype(int)
        test_score = dl.evaluate(test_pred)
        print('Test loss {:.4f} | Test Score {} '.format(
            test_loss.item(), test_score))


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
    parser.add_argument('--dataset', type=str, default='IMDB',
                        choices=['IMDB'])
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=100)
    args = parser.parse_args().__dict__

    args = setup(args)

    main(args)
