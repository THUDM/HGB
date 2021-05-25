from collections import Counter
from sklearn.metrics import f1_score
import numpy as np

class F1:
    def __init__(self, true_file, pred_files):
        self.ture_label = self.load_labels(true_file)
        self.F1_list={'macro':[], 'micro':[]}
        for pred_file in pred_files:
            pred_label = self.load_labels(pred_file)
            ans = self.evaluate_F1(pred_label)
            self.F1_list['macro'].append(ans['macro'])
            self.F1_list['micro'].append(ans['micro'])
        self.F1_mean = {'macro':np.mean(self.F1_list['macro']), 'micro':np.mean(self.F1_list['micro'])}
        self.F1_std = {'macro': np.std(self.F1_list['macro']), 'micro': np.std(self.F1_list['micro'])}

    def load_labels(self, name):
        """
        return labels dict
            num_classes: total number of labels
            total: total number of labeled data
            count: number of labeled data for each node type
            data: a numpy matrix with shape (self.nodes['total'], self.labels['num_classes'])
        """
        labels = {'num_classes': 0, 'total': 0, 'count': Counter(), 'data': {}}
        nc = 0
        with open(name, 'r', encoding='utf-8') as f:
            for line in f:
                th = line.split('\t')
                node_id, node_name, node_type, node_label = int(th[0]), th[1], int(th[2]), list(map(float, th[3].rstrip().split(',')))

                for i in range(len(node_label)):
                    node_label[i]=int(node_label[i])
                    nc = max(nc, node_label[i]+1)
                labels['data'][node_id] = node_label
                labels['count'][node_type] += 1
                labels['total'] += 1
        labels['num_classes'] = nc

        return labels

    def evaluate_F1(self,pred_label):
        y_true, y_pred =[], []
        ans={'macro':0, 'micro':0}
        for k in self.ture_label['data'].keys():
            y_true.append(self.ture_label['data'][k])
            if k not in pred_label['data']:
                return ans
            y_pred.append(pred_label['data'][k])
        ans['macro'] = f1_score(y_true, y_pred, average='macro')
        ans['micro'] = f1_score(y_true, y_pred, average='micro')
        return ans

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate F1 for NC result.")
    parser.add_argument('--true', type=str,
                        help='true label file.')
    parser.add_argument('--pred', nargs='+',
                        help='prediction files.')
    return parser.parse_args()

if __name__ == '__main__':
    # get argument settings.
    args = parse_args()

    com = F1(args.true, args.pred)
    print(com.F1_list)
    print(com.F1_mean)
    print(com.F1_std)