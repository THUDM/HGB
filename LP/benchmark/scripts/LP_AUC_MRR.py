from collections import Counter
from sklearn.metrics import f1_score,roc_auc_score
import numpy as np
import os
from collections import defaultdict

class AUC_MRR:
    def __init__(self, true_file, pred_files):
        self.links_true = self.load_links(true_file)

        self.AUC_list=[]
        self.MRR_list=[]
        for pred_file in pred_files:
            self.links_test = self.load_links(pred_file)
            ans = self.evaluate_AUC_MRR()
            self.AUC_list.append(ans['AUC'])
            self.MRR_list.append(ans['MRR'])
        self.AUC_mean = np.mean(self.AUC_list)
        self.MRR_mean = np.mean(self.MRR_list)
        self.AUC_std = np.std(self.AUC_list)
        self.MRR_std = np.std(self.MRR_list)

    def load_links(self, file_name):
        """
        return links dict
            total: total number of links
            count: a dict of int, number of links for each type
            data: a list, where each link type have a dict with {(head_id, tail_id): confidence}
        """
        links = {'total': 0, 'count': Counter(), 'data': defaultdict(dict)}
        with open(file_name, 'r', encoding='utf-8') as f:
            for line in f:
                th = line.split('\t')
                h_id, t_id, r_id, link_weight = int(th[0]), int(th[1]), int(th[2]), float(th[3])
                links['data'][r_id][(h_id, t_id)] = link_weight
                links['count'][r_id] += 1
                links['total'] += 1
        return links


    def evaluate_AUC_MRR(self):
        ans_all_link_type = {'AUC':[],'MRR':[]}
        for link_type in  self.links_test['count'].keys():
            edge_list = {0: [], 1: []}
            confidence = []
            labels = []
            for h_t in self.links_test['data'][link_type].keys():
                edge_list[0].append(h_t[0])
                edge_list[1].append(h_t[1])
                confidence.append(self.links_test['data'][link_type][h_t])
                if h_t in self.links_true['data'][link_type]:
                    labels.append(1.0)
                else:
                    labels.append(0.)
            ans = self.evaluate(edge_list, confidence, labels)
            ans_all_link_type['AUC'].append(ans['AUC'])
            ans_all_link_type['MRR'].append(ans['MRR'])
        ans_all_link_type['AUC'] = np.mean(ans_all_link_type['AUC'])
        ans_all_link_type['MRR'] = np.mean(ans_all_link_type['MRR'])
        return ans_all_link_type

    @staticmethod
    def evaluate(edge_list, confidence, labels):
        """
        :param edge_list: shape(2, edge_num)
        :param confidence: shape(edge_num,)
        :param labels: shape(edge_num,)
        :return: dict with all scores we need
        """
        confidence = np.array(confidence)
        labels = np.array(labels)
        roc_auc = roc_auc_score(labels, confidence)
        mrr_list, cur_mrr = [], 0
        t_dict, labels_dict, conf_dict = defaultdict(list), defaultdict(list), defaultdict(list)
        for i, h_id in enumerate(edge_list[0]):
            t_dict[h_id].append(edge_list[1][i])
            labels_dict[h_id].append(labels[i])
            conf_dict[h_id].append(confidence[i])
        for h_id in t_dict.keys():
            conf_array = np.array(conf_dict[h_id])
            rank = np.argsort(-conf_array)
            sorted_label_array = np.array(labels_dict[h_id])[rank]
            pos_index = np.where(sorted_label_array == 1)[0]
            if len(pos_index) == 0:
                continue
            pos_min_rank = np.min(pos_index)
            cur_mrr = 1 / (1 + pos_min_rank)
            mrr_list.append(cur_mrr)
        mrr = np.mean(mrr_list)

        return {'AUC': roc_auc, 'MRR': mrr}

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate AUC and MRR for LP result.")
    parser.add_argument('--true', type=str,
                        help='true link file.')
    parser.add_argument('--pred', nargs='+',
                        help='prediction files.')

    return parser.parse_args()

if __name__ == '__main__':
    # get argument settings.
    args = parse_args()

    res = AUC_MRR(args.true, args.pred)
    print(f'AUC list: {res.AUC_list}, MRR list: {res.MRR_list}')
    print(f'AUC mean: {res.AUC_mean}, MRR mean: {res.MRR_mean}')
    print(f'AUC std: {res.AUC_std}, MRR std: {res.MRR_std}')
