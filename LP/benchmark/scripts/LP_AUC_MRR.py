from collections import Counter
from sklearn.metrics import f1_score,roc_auc_score
import numpy as np
import os
from collections import defaultdict
import sys
import json

class AUC_MRR:
    def __init__(self, data_name, pred_files):
        self.AUC_list = []
        self.MRR_list = []
        if len(pred_files)==0:
            self.AUC_mean = np.mean(0)
            self.MRR_mean = np.mean(0)
            self.AUC_std = np.std(0)
            self.MRR_std = np.std(0)
        else:
            true_file = os.path.join(args.ground_dir, data_name, 'link.dat.test')
            self.links_true = self.load_links(true_file)
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
    parser.add_argument('--pred_zip', type=str, default="lp.zip",
                        help='Compressed pred files.')
    parser.add_argument('--ground_dir', type=str, default="../data",
                        help='Dir of ground files.')
    parser.add_argument('--log', type=str, default="lp.log",
                        help='output file')
    return parser.parse_args()

import zipfile

def extract_zip(zip_path, extract_path):
    zip = zipfile.ZipFile(zip_path, 'r')
    zip.extractall(extract_path)
    return zip.namelist()

def write_log(log_file, log_msg):
    with open(log_file, 'w') as log_handle:
        log_handle.write(log_msg)

def delete_files(files_):
    for f in files_:
        if os.path.exists(f):
            os.remove(f)

if __name__ == '__main__':
    # get argument settings.
    args = parse_args()

    zip_path = args.pred_zip
    if not os.path.exists(zip_path):
        log_msg = 'ERROR: No such zip file!'
        write_log(args.log, log_msg)
        sys.exit()
    extract_path='lp'
    extract_file_list = extract_zip(zip_path, extract_path)
    extract_file_list = [os.path.join(extract_path,f_) for f_ in extract_file_list]

    data_list = ['amazon', 'LastFM', 'PubMed']

    res={}
    detect_data_files = []
    for data_name in data_list:
        pred_files = []
        for i in range(1, 6):
            file_name = os.path.join(extract_path, f'{data_name}_{i}.txt')
            if not os.path.exists(file_name):
                continue
            pred_files.append(file_name)
            detect_data_files.append(file_name)
        if len(pred_files) > 0 and len(pred_files) != 5:
            log_msg = f'ERROR: Please check the size of {data_name} dataset!'
            write_log(args.log, log_msg)
            delete_files(extract_file_list)
            sys.exit()
        res[data_name] = AUC_MRR(data_name, pred_files)
    if len(detect_data_files) == 0:
        log_msg = f'ERROR: No file detected, please confirm that ' \
                  f'the data file is in the top directory of the compressed package!'
        write_log(args.log, log_msg)
        sys.exit()
    delete_files(extract_file_list)

    hgb_score_list = []
    for data_name in data_list:
        hgb_score_list.append(res[data_name].AUC_mean)
        hgb_score_list.append(res[data_name].MRR_mean)
    hgb_score = np.mean(hgb_score_list)

    detail_json = {}
    log_msg = f'{hgb_score}###'
    for data_name in data_list:
        detail_json[data_name] = {}
        detail_json[data_name]['AUC_mean'] = res[data_name].AUC_mean
        detail_json[data_name]['AUC_std'] = res[data_name].AUC_std
        detail_json[data_name]['MRR_mean'] = res[data_name].MRR_mean
        detail_json[data_name]['MRR_std'] = res[data_name].MRR_std
    log_msg += json.dumps(detail_json)
    write_log(args.log, log_msg)
    sys.exit()

