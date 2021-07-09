import json
from collections import Counter
from sklearn.metrics import f1_score
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import os
import sys

class F1:
    def __init__(self, data_name, pred_files, label_classes, args):
        self.label_classes = label_classes
        self.F1_list = {'macro': [], 'micro': []}
        if len(pred_files)==0:
            self.F1_mean = {'macro_mean': 0, 'micro_mean': 0}
            self.F1_std = {'macro_std': 0, 'micro_std': 0}
        else:
            true_file = os.path.join(args.ground_dir, data_name, 'label.dat.test')
            self.ture_label = self.load_labels(true_file)

            for pred_file in pred_files:
                pred_label = self.load_labels(pred_file)
                ans = self.evaluate_F1(pred_label)
                self.F1_list['macro'].append(ans['macro'])
                self.F1_list['micro'].append(ans['micro'])
            self.F1_mean = {'macro_mean':np.mean(self.F1_list['macro']), 'micro_mean':np.mean(self.F1_list['micro'])}
            self.F1_std = {'macro_std': np.std(self.F1_list['macro']), 'micro_std': np.std(self.F1_list['micro'])}

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
                node_id, node_name, node_type, node_label = int(th[0]), th[1], int(th[2]), th[3]
                if node_label.strip()=="":
                    node_label = []
                else:
                    node_label = list(map(float, node_label.rstrip().split(',')))

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
        if self.label_classes>2:
            mlp = MultiLabelBinarizer([i for i in range(self.label_classes)])
            y_pred = mlp.fit_transform(y_pred)
            y_true = mlp.fit_transform(y_true)
        ans['macro'] = f1_score(y_true, y_pred, average='macro')
        ans['micro'] = f1_score(y_true, y_pred, average='micro')
        return ans

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate F1 for NC result.")
    parser.add_argument('--pred_zip', type=str, default="nc.zip",
                        help='Compressed pred files.')
    parser.add_argument('--ground_dir', type=str, default="../data",
                        help='Dir of ground files.')
    parser.add_argument('--log', type=str, default="nc.log",
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
    log_msg = ''
    if not os.path.exists(zip_path):
        log_msg = 'ERROR: No such zip file!'
        write_log(args.log, log_msg)
        sys.exit()
    extract_path = 'nc'
    extract_file_list = extract_zip(zip_path, extract_path)
    extract_file_list = [os.path.join(extract_path, f_) for f_ in extract_file_list]

    data_list = ['DBLP', 'IMDB', 'ACM', 'Freebase']
    class_count = {'DBLP':2, 'IMDB':5,'ACM':2,'Freebase':2}

    res={}
    detect_data_files = []
    for data_name in data_list:
        pred_files = []
        for i in range(1,6):
            file_name = os.path.join(extract_path, f'{data_name}_{i}.txt')
            if not os.path.exists(file_name):
                continue
            pred_files.append(file_name)
            detect_data_files.append(file_name)
        if len(pred_files)>0 and len(pred_files)!=5:
            log_msg = f'ERROR: Please check the size of {data_name} dataset!'
            write_log(args.log, log_msg)
            delete_files(extract_file_list)
            sys.exit()
        res[data_name] = F1(data_name,pred_files,class_count[data_name],args)
    if len(detect_data_files) == 0:
        log_msg = f'ERROR: No file detected, please confirm that ' \
                  f'the data file is in the top directory of the compressed package!'
        write_log(args.log, log_msg)
        sys.exit()
    delete_files(extract_file_list)

    hgb_score_list = []
    for data_name in data_list:
        hgb_score_list.append(res[data_name].F1_mean['macro_mean'])
        hgb_score_list.append(res[data_name].F1_mean['micro_mean'])
    hgb_score = np.mean(hgb_score_list)

    detail_json = {}
    log_msg = f'{hgb_score}###'
    for data_name in data_list:
        detail_json[data_name] = {}
        detail_json[data_name]["F1 mean"] = res[data_name].F1_mean
        detail_json[data_name]["F1 std"] = res[data_name].F1_std
    log_msg += json.dumps(detail_json)
    write_log(args.log, log_msg)
    sys.exit()



