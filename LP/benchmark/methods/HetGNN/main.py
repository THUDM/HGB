import torch
import torch.optim as optim

torch.set_num_threads(2)
from args import read_args
from torch.autograd import Variable
import numpy as np
import random
import pickle
import os
import data_generator
import tools
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '9'
sys.path.append(f'../../')
from scripts.data_loader import data_loader

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Use device: {device}')

if __name__ == '__main__':

    args = read_args()
    print("------arguments-------")
    for k, v in vars(args).items():
        print(k + ': ' + str(v))
    data_name = args.data
    temp_dir = os.path.join(sys.path[0], f'{data_name}-temp')
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    model_save_dir = os.path.join(sys.path[0], f'{data_name}-temp')
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    # fix random seed
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)

    data_path = f'../../data/{data_name}'
    dl_pickle_f = os.path.join(data_path, 'dl_pickle')
    if os.path.exists(dl_pickle_f):
        dl = pickle.load(open(dl_pickle_f, 'rb'))
        print(f'Info: load {data_name} from {dl_pickle_f}')
    else:
        dl = data_loader(data_path)
        pickle.dump(dl, open(dl_pickle_f, 'wb'))
        print(f'Info: load {data_name} from original data and generate {dl_pickle_f}')
    # reverse links
    if args.data == "amazon":
        for r_id in dl.links['data'].keys():
            dl.links['data'][r_id] += dl.links['data'][r_id].T
    elif args.data == 'LastFM' or args.data == 'PubMed' or args.data == 'LastFM_magnn':
        r_ids=list(dl.links['data'].keys())
        for r_id in r_ids:
            h_type, t_type = dl.links['meta'][r_id]
            if h_type==t_type:
                continue
            dl.links['data'][-r_id-1]=dl.links['data'][r_id].T
            dl.links['meta'][-r_id-1]=dl.links['meta'][r_id][::-1]
            dl.links['count'][-r_id-1]=dl.links['count'][r_id]
            dl.links['total'] = dl.links['total']+dl.links['data'][r_id].nnz

    input_data = data_generator.input_data(args, dl)
    '''genarate het_neigh_train.txt with walk_restart, and het_random_walk.txt with random walk'''
    het_neigh_train_f = os.path.join(temp_dir, 'het_neigh_train.txt')
    if not os.path.exists(het_neigh_train_f):
        input_data.gen_het_w_walk_restart(het_neigh_train_f)
    het_random_walk_f = os.path.join(temp_dir, 'het_random_walk.txt')
    if not os.path.exists(het_random_walk_f):
        input_data.gen_het_w_walk(het_random_walk_f)
    # exit('Info: generate file done.')

    '''genarate embeds and neighs'''
    input_data.gen_embeds_w_neigh()
    feature_list = input_data.feature_list

    for node_type in feature_list.keys():
        feature_list[node_type] = feature_list[node_type].to(device)

    ''' top K of each node_type in het_neigh_train.txt'''
    model = tools.HetAgg(args, feature_list, neigh_list_train=input_data.neigh_list_train,
                         train_id_list=input_data.train_id_list, dl=dl, input_data=input_data, device=device).to(
        device)

    optim = optim.Adam([{'params': model.parameters()}], lr=args.lr, weight_decay=0)
    model.init_weights()

    print('model training ...')
    if args.checkpoint != '':
        model.load_state_dict(torch.load(args.checkpoint))

    model.train()
    mini_batch_s = args.mini_batch_s
    embed_d = args.embed_d

    '''train model and get embed file'''
    for iter_i in range(args.train_iter_n):
        print(f'Info: iteration {iter_i}  out of {args.train_iter_n}')
        triple_list = input_data.sample_het_walk_triple()
        triple_keys = list(triple_list.keys())
        for k in triple_keys:
            if len(triple_list[k]) < 1000:
                triple_list.pop(k)
        min_len = 1e10
        for ii in triple_list.values():
            if len(ii) < min_len:
                min_len = len(ii)
        batch_n = int(min_len / mini_batch_s)
        print(f'Info: batch_n = {batch_n}')
        for k in range(batch_n):
            c_out = torch.zeros([len(triple_list), mini_batch_s, embed_d])
            p_out = torch.zeros([len(triple_list), mini_batch_s, embed_d])
            n_out = torch.zeros([len(triple_list), mini_batch_s, embed_d])

            for triple_pair_index, triple_pair in enumerate(triple_list.keys()):
                triple_list_batch = triple_list[triple_pair][k * mini_batch_s: (k + 1) * mini_batch_s]
                c_out_temp, p_out_temp, n_out_temp = model(triple_list_batch, triple_pair)

                c_out[triple_pair_index] = c_out_temp
                p_out[triple_pair_index] = p_out_temp
                n_out[triple_pair_index] = n_out_temp

            loss = tools.cross_entropy_loss(c_out, p_out, n_out, embed_d)

            optim.zero_grad()
            loss.backward()
            optim.step()

            if k % 50 == 0:
                print(f"Train: loss= {loss}")

        if iter_i % args.save_model_freq == 0:
            torch.save(model.state_dict(),
                       os.path.join(model_save_dir, f"iter-{str(iter_i)}.pt"))
            # save embeddings for evaluation
            model.save_embed(os.path.join(temp_dir, f'node_embedding{iter_i}.txt'))
            print(f'Info: save model and  node_embedding{iter_i}.txt done')
        print(f'Info: iteration {iter_i} finish.')

    exit("HetGNN\'s traing end.")
