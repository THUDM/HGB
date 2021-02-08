import torch
import torch.optim as optim
torch.set_num_threads(2)
from args import read_args
from torch.autograd import Variable
import numpy as np
import random
import pickle
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


import data_generator
import tools
import sys
sys.path.append('../../')
from scripts.data_loader import data_loader


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Use device: {device}')

if __name__ == '__main__':
    args = read_args()
    print("------arguments-------")
    for k, v in vars(args).items():
        print(k + ': ' + str(v))

    temp_dir = os.path.join(sys.path[0], 'temp')
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    model_save_dir = os.path.join(sys.path[0], 'model_save')
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    # fix random seed
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)

    dl_pickle_f=os.path.join(temp_dir, 'dl_pickle')
    if os.path.exists(dl_pickle_f):
        dl = pickle.load(open(dl_pickle_f, 'rb'))
        print(f'Info: load DBLP from {dl_pickle_f}')
    else:
        dl = data_loader('../../data/DBLP')
        pickle.dump(dl, open(dl_pickle_f, 'wb'))
        print(f'Info: load DBLP from original data and generate {dl_pickle_f}')

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
    for i in range(len(feature_list)):
        feature_list[i] = torch.from_numpy(np.array(feature_list[i])).float().to(device)

    '''het_neigh_train.txt中的top K(10, 10, 10, 3)'''
    model = tools.HetAgg(args, feature_list, input_data.a_neigh_list_train, input_data.p_neigh_list_train,
                         input_data.t_neigh_list_train, input_data.v_neigh_list_train, input_data.a_train_id_list,
                         input_data.p_train_id_list, input_data.t_train_id_list, input_data.v_train_id_list, dl).to(
        device)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optim = optim.Adam(parameters, lr=args.lr, weight_decay=0)
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
        min_len = 1e10
        for ii in range(len(triple_list)):
            if len(triple_list[ii]) < min_len:
                min_len = len(triple_list[ii])
        batch_n = int(min_len / mini_batch_s)
        print(f'Info: batch_n = {batch_n}')
        for k in range(batch_n):
            c_out = torch.zeros([len(triple_list), mini_batch_s, embed_d])
            p_out = torch.zeros([len(triple_list), mini_batch_s, embed_d])
            n_out = torch.zeros([len(triple_list), mini_batch_s, embed_d])

            for triple_index in range(len(triple_list)):
                triple_list_temp = triple_list[triple_index]
                triple_list_batch = triple_list_temp[k * mini_batch_s: (k + 1) * mini_batch_s]
                c_out_temp, p_out_temp, n_out_temp = model(triple_list_batch, triple_index)

                c_out[triple_index] = c_out_temp
                p_out[triple_index] = p_out_temp
                n_out[triple_index] = n_out_temp

            loss = tools.cross_entropy_loss(c_out, p_out, n_out, embed_d)

            optim.zero_grad()
            loss.backward()
            optim.step()

            if k % 50 == 0:
                print(f"Train: loss= {loss}")

        if iter_i % args.save_model_freq == 0:
            torch.save(model.state_dict(), os.path.join(sys.path[0], 'model_save', f"HetGNN_DBLP_{str(iter_i)}.pt"))
            # save embeddings for evaluation
            model.save_embed(os.path.join(temp_dir, f'node_embedding-{iter_i}.txt'))
            print(f'Info: save model and  node_embedding-{iter_i}.txt done')
        print('Info: iteration ' + str(iter_i) + ' finish.')

    exit('End.')
