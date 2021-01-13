"""
define model
"""
weight_size = eval(args.layer_size)
num_layers = len(weight_size) - 2
heads = [args.heads] * num_layers + [1]
model = myGAT(config['n_users']+config['n_entities'], args.kge_size, config['n_relations']*2+1, args.embed_size, weight_size[-2], weight_size[-1], num_layers, heads, F.elu, 0.1, 0., 0.05, False, pretrain=pretrain_data, alpha=1.0).cuda()


"""
build feed input
"""
edge2type = {}
for i,mat in enumerate(data_generator.lap_list):
    for u,v in zip(*mat.nonzero()):
        edge2type[(u,v)] = i
for i in range(data_generator.n_users+data_generator.n_entities):
    edge2type[(i,i)] = len(data_generator.lap_list)

adjM = sum(data_generator.lap_list)
adjM[adjM>1.] = 1.
print(len(adjM.nonzero()[0]))
g = dgl.from_scipy(adjM, eweight_name='weight')
g = dgl.remove_self_loop(g) # these two lines are vital, because we want self-loop to be the last edges
g = dgl.add_self_loop(g)
g.edata['weight'][g.edata['weight']==0.] = 1.
e_feat = []
edge2id = {}
for u, v in zip(*g.edges()):
    u = u.item()
    v = v.item()
    if u == v:
        break
    e_feat.append(edge2type[(u,v)])
    edge2id[(u,v)] = len(edge2id)
no_self_loop = len(e_feat)
for i in range(data_generator.n_users+data_generator.n_entities):
    e_feat.append(edge2type[(i,i)])
    edge2id[(i,i)] = len(edge2id)
self_loop = len(e_feat) - no_self_loop
must = torch.tensor([True]*self_loop)
e_feat = torch.tensor(e_feat, dtype=torch.long)


"""
call model
"""
model(g, e_feat)