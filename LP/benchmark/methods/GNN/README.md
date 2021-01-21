in dir `GNN`: 
* python homoGNN.py --data amazon   
* python homoGNN.py --data youtube  --model GAT --edge_sample_ratio 0.5 --test_with_CPU True  --n_heads [1]

train history  
batch-size=10  
Sec neigh test:average score: {'auc_score': 0.8751442534978069, 'roc_auc': 0.5836778244268963, 'F1': 0.7869757174392935, 'MRR': 0.9237664181753639}
Random test: average score: {'auc_score': 0.8295940337300879, 'roc_auc': 0.763620719772891, 'F1': 0.743441993195039, 'MRR': 0.9845429789119885}

batch-size=50
Sec neigh test:average score: {'auc_score': 0.8799123893425139, 'roc_auc': 0.5965887050513258, 'F1': 0.7716060443170281, 'MRR': 0.9292687255946043}
Random test: average score: {'auc_score': 0.8526468205992548, 'roc_auc': 0.7828092471747066, 'F1': 0.7464609695498932, 'MRR': 0.9926856386720603}

batch-size=100
Sec neigh test:average score: {'auc_score': 0.8847749103595013, 'roc_auc': 0.6111053466497554, 'F1': 0.7987539300239407, 'MRR': 0.9345047923322684}
Random test: average score: {'auc_score': 0.8636440392672693, 'roc_auc': 0.8208072802001548, 'F1': 0.7801882008226743, 'MRR': 0.9906412580733667}

batch-size=1k
Sec neigh test:average score: {'auc_score': 0.8785125462027215, 'roc_auc': 0.5893031572098241, 'F1': 0.8046580659892681, 'MRR': 0.9225239616613419}
Random test: average score: {'auc_score': 0.880089546574293, 'roc_auc': 0.8246814918523915, 'F1': 0.7823070732857896, 'MRR': 0.9904182478862991}
~   early_stop/10 batch
Sec neigh test:average score: {'auc_score': 0.8708616540710856, 'roc_auc': 0.5819925511687718, 'F1': 0.8172809701597282, 'MRR': 0.8942137025204118}
Random test: average score: {'auc_score': 0.7696700041198933, 'roc_auc': 0.6048900198379044, 'F1': 0.5636567799891582, 'MRR': 0.9767789991631844}
~      threshold = 0.2
Sec neigh test:average score: {'auc_score': 0.8660700793859999, 'roc_auc': 0.5684991444793697, 'F1': 0.9077694848892696, 'MRR': 0.8848952786652466}
Random test: average score: {'auc_score': 0.7716715262398035, 'roc_auc': 0.6137836858083723, 'F1': 0.6662117021658706, 'MRR': 0.9711491923352945}
~      threshold = 0.3
Sec neigh test:average score: {'auc_score': 0.874177972633797, 'roc_auc': 0.5888394191992583, 'F1': 0.8500641436818474, 'MRR': 0.9014022009229676}
Random test: average score: {'auc_score': 0.8196741998700632, 'roc_auc': 0.723661838667493, 'F1': 0.6215265564544495, 'MRR': 0.9866430509202074}
~      threshold = median
Sec neigh test:average score: {'auc_score': 0.8737317446113733, 'roc_auc': 0.5898792448590038, 'F1': 0.6506105149853678, 'MRR': 0.8962548810791622}
Random test: average score: {'auc_score': 0.8122197689978679, 'roc_auc': 0.6652681442852221, 'F1': 0.6616415861307777, 'MRR': 0.9842849923121489}

batch-size=10k
Sec neigh test:average score: {'auc_score': 0.8618617112307656, 'roc_auc': 0.5672916905777097, 'F1': 0.7982315112540194, 'MRR': 0.8793042243521476}
Random test: average score: {'auc_score': 0.8073750578882525, 'roc_auc': 0.7040701578023945, 'F1': 0.6199745802394809, 'MRR': 0.980289553512397}

batch-size=100k
Sec neigh test:average score: {'auc_score': 0.8650793959796585, 'roc_auc': 0.5632306132130747, 'F1': 0.8293161740647835, 'MRR': 0.8775292864749734}
Random test: average score: {'auc_score': 0.8204892475173973, 'roc_auc': 0.6822746933646473, 'F1': 0.5811678385814736, 'MRR': 0.9773459035639547}

batch-size=epoch(130k)
Sec neigh test:average score: {'auc_score': 0.8692902382207435, 'roc_auc': 0.5696247631050153, 'F1': 0.8368951449490719, 'MRR': 0.886670216542421}
Random test: average score: {'auc_score': 0.8137419005837101, 'roc_auc': 0.6813740425433399, 'F1': 0.5940358615399367, 'MRR': 0.981808260222797}