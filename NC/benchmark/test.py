from scripts.data_loader import data_loader

dl = data_loader('./data/Yelp')
print(dl.nodes)
print(dl.links)
print(dl.labels_train)
print(dl.labels_train['data'][dl.labels_train['mask']])
pred = dl.labels_test['data'][dl.labels_test['mask']]
print(dl.evaluate(pred))

train_idx = np.nonzero(dl.labels_train['mask'])[0]
test_idx = np.nonzero(dl.labels_test['mask'])[0]
print(train_idx)
print(train_idx.shape)
print(test_idx)
print(test_idx.shape)