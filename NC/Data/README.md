## Data

We provide 1 HIN benchmark datasets: ```DBLP```,

`todo` Users can retrieve them <a href="">here</a> and unzip the downloaded file to the current folder.

The statistics of each dataset are as follows.

**Dataset** | #node types | #nodes | #link types | #links | #attributes | #attributed nodes | #label types | #labeled nodes
--- | --- | --- | --- | --- | --- | --- | --- | ---
**DBLP** | 4 | 26,128 | 3 | 239,566 | 334/4231/50/20 | ALL | 4 | 4057


Each dataset contains:
- 3 data files (```node.dat```, ```link.dat```, ```label.dat```);
- 2 evaluation files (```label.dat.test```);
- 2 description files (```meta.dat```, ```info.dat```);

### node.dat

- In each line, there are 4 elements (```node_id```,  ```node_type```, ```node_attributes```) separated by ``` ```.
- In ```node_attributes```, attributes are separated by comma (```,```).

### link.dat

- In each line, there are 4 elements (```node_id```, ```node_id```, ```link_type```) separated by ``` ```.  
- `todo`All links are undirected. Each node is connected by at least one link.

### label.dat

- In each line, there are 4 elements (```node_id```, ```node_type```, ```node_label```) separated by ``` ```.
- All labeled nodes are of the same ```node_type```.
- For ```DBLP```, each labeled node only has one label.
- `todo`For unsupervised training, ```label.dat``` and ```label.dat.test``` are merged for five-fold cross validation. For semi-supervised training, ```label.dat``` is used for training and ```label.dat.test``` is used for testing.

### label.dat.test

- In each line, there are 4 elements (```node_id```, ```node_type```, ```node_label```) separated by ``` ```.
- All labeled nodes are of the same ```node_type```.
- Number of labeled nodes in ```label.dat.test``` = One fourth of the number of labeled nodes in ```label.dat```.
- For ```DBLP```, each labeled node only has one label. 
- `todo`For unsupervised training, ```label.dat``` and ```label.dat.test``` are merged for five-fold cross validation. For semi-supervised training, ```label.dat``` is used for training and ```label.dat.test``` is used for testing.

### meta.note

- This file describes the number of instances of each node type, link type, and label type in the corresponding dataset.

### info.note

- This file describes the meaning of each node type, link type, and label type in the corresponding dataset.
