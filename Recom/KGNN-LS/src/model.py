import tensorflow as tf
import numpy as np
import scipy.sparse as sp
from aggregators import SumAggregator, ConcatAggregator, NeighborAggregator, LabelAggregator
from sklearn.metrics import f1_score, roc_auc_score


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj_bias(adj, norm=False):
    num_nodes = adj.shape[0]
    adj = adj + sp.eye(num_nodes)  # self-loop
    adj[adj > 0.0] = 1.0
    if not sp.isspmatrix_coo(adj):
        adj = adj.tocoo()
    if norm:
        adj = normalize_adj(adj)
    adj = adj.astype(np.float32)
    # This is where I made a mistake, I used (adj.row, adj.col) instead
    indices = np.vstack((adj.col, adj.row)).transpose()
    return tf.SparseTensor(indices=indices, values=adj.data, dense_shape=adj.shape)


class KGCN(object):
    def __init__(self, args, n_user, n_entity, n_relation, adj_entity, adj_relation):
        self._parse_args(args, adj_entity, adj_relation)
        self._build_inputs()
        self._build_model(n_user, n_entity, n_relation)
        self._build_train()

    @staticmethod
    def get_initializer():
        return tf.contrib.layers.xavier_initializer()

    def _parse_args(self, args, adj_entity, adj_relation):
        # [entity_num, neighbor_sample_size]
        self.adj_entity = adj_entity
        self.adj_relation = adj_relation

        self.n_iter = args.n_iter
        self.batch_size = args.batch_size
        self.n_neighbor = args.neighbor_sample_size
        self.dim = args.dim
        self.l2_weight = args.l2_weight
        self.lr = args.lr
        if args.aggregator == 'sum':
            self.aggregator_class = SumAggregator
        elif args.aggregator == 'concat':
            self.aggregator_class = ConcatAggregator
        elif args.aggregator == 'neighbor':
            self.aggregator_class = NeighborAggregator
        else:
            raise Exception("Unknown aggregator: " + args.aggregator)
        if args.model not in ['gcn', 'gat', 'kgcn']:
            raise Exception("model type should be in ['gcn', 'gat', 'kgcn']")
        self.model_type = args.model

    def _build_inputs(self):
        self.user_indices = tf.placeholder(
            dtype=tf.int64, shape=[None], name='user_indices')
        self.item_indices = tf.placeholder(
            dtype=tf.int64, shape=[None], name='item_indices')
        self.labels = tf.placeholder(
            dtype=tf.float32, shape=[None], name='labels')

    def _build_model(self, n_user, n_entity, n_relation):
        self.user_emb_matrix = tf.get_variable(
            shape=[n_user, self.dim], initializer=KGCN.get_initializer(), name='user_emb_matrix')
        self.entity_emb_matrix = tf.get_variable(
            shape=[n_entity, self.dim], initializer=KGCN.get_initializer(), name='entity_emb_matrix')
        model_type = self.model_type
        if model_type == 'kgcn':
            self.relation_emb_matrix = tf.get_variable(
                shape=[n_relation, self.dim], initializer=KGCN.get_initializer(), name='relation_emb_matrix')

        # [batch_size, dim]
        self.user_embeddings = tf.nn.embedding_lookup(
            self.user_emb_matrix, self.user_indices)

        # entities is a list of i-iter (i = 0, 1, ..., n_iter) neighbors for the batch of items
        # dimensions of entities:
        # {[batch_size, 1], [batch_size, n_neighbor], [batch_size, n_neighbor^2], ..., [batch_size, n_neighbor^n_iter]}
        if model_type == 'kgcn':
            entities, relations = self.get_neighbors(self.item_indices)

            # [batch_size, dim]
            self.item_embeddings, self.aggregators = self.aggregate(
                entities, relations)  # Change to GAT and embedding_lookup
        else:
            edges = []
            for i in range(n_entity):
                nei = list(set(self.adj_entity[i].tolist()))
                for n in nei:
                    edges.append((i, n))
                    edges.append((n, i))
            edges = np.array(list(set(edges)))
            sp_mat = sp.coo_matrix(
                ([1.0]*edges.shape[0], (edges[:, 0], edges[:, 1])), shape=(n_entity, n_entity))
        if model_type == 'gat':
            from utils.sp_gat import SpGAT
            self.attn_drop = tf.placeholder(dtype=tf.float32, shape=())
            self.ffd_drop = tf.placeholder(dtype=tf.float32, shape=())
            self.is_train = tf.placeholder(dtype=tf.bool, shape=())
            sp_tensor = preprocess_adj_bias(sp_mat)
            self.entity_emb = SpGAT.inference(tf.expand_dims(self.entity_emb_matrix, axis=0), self.dim, n_entity,
                                              self.is_train, self.attn_drop, self.ffd_drop, sp_tensor, [self.dim]*(self.n_iter-1), [8]*(self.n_iter-1)+[1])
            self.item_embeddings = tf.nn.embedding_lookup(
                tf.squeeze(self.entity_emb, axis=0), self.item_indices)
        elif model_type == 'gcn':
            from gcn.layers import GCN
            self.drop = tf.placeholder(dtype=tf.float32, shape=())
            sp_tensor = preprocess_adj_bias(sp_mat, norm=True)
            self.entity_emb = GCN(self.entity_emb_matrix,
                                  self.dim, self.drop, sp_tensor, self.n_iter)
            self.item_embeddings = tf.nn.embedding_lookup(
                self.entity_emb, self.item_indices)

        # [batch_size]
        self.scores = tf.reduce_sum(
            self.user_embeddings * self.item_embeddings, axis=1)
        self.scores_normalized = tf.sigmoid(self.scores)

    def get_neighbors(self, seeds):
        seeds = tf.expand_dims(seeds, axis=1)
        entities = [seeds]
        relations = []
        for i in range(self.n_iter):
            neighbor_entities = tf.reshape(
                tf.gather(self.adj_entity, entities[i]), [self.batch_size, -1])
            neighbor_relations = tf.reshape(
                tf.gather(self.adj_relation, entities[i]), [self.batch_size, -1])
            entities.append(neighbor_entities)
            relations.append(neighbor_relations)
        return entities, relations

    def aggregate(self, entities, relations):
        aggregators = []  # store all aggregators
        entity_vectors = [tf.nn.embedding_lookup(
            self.entity_emb_matrix, i) for i in entities]
        relation_vectors = [tf.nn.embedding_lookup(
            self.relation_emb_matrix, i) for i in relations]

        for i in range(self.n_iter):
            if i == self.n_iter - 1:
                aggregator = self.aggregator_class(
                    self.batch_size, self.dim, act=tf.nn.tanh)
            else:
                aggregator = self.aggregator_class(self.batch_size, self.dim)
            aggregators.append(aggregator)

            entity_vectors_next_iter = []
            for hop in range(self.n_iter - i):
                shape = [self.batch_size, -1, self.n_neighbor, self.dim]
                vector = aggregator(self_vectors=entity_vectors[hop],
                                    neighbor_vectors=tf.reshape(
                                        entity_vectors[hop + 1], shape),
                                    neighbor_relations=tf.reshape(
                                        relation_vectors[hop], shape),
                                    user_embeddings=self.user_embeddings)
                entity_vectors_next_iter.append(vector)
            entity_vectors = entity_vectors_next_iter

        res = tf.reshape(entity_vectors[0], [self.batch_size, self.dim])

        return res, aggregators

    def _build_train(self):
        self.base_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=self.labels, logits=self.scores))
        vvars = tf.trainable_variables()
        self.l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in vvars if v.name not
                                 in ['bias', 'gamma', 'b', 'g', 'beta']])  # * l2_coef
        # self.l2_loss = tf.nn.l2_loss(self.user_emb_matrix) + tf.nn.l2_loss(
        #    self.entity_emb_matrix) + tf.nn.l2_loss(self.relation_emb_matrix)
        # for aggregator in self.aggregators:
        #    self.l2_loss = self.l2_loss + tf.nn.l2_loss(aggregator.weights)
        self.loss = self.base_loss + self.l2_weight * self.l2_loss

        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def train(self, sess, feed_dict):
        return sess.run([self.optimizer, self.loss], feed_dict)

    def eval(self, sess, feed_dict):
        labels, scores = sess.run(
            [self.labels, self.scores_normalized], feed_dict)
        auc = roc_auc_score(y_true=labels, y_score=scores)
        scores[scores >= 0.5] = 1
        scores[scores < 0.5] = 0
        f1 = f1_score(y_true=labels, y_pred=scores)
        return auc, f1

    def get_scores(self, sess, feed_dict):
        return sess.run([self.item_indices, self.scores_normalized], feed_dict)


class KGNN_LS(object):
    def __init__(self, args, n_user, n_entity, n_relation, adj_entity, adj_relation, interaction_table, offset):
        self._parse_args(args, adj_entity, adj_relation,
                         interaction_table, offset)
        self._build_inputs()
        self._build_model(n_user, n_entity, n_relation)
        self._build_train()

    @staticmethod
    def get_initializer():
        return tf.contrib.layers.xavier_initializer()

    def _parse_args(self, args, adj_entity, adj_relation, interaction_table, offset):
        # [entity_num, neighbor_sample_size]
        self.adj_entity = adj_entity
        self.adj_relation = adj_relation

        # LS regularization
        self.interaction_table = interaction_table
        self.offset = offset
        self.ls_weight = args.ls_weight

        self.n_iter = args.n_iter
        self.batch_size = args.batch_size
        self.n_neighbor = args.neighbor_sample_size
        self.dim = args.dim
        self.l2_weight = args.l2_weight
        self.lr = args.lr
        self.model_type = args.model

    def _build_inputs(self):
        self.user_indices = tf.placeholder(
            dtype=tf.int64, shape=[None], name='user_indices')
        self.item_indices = tf.placeholder(
            dtype=tf.int64, shape=[None], name='item_indices')
        self.labels = tf.placeholder(
            dtype=tf.float32, shape=[None], name='labels')

    def _build_model(self, n_user, n_entity, n_relation):
        self.user_emb_matrix = tf.get_variable(
            shape=[n_user, self.dim], initializer=KGNN_LS.get_initializer(), name='user_emb_matrix')
        self.entity_emb_matrix = tf.get_variable(
            shape=[n_entity, self.dim], initializer=KGNN_LS.get_initializer(), name='entity_emb_matrix')
        self.relation_emb_matrix = tf.get_variable(
            shape=[n_relation, self.dim], initializer=KGNN_LS.get_initializer(), name='relation_emb_matrix')

        # [batch_size, dim]
        self.user_embeddings = tf.nn.embedding_lookup(
            self.user_emb_matrix, self.user_indices)

        # entities is a list of i-iter (i = 0, 1, ..., n_iter) neighbors for the batch of items
        # dimensions of entities:
        # {[batch_size, 1], [batch_size, n_neighbor], [batch_size, n_neighbor^2], ..., [batch_size, n_neighbor^n_iter]}
        entities, relations = self.get_neighbors(self.item_indices)

        # [batch_size, dim]
        self.item_embeddings, self.aggregators = self.aggregate(
            entities, relations)

        # LS regularization
        self._build_label_smoothness_loss(entities, relations)

        # [batch_size]
        self.scores = tf.reduce_sum(
            self.user_embeddings * self.item_embeddings, axis=1)
        self.scores_normalized = tf.sigmoid(self.scores)

    def get_neighbors(self, seeds):
        seeds = tf.expand_dims(seeds, axis=1)
        entities = [seeds]
        relations = []
        for i in range(self.n_iter):
            neighbor_entities = tf.reshape(
                tf.gather(self.adj_entity, entities[i]), [self.batch_size, -1])
            neighbor_relations = tf.reshape(
                tf.gather(self.adj_relation, entities[i]), [self.batch_size, -1])
            entities.append(neighbor_entities)
            relations.append(neighbor_relations)
        return entities, relations

    # feature propagation
    def aggregate(self, entities, relations):
        aggregators = []  # store all aggregators
        entity_vectors = [tf.nn.embedding_lookup(
            self.entity_emb_matrix, i) for i in entities]
        relation_vectors = [tf.nn.embedding_lookup(
            self.relation_emb_matrix, i) for i in relations]

        for i in range(self.n_iter):
            if i == self.n_iter - 1:
                aggregator = SumAggregator(
                    self.batch_size, self.dim, act=tf.nn.tanh)
            else:
                aggregator = SumAggregator(self.batch_size, self.dim)
            aggregators.append(aggregator)

            entity_vectors_next_iter = []
            for hop in range(self.n_iter - i):
                shape = [self.batch_size, -1, self.n_neighbor, self.dim]
                vector = aggregator(self_vectors=entity_vectors[hop],
                                    neighbor_vectors=tf.reshape(
                                        entity_vectors[hop + 1], shape),
                                    neighbor_relations=tf.reshape(
                                        relation_vectors[hop], shape),
                                    user_embeddings=self.user_embeddings,
                                    masks=None)
                entity_vectors_next_iter.append(vector)
            entity_vectors = entity_vectors_next_iter

        res = tf.reshape(entity_vectors[0], [self.batch_size, self.dim])

        return res, aggregators

    # LS regularization
    def _build_label_smoothness_loss(self, entities, relations):

        # calculate initial labels; calculate updating masks for label propagation
        entity_labels = []
        # True means the label of this item is reset to initial value during label propagation
        reset_masks = []
        holdout_item_for_user = None

        for entities_per_iter in entities:
            # [batch_size, 1]
            users = tf.expand_dims(self.user_indices, 1)
            # [batch_size, n_neighbor^i]
            user_entity_concat = users * self.offset + entities_per_iter

            # the first one in entities is the items to be held out
            if holdout_item_for_user is None:
                holdout_item_for_user = user_entity_concat

            # [batch_size, n_neighbor^i]
            initial_label = self.interaction_table.lookup(user_entity_concat)
            # False if the item is held out
            holdout_mask = tf.cast(
                holdout_item_for_user - user_entity_concat, tf.bool)
            # True if the entity is a labeled item
            reset_mask = tf.cast(initial_label - tf.constant(0.5), tf.bool)
            reset_mask = tf.logical_and(
                reset_mask, holdout_mask)  # remove held-out items
            initial_label = tf.cast(holdout_mask, tf.float32) * initial_label + tf.cast(
                tf.logical_not(holdout_mask), tf.float32) * tf.constant(0.5)  # label initialization

            reset_masks.append(reset_mask)
            entity_labels.append(initial_label)
        # we do not need the reset_mask for the last iteration
        reset_masks = reset_masks[:-1]

        # label propagation
        relation_vectors = [tf.nn.embedding_lookup(
            self.relation_emb_matrix, i) for i in relations]
        aggregator = LabelAggregator(self.batch_size, self.dim)
        for i in range(self.n_iter):
            entity_labels_next_iter = []
            for hop in range(self.n_iter - i):
                vector = aggregator(self_vectors=entity_labels[hop],
                                    neighbor_vectors=tf.reshape(
                                        entity_labels[hop + 1], [self.batch_size, -1, self.n_neighbor]),
                                    neighbor_relations=tf.reshape(
                                        relation_vectors[hop], [self.batch_size, -1, self.n_neighbor, self.dim]),
                                    user_embeddings=self.user_embeddings,
                                    masks=reset_masks[hop])
                entity_labels_next_iter.append(vector)
            entity_labels = entity_labels_next_iter

        self.predicted_labels = tf.squeeze(entity_labels[0], axis=-1)

    def _build_train(self):
        # base loss
        self.base_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=self.labels, logits=self.scores))

        # L2 loss
        self.l2_loss = tf.nn.l2_loss(self.user_emb_matrix) + tf.nn.l2_loss(
            self.entity_emb_matrix) + tf.nn.l2_loss(self.relation_emb_matrix)
        for aggregator in self.aggregators:
            self.l2_loss = self.l2_loss + tf.nn.l2_loss(aggregator.weights)
        self.loss = self.base_loss + self.l2_weight * self.l2_loss

        # LS loss
        self.ls_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=self.labels, logits=self.predicted_labels))
        self.loss += self.ls_weight * self.ls_loss

        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def train(self, sess, feed_dict):
        return sess.run([self.optimizer, self.loss], feed_dict)

    def eval(self, sess, feed_dict):
        labels, scores = sess.run(
            [self.labels, self.scores_normalized], feed_dict)
        auc = roc_auc_score(y_true=labels, y_score=scores)
        scores[scores >= 0.5] = 1
        scores[scores < 0.5] = 0
        f1 = f1_score(y_true=labels, y_pred=scores)
        return auc, f1

    def get_scores(self, sess, feed_dict):
        return sess.run([self.item_indices, self.scores_normalized], feed_dict)
