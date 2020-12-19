from gcn.inits import *
import tensorflow as tf


def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res


class GraphConvolution:
    """Graph convolution layer."""

    def __init__(self, input_dim, output_dim, placeholders, dropout=0., act=tf.nn.relu, bias=False,
                 featureless=False, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.support = placeholders['support']
        self.featureless = featureless
        self.bias = bias
        self.vars = {}

        self.vars['weights'] = glorot([input_dim, output_dim])
        if self.bias:
            self.vars['bias'] = zeros([output_dim])

    def _call(self, inputs):
        x = inputs

        # dropout
        x = tf.nn.dropout(x, 1-self.dropout)

        # convolve
        for i in range(1):
            if not self.featureless:
                pre_sup = dot(x, self.vars['weights'])
            else:
                pre_sup = self.vars['weights']
            support = dot(self.support, pre_sup, sparse=True)
        output = support

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)


def GCN(inputs, dim, drop, A, n_layer):
    placeholders = {'dropout': drop, 'support': A}

    if n_layer == 1:
        return GraphConvolution(dim, dim, placeholders, act=lambda x: x)._call(inputs)

    for _ in range(n_layer-1):
        x = GraphConvolution(dim, dim, placeholders)._call(inputs)
    x = GraphConvolution(dim, dim, placeholders, act=lambda x: x)._call(x)
    return x
