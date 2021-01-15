import tensorflow as tf

import pdb

TINY = 1e-5
class BaseGAttN:
    def loss(logits, labels, nb_classes, class_weights):
        sample_wts = tf.reduce_sum(tf.multiply(tf.one_hot(labels, nb_classes), class_weights), axis=-1)
        xentropy = tf.multiply(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels, logits=logits), sample_wts)
        return tf.reduce_mean(xentropy, name='xentropy_mean')

    def training(loss, lr, l2_coef):
        # weight decay
        vars = tf.trainable_variables()
        lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars if v.name not
                           in ['bias', 'gamma', 'b', 'g', 'beta']]) * l2_coef

        # optimizer
        opt = tf.train.AdamOptimizer(learning_rate=lr)

        # training op
        train_op = opt.minimize(loss+lossL2)
        
        return train_op

    def preshape(logits, labels, nb_classes):
        new_sh_lab = [-1]
        new_sh_log = [-1, nb_classes]
        log_resh = tf.reshape(logits, new_sh_log)
        lab_resh = tf.reshape(labels, new_sh_lab)
        return log_resh, lab_resh

    def confmat(logits, labels):
        preds = tf.argmax(logits, axis=1)
        return tf.confusion_matrix(labels, preds)

##########################
# Adapted from tkipf/gcn #
##########################

    def masked_softmax_cross_entropy(logits, labels, mask):
        """Softmax cross-entropy loss with masking."""
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        mask = tf.cast(mask, dtype=tf.float32)
        mask /= tf.reduce_mean(mask)
        loss *= mask
        return tf.reduce_mean(loss)

    def masked_sigmoid_cross_entropy(logits, labels, mask):
        """Softmax cross-entropy loss with masking."""
        labels = tf.cast(labels, dtype=tf.float32)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
        loss=tf.reduce_mean(loss,axis=1)
        mask = tf.cast(mask, dtype=tf.float32)
        mask /= tf.reduce_mean(mask)
        loss *= mask
        # pdb.set_trace()
        return tf.reduce_mean(loss)

    def masked_accuracy(logits, labels, mask):
        """Accuracy with masking."""
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        accuracy_all = tf.cast(correct_prediction, tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        mask /= tf.reduce_mean(mask)
        accuracy_all *= mask
        return tf.reduce_mean(accuracy_all)

    def micro_f1(logits, labels, mask):
        """Accuracy with masking."""
        predicted = tf.round(tf.nn.sigmoid(logits))
        #pdb.set_trace()
        # Use integers to avoid any nasty FP behaviour
        predicted = tf.cast(predicted, dtype=tf.int32)
        labels = tf.cast(labels, dtype=tf.int32)
        mask = tf.cast(mask, dtype=tf.int32)

        # expand the mask so that broadcasting works ([nb_nodes, 1])
        mask = tf.expand_dims(mask, -1)
        
        # Count true positives, true negatives, false positives and false negatives.
        tp = tf.cast(tf.count_nonzero(predicted * labels * mask), tf.float32)
        tn = tf.cast(tf.count_nonzero((predicted - 1) * (labels - 1) * mask), tf.float32)
        fp = tf.cast(tf.count_nonzero(predicted * (labels - 1) * mask), tf.float32)
        fn = tf.cast(tf.count_nonzero((predicted - 1) * labels * mask), tf.float32)
        
        # Calculate accuracy, precision, recall and F1 score.
        precision = tp / (tp + fp + TINY)
        recall = tp / (tp + fn + TINY)
        fmeasure = (2 * precision * recall) / (precision + recall + TINY)
        #pdb.set_trace()
        return fmeasure


    def micro_f1_onelabel(logits, labels, mask):
        predicted = tf.argmax(tf.nn.softmax(logits), axis=1)
        pre = tf.one_hot(predicted, depth=tf.shape(labels)[-1], dtype=tf.int32)
        labels = tf.cast(labels, dtype=tf.int32)
        # expand the mask so that broadcasting works ([nb_nodes, 1])
        mask = tf.expand_dims(tf.cast(mask, dtype=tf.int32), -1)
        # Count true positives, true negatives, false positives and false negatives.
        tp = tf.cast(tf.count_nonzero(pre * labels * mask), tf.float32)
        tn = tf.cast(tf.count_nonzero((pre - 1) * (labels - 1) * mask), tf.float32)
        fp = tf.cast(tf.count_nonzero(pre * (labels - 1) * mask), tf.float32)
        fn = tf.cast(tf.count_nonzero((pre - 1) * labels * mask), tf.float32)
        
        # Calculate accuracy, precision, recall and F1 score.
        precision = tp / (tp + fp + TINY)
        recall = tp / (tp + fn + TINY)
        fmeasure = (2 * precision * recall) / (precision + recall + TINY)
        return fmeasure


    def macro_f1(logits, labels, mask):
        """Accuracy with masking."""
        predicted = tf.round(tf.nn.sigmoid(logits))

        # Use integers to avoid any nasty FP behaviour
        predicted = tf.cast(predicted, dtype=tf.int32)
        labels = tf.cast(labels, dtype=tf.int32)
        mask = tf.cast(mask, dtype=tf.int32)

        # expand the mask so that broadcasting works ([nb_nodes, 1])
        mask = tf.expand_dims(mask, -1)
        
        # Count true positives, true negatives, false positives and false negatives.
        tp = tf.cast(tf.count_nonzero(predicted * labels * mask, axis=0), tf.float32)
        tn = tf.cast(tf.count_nonzero((predicted - 1) * (labels - 1) * mask, axis=0), tf.float32)
        fp = tf.cast(tf.count_nonzero(predicted * (labels - 1) * mask, axis=0), tf.float32)
        fn = tf.cast(tf.count_nonzero((predicted - 1) * labels * mask, axis=0), tf.float32)
        
        # Calculate accuracy, precision, recall and F1 score.
        precision = tf.reduce_mean(tf.divide(tp, tp + fp + TINY))
        recall = tf.reduce_mean(tf.divide(tp, tp + fn + TINY))
        fmeasure = (2 * precision * recall) / (precision + recall + TINY)
        return fmeasure


    def macro_f1_onelabel(logits, labels, mask):
        predicted = tf.argmax(tf.nn.softmax(logits), axis=1)
        pre = tf.one_hot(predicted, depth=tf.shape(labels)[-1], dtype=tf.int32)
        labels = tf.cast(labels, dtype=tf.int32)
        # expand the mask so that broadcasting works ([nb_nodes, 1])
        mask = tf.expand_dims(tf.cast(mask, dtype=tf.int32), -1)
        # Count true positives, true negatives, false positives and false negatives.
        tp = tf.cast(tf.count_nonzero(pre * labels * mask, axis=0), tf.float32)
        tn = tf.cast(tf.count_nonzero((pre - 1) * (labels - 1) * mask, axis=0), tf.float32)
        fp = tf.cast(tf.count_nonzero(pre * (labels - 1) * mask, axis=0), tf.float32)
        fn = tf.cast(tf.count_nonzero((pre - 1) * labels * mask, axis=0), tf.float32)
        
        # Calculate accuracy, precision, recall and F1 score.
        precision = tf.reduce_mean(tf.divide(tp, tp + fp + TINY))
        recall = tf.reduce_mean(tf.divide(tp, tp + fn + TINY))
        fmeasure = (2 * precision * recall) / (precision + recall + TINY)
        return fmeasure
