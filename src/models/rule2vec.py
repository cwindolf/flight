import tensorflow as tf


class Rule2Vec():

    # 2d 9 neighbor outer totalistic
    NUM_RULES = 2 ** 18

    def __init__(self, prev_states_placeholder, rule_code_placeholder,
                 next_centers_placeholder, embedding_dims, hidden_layer_size,
                 learning_rate, batch_size):
        prev_outer_totals = tf.reduce_sum(
            tf.tile(tf.constant([[
                [1., 1., 1.],
                [1., 0., 1.],
                [1., 1., 1.]]]),
                (batch_size, 1, 1))
            * tf.to_float(tf.stop_gradient(prev_states_placeholder)),
            axis=(1, 2))
        prev_centers = tf.squeeze(tf.to_float(tf.stop_gradient(prev_states_placeholder[:, 1, 1])))
        float_labels = tf.to_float(tf.stop_gradient(next_centers_placeholder))
        self.rule_embedding = self.embed_rule(rule_code_placeholder,
                                              embedding_dims)
        guesses = self.forward_pass(embedding_dims, hidden_layer_size,
                                   prev_outer_totals, prev_centers, batch_size)
        self.loss = self.calc_loss(guesses, float_labels)
        self.train_op = self.trainer(learning_rate)
        self.percent_correct = tf.reduce_sum(
            tf.to_float(
                tf.logical_and(
                    guesses > 0.5,
                    float_labels > 0.5))) * (100 / batch_size)



    def embed_rule(self, rule_code_placeholder, embedding_dims):
        self.E = tf.get_variable('E',
            shape=(self.NUM_RULES, embedding_dims),
            initializer=tf.truncated_normal_initializer())
        return tf.nn.embedding_lookup(self.E, rule_code_placeholder)


    def forward_pass(self, embedding_dims, hidden_layer_size,
                     prev_outer_total, prev_center, batch_size):
        # first layer, relu
        in_vector = tf.expand_dims(
                        tf.concat([
                                prev_outer_total[:, None],
                                prev_center[:, None],
                                self.rule_embedding
                            ], 1), 1)
        fc_1_W = tf.get_variable('fc_1_W',
                    shape=(1, 2 + embedding_dims, hidden_layer_size * 2),
                    initializer=tf.truncated_normal_initializer())
        relus = tf.nn.relu(tf.matmul(in_vector, tf.tile(fc_1_W, (batch_size, 1, 1))))
        # second hidden layer with relu
        fc_2_W = tf.get_variable('fc_2_W',
                    shape=(1, 2 + hidden_layer_size * 2, hidden_layer_size),
                    initializer=tf.truncated_normal_initializer())
        relus = tf.nn.relu(tf.matmul(
            tf.concat([
                prev_outer_total[:, None, None],
                prev_center[:, None, None],
                relus], 2),
            tf.tile(fc_2_W, (batch_size, 1, 1))))
        # third hidden layer with relu
        # fc_3_W = tf.get_variable('fc_3_W',
        #             shape=(1, hidden_layer_size, hidden_layer_size),
        #             initializer=tf.truncated_normal_initializer())
        # relus = tf.nn.relu(tf.matmul(relus, tf.tile(fc_3_W, (batch_size, 1, 1))))
        # second layer for logit. we only have one logit per example
        fc_4_W = tf.get_variable('fc_4_W',
            shape=(1, hidden_layer_size, 1),
            initializer=tf.truncated_normal_initializer())
        return tf.squeeze(tf.nn.softmax(tf.matmul(relus, tf.tile(fc_4_W, (batch_size, 1, 1)))))


    def calc_loss(self, guesses, next_centers):
        return tf.nn.l2_loss(guesses - next_centers)


    def trainer(self, learning_rate):
        return tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
