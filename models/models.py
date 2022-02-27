from tensorflow.keras import Model
from Models.modules import *


class DIN(Model):
    def __init__(
            self,
            ffn_hidden_units, att_hidden_units,
            att_activation='sigmoid', dnn_dropout=0.5
    ):
        super(DIN, self).__init__()

        self.attention_layer = AttentionLayer(att_hidden_units, att_activation)

        self.bn = BatchNormalization(trainable=True)

        self.ffn = [Dense(unit, activation=PReLU() ) for unit in ffn_hidden_units]
        self.dropout = Dropout(dnn_dropout)
        self.dense_final = Dense(2)
        self.dropout2 = Dropout(dnn_dropout)

    def call(self, inputs):

        mask_bool, user_side, seq_embed, target_embed_seq, target_embed_side = inputs
        mask_value = tf.cast(mask_bool, dtype=tf.float32)
        user_info = self.attention_layer([target_embed_seq, seq_embed, seq_embed, mask_value])
        info_all = tf.concat([user_info, target_embed_seq, target_embed_side, user_side], axis=-1)
        info_all = self.bn(info_all)
        # info_all = self.dropout2(info_all)
        for dense in self.ffn:
            info_all = dense(info_all)

        logits = self.dense_final(info_all)
        outputs = tf.nn.softmax(logits)
        return outputs


class DIEN(Model):
    def __init__(self, att_hidden_units, ffn_hidden_units, dnn_dropout=0.5):

        super(DIEN, self).__init__()

        # attention layer
        self.attention_layer = AttentionLayer4AUGRU(
            att_hidden_units=att_hidden_units,
        )

        self.hist_gru = GRU(32, return_sequences=True)
        self.hist_augru = AUGRU(32)
        self.bn = BatchNormalization(trainable=True)
        # ffn

        self.ffn = [Dense(unit, activation=PReLU()) for unit in ffn_hidden_units]
        self.dropout = Dropout(dnn_dropout)
        self.dense_final = Dense(2)
        self.dropout2 = Dropout(dnn_dropout)

    def call(self, inputs):
        mask_bool, user_side, seq_embed, target_embed_seq, target_embed_side = inputs
        gru_embed = self.hist_gru(seq_embed, mask=mask_bool)
        mask_value = tf.cast(mask_bool, dtype=tf.float32)

        # att_score : None, 1, maxlen
        att_score = self.attention_layer([target_embed_seq, gru_embed, gru_embed, mask_value])
        augru_hidden_state = tf.zeros([gru_embed.shape[0], 32])
        augru_hidden_state = self.hist_augru(
            tf.transpose(gru_embed, [1, 0, 2]),
            # gru_embed: (None, maxlen, gru_hidden) -> (maxlen, None, gru_hidden)
            augru_hidden_state,
            tf.transpose(att_score, [2, 0, 1]),  # None, 1, maxlen -> maxlen, None, 1 1
            mask=mask_value,
        )
        # concat user_info(att hist), cadidate item embedding, other features
        info_all = tf.concat([augru_hidden_state, target_embed_seq, target_embed_side, user_side], axis=-1)

        info_all = self.bn(info_all)
        # ffn
        for dense in self.ffn:
            info_all = dense(info_all)

        # info_all = self.dropout(info_all)
        logits = self.dense_final(info_all)
        outputs = tf.nn.softmax(logits)
        return outputs

    # Not used
    def compute_auxiliary(self, h_states, click_seq, noclick_seq, mask):
        mask = tf.cast(mask, tf.float32)
        click_input_ = tf.concat([h_states, click_seq], -1)
        noclick_input_ = tf.concat([h_states, noclick_seq], -1)
        click_prop_ = self.aux_net(click_input_)[:, :, 0]
        noclick_prop_ = self.aux_net(noclick_input_)[:, :, 0]
        click_loss_ = - tf.reshape(tf.math.log(click_prop_), [-1, tf.shape(click_seq)[1]]) * mask
        noclick_loss_ = - tf.reshape(tf.math.log(1.0 - noclick_prop_), [-1, tf.shape(noclick_seq)[1]]) * mask
        loss_ = tf.reduce_mean(click_loss_ + noclick_loss_)
        return loss_


class BaseModel(Model):
    def __init__(
            self, ffn_hidden_units, dnn_dropout
    ):

        super(BaseModel, self).__init__()

        self.bn = BatchNormalization(trainable=True)
        self.ffn = [Dense(unit, activation=PReLU()) for unit in ffn_hidden_units]
        self.dropout = Dropout(dnn_dropout)
        self.dense_final = Dense(2)

    def call(self, inputs):
        mask_bool, user_side, seq_embed, \
            target_embed_seq, target_embed_side = inputs

        mask_value = tf.cast(mask_bool, dtype=tf.float32)
        seq_embed_masked = seq_embed * tf.expand_dims(mask_value, axis=-1)
        seq_embed_sum = tf.reduce_mean(seq_embed_masked, axis=1)

        info_all = tf.concat([seq_embed_sum, target_embed_seq,
                              target_embed_side, user_side], axis=-1)
        info_all = self.bn(info_all)

        for dense in self.ffn:
            info_all = dense(info_all)

        # info_all = self.dropout(info_all)
        logits = self.dense_final(info_all)
        outputs = tf.nn.softmax(logits)
        return outputs


class GruDNN(Model):
    def __init__(
            self, ffn_hidden_units, dnn_dropout
    ):

        super(GruDNN, self).__init__()

        self.bn = BatchNormalization(trainable=True)
        self.ffn = [Dense(unit, activation=PReLU()) for unit in ffn_hidden_units]
        self.gru = GRU(units=64, return_sequences=False)
        self.dropout = Dropout(dnn_dropout)
        self.dense_final = Dense(2)

    def call(self, inputs):

        mask_bool, user_side, seq_embed, \
            target_embed_seq, target_embed_side = inputs

        gru_info = self.gru(inputs=seq_embed, mask=mask_bool)

        info_all = tf.concat([gru_info, target_embed_seq,
                              target_embed_side, user_side], axis=-1)
        info_all = self.bn(info_all)

        for dense in self.ffn:
            info_all = dense(info_all)

        # info_all = self.dropout(info_all)
        logits = self.dense_final(info_all)
        outputs = tf.nn.softmax(logits)
        return outputs


class MyGRUFm(Model):
    def __init__(
            self, ffn_hidden_units, dnn_dropout
    ):

        super(MyGRUFm, self).__init__()

        self.bn = BatchNormalization(trainable=True)
        self.ffn = [Dense(unit, activation=PReLU()) for unit in ffn_hidden_units]
        self.gru = MyGRU(units=64)
        self.dropout = Dropout(dnn_dropout)
        self.dense_final = Dense(2)

    def call(self, inputs):

        mask_bool, user_side, seq_embed, \
            target_embed_seq, target_embed_side = inputs
        mask_value = tf.cast(mask_bool, dtype=tf.float32)
        augru_hidden_state = tf.zeros([user_side.shape[0], 64])
        augru_hidden_state = self.gru(
            tf.transpose(seq_embed, [1, 0, 2]),
            # gru_embed: (None, maxlen, gru_hidden) -> (maxlen, None, gru_hidden)
            augru_hidden_state,
            mask=mask_value,
        )

        info_all = tf.concat([augru_hidden_state, target_embed_seq,
                              target_embed_side, user_side], axis=-1)
        info_all = self.bn(info_all)

        for dense in self.ffn:
            info_all = dense(info_all)

        # info_all = self.dropout(info_all)
        logits = self.dense_final(info_all)
        outputs = tf.nn.softmax(logits)
        return outputs


class EmbeddingLayer(Model):
    def __init__(self, feature_columns, use_fm=False):
        super(EmbeddingLayer, self).__init__()
        self.dense_feature_columns, self.sparse_seq_columns, \
            self.sparse_item_side, self.sparse_user_side, = feature_columns

        self.dense_len = len(self.dense_feature_columns)

        # seq_embedding_layer
        self.embed_seq_layers = [Embedding(input_dim=feat['feat_num'],
                                           input_length=1,
                                           output_dim=feat['embed_dim'],
                                           embeddings_initializer='random_uniform')
                                 for feat in self.sparse_seq_columns]
        self.n_item = self.sparse_seq_columns[0]['feat_num']
        self.use_fm = use_fm

        # behavior embedding layers, item id and category id
        self.embed_user_side = [Embedding(input_dim=feat['feat_num'],
                                          input_length=1,
                                          output_dim=feat['embed_dim'],
                                          embeddings_initializer='random_uniform'
                                          )
                                for feat in self.sparse_user_side]
        # behavior embedding layers, item id and category id
        self.embed_item_side = [Embedding(input_dim=feat['feat_num'],
                                          input_length=1,
                                          output_dim=feat['embed_dim'],
                                          embeddings_initializer='random_uniform'
                                          )
                                for feat in self.sparse_item_side]

    def call(self, inputs):

        dense_inputs, target_user_side, seq_inputs, \
            target_item_seq, target_item_side = inputs

        mask_bool = tf.not_equal(seq_inputs[:, :, 0], 0)  # (None, maxlen)
        user_side = tf.concat([self.embed_user_side[i](target_user_side[:, i]) for i in range(5)], axis=-1)
        seq_embed = tf.concat([self.embed_seq_layers[i](seq_inputs[:, :, i]) for i in range(3)],
                              axis=-1)

        target_embed_seq = tf.concat([self.embed_seq_layers[i](target_item_seq[:, i]) for i in range(3)], axis=-1)

        target_embed_side = tf.concat([self.embed_item_side[i](target_item_side[:, i]) for i in range(3)],
                                      axis=-1)
        if self.use_fm:
            fm_sparse_input = seq_inputs[:, :, 0]
            fm_input = [fm_sparse_input, seq_embed, target_embed_seq]
        if self.use_fm:
            return mask_bool, user_side, seq_embed, target_embed_seq, target_embed_side, fm_input
        return mask_bool, user_side, seq_embed, target_embed_seq, target_embed_side


class TestModel(Model):
    def __init__(
            self, ffn_hidden_units, dnn_dropout, n_item=378458, use_fm=False
    ):

        super(TestModel, self).__init__()
        self.use_fm = use_fm
        self.bn = BatchNormalization(trainable=True)
        self.ffn = [Dense(unit, activation=PReLU()) for unit in ffn_hidden_units]
        self.dropout = Dropout(dnn_dropout)
        if self.use_fm:
            self.fm = FM(n_item)
        self.dense_final = Dense(2)

    def call(self, inputs):
        if self.use_fm:
            mask_bool, user_side, seq_embed, \
                target_embed_seq, target_embed_side, fm_input = inputs
        else:
            mask_bool, user_side, seq_embed, \
            target_embed_seq, target_embed_side = inputs

        mask_value = tf.cast(mask_bool, dtype=tf.float32)
        seq_embed_masked = seq_embed * tf.expand_dims(mask_value, axis=-1)
        seq_embed_sum = tf.reduce_mean(seq_embed_masked, axis=1)
        if self.use_fm:
            fm_out = self.fm(fm_input, mask_value)

        info_all = tf.concat([seq_embed_sum, target_embed_seq,
                              target_embed_side, user_side], axis=-1)
        info_all = self.bn(info_all)

        for dense in self.ffn:
            info_all = dense(info_all)
        if self.use_fm:
            info_all = tf.concat([fm_out, info_all], axis=-1)

        # info_all = self.dropout(info_all)
        logits = self.dense_final(info_all)
        outputs = tf.nn.softmax(logits)
        return outputs



