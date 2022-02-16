from Models.models import *


class ESSM(Model):
    def __init__(self, feature_columns, ctr_model, cvr_model):

        super(ESSM, self).__init__()
        self.embedding_layer = EmbeddingLayer(feature_columns)
        self.CTR_model = ctr_model
        self.CVR_model = cvr_model

    def call(self, inputs):
        mask_bool, user_side, seq_embed, target_embed_seq, target_embed_side = \
            self.embedding_layer(inputs)

        prob_ctr = self.CTR_model(
            [mask_bool, user_side, seq_embed[:, :, :-16],
             target_embed_seq[:, :-16], target_embed_side]
        )
        prob_cvr = self.CVR_model(
            [mask_bool, user_side, seq_embed, target_embed_seq, target_embed_side]
        )
        prob_final = prob_ctr * prob_cvr

        return prob_final, prob_cvr, prob_ctr,


class SingleModel(Model):
    def __init__(self, feature_columns, single_model):

        super(SingleModel, self).__init__()
        self.embedding_layer = EmbeddingLayer(feature_columns)
        self.single_model = single_model

    def call(self, inputs):
        mask_bool, user_side, seq_embed, target_embed_seq, target_embed_side = \
            self.embedding_layer(inputs)

        prob_final = self.single_model(
            [mask_bool, user_side, seq_embed, target_embed_seq, target_embed_side]
        )

        return prob_final
