from sklearn import metrics
from Models.ensemble import *
from prepare_data import *
from data_iterator import *
import os
import tensorflow as tf
import datetime


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'


setup_seed(16)
# ========================= File Paths =======================
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/test/' + current_time + '/train'
summary_writer = tf.summary.create_file_writer(train_log_dir)
mode_save_path = "./saved_model"
# ========================= Training Setting =======================
func_span = 100   # How many iteration to call functions
global_best_auc = -1  # Keep to -1
global_patients = 0  # Keep to 0 q
epochs = 100
# ========================= Hyper Parameters =======================
max_len = 100  #100
embed_dim = 16 # to make life easier all sparse features have same embedding dim 16
att_hidden_units = [80, 80, 40]  # FFN for Attention Layer
# ffn_hidden_units = [256, 128, 64] # FFN for final output[568, 256, 128, 64]
ffn_hidden_units = [568, 256, 128, 64] # FFN for final output[568, 256, 128, 64]
dnn_dropout = 0 # Need to ensure this
att_activation = 'sigmoid'
ffn_activation = 'prelu' #prelu
train_batch_size = 2048 # 128
test_val_batch_size = 4096
learning_rate = 0.01
ctr_weight = 1
cvr_weight = 0
# ========================== Create dataset =======================
feature_columns, train, dev, test = process_data(embed_dim, max_len)
train_X, train_y, train_y_click = train
dev_X, dev_y, dev_y_click = dev
test_X, test_y, test_y_click = test
train_data_all = get_dataloader(train_batch_size, train_X, train_y, train_y_click)
dev_data_all = get_dataloader(test_val_batch_size, dev_X, dev_y, dev_y_click)
test_data_all = get_dataloader(test_val_batch_size, test_X, test_y, test_y_click)
num_instance = len(train_X[0])
# ========================== Evaluation Recorders =======================
train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
train_cvr_loss = tf.keras.metrics.Mean('train_cvr_loss', dtype=tf.float32)
train_ctr_loss = tf.keras.metrics.Mean('train_ctr_loss', dtype=tf.float32)

dev_loss = tf.keras.metrics.Mean('dev_loss', dtype=tf.float32)
dev_cvr_loss = tf.keras.metrics.Mean('dev_cvr_loss', dtype=tf.float32)
dev_ctr_loss = tf.keras.metrics.Mean('dev_ctr_loss', dtype=tf.float32)
# =========================Initialize Models=========================================
modes = ["Single", "ESSM"]
mode = modes[1]

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# cvr_model2 = GruFM(ffn_hidden_units=ffn_hidden_units, dnn_dropout=dnn_dropout)
# ctr_model = DIEN(att_hidden_units=att_hidden_units, ffn_hidden_units=ffn_hidden_units)
# ctr_model = TestModel(ffn_hidden_units=ffn_hidden_units, dnn_dropout=dnn_dropout, use_fm=True)
# model = SingleModel_t(feature_columns=feature_columns, single_model=ctr_model, use_fm=True)
cvr_model = DIN(att_hidden_units=att_hidden_units, ffn_hidden_units=ffn_hidden_units)
ctr_model = DIN(att_hidden_units=att_hidden_units, ffn_hidden_units=ffn_hidden_units)

model = ESSM(feature_columns=feature_columns, ctr_model=cvr_model, cvr_model=ctr_model)
# model = SingleModel(feature_columns=feature_columns, single_model=ctr_model)

model_name = "cvr_model2"
ctr_loss_func = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
loss_func = tf.keras.losses.CategoricalCrossentropy()

# ESSM DIN + DIN 0.01 [256, 128, 64]  [80, 80, 40]  0.91366563309948
# Single DIN 0.88

def evaluate():
    """
    Helper function of main_train.
    This function evaluate current model against the validation set.

    :return: AUC on validation set.
    """
    global global_best_auc
    global ctr_weight
    outputs = []
    labels = []
    cvr_labels = []
    cvr_outputs = []
    ctr_labels = []
    ctr_outputs = []
    for step, (mini_batch, label, label_ctr) in enumerate(dev_data_all, start=1):
        if mode == "ESSM":
            main_prob, cvr_prob, ctr_prob = model(mini_batch)

            main_loss = loss_func(y_true=label, y_pred=main_prob)
            ctr_loss = loss_func(y_true=label_ctr, y_pred=ctr_prob)
            cvr_loss = loss_func(y_true=label, y_pred=cvr_prob, sample_weight=label_ctr[:, 0]) #

            dev_loss(main_loss)
            dev_cvr_loss(cvr_loss)
            dev_ctr_loss(ctr_loss)

            outputs.append(main_prob)
            labels.append(label)

            ctr_labels.append(label_ctr)
            ctr_outputs.append(ctr_prob)

            click_mask = tf.cast(label_ctr[:, 0], tf.bool)
            click_part_label = label[click_mask]
            click_part_pred = cvr_prob[click_mask]
            cvr_labels.append(click_part_label)
            cvr_outputs.append(click_part_pred)

        else:
            main_prob = model(mini_batch)
            main_loss = loss_func(y_true=label, y_pred=main_prob)
            final_loss = main_loss
            dev_loss(final_loss)
            outputs.append(main_prob)
            labels.append(label)

    pred = tf.concat(outputs, 0)[:, 0]
    y = tf.concat(labels, 0)[:, 0]
    fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    if mode == "ESSM":
        pred_c = tf.concat(ctr_outputs , 0)[:, 0]
        y_c = tf.concat(ctr_labels, 0)[:, 0]
        fpr, tpr, thresholds = metrics.roc_curve(y_c, pred_c, pos_label=1)
        auc_ctr = metrics.auc(fpr, tpr)

        pred_cv = tf.concat(cvr_outputs, 0)[:, 0]
        y_cv = tf.concat(cvr_labels, 0)[:, 0]
        fpr, tpr, thresholds = metrics.roc_curve(y_cv, pred_cv, pos_label=1)
        auc_cvr = metrics.auc(fpr, tpr)
        print(f"Current Validation Main_AUC is {auc}")
        print(f"Current Validation CVR_AUC is {auc_cvr}")
        print(f"Current Validation CTR_AUC is {auc_ctr}")

    if auc > global_best_auc:
        print(f"Validation AUC improved from {global_best_auc} to {auc}")
    else:
        print(f"No improvement...")
    return auc


def early_stopping(auc, patients):
    """
    Helper function of main_train.
    This function monitors the AUC on validation set, and stops the training process when
    the AUC doesn't improved for a certain global steps which is controlled by global_patients.

    :param auc: Current AUC on validation set.
    :param patients: The number of global steps since the last time model improved.
    :return: Bool value indicates weather to stop training.
    """
    global global_best_auc
    global global_patients
    if auc < global_best_auc:
        global_patients += 1
        if global_patients == patients:
            return True
        print(f"Wait for {global_patients - patients}, current best auc is {global_best_auc}")
    else:
        global_patients = 0
        return False


def save_model(auc):
    """
    Helper function of main_train.
    Save the model with the best AUC on the validation set.
    :param auc: Current AUC.
    :return: None
    """
    global global_best_auc
    if auc > global_best_auc:
        print(f"A better model have been saved with AUC:{auc}")
        model_cur_path = f"./saved_weights/{model_name}_{auc}.ckpt"
        model.save_weights(model_cur_path)
        global_best_auc = max(auc, global_best_auc)
        return
    else:
        print(f"This model is not better than before!---auc:{auc}")


def train_one_step(mini_batch, label, label_ctr):
    with tf.GradientTape() as tape:
        if mode == "ESSM":
            main_prob, cvr_prob, ctr_prob = model(mini_batch)
            main_loss = loss_func(y_true=label, y_pred=main_prob)

            cvr_loss = loss_func(y_true=label, y_pred=cvr_prob, sample_weight=label_ctr[:, 0])
            ctr_loss = ctr_loss_func(y_true=label_ctr, y_pred=ctr_prob)

            final_loss = main_loss + ctr_weight * ctr_loss + cvr_loss * cvr_weight
            train_loss(main_loss)
            train_cvr_loss(cvr_loss)
            train_ctr_loss(ctr_loss)
        else:
            main_prob = model(mini_batch)
            main_loss = loss_func(y_true=label, y_pred=main_prob)
            final_loss = main_loss
            train_loss(final_loss)

    gradient = tape.gradient(final_loss, model.trainable_variables)
    clip_gradient, _ = tf.clip_by_global_norm(gradient, 5.0)
    optimizer.apply_gradients(zip(clip_gradient, model.trainable_variables))


def reset_states(t_loss, t_cvr_loss, t_ctr_loss, d_loss, d_cvr_loss, d_ctr_loss):
    t_loss.reset_states()
    t_ctr_loss.reset_states()
    t_cvr_loss.reset_states()
    d_loss.reset_states()
    d_ctr_loss.reset_states()
    d_cvr_loss.reset_states()


def reset_global():
    global global_best_auc
    global global_patients
    global_best_auc = -1  # Keep to -1
    global_patients = 0


def main_train():
    reset_global()
    global learning_rate
    global_step = 0
    early_stopping_flag = False
    patients = 30

    for epoch in range(1, epochs + 1):
        print("New epoch start, reshuffling...")
        for step, (mini_batch, label, label_click) in enumerate(train_data_all, start=1):
            if step % 10 == 0:
                print(f"{step} / {num_instance // train_batch_size} epoch {epoch}")
            train_one_step(mini_batch, label, label_click)
            global_step += 1
            if global_step % func_span == 0:
                auc = evaluate()
                print("============================================")
                print(f"train_main_loss is {train_loss.result().numpy()}, "
                      f"train_cvr_loss is {train_cvr_loss.result().numpy()} ,train_ctr_loss is {train_ctr_loss.result().numpy()}")
                print(f"dev_main_loss is {dev_loss.result().numpy()}, dev_cvr_loss is {dev_cvr_loss.result().numpy()}, "
                      f"dev_ctr_loss is {dev_ctr_loss.result().numpy()}")
                print("============================================")
                reset_states(train_loss, train_cvr_loss, train_ctr_loss, dev_loss, dev_cvr_loss, dev_ctr_loss)

                if early_stopping(auc, patients):
                    print(
                        f"The AUC for validation set have not been improved for {patients * func_span} iterations, stopping training...")
                    early_stopping_flag = True
                    break
                save_model(auc)
                print("============================================")
        if early_stopping_flag:
            print(f"Stop training... The best AUC is {global_best_auc}")
            break
        learning_rate *= 0.5


main_train()
