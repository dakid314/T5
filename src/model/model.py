from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.model_selection import  StratifiedKFold
from sklearn.metrics import roc_curve, confusion_matrix, precision_score, accuracy_score, f1_score, matthews_corrcoef, auc,precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dropout,Dense
from sklearn.metrics import roc_curve, auc
import pickle
import os 

def get_evaluation(label: list, pred: list, pro_cutoff: float = None):
    fpr, tpr, thresholds = roc_curve(label, pred)
    if pro_cutoff is None:
        best_one_optimal_idx = np.argmax(tpr - fpr)
        pro_cutoff = thresholds[best_one_optimal_idx]
    pred_l = [1 if i >= pro_cutoff else 0 for i in pred]
    #后面新增的计算prAUC
    confusion_matrix_1d = confusion_matrix(label, pred_l).ravel()
    confusion_dict = {N: n for N, n in zip(['tn', 'fp', 'fn', 'tp'], list(
        confusion_matrix_1d * 2 / np.sum(confusion_matrix_1d)))}
    
    precision, recall, _ = precision_recall_curve(label, pred)
    pr_auc = auc(recall, precision)
    
    evaluation = {
        "accuracy": accuracy_score(label, pred_l),
        "precision": precision_score(label, pred_l),
        "f1_score": f1_score(label, pred_l),
        "mmc": matthews_corrcoef(label, pred_l),
        "rocAUC": auc(fpr, tpr),
        "prAUC": pr_auc,
        "specificity": confusion_dict['tn'] / (confusion_dict['tn'] + confusion_dict['fp']),
        "sensitivity": confusion_dict['tp'] / (confusion_dict['tp'] + confusion_dict['fn']),
        'pro_cutoff': pro_cutoff
    }
    return evaluation

def plot_roc_curve(target, pred, path_to_: str):
    fpr, tpr, thresholds = roc_curve(target, pred)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(19.2, 10.8))
    plt.plot(fpr, tpr, color='red', lw=2,
             label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='blue', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic (ROC) curve')
    plt.legend(loc="lower right")

    plt.savefig(f"{path_to_}")
    plt.clf()
    
first_greater_than_found = False

def get_model():
    input1 = tf.keras.Input(shape=(100,))
       
    embedding_layer = tf.keras.layers.Embedding(21, 100)(input1)
    
    dorp1 = tf.keras.layers.Dropout(rate=0.2)(embedding_layer)
    lstm1 = tf.keras.layers.LSTM(units=100,input_shape = (100,100),return_sequences=True)(dorp1)
    
    dorp2 = tf.keras.layers.Dropout(rate=0.2)(lstm1)
    lstm2 = tf.keras.layers.LSTM(units=100,input_shape = (100,100),return_sequences=True)(dorp2)
    
    dorp3 = tf.keras.layers.Dropout(rate=0.2)(lstm2)
    lstm3 = tf.keras.layers.LSTM(units=100,input_shape = (100,100),return_sequences=True)(dorp3)
    
    flatten_layer = tf.keras.layers.Flatten()(lstm3)
    output_layer = tf.keras.layers.Dense(1,activation='sigmoid')(flatten_layer)
    
    model = tf.keras.models.Model(inputs=input1, outputs=output_layer, name='Rnn')
    
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=[
                      tf.keras.metrics.BinaryAccuracy(),
                      tf.keras.metrics.AUC(name='auc')
                  ])
    
    return model

class MyOptimitzer:
    def __init__(self):
        self.train_best_predicted_pair = []
        self.train_best_5C_predicted_pair = []
        self.best_predicted_pair = []
        self.best_5C_predicted_pair = []
        self.fiveC_model = None
        self.total_model = None
        pass
    def fit_total(self,full_X, full_y):  
        self.total_model = get_model()
        self.total_model.fit(
                full_X,
                full_y,
                epochs=20,
                batch_size=15,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        monitor='val_auc', min_delta=0, patience=3, verbose=0,
                        mode='auto', baseline=None, restore_best_weights=True
                    ),
                ]
            )
    def fit(self,full_X, full_y):
        for Kfold_id, (train_id, test_id) in enumerate(
            StratifiedKFold(
                n_splits=5,
                shuffle=True,
                random_state=42
            ).split(full_X, full_y)
        ):
            y_to_train = full_y[train_id].copy()
            self.fiveC_model = get_model()  # 假定 get_model() 函数能够返回一个神经网络模型

            # 训练模型
            self.fiveC_model.fit(
                full_X[train_id],
                y_to_train,
                epochs=20,
                batch_size=15,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        monitor='val_auc', min_delta=0, patience=3, verbose=0,
                        mode='auto', baseline=None, restore_best_weights=True
                    ),
                ]
            )

            self.best_predicted_pair = [
            np.nan_to_num(self.fiveC_model.predict(
                full_X[test_id]
            ), nan=0.0),
            full_y[test_id]
            ]
            self.train_best_predicted_pair = [
                np.nan_to_num(self.fiveC_model.predict(
                    full_X[train_id]
                ), nan=0.0),
                full_y[train_id]
            ]
            # 预测并记录
            self.best_5C_predicted_pair.append([
                np.nan_to_num(self.fiveC_model.predict(
                    full_X[test_id]
                ), nan=0.0),
                full_y[test_id]
            ])
            self.train_best_5C_predicted_pair.append([
                np.nan_to_num(self.fiveC_model.predict(
                    full_X[train_id]
                ), nan=0.0),
                y_to_train
            ])
        return self
    
    def get_summary(self, path_to_dir: str = None):
        os.makedirs(path_to_dir, exist_ok=True)
        model_path = "-"
        

        model_path = f"{path_to_dir}/RNN.pkl"
        if path_to_dir is not None:
            with open(model_path, "bw+") as f:
                pickle.dump(self.total_model, f)
            
        training_testing_performance = get_evaluation(
            label=self.best_predicted_pair[1],
            pred=self.best_predicted_pair[0],
        )

        # 计算5C中的平均表现
        FiveFold_result = {}
        for keys in training_testing_performance.keys():
            value_list = []
            for item in self.best_5C_predicted_pair:

                item_performance = get_evaluation(
                    label=item[1],
                    pred=item[0],
                )
                value_list.append(item_performance[keys])

            if keys == "pro_cutoff":
                FiveFold_result[keys] = value_list
            else:
                FiveFold_result[keys] = sum(value_list) / len(value_list)


        return pd.Series({
                        "Classifier_Name": 'RNN',
                        "Model_Path": model_path
                    } | FiveFold_result
                        )
    
def load_data(path):
    seq = []
    with open(path) as pos:
        sequence = ""
        for line in pos:
            line = line.strip()
            if line.startswith(">"):
                if sequence and len(sequence) >= 100:  # 检查上一个序列是否符合条件
                    seq.append(sequence[:100])  # 存储前 100 个字符的部分
                sequence = ""  # 重置序列
            else:
                sequence += line
        if sequence and len(sequence) >= 100:  # 检查最后一条序列是否符合条件
            seq.append(sequence[:100])
    return seq
a = 0
rate_list = ['1_2','1_10','1_100']
while a < 5:
    for rate in rate_list:
        pos_sequence = pd.Series(load_data('data/pos/T5_training_70.fasta'))
        neg_sequence = pd.Series(load_data(f'data/T5/70/{a}/all_nT5_70_{rate}.fasta'))
        pos_label = pd.Series(np.ones(len(pos_sequence)))
        neg_label = pd.Series(np.zeros(len(neg_sequence)))

        full_y = pd.concat([pos_label, neg_label])
        full_X = pd.concat([pos_sequence, neg_sequence])

        full_y = np.vstack(full_y.values)
        full_X = np.array(full_X)
        full_X = [list(string[:100]) for string in full_X]

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(full_X)
        full_X = tokenizer.texts_to_sequences(full_X)
        full_X = np.array(full_X)

        model_path_to_save = f'/mnt/md0/Public/T5/model/{rate}/{a}'
        os.makedirs(model_path_to_save, exist_ok=True)

        result_list = []
        
        fivecross_result = pd.concat([
            MyOptimitzer(
            ).fit(full_X=full_X,
                  full_y=full_y
            ).get_summary(path_to_dir=f"{model_path_to_save}/")], axis=1).T

        fivecross_result_splited = fivecross_result.loc[:, [
                        "accuracy", "precision", "f1_score", "mmc", "rocAUC", "specificity", "sensitivity", "pro_cutoff","prAUC"]]
        fivecross_result_splited.to_csv(
            f"{model_path_to_save}/RNN_5Fold.csv"
        )

        series = fivecross_result_splited.sum(axis=0)
        series.name = 'RNN'
        result_list.append(series)


    pd.concat(result_list).reset_index(drop=True).to_csv(f"{model_path_to_save}/5fold_results.csv", index=False)
    a += 1
    