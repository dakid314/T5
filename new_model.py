from sklearn.model_selection import StratifiedKFold
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
from sklearn.base import BaseEstimator, ClassifierMixin
from tensorflow.keras.layers import Dropout,Dense
from sklearn.metrics import roc_curve, auc
import pickle
from sklearn.metrics import roc_auc_score
import os 
from sklearn.metrics import make_scorer
from tqdm import tqdm
import typing
from datetime import datetime
from sklearn.base import ClassifierMixin
from sklearn.model_selection import GridSearchCV, StratifiedKFold,StratifiedShuffleSplit
from sklearn.model_selection._search import BaseSearchCV
from skopt import BayesSearchCV
from skopt.space import Real, Categorical
from skopt import BayesSearchCV
from sklearn.model_selection import cross_val_score, GridSearchCV, RepeatedStratifiedKFold
from sklearn.metrics import roc_curve, confusion_matrix, precision_score, accuracy_score, f1_score, matthews_corrcoef, auc,precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import warnings
warnings.filterwarnings("ignore")


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
class MyOptimitzer:
    def __init__(self, classifier_name: str, classifier_class: ClassifierMixin, classifier_param_dict: dict) -> None:
        self.classifier_name = classifier_name
        self.classifier_class = classifier_class
        self.classifier_param_dict = classifier_param_dict

        self.grid_search: BaseSearchCV = None
        self.train_best_predicted_pair = None
        self.train_best_5C_predicted_pair = None
        self.best_predicted_pair = None
        self.best_5C_predicted_pair = None
        self.start_to_train_time = datetime.now()
        self.end_of_train_time = None
        pass

    def find_best(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation: tuple,
        search_method: typing.Literal["GridSearchCV", "BayesSearchCV"],
        n_jobs: int = 26
    ):

        

        if search_method == "GridSearchCV":
            self.grid_search = GridSearchCV(
                self.classifier_class(),
                param_grid=self.classifier_param_dict,
                cv=StratifiedKFold(
                    n_splits=5,
                    shuffle=True,
                    random_state=42
                ),
                scoring='roc_auc',
                n_jobs=n_jobs,
                refit=True
            )
        elif search_method == "BayesSearchCV":
            self.grid_search = BayesSearchCV(
                self.classifier_class(),
                search_spaces=self.classifier_param_dict,
                cv=StratifiedKFold(
                    n_splits=5,
                    shuffle=True,
                    random_state=42
                ),
                scoring='roc_auc',
                n_jobs=n_jobs,
                n_points=n_jobs,
                n_iter=5,
                refit=True
            )
        else:
            raise ValueError(
                'search_method: typing.Literal["GridSearchCV", "BayesSearchCV"]'
            )
        y_origin = y
        
        full_X = np.concatenate([
            X, validation[0]
        ])
        full_y = np.concatenate([
            y_origin, validation[1]
        ])

        self.grid_search.fit(full_X, full_y)
        self.best_predicted_pair = [
            np.nan_to_num(self.grid_search.predict_proba(
                X=validation[0]
            ), nan=0.0),
            validation[1]
        ]
        self.train_best_predicted_pair = [
            np.nan_to_num(self.grid_search.predict_proba(
                X=X
            ), nan=0.0),
            y
        ]

        # 5倍交叉验证
        
        # 跑模型
        self.best_5C_predicted_pair = []
        self.train_best_5C_predicted_pair = []
        for Kfold_id, (train_id, test_id) in enumerate(
            StratifiedKFold(
                n_splits=5,
                shuffle=True,
                random_state=42
            ).split(full_X, full_y)
        ):
            

            # 定义模型并加载参数
            fiveC_model = self.classifier_class(
                **self.grid_search.best_params_,
            )
            y_to_train = full_y[train_id].copy()

            
            fiveC_model.fit(
                full_X[train_id],
                y_to_train,
                epochs=20,
                batch_size=100,
            )

            # 预测并记录
            self.best_5C_predicted_pair.append([
                np.nan_to_num(fiveC_model.predict_proba(
                    X=full_X[test_id]
                ), nan=0.0),
                full_y[test_id]
            ])
            self.train_best_5C_predicted_pair.append([
                np.nan_to_num(fiveC_model.predict_proba(
                    X=full_X[train_id]
                ), nan=0.0),
                y_to_train
            ])

        return self

    def get_summary(self, path_to_dir: str = None):
        os.makedirs(path_to_dir, exist_ok=True)
        model_path = "-"
        

        model_path = f"{path_to_dir}/{self.classifier_name}.pkl"
        if path_to_dir is not None:
            with open(model_path, "bw+") as f:
                pickle.dump(
                    self.grid_search, f
                )
            
        training_testing_performance = get_evaluation(
            label=self.best_predicted_pair[1],
            pred=self.best_predicted_pair[0][:, 1],
        )

        # 计算5C中的平均表现
        FiveFold_result = {}
        for keys in training_testing_performance.keys():
            value_list = []
            for item in self.best_5C_predicted_pair:

                item_performance = get_evaluation(
                    label=item[1],
                    pred=item[0][:, 1],
                )
                value_list.append(item_performance[keys])

            if keys == "pro_cutoff":
                FiveFold_result[keys] = value_list
            else:
                FiveFold_result[keys] = sum(value_list) / len(value_list)

        self.end_of_train_time = datetime.now()

        return pd.Series({
                        "Classifier_Name": self.classifier_name,
                        "Optimitied_Param": dict(self.grid_search.best_params_),
                        "Model_Path": model_path
                    } | FiveFold_result
                        )
class CustomLSTMClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, units=64, dropout_rate=0.2, verbose=0, optimizer='adam', activation='sigmoid',
                 early_stopping_min_delta=0.0001, early_stopping_patience=5, early_stopping_mode='auto'):
        self.units = units
        self.dropout_rate = dropout_rate
        self.verbose = verbose
        self.optimizer = optimizer
        self.activation = activation
        self.early_stopping_min_delta = early_stopping_min_delta
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_mode = early_stopping_mode
        self.model = None
        self.callbacks = None
        self.classes_ = None
    def build_model(self):
        input1 = tf.keras.Input(shape=(100,))
        
        embedding_layer = tf.keras.layers.Embedding(21, 100)(input1)
        
        dorp1 = tf.keras.layers.Dropout(self.dropout_rate)(embedding_layer)
        lstm1 = tf.keras.layers.LSTM(self.units, input_shape=(100,100), return_sequences=True)(dorp1)
        
        dorp2 = tf.keras.layers.Dropout(self.dropout_rate)(lstm1)
        lstm2 = tf.keras.layers.LSTM(self.units, input_shape=(100,100), return_sequences=True)(dorp2)
        
        dorp3 = tf.keras.layers.Dropout(self.dropout_rate)(lstm2)
        lstm3 = tf.keras.layers.LSTM(self.units, input_shape=(100,100), return_sequences=True)(dorp3)
        
        flatten_layer = tf.keras.layers.Flatten()(lstm3)
        output_layer = tf.keras.layers.Dense(1, activation=self.activation)(flatten_layer)
        
        model = tf.keras.models.Model(inputs=input1, outputs=output_layer, name='Rnn')
        
        model.compile(optimizer=self.optimizer,
                    loss=tf.keras.losses.BinaryCrossentropy(),
                    metrics=[
                        tf.keras.metrics.BinaryAccuracy(),
                        tf.keras.metrics.AUC(name='auc')
                    ])
        
        self.model = model

        self.callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_auc',
                min_delta=self.early_stopping_min_delta,
                patience=self.early_stopping_patience,
                verbose=self.verbose,
                mode=self.early_stopping_mode,
                baseline=None,
                restore_best_weights=True
            )
        ]
        
        return model

    def fit(self, X, y,epochs=10, batch_size=3):
        self.build_model()
        self.model.fit(X, y, callbacks=self.callbacks,epochs=epochs, batch_size=batch_size)
        self.classes_ = np.unique(y)

    def predict(self, X):
        return (self.model.predict(X) > 0.5).astype(int)

    def predict_proba(self, X):
        predictions = self.model.predict(X)
        # 将预测概率转换为二维数组形式
        proba = np.zeros((X.shape[0], 2))
        proba[:, 1] = predictions.reshape(-1)  # 填充预测概率
        proba[:, 0] = 1 - proba[:, 1]  # 计算第二个类别的预测概率
        return proba

    def score(self, X, y):
        predictions = self.predict_proba(X)
        return roc_auc_score(y, predictions)

    def get_params(self, deep=True):
        return {
            'units': self.units,
            'dropout_rate': self.dropout_rate,
            'verbose': self.verbose,
            'optimizer': self.optimizer,
            'activation': self.activation,
            'early_stopping_min_delta': self.early_stopping_min_delta,
            'early_stopping_patience': self.early_stopping_patience,
            'early_stopping_mode': self.early_stopping_mode
        }

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self
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
        find_space = [
            {
                "name": "RNN",
                "class": CustomLSTMClassifier,
                "param": {
                'units': [50, 100, 150],
                'dropout_rate': [0.2, 0.3, 0.4],
                'verbose': [0, 1, 2],  # Different values for verbose parameter
                'optimizer': ['adam', 'rmsprop', 'sgd'],
                'activation': ['relu', 'tanh', 'sigmoid'],
                'early_stopping_min_delta': [0, 0.1, 0.2],
                'early_stopping_patience': [2, 3, 4],
                'early_stopping_mode': ['auto', 'min', 'max']
                }
            }
        ]
        model_path_to_save = f'/mnt/md0/Public/T5/model/{rate}/{a}'
        os.makedirs(model_path_to_save, exist_ok=True)

        result_list = []
        
        for model_index in tqdm(range(len(find_space))):
            fivecross_result = pd.concat([
                MyOptimitzer(
                    find_space[model_index]["name"],
                    find_space[model_index]["class"],
                    find_space[model_index]["param"],
                ).find_best(
                    X=full_X[train_id],
                    y=full_y[train_id],
                    search_method = "BayesSearchCV",
                    validation=(full_X[test_id], full_y[test_id])
                ).get_summary(
                    path_to_dir=f"{model_path_to_save}/"
                )
                for Kfold_id, (train_id, test_id) in enumerate(StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(full_X, full_y))
            ], axis=1).T

            print(fivecross_result)

            fivecross_result.loc[:, ["Classifier_Name", "Optimitied_Param", "Model_Path"]].to_csv(
                f"{model_path_to_save}/{find_space[model_index]['name']}_Param.csv"
            )
            fivecross_result_splited = fivecross_result.loc[:, [
                "accuracy", "precision", "f1_score", "mmc", "rocAUC", "specificity", "sensitivity", "pro_cutoff","prAUC"]]
            fivecross_result_splited.to_csv(
                f"{model_path_to_save}/{find_space[model_index]['name']}_5Fold.csv"
            )

            series = fivecross_result_splited.sum(axis=0)
            series.name = find_space[model_index]["name"]
            result_list.append(series)

            pd.concat(
                result_list, axis=1,
            ).T.to_csv(
                f"{model_path_to_save}/5fold_results.csv",
                index=True
            )


    
    a += 1