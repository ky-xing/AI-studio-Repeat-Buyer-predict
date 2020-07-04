import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,KFold,StratifiedKFold
from sklearn.metrics import auc,accuracy_score,log_loss
import catboost as cbt
import lightgbm as lgb
import xgboost as xbg
pd.set_option('display.max_columns',1000)

import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('train_features.csv')
test = pd.read_csv('test_features.csv')

##对数据进行转换，节省内存
def formatTrans(df):
    col = list(df.columns)
    for i in col:
        if df[i].dtypes == 'float64':
            df[i] = df[i].astype(np.float32)
        if df[i].dtypes == 'int64':
            df[i] = df[i].astype(np.int32)

formatTrans(train)
formatTrans(test)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# train.drop(['recency_z-score', 'frequency_z-score', 'monetary_z-score', 'rfm_z-score'],axis=1,inplace=True)
# test.drop(['recency_z-score', 'frequency_z-score', 'monetary_z-score', 'rfm_z-score'],axis=1,inplace=True)

labels = [ 'gender', 'province', 'city', 'city_grade', 'is_rated']
rfm_labels = ['rec_score', 'fre_sore', 'mone_score', 'rfms', 'customer_id_cut']


labels = labels + rfm_labels

print('Encoding labels: ', labels)

# 将某些特征进行Label-Encoder
label_list = []
for label in labels:
    lbe = LabelEncoder()
    print(label)
    train[label] = lbe.fit_transform(train[label])
    test[label] = lbe.transform(test[label])
    label_list.append(lbe)

y_train = train['label']
x_train = train.drop(['customer_id','label'],axis = 1)
x_test = test.drop(['customer_id','label'],axis = 1)
x_train = x_train.reset_index(drop= True)
x_test = x_test.reset_index(drop= True)
y_train = y_train.reset_index(drop= True)

feature_name = [i for i in data.columns if i not in ['customer_id','label']]

def model_stacking(model, x_train, y_train, x_test, feature_name, n_folds=5):
    print("model:"' ' + 'training start...')
    print('len_x_train:', len(x_train))
    train_num = x_train.shape[0]
    test_num = x_test.shape[0]
    layer2_train_set = np.zeros((train_num))
    layer2_test_set = np.zeros((test_num))
    test_nfolds_sets = np.zeros((test_num, n_folds))
    kf = KFold(n_splits=n_folds)

    for i, (train_index, test_index) in enumerate(kf.split(x_train)):
        x_tr, y_tr = x_train[feature_name].iloc[train_index], y_train[train_index]
        x_te, y_te = x_train[feature_name].iloc[test_index], y_train[test_index]
        model.fit(x_tr[feature_name], y_tr, eval_set=[(x_te[feature_name], y_te)])

        layer2_train_set[test_index] = model.predict(x_te[feature_name])
        test_nfolds_sets[:, i] = model.predict(x_test)

    layer2_test_set[:] = test_nfolds_sets.mean(axis=1)
    print('training finished!')
    return layer2_train_set, layer2_test_set


lgb_model = lgb.LGBMClassifier(random_seed=2019, n_jobs=-1, objective='binary', learning_rate=0.02, n_estimators=1000,
                               num_leaves=31, max_depth=8, early_stopping_rounds=200, verbose=50)
                               sdfa


xgb_model = xbg.XGBClassifier(objective='binary:logistic', eval_metric='auc', learning_rate=0.015, max_depth=7,
                              n_estimators=1000,early_stopping_rounds =200, feval=lgb_f1, verbose=50)

cbt_model = cbt.CatBoostClassifier(iterations=1000, learning_rate=0.02, max_depth=10, l2_leaf_reg=1, verbose=50,
                                   early_stopping_rounds=200, task_type='GPU', eval_metric='F1')
#通过model训练得到第二层的训练集和测试集
train_sets = []
test_sets = []
for model in [xgb_model,lgb_model,cbt_model]:
    train_set,test_set = model_stacking(model,x_train,y_train,x_test,feature_name)
    train_sets.append(train_set)
    test_sets.append(test_set)
##第二层训练采用logistic回归进行训练
from sklearn.linear_model import LogisticRegression

meta_train = np.concatenate([result_set.reshape(-1, 1) for result_set in train_sets], axis=1)
meta_test = np.concatenate([y_test_set.reshape(-1, 1) for y_test_set in test_sets], axis=1)

# 使用逻辑回归作为第二层模型
bclf = LogisticRegression()
bclf.fit(meta_train, y_train)
test_pred = bclf.predict_proba(meta_test)[:, 1]

# 提交结果

submit = test[['customer_id']]
submit['label'] = (test_pred >= 0.5).astype(int)
print(submit['label'].value_counts())
submit.to_csv("Stacking_Model_result.csv", index=False)