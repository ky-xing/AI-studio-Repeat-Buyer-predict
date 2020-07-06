import pandas as pd
import numpy as np
import datetime
import os
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
rfm_labels = ['rs', 'fs', 'ms', 'rfms', 'customer_id_cut']

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
##去除一些缺失值较多的特征
train.drop(['avg_discount','goods_price_std','goods_price_cv','payment_std','payment_cv'],axis=1,inplace=True)
test.drop(['avg_discount','goods_price_std','goods_price_cv','payment_std','payment_cv'],axis=1,inplace=True)

import xgboost as xgb
from sklearn.model_selection import train_test_split
##划分数据集
X_train, X_valid, y_train, y_valid = train_test_split(train.drop(['labels', 'customer_id'], axis=1),
                                                     train['labels'],test_size=0.05, random_state=42)

# 自定义损失函数
penalty = 30 #惩罚因子
def Weighted_LogLoss(preds, xgbtrain):
    labels = xgbtrain.get_label()
    sig_preds = 1/(1+np.exp(-preds))
    sig_clip_preds = np.clip(preds,0.05,0.95)
    score = -np.mean(labels*np.log(sig_clip_preds)*penalty+(1-labels)*np.log(1.-sig_clip_preds))
    return 'WLL', score

parameters = {'boosting_type':'gbdt',
        #'objective' : 'binary:logistic',
        # 'eval_metric' : 'auc',
        'eta' : 0.015,
        'max_depth' : 7,
        'colsample_bytree':0.8,
        'subsample': 0.9,
        'subsample_freq': 8,
        'alpha': 0.5,
        'lambda': 1,
        }

train_data = xgb.DMatrix(X_train, label=y_train)
valid_data = xgb.DMatrix(X_valid, label=y_valid)
test_data = xgb.DMatrix(test.drop('customer_id',axis=1))

# 模型训练与预测
model = xgb.train(parameters, train_data, evals=[(train_data, 'train'), (valid_data, 'valid')],
                  num_boost_round = 500, early_stopping_rounds=50, verbose_eval=50,
                  obj=custo,feval=Weighted_LogLoss,maximize=False)

predict=model.predict(test_data)
test['result']=predict.round(decimals=6)
# 输出csv
test.reset_index(drop=True).sort_values('customer_id', ascending=True, inplace=True,)
test[['customer_id','result']].to_csv('submission_xgboost.csv', float_format='%.5f',index=False)

