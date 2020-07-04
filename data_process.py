import pandas as pd
import numpy as np
from tqdm import tqdm
import datetime
import time
#导入数据
data = pd.read_csv('train.csv')
print(data.columns)

##清洗数据
data['customer_gender'].fillna(value =-1,inplace =True)
data.sort_values(['goods_id'],ascending=True,inplace=True,na_position='last')
data['goods_price'].fillna(method='ffill',inplace=True)
data.sort_values(['order_id'],ascending=True,inplace=True,na_position='last')

data['customer_city'].fillna(value='other',inplace=True)
data['customer_province'].fillna(value='other',inplace=True)

#根据订单量来划分省份和城市，省份少于5000和城市少于2000不做区分
temp = data.groupby(['customer_province'])[['order_id']].agg(np.size).reset_index()
other_province_list = list(temp[temp['order_id'] < 5000]['customer_province'])

temp = data.groupby(['customer_city'])[['order_id']].agg(np.size).reset_index()
other_city_list = list(temp[temp['order_id'] < 150]['customer_city'])
del temp # 清内存

data['customer_province'] = data['customer_province'].map(lambda x: 'other' if x in other_province_list else x)
data['customer_city'] = data['customer_city'].map(lambda x: 'other' if x in other_city_list else x)

# 针对customer_id做一个分桶，区分客户注册时长

bins = data['customer_id'].quantile(q = [0, 0.05, 0.1 , 0.15, 0.2 , 0.25, 0.3 , 0.35, 0.4 , 0.45, 0.5, 0.55, 0.6 , 0.65, 0.7 , 0.75, 0.8 , 0.85, 0.9 , 0.95, 1],
                                  interpolation = 'nearest')
# 分20桶
bins[0] = 0
labels = [x for x in range(1 , 21)]
data['customer_id_cut'] = pd.cut(data['customer_id'], bins, labels=labels,  include_lowest=True)
data['customer_id_cut'] = data['customer_id_cut'].astype(int)
#分100桶
bins = data['customer_id'].quantile(q=np.arange(0, 1.01, 0.01), interpolation='nearest')
labels = [x for x in range(1 , 101)]
data['customer_id_cut_100'] = pd.cut(data['customer_id'], bins, labels=labels,  include_lowest=True)
data['customer_id_cut_100'] = data['customer_id_cut_100'].astype(int)
#分1000桶
bins = data['customer_id'].quantile(q=np.arange(0, 1.001, 0.001), interpolation='nearest')
labels = [x for x in range(1 , 1001)]
data['customer_id_cut_1000'] = pd.cut(data['customer_id'], bins, labels=labels,  include_lowest=True)
data['customer_id_cut_1000'] = data['customer_id_cut_1000'].astype(int)
##对订单状态不同值进行处理
data['order_status'] = data['order_status'].map(lambda x: 1 if x == 0 else x)
data['order_status'] = data['order_status'].map(lambda x: 6 if x > 6 else x)
##看双11是否购买商品
date_20121111 = '2012-11-11 00:00:00'
data['double_11'] = data['order_pay_time'].map(lambda x:1 if datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').date() == \
                                               datetime.datetime.strptime(date_20121111, '%Y-%m-%d %H:%M:%S').date() else 0)
##释放内存
import gc
gc.collect()

###特征工程，对主要特征进行提取
def data_preprocess(raw, end_date='2013-06-30 23:59:59'):
    data = pd.DataFrame()
    data['customer_id'] = raw.groupby('customer_id')['customer_id'].agg('last')
    data['customer_id_cut'] = raw.groupby('customer_id')['customer_id_cut'].agg('last')
    data['customer_id_cut_100'] = raw.groupby('customer_id')['customer_id_cut_100'].agg('last')
    data['customer_id_cut_1000'] = raw.groupby('customer_id')['customer_id_cut_1000'].agg('last')

    # —————————RFM指标构建—————————————————————————————————————
    # Recency 计算最近一次消费据最终日期的时间
    edge_date = pd.to_datetime(end_date, format='%Y-%m-%d %H:%M:%S')
    data['recency'] = raw.groupby('customer_id')['order_pay_time'].last()
    data['recency'] = pd.to_datetime(data['recency'])
    data['recency'] = edge_date - data['recency']
    data['recency'] = data['recency'].astype('timedelta64[D]').astype('int')
    # Frequency 数据包含时间内的购买次数
    dedup = raw.drop_duplicates(['order_id'])  # 先对order—id进行去重
    data['frequency'] = dedup.groupby('customer_id')['order_id'].nunique()
    # Monetary 统计消费金额
    data['monetary'] = dedup.groupby('customer_id')['order_total_payment'].sum()
    # Discount-打折率统计
    data['total_discount'] = dedup.groupby('customer_id')['order_total_discount'].sum()
    data['avg_discount'] = data['total_discount'] / data['monetary']
    data['avg_discount'].fillna(0, inplace=True)
    gc.collect()
    # ——————————统计商品信息——————————————————————————————————
    data['items_total'] = dedup.groupby('customer_id')['order_total_num'].agg(np.sum)
    data['items_last'] = dedup.groupby('customer_id')['order_total_num'].agg('last')
    data['items_max'] = dedup.groupby('customer_id')['order_total_num'].agg(np.max)
    data['items_min'] = dedup.groupby('customer_id')['order_total_num'].agg(np.min)
    data['avg_items_per_order'] = data['items_total'] / dedup.groupby('customer_id')['order_id'].size()
    gc.collect()
    # ———————————商品统计—————————————————————————————————
    data['goods_id_last'] = raw.groupby('customer_id')['goods_id'].last()
    data['goods_status_last'] = raw.groupby('customer_id')['goods_status'].last()
    data['goods_price_last'] = raw.groupby('customer_id')['goods_price'].last()
    data['goods_list_time_last'] = raw.groupby('customer_id')['goods_list_time'].last()
    gc.collect()
    # ———————————商品价格统计——————————————————————————————
    data[['goods_price_max', 'goods_price_min',
          'goods_price_avg', 'goods_price_std', ]] = raw.groupby('customer_id')['goods_price'].agg(
        [np.max, np.min, np.mean, np.std])
    data['goods_price_cv'] = data['goods_price_std'] / data['goods_price_avg']
    data['goods_price_std'].fillna(0, inplace=True)
    data['goods_price_cv'].fillna(0, inplace=True)
    gc.collect()
    # ——————————用户特征处理
    data['gender'] = raw.groupby('customer_id')['customer_gender'].agg(np.max)
    data['province'] = raw.groupby('customer_id')['customer_province'].last()
    data['city'] = raw.groupby('customer_id')['customer_city'].last()
    gc.collect()
    # ——————————评价处理 ————————————————————
    # 是否评价过，评价总数，评价订单比例
    data['is_rated'] = raw.groupby('customer_id')['is_customer_rate'].max()
    data['rated_num'] = dedup.groupby('customer_id')['is_customer_rate'].sum()
    data['avg_rated'] = data['rated_num'] / dedup.groupby('customer_id')['order_id'].agg(np.size)

    # ————————-时间特征处理—————————————————————
    set_time = pd.to_datetime('2013-01-01 00:00:00', format='%Y-%m-%d %H:%M:%S')

    # 最后一单、倒数第二弹、第一单的日期（如果只有一单，三个数默认一样）
    data['order_pay_time_last'] = raw.groupby('customer_id')['order_pay_time'].last()
    data['order_pay_time_last2'] = dedup.groupby('customer_id')['order_pay_time'].apply(
        lambda x: x.iloc[len(x) - 1] if len(x) == 1 else x.iloc[len(x) - 2])
    data['order_pay_time_first'] = dedup.groupby('customer_id')['order_pay_time'].first()

    gc.collect()

    def time_feature(time):
        t = datetime.datetime.strptime(time, '%Y-%m-%d %H:%M:%S')
        return pd.Series([t.month, t.day, t.weekday(), t.hour, t.minute])

    # 提取支付订单的月份，日期，周，小时，分钟等时间特征
    data[['order_pay_time_last_month',
          'order_pay_time_last_day',
          'order_pay_time_last_weekday',
          'order_pay_time_last_hour',
          'order_pay_time_last_minute']] = data['order_pay_time_last'].apply(time_feature)
    gc.collect()

    # 最后一单和倒数第二单的时间间隔
    # 最后一单和第一单的时间间隔
    data['order_pay_time_last'] = pd.to_datetime(data['order_pay_time_last'], format='%Y-%m-%d %H:%M:%S')
    data['order_pay_time_last2'] = pd.to_datetime(data['order_pay_time_last2'], format='%Y-%m-%d %H:%M:%S')
    data['order_pay_time_first'] = pd.to_datetime(data['order_pay_time_first'], format='%Y-%m-%d %H:%M:%S')

    data['order_pay_time_diff12'] = ((data['order_pay_time_last'].dt.date - data[
        'order_pay_time_last2'].dt.date).astype('timedelta64[D]').astype('int')) / 365
    data['order_pay_time_diff_max'] = ((data['order_pay_time_last'].dt.date - data[
        'order_pay_time_first'].dt.date).astype('timedelta64[D]').astype('int')) / 365
    gc.collect()

    # 商品list 和 delist last time
    data['goods_list_time_last'] = dedup.groupby('customer_id')['goods_list_time'].last().astype('str')
    data['goods_delist_time_last'] = dedup.groupby('customer_id')['goods_delist_time'].last().astype('str')
    data['goods_list_time_diff'] = data['goods_list_time_last'].map(
        lambda x: (datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S') - set_time).days / 365)
    data['goods_delist_time_diff'] = data['goods_delist_time_last'].map(
        lambda x: (datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S') - set_time).days / 365)
    data['goods_diff'] = data['goods_delist_time_diff'] - data['goods_list_time_diff']

    data['order_pay_time_last'] = data['order_pay_time_last'].astype('str')
    data['order_pay_time_last_diff'] = data['order_pay_time_last'].map(
        lambda x: (datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S') - set_time).days / 365)
    gc.collect()
    # 最后一单和2013年初的距离：recency
    # 倒数第二单和2013年初的距离
    # 倒数第二单和2013年末的距离
    # 最后一单和2013年末的距离
    data['order_pay_time_last'] = pd.to_datetime(data['order_pay_time_last'], format='%Y-%m-%d %H:%M:%S')
    data['order_pay_time_diff_st-to-last'] = ((data['order_pay_time_last'] - set_time).astype('timedelta64[D]').astype(
        'int')) / 365
    data['order_pay_time_diff_st-to-last2'] = ((data['order_pay_time_last2'] - set_time).astype(
        'timedelta64[D]').astype('int')) / 365
    data['order_pay_time_diff_end-to-last2'] = ((data['order_pay_time_last2'] - edge_date).astype(
        'timedelta64[D]').astype('int')) / 365
    gc.collect()
    # rfmz-score
    data['recency_z-score'] = (data['recency'] - data['recency'].mean()) / data['recency'].std()
    data['frequency_z-score'] = (data['frequency'] - data['frequency'].mean()) / data['frequency'].std()
    data['monetary_z-score'] = (data['monetary'] - data['monetary'].mean()) / data['monetary'].std()
    gc.collect()
    # 2012年双11是否有购物
    # 2012年双11购物单数超过1单的
    data[['order_double11',
          'order_double11_sum']] = dedup.groupby('customer_id')['double_11'].agg([np.max, np.sum])
    gc.collect()
    return data

###构建RFM评分体系，区分用户
def RFM_model(recency, frequency, monetary):
    RFM = pd.DataFrame()

    # Recency-Score
    bins = recency.quantile(q=[0, 0.2, 0.4, 0.6, 0.8, 1], interpolation='nearest')
    bins[0] = 0
    labels = [5, 4, 3, 2, 1]
    RFM['rec_score'] = pd.cut(recency, bins, labels=labels, include_lowest=True)
    RFM['rec_score'] = RFM['rec_score'].astype(int)

    # Frequency-Score
    bins = [0, 1, 2, 4, 10, 500]
    labels = [1, 2, 3, 4, 5]
    RFM['fre_score'] = pd.cut(frequency, bins, labels=labels, include_lowest=True)
    RFM['fre_score'] = RFM['fre_score'].astype(int)

    # Monetary-Score
    bins = monetary.quantile(q=[0, 0.2, 0.4, 0.6, 0.8, 1], interpolation='nearest')
    bins[0] = 0
    labels = [1, 2, 3, 4, 5]
    RFM['mon_score'] = pd.cut(monetary, bins, labels=labels, include_lowest=True)
    RFM['mon_score'] = RFM['mon_score'].astype(int)

    # RFM and RFM score
    RFM['RFM'] = 100 * RFM['rec_score'].astype(int) + 10 * RFM['fre_score'].astype(int) + RFM['mon_score'].astype(int)
    bins = RFM['RFM'].quantile(q=[0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1], interpolation='nearest')
    bins[0] = 0
    labels = [1, 2, 3, 4, 5, 6, 7, 8]
    RFM['rfms'] = pd.cut(RFM['RFM'], bins, labels=labels, include_lowest=True)
    RFM['rfms'] = RFM['rfms'].astype(int)

    # RFM z-score
    RFM['rfm_z-score'] = (RFM['rfms'] - RFM['rfms'].mean()) / RFM['rfms'].std()

    return RFM


#构建训练数据
print('Start data preprocessing...pls wait...')

# 使用8月1日之前的数据来训练

edge_time = '2013-06-30 23:59:59'
predict_time = '2013-07-31 23:59:59'
label = set(data[data['order_pay_time'] > predict_time]['customer_id'])      # 给原始数据打标签
train_data = data[data['order_pay_time'] <= edge_time]
gc.collect()

print('Start data preprocessing train...pls wait...')
st1 = time.time()
train = data_preprocess(train_data)
print('train data runtime: ', time.time()-st1)
gc.collect()


print('Start data preprocessing test...pls wait...')
st = time.time()
test = data_preprocess(data, end_date='2013-08-31 23:59:59')
print('test data runtime: ', time.time()-st)
gc.collect()
# 打标签
train['labels']=train.index.map(lambda x:int(x in label))
gc.collect()
print(f'Data preprocessing done, total time cost {time.time() - st1}s.')


#增加RFM评分
st = time.time()

temp = RFM_model(train['recency'], train['frequency'], train['monetary'])
train = train.join(temp)
print('train data set rfm runtime:', time.time() - st)
st = time.time()
del temp
gc.collect()

temp = RFM_model(test['recency'], test['frequency'], test['monetary'])
test = test.join(temp)
del temp
gc.collect()

print('test data set rfm runtime:', time.time() - st)

# 获取每一单的时间间隔的均值，如果只有1单，则返回10**8
edge_time = '2013-07-31 23:59:59'
train_data = data[data['order_pay_time'] <= edge_time]

# 构建train test基础数据，再按付款时间去重后排序
train_data = train_data.drop_duplicates('order_pay_time')
test_data = data.drop_duplicates('order_pay_time')
train_data.sort_values(['order_pay_time'], ascending=True, inplace=True)
test_data.sort_values(['order_pay_time'], ascending=True, inplace=True)

# 让每一个特征值转为一个list，nan转为-1
train['paytime_list'] = train_data.groupby('customer_id')['order_pay_time'].apply(list)
test['paytime_list'] = test_data.groupby('customer_id')['order_pay_time'].apply(list)

train['paytime_list'].fillna(-1, inplace=True)
test['paytime_list'].fillna(-1, inplace=True)


def get_interval(df):
    total = 0
    if df == -1 or len(df) == 1:
        return 10000000

    for i in range(1, len(df)):
        total += (datetime.datetime.strptime(df[i], '%Y-%m-%d %H:%M:%S') - datetime.datetime.strptime(df[i - 1],
                                                                                                      '%Y-%m-%d %H:%M:%S')).days
    total = total / len(df)
    return total


print('Calculating order intervales...')
train['order_interval'] = train['paytime_list'].apply(get_interval)
test['order_interval'] = test['paytime_list'].apply(get_interval)

train.drop('paytime_list', axis=1, inplace=True)
test.drop('paytime_list', axis=1, inplace=True)

print('Done.')

# 去掉掉时间戳列
train.drop(['goods_list_time_last', 'goods_delist_time_last', 'order_pay_time_last', 'order_pay_time_last2',
            'order_pay_time_first'], axis=1, inplace=True)
test.drop(['goods_list_time_last', 'goods_delist_time_last', 'order_pay_time_last', 'order_pay_time_last2',
           'order_pay_time_first'], axis=1, inplace=True)

# 输出特征保存
print(f'Outputing train dataset, {len(set(train.columns))} columns involved.')
train.to_csv('train_features.csv', index=False)

print(f'Outputing test dataset, {len(set(test.columns))} columns involved.')
test.to_csv('test_features.csv', index=False)

print('Done.')
