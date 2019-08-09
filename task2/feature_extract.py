
import numpy as np
import pandas as pd
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold # 计算方差
import matplotlib.pyplot as plt
from sklearn.feature_extraction import DictVectorizer
from scipy.stats import pearsonr    # 计算相关系数
from sklearn.feature_selection import SelectKBest,SelectPercentile

data = pd.read_csv("data.csv",encoding="gbk")
print(data.head())
# print(data.columns)

# 人工去除无用特征、特征过滤
data = data.drop(["is_high_user","student_feature","Unnamed: 0","custid","trade_no","bank_card_no","source","id_name"],axis=1)

# 时间处理?
def deal_time(t):
    try:
        t = datetime.datetime.strptime(str(int(t)),"%Y%M%d")
        t = t.strftime("%Y-%M-%d")
    except:
        pass
    return t

data['first_transaction_time']=pd.to_datetime(data['first_transaction_time'].apply(lambda x:deal_time(x)))
data['latest_query_time']=pd.to_datetime(data['latest_query_time'])
data['loans_latest_time']=pd.to_datetime(data['loans_latest_time'])
data.loc[:,'first_transaction_time_year']=data['first_transaction_time'].apply(lambda x : x.year)
data.loc[:,'first_transaction_time_month']=data['first_transaction_time'].apply(lambda x : x.month)
data.loc[:,'first_transaction_time_day']=data['first_transaction_time'].apply(lambda x : x.day)
# data.loc[:,'latest_query_time_year']=data['latest_query_time'].apply(lambda x : x.year)
data.loc[:,'latest_query_time_month']=data['latest_query_time'].apply(lambda x : x.month)
data.loc[:,'latest_query_time_day']=data['latest_query_time'].apply(lambda x : x.day)
# data.loc[:,'loans_latest_time_year']=data['loans_latest_time'].apply(lambda x : x.year)
data.loc[:,'loans_latest_time_month']=data['loans_latest_time'].apply(lambda x : x.month)
data.loc[:,'loans_latest_time_day']=data['loans_latest_time'].apply(lambda x : x.day)
data.drop(["first_transaction_time","latest_query_time","loans_latest_time"],axis=1,inplace=True)

# 查看每一列种类数量
view_list = sorted([[column,data[column].value_counts().shape[0]] for column in data.columns],key=lambda x:x[1],reverse=True)
for item in view_list:
    print(item[0],item[1])

# null值替换
for column in data.columns:
    if data[column].dtype in ["float64","int64"]:
        data[column].fillna(data[column].mean(),inplace=True)
    else:
        print(column,data[column].dtype)

data.dropna(inplace=True)
for column in data.columns:
    if column !="reg_preference_for_trad":
        a = pearsonr(data[column],data["status"])
        print(column,a)
data.drop(["take_amount_in_later_12_month_highest","trans_amount_increase_rate_lately","transd_mcc","trans_days_interval_filter",
           "jewelry_consume_count_last_6_month","latest_six_month_apply","loans_credibility_behavior","loans_latest_time_day",
           "first_transaction_time_day","first_transaction_time_month"],axis=1,inplace=True)
# 切分数据集
x_train,x_test,y_train,y_test = train_test_split(data.drop("status",axis=1),data["status"],random_state=2018,test_size=0.3)

# 把其中的非数字型转为one-hot
x_train = x_train.to_dict(orient="records")
x_test = x_test.to_dict(orient="records")
trans = DictVectorizer()
x_train = trans.fit_transform(x_train)
x_test = trans.transform(x_test)

# 标准化
trans = StandardScaler(with_mean=False)
x_train = trans.fit_transform(x_train)
x_test = trans.transform(x_test)


def CalcIV(Xvar, Yvar):
    N_0 = np.sum(Yvar == 0)
    N_1 = np.sum(Yvar == 1)
    N_0_group = np.zeros(np.unique(Xvar).shape)

    N_1_group = np.zeros(np.unique(Xvar).shape)
    for i in range(len(np.unique(Xvar))):
        N_0_group[i] = Yvar[(Xvar == np.unique(Xvar)[i]) & (Yvar == 0)].count()
        N_1_group[i] = Yvar[(Xvar == np.unique(Xvar)[i]) & (Yvar == 1)].count()
    iv = np.sum((N_0_group / N_0 - N_1_group / N_1) * np.log((N_0_group / N_0) / (N_1_group / N_1)))
    if iv >= 1.0:  ## 处理极端值
        iv = 1
    return iv

def caliv_batch(df, Yvar):
    ivlist = []
    for col in df.columns:
        iv = CalcIV(df[col], Yvar)
        ivlist.append(iv)
    names = list(df.columns)
    iv_df = pd.DataFrame({'Var': names, 'Iv': ivlist}, columns=['Var', 'Iv'])

    return iv_df, ivlist

im_iv, ivl = im_iv, ivl = caliv_batch(data.drop("status",axis=1),data["status"])
print(im_iv, ivl)


from sklearn.ensemble import RandomForestClassifier

feat_labels=data.drop("status",axis=1).columns[1:]
forest=RandomForestClassifier(n_estimators=100,n_jobs=-1,random_state=0)
forest.fit(x_train,y_train)
importances=forest.feature_importances_
indices=np.argsort(importances)[::-1]
for f in range(x_train.shape[1]):
    #给予10000颗决策树平均不纯度衰减的计算来评估特征重要性
    print ("%2d) %-*s %f" % (f+1,30,feat_labels[f],importances[indices[f]]) )
#可视化特征重要性-依据平均不纯度衰减
plt.title('Feature Importance-RandomForest')
plt.bar(range(x_train.shape[1]),importances[indices],color='lightblue',align='center')
plt.xticks(range(x_train.shape[1]),feat_labels,rotation=90)
plt.xlim([-1,x_train.shape[1]])
plt.tight_layout()
plt.show()
