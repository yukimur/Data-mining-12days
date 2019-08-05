
import pandas as pd
import datetime
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold # 计算方差
import matplotlib.pyplot as plt
from sklearn.feature_extraction import DictVectorizer


data = pd.read_csv("data.csv",encoding="gbk")
print(data.head())
# print(data.columns)

# 人工去除无用特征、特征过滤
data = data.drop(["custid","trade_no","bank_card_no","source","id_name"],axis=1)

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
data.loc[:,'latest_query_time_year']=data['latest_query_time'].apply(lambda x : x.year)
data.loc[:,'latest_query_time_month']=data['latest_query_time'].apply(lambda x : x.month)
data.loc[:,'latest_query_time_day']=data['latest_query_time'].apply(lambda x : x.day)
data.loc[:,'loans_latest_time_year']=data['loans_latest_time'].apply(lambda x : x.year)
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
# 切分数据集
x_train,x_test,y_train,y_test = train_test_split(data.drop("status",axis=1),data["status"],random_state=2018)

# 把其中的非数字型转为one-hot
x_train = x_train.to_dict(orient="records")
x_test = x_test.to_dict(orient="records")
trans = DictVectorizer()
x_train = trans.fit_transform(x_train)
x_test = trans.transform(x_test)

# 过滤低方差特征
trans = VarianceThreshold(threshold=2)
x_train = trans.fit_transform(x_train)
print(x_train.shape)