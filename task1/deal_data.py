
import pandas as pd
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold # 计算方差
import matplotlib.pyplot as plt
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from scipy.stats import pearsonr    # 计算相关系数

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

# 过滤低方差特征
trans = VarianceThreshold(threshold=1)
x_train = trans.fit_transform(x_train)
x_test = trans.transform(x_test)
print(x_train.shape)

estimator = RandomForestClassifier(n_estimators=200,max_depth=80)
# estimator = GradientBoostingClassifier(random_state=10)
# estimator = KNeighborsClassifier(n_neighbors=50)
# estimator = LogisticRegression()
# estimator = XGBClassifier(learning_rate=0.01,
#                       n_estimators=200,           # 树的个数-10棵树建立xgboost
#                       max_depth=30,               # 树的深度
#                       min_child_weight = 1,      # 叶子节点最小权重
#                       gamma=0.,                  # 惩罚项中叶子结点个数前的参数
#                       subsample=1,               # 所有样本建立决策树
#                       colsample_btree=1,         # 所有特征建立决策树
#                       scale_pos_weight=1,        # 解决样本个数不平衡的问题
#                       random_state=27,           # 随机数
#                       slient = 0
#                       )
estimator.fit(x_train,y_train)
# 模型评估
y_pred = estimator.predict(x_test)
score = estimator.score(x_test,y_test)
print(score)

# 查看精确率，召回率，f1  准确率？？
from sklearn.metrics import classification_report

report = classification_report(y_test,y_pred,labels=[0,1],target_names=['否','是'])
print(report)