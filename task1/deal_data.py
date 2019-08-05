
import pandas as pd
import datetime
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold # 计算方差


data = pd.read_csv("data.csv",encoding="gbk")
print(data.head())
print(data.columns)

# 人工去除无用特征、特征过滤
data = data.drop(["custid","trade_no","bank_card_no","source","id_name","status"],axis=1)
#
data['latest_query_time'] = data['latest_query_time'].str.replace("-","")
data['loans_latest_time'] = data['loans_latest_time'].str.replace("-","")

# 过滤低方差特征
trans = VarianceThreshold(threshold=1)
data = trans.fit_transform(data.drop("reg_preference_for_trad",axis=1))
print(data.columns)