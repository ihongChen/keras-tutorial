#! encoding=utf8
from sqlalchemy import create_engine
import pandas as pd
import preload_mnist


X_train,y_train,X_test,y_test = preload_mnist.run()

X_train = X_train.reshape(60000,28*28)
X_test = X_test.reshape(10000,28*28)

df_train = pd.DataFrame(X_train.astype('float32'))
df_train['label'] = y_train.astype('int')

# import sqlite3
# import sqlalchemy
# with sqlite3.connect('my_db.sqlite') as cnx:
#     df_train.to_sql(u'MNIST手寫辨識集_TRAIN',cnx,if_exists='replace')

# import pymysql
connstr = "mysql+pymysql://user:pwd@host:3306/test?charset=utf8" # 要加入?charset=utf8才不會亂碼
uri = 'mysql+pymysql://user:pwd@host:3306/test::mnist-train'
engine = create_engine(connstr)
conn = engine.connect()

import pymysql
## save to table
# df.to_sql('df_test',mysql_cn,flavor='mysql',if_exists='replace')
# conn= pymysql.connect(host='localhost', port=3306, user='root', passwd='',charset='UTF8')

# df_train.to_sql(u'MNIST手寫辨識_TRAIN',conn,if_exists='replace',index=False)
%time df_train.to_sql(u'MNIST手寫辨識_TRAIN',conn,if_exists='replace',index=False)

from odo import odo
%time odo(df_train,uri)
