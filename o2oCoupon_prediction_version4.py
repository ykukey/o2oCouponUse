#-*- coding:utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
import datetime
from datetime import datetime as dt
from random import sample

#数据预处理
from sklearn.model_selection import KFold
import imblearn
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

#特征评估
from sklearn.feature_selection import SelectKBest,chi2
from minepy import MINE
from sklearn.model_selection import cross_val_predict 
from sklearn.model_selection import StratifiedKFold
#建模
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb

#验证，调参
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

#评估指标
from sklearn.model_selection import learning_curve
from sklearn.metrics import make_scorer,confusion_matrix,precision_score,recall_score,f1_score,auc,roc_auc_score,roc_curve,auc

print("读取文件")
finPath="~/projects/MachineLearning/classifier/o2o"

offline=pd.read_csv(finPath+"/ccf_offline_stage1_train.csv")
online=pd.read_csv(finPath+"/ccf_online_stage1_train.csv")
test=pd.read_csv(finPath+"/ccf_offline_stage1_test_revised.csv")

print("浮点数转日期")
offline['Date_received']=offline['Date_received'].map(lambda x:dt.strptime(str(int(x)),'%Y%m%d') if x==x else x)
offline['Date']=offline['Date'].map(lambda x:dt.strptime(str(int(x)),'%Y%m%d') if x==x else x)
online['Date_received']=online['Date_received'].map(lambda x:dt.strptime(str(int(x)),'%Y%m%d') if x==x else x)
online['Date']=online['Date'].map(lambda x:dt.strptime(str(int(x)),'%Y%m%d') if x==x else x)
test['Date_received']=test['Date_received'].map(lambda x:dt.strptime(str(int(x)),'%Y%m%d') if x==x else x)

###################################################################线下消费标签
#"0":领券不消费（15天外消费）
#"1":领券15天内消费
#"2":不领券消费
offline['TN']=np.where(offline['Coupon_id'].notnull() & offline['Date'].notnull() & (offline['Date']<=offline['Date_received']+datetime.timedelta(days=15)),1,np.where(offline['Coupon_id'].isnull() & offline['Date'].notnull(),2,0))

################################################################线上消费标签
online['TN']=np.where(online['Coupon_id'].notnull() & online['Date'].notnull() & (online['Date']<=online['Date_received']+datetime.timedelta(days=15)),1,np.where(online['Coupon_id'].isnull() & online['Date'].notnull(),2,0))

#################################################################提取训练样本
print("提取线下训练样本：")
#提取训练样本：领券行为（领券15天内消费，领券15外消费）
off_train=offline.loc[offline['TN']!=2,:]
off_train['tag']='offt'
test['tag']='test'
test['TN']=2

#融合训练集和测试集
merge_df=pd.concat([off_train,test],axis=0,sort=False,ignore_index=True)

print("数据预处理")
#优惠券特征构造:（1）自身特征；（2）根据训练集得出的画像
#优惠类型：满减:mj，限时优惠:fixed，折扣:ratio  (线下无“fixed”类型的优惠券)
merge_df["DiscountType"]=np.where(merge_df.Discount_rate.map(lambda x:":" in x),'mj',np.where(merge_df.Discount_rate=='fixed','fixed','ratio'))
merge_df.loc[merge_df['tag']!='test',:].groupby('DiscountType')[['TN']].mean()

#满减门槛
merge_df['mjThresh']=merge_df.Discount_rate.map(lambda x:float(x.split(":")[0]) if ":" in x else 0)
merge_df.loc[merge_df['tag']!='test',:].groupby('mjThresh')[['TN']].mean()

#满减力度
merge_df['mjratio']=merge_df.Discount_rate.map(lambda x:float(x.split(":")[1])/float(x.split(":")[0]) if ":" in x else 0)
merge_df.loc[merge_df['tag']!='test',:].groupby('mjratio')[['TN']].mean()

#折扣力度
merge_df['ratioIntense']=merge_df.Discount_rate.map(lambda x:float(x) if (":" not in x) and (x!="fixed") else 0)
merge_df.loc[merge_df['tag']!='test',:].groupby('ratioIntense')[['TN']].mean()

#优惠券接受时间
merge_df['received_weekday']=merge_df['Date_received'].map(lambda x:x.weekday())
merge_df.loc[merge_df['tag']!='test',:].groupby('received_weekday')[['TN']].mean()

#商家特征构造
#线下商家距离
merge_df.loc[(merge_df['tag']=='offt')&(merge_df['Distance'].notnull()),:].groupby('Distance')[['TN']].mean()

#距离缺失值预测	
#数据类型转换
temp=pd.get_dummies(merge_df,columns=['DiscountType','received_weekday'])

#######################################################################################初步模型调试
#训练集，测试集划分
train_x_ori,train_y_ori=temp[temp.tag=='offt'].drop(["User_id","Merchant_id","Coupon_id","Discount_rate","Date_received","tag","TN","Date"],axis=1),temp[temp.tag=="offt"].TN
test_x,test_y=temp[temp.tag=='test'].drop(["User_id","Merchant_id","Coupon_id","Discount_rate","Date_received","tag","TN"],axis=1),temp[temp.tag=="test"].TN
train_x_ori.fillna({"Distance":-1},inplace=True)
test_x.fillna({"Distance":-1},inplace=True)

#训练集继续划分为 训练集，验证集划分（8:2）
train_x,valid_x,train_y,valid_y=train_test_split(train_x_ori,train_y_ori,test_size=0.2,random_state=42)

train_y.value_counts()#正负：1:10

#不平衡抽样
#欠抽样处理:随机欠抽样
model_RandomUnderSampler = RandomUnderSampler()
train_x_undersam_arr, train_y_undersam_arr =model_RandomUnderSampler.fit_sample(train_x,train_y)
train_x_undersam=pd.DataFrame(train_x_undersam_arr,columns=train_x.columns)
train_y_undersam=pd.Series(train_y_undersam_arr)

#洗牌（不平衡样本处理后按标签顺序排列，需要随机打乱）
np.random.seed(42)
shuffle_index = np.random.permutation(train_x_undersam.shape[0])
train_x_undersam,train_y_undersam=train_x_undersam.loc[shuffle_index,:],train_y_undersam.loc[shuffle_index]

#模型测试：
#训练集
#(1)提取特征1:前3个月消费行为 20160215-20160515用户消费行为
offline_all_pre=offline.loc[((offline.Date<dt.strptime("20160515","%Y%m%d")) & (offline.Date>=dt.strptime("20160215","%Y%m%d")))|((offline.Date_received>=dt.strptime("20160215","%Y%m%d")) & (offline.Date_received<dt.strptime("20160515","%Y%m%d"))),:]#领券或消费时间在前五个月
online_all_pre=online.loc[((online.Date<dt.strptime("20160515","%Y%m%d")) & (online.Date>=dt.strptime("20160215","%Y%m%d")))|((online.Date_received>=dt.strptime("20160215","%Y%m%d")) & (online.Date_received<dt.strptime("20160515","%Y%m%d"))),:]# 线上领券或消费时间在前五个月
#(1)训练集1：1个月领券数据 20160515-20160615领券用户消费数据（训练集）
offline_train=temp.loc[(temp.Date_received>=dt.strptime("20160515","%Y%m%d"))&(temp.tag!="test"),:]

#用户重叠
len(set(offline_train.User_id.unique())-set(offline_all_pre.User_id.unique()))#98064,0.56
#商户重叠
len(set(offline_train.Merchant_id.unique())-set(offline_all_pre.Merchant_id.unique()))#438, 0.10
#优惠券重叠
len(set(offline_train.Coupon_id.unique())-set(offline_all_pre.Coupon_id.unique()))#3284,0.52

#(2)提取特征2:前3个月消费行为 20160115-20160415所有用户消费行为
test_offline_all_pre=offline.loc[((offline.Date<dt.strptime("20160415","%Y%m%d")) & (offline.Date>=dt.strptime("20160115","%Y%m%d")))|((offline.Date_received>=dt.strptime("20160115","%Y%m%d")) & (offline.Date_received<dt.strptime("20160415","%Y%m%d"))),:]#领券或消费时间在前五个月
test_online_all_pre=online.loc[((online.Date<dt.strptime("20160415","%Y%m%d")) & (online.Date>=dt.strptime("20160115","%Y%m%d")))|((online.Date_received>=dt.strptime("20160115","%Y%m%d")) & (online.Date_received<dt.strptime("20160415","%Y%m%d"))),:]# 线上领券或消费时间在前五个月
#(2)训练集2：1个月领券数据 20160415-20160515领券用户消费数据（训练集）
test_offline_train=temp.loc[(temp.Date_received>=dt.strptime("20160415","%Y%m%d"))&(temp.Date_received<dt.strptime("20160515","%Y%m%d"))&(temp.tag!="test"),:]

#用户重叠
len(set(test_offline_train.User_id.unique())-set(test_offline_all_pre.User_id.unique()))#55540,0.54
#商户重叠
len(set(test_offline_train.Merchant_id.unique())-set(test_offline_all_pre.Merchant_id.unique()))#732,0.21
#优惠券重叠
len(set(test_offline_train.Coupon_id.unique())-set(test_offline_all_pre.Coupon_id.unique()))#3540,0.75

#增加线下特征
trainU_O=o2oFunc.addUFeatures(offline_all_pre,offline_train)#添加用户特征
trainU_M_O=o2oFunc.addMFeatures(offline_all_pre,trainU_O["train"][-1])#添加商户特征
trainU_M_C_O=o2oFunc.addCFeatures(offline_all_pre,trainU_M_O["train"][-1])#添加优惠券特征
trainU_M_UM_O=o2oFunc.addUMFeatures(offline_all_pre,trainU_M_C_O)#添加用户商家特征
trainU_M_UM_UC_O=o2oFunc.addUCFeatures(offline_all_pre,trainU_M_UM_O)#添加用户优惠券特征

test_trainU_O=o2oFunc.addUFeatures(test_offline_all_pre,test_offline_train)#添加用户特征
test_trainU_M_O=o2oFunc.addMFeatures(test_offline_all_pre,test_trainU_O["train"][-1])#添加商户特征
test_trainU_M_C_O=o2oFunc.addCFeatures(test_offline_all_pre,test_trainU_M_O["train"][-1])#添加优惠券特征
test_trainU_M_UM_O=o2oFunc.addUMFeatures(test_offline_all_pre,test_trainU_M_C_O)#添加用户商家特征
test_trainU_M_UM_UC_O=o2oFunc.addUCFeatures(test_offline_all_pre,test_trainU_M_UM_O)#添加用户优惠券特征

#增加线上用户特征
len(set(offline_train.User_id)-set(online_all_pre.User_id))#109277,存在重合用户
len(set(offline_train.Merchant_id)-set(online_all_pre.Merchant_id))#4660，不存在重合商户
len(set(offline_train.Coupon_id.unique())-set(online_all_pre.Coupon_id.unique()))#7160，不存在重合优惠券
trainU_OL=o2oFunc.addUFeatures(online_all_pre,offline_train)["train"][-1].rename(columns=lambda x:x+"_OL")#添加用户线上特征
trainU_OL.drop_duplicates(inplace=True)#去重
trainU_M_UM_UC_UOL=pd.merge(trainU_M_UM_UC_O,trainU_OL,left_on=list(offline_train.columns),right_on=[i+"_OL" for i in offline_train.columns],how="left").drop([i+"_OL" for i in offline_train.columns],axis=1)#增添线上用户特征


test_trainU_OL=o2oFunc.addUFeatures(test_online_all_pre,test_offline_train)["train"][-1].rename(columns=lambda x:x+"_OL")#添加用户线上特征
test_trainU_OL.drop_duplicates(inplace=True)#去重
test_trainU_M_UM_UC_UOL=pd.merge(test_trainU_M_UM_UC_O,test_trainU_OL,left_on=list(test_offline_train.columns),right_on=[i+"_OL" for i in test_offline_train.columns],how="left").drop([i+"_OL" for i in test_offline_train.columns],axis=1)#增添线上用户特征

#增加线下特征（用户，优惠券，用户X优惠券）
offline_train_OL=o2oFunc.addOLFeatures(online_all_pre,offline_train)#增添线上特征
offline_train_OL.drop_duplicates(inplace=True)#去重

test_offline_train_OL=o2oFunc.addOLFeatures(test_online_all_pre,test_offline_train)#增添线上特征
test_offline_train_OL.drop_duplicates(inplace=True)#去重

#合并线上，线下特征
trainU_M_UM_UC_OL_O=pd.merge(trainU_M_UM_UC_UOL,offline_train_OL,on=list(offline_train.columns),how="left")

test_trainU_M_UM_UC_OL_O=pd.merge(test_trainU_M_UM_UC_UOL,test_offline_train_OL,on=list(test_offline_train.columns),how="left")
#增加预测区间特征
#trainU_M_UM_UC_OL_Pre_O=addPreFeatures(trainU_M_UM_UC_OL_O)
dftest=o2oFunc.addPreFeatures(offline_train)
dftest.drop_duplicates(inplace=True)
trainU_M_UM_UC_OL_Pre_O=pd.merge(left=trainU_M_UM_UC_OL_O,right=dftest,on=list(offline_train.columns),how="left")
trainU_M_UM_UC_OL_Pre_O["cvTag"]="train"

test_dftest=o2oFunc.addPreFeatures(test_offline_train)
test_dftest.drop_duplicates(inplace=True)
test_trainU_M_UM_UC_OL_Pre_O=pd.merge(left=test_trainU_M_UM_UC_OL_O,right=test_dftest,on=list(test_offline_train.columns),how="left")
test_trainU_M_UM_UC_OL_Pre_O["cvTag"]="test"

#训练集，测试集融合
merge_train_test=pd.concat([trainU_M_UM_UC_OL_Pre_O,test_trainU_M_UM_UC_OL_Pre_O],axis=0,ignore_index=True)
'''
#查看当前所有变量（释放内存）
dir()
#删除中间变量,节省内存

'''

#规范化
cols=["cvTag","User_id","Merchant_id","Coupon_id","Discount_rate","Date_received","Date","TN","tag","DiscountType_mj","DiscountType_ratio","received_weekday_0","received_weekday_1","received_weekday_2","received_weekday_3","received_weekday_4","received_weekday_5","received_weekday_6","new_user_0.0","new_user_1.0"]

#添加User,Merchant,Coupon,UM,UC,OnLine特征后stacking模型效果的验证
merge_train_test_std=merge_train_test.fillna({"Distance":-1})
merge_train_test_stdTemp=merge_train_test_std.drop(cols,axis=1).apply(lambda x:o2oFunc.dataStand(x),axis=0)
merge_train_test_std.loc[:,merge_train_test_stdTemp.columns]=merge_train_test_stdTemp

merge_train_test_std_drop=merge_train_test_std.drop(["User_id","Merchant_id","Coupon_id","Discount_rate","Date_received","Date","tag"],axis=1)

#过拟合训练：判断特征信息是否足够
gnh_trainU_M_UM_UC_OL_Pre_x,gnh_trainU_M_UM_UC_OL_Pre_y=merge_train_test_std_drop.drop(["TN","cvTag"],axis=1),merge_train_test_std_drop.TN
gnh_undersam=o2oFunc.underSampler(gnh_trainU_M_UM_UC_OL_Pre_x,gnh_trainU_M_UM_UC_OL_Pre_y)
gnh_undersam_x=gnh_undersam["train_x"]
gnh_undersam_y=gnh_undersam["train_y"]

#xgboost
xgbc=xgb.XGBClassifier(n_jobs=-1,random_state=42)
xgbc.fit(gnh_undersam_x,gnh_undersam_y)
roc_auc_score(gnh_undersam_y,xgbc.predict(gnh_undersam_x))#0.7986 #0.8331 #0.8098 batch
pd.Series(xgbc.feature_importances_,index=gnh_undersam_x.columns).sort_values(ascending=False)

#stackingModel
rfr=RandomForestClassifier(n_jobs=-1,random_state=42,n_estimators=200)
ext=ExtraTreesClassifier(n_estimators=200,n_jobs=-1,random_state=42,oob_score=True,bootstrap=True)
gbmc=GradientBoostingClassifier(random_state=42)
xgbc=xgb.XGBClassifier(n_jobs=-1,random_state=42)

gnh_P1_output=o2oFunc.stackingP1(rfr,ext,gbmc,xgbc,gnh_undersam_x,gnh_undersam_y,gnh_undersam_x,cv=10)
gnh_stack_train_x,gnh_stack_train_y=gnh_P1_output["train_pred"].drop("realY",axis=1),gnh_P1_output["train_pred"]["realY"]
gnh_test_x=gnh_P1_output["test_pred"]

paramdic={"C":[0.1,1,10]}
clf=LogisticRegression(random_state=42,max_iter=10000,penalty="l2",class_weight="balanced")
grid=GridSearchCV(clf,param_grid=paramdic,scoring="roc_auc",cv=10)
grid.fit(gnh_stack_train_x,gnh_stack_train_y)
grid.best_params_#C=10 #C=1
grid.best_score_#0.8783 #0.9165 #0.9300 #0.8978 batch

clf1=LogisticRegression(random_state=42,max_iter=10000,penalty="l2",class_weight="balanced",C=1)
clf1.fit(gnh_stack_train_x,gnh_stack_train_y)
roc_auc_score(gnh_undersam_y,clf1.predict(gnh_test_x))#0.7992 #0.8396 #0.9199 #0.9258 batch

#含标签数据集划分为训练集，测试集
trainU_M_UM_UC_OL_Pre_x,validU_M_UM_UC_OL_Pre_x,trainU_M_UM_UC_OL_Pre_y,validU_M_UM_UC_OL_Pre_y=merge_train_test_std_drop[merge_train_test_std_drop.cvTag=="train"].drop(["TN","cvTag"],axis=1),merge_train_test_std_drop[merge_train_test_std_drop.cvTag=="test"].drop(["TN","cvTag"],axis=1),merge_train_test_std_drop[merge_train_test_std_drop.cvTag=="train"].TN,merge_train_test_std_drop[merge_train_test_std_drop.cvTag=="test"].TN

trainU_M_UM_UC_OL_Pre_y.value_counts()

train_undersam=o2oFunc.underSampler(trainU_M_UM_UC_OL_Pre_x,trainU_M_UM_UC_OL_Pre_y)
trainU_M_UM_UC_OL_Pre_x_undersam=train_undersam["train_x"]
trainU_M_UM_UC_OL_Pre_y_undersam=train_undersam["train_y"]

#集成学习器调参
#未调参：n_estimators=200,auc=0.84197
#未调batch: n_etimators=200,auc=0.7822
#调参batch：0.7853
#(1)RandomForest:n_estimators,max_depth,max_features,min_samples_leaf
n_estimators=[200]
max_depth=[i for i in range(10,30,2)]
max_features=["sqrt"]
min_samples_leaf=[1]
rfr_res=pd.DataFrame(columns=["params","precision","recall","f1","auc"])
for esNum in n_estimators:
    for dep in max_depth:
        for feNum in max_features:
            for leafNum in min_samples_leaf:
                param=("n_estimators:"+str(esNum),"max_depth:"+str(dep),"max_features:"+str(feNum),"min_samples_leaf:"+str(leafNum))
                rfr=RandomForestClassifier(n_jobs=-1,random_state=42,oob_score=True,n_estimators=esNum,max_depth=dep,max_features=feNum,min_samples_leaf=leafNum)
                rfr.fit(trainU_M_UM_UC_OL_Pre_x_undersam,trainU_M_UM_UC_OL_Pre_y_undersam)
                y_pred=rfr.predict(validU_M_UM_UC_OL_Pre_x)
                precision=precision_score(validU_M_UM_UC_OL_Pre_y,y_pred)
                recall=recall_score(validU_M_UM_UC_OL_Pre_y,y_pred)
                f1=f1_score(validU_M_UM_UC_OL_Pre_y,y_pred)
                auc=roc_auc_score(validU_M_UM_UC_OL_Pre_y,y_pred)
                se=pd.DataFrame([[param,precision,recall,f1,auc]],columns=["params","precision","recall","f1","auc"])
                rfr_res=pd.concat([rfr_res,se],axis=0,ignore_index=True)

rfr_res.sort_values(by="auc",ascending=False)#(200,18,None,1)  auc:0.84678 #batch (200,14,"sqrt",1) auc:0.785355

#交叉验证调参
paramdic={"n_estimators":[200],"max_depth":[i for i in range(12,26,2)],"max_features":[None],"min_samples_leaf":[1]}
rfr_undersam=RandomForestClassifier(random_state=42,n_jobs=-1,oob_score=True,class_weight="balanced")
grid=GridSearchCV(rfr_undersam,param_grid=paramdic,scoring="roc_auc",cv=5)
grid.fit(trainU_M_UM_UC_OL_Pre_x_undersam,trainU_M_UM_UC_OL_Pre_y_undersam)
grid.best_params_#(200,14,None,1)
grid.best_score_#auc:0.9210
#交叉验证获取最优参数后应用于指定验证集
roc_auc_score(validU_M_UM_UC_OL_Pre_y,grid.predict(validU_M_UM_UC_OL_Pre_x))#(200,14,None,1) auc:0.84638

#(2)Extre-RandomForest
#未调参：batch n_estimators=200,quc=0.77429
n_estimators=[200]
max_depth=[i for i in range(10,30,2)]
max_features=["sqrt"]
min_samples_leaf=[1]
ext_res=pd.DataFrame(columns=["params","precision","recall","f1","auc"])
for esNum in n_estimators:
    for dep in max_depth:
        for feNum in max_features:
            for leafNum in min_samples_leaf:
                param=("n_estimators:"+str(esNum),"max_depth:"+str(dep),"max_features:"+str(feNum),"min_samples_leaf:"+str(leafNum))
                ext=ExtraTreesClassifier(n_estimators=esNum,max_depth=dep,max_features=feNum,min_samples_leaf=leafNum,n_jobs=-1,random_state=42,oob_score=True,bootstrap=True)
                ext.fit(trainU_M_UM_UC_OL_Pre_x_undersam,trainU_M_UM_UC_OL_Pre_y_undersam)
                y_pred=ext.predict(validU_M_UM_UC_OL_Pre_x)
                precision=precision_score(validU_M_UM_UC_OL_Pre_y,y_pred)
                recall=recall_score(validU_M_UM_UC_OL_Pre_y,y_pred)
                f1=f1_score(validU_M_UM_UC_OL_Pre_y,y_pred)
                auc=roc_auc_score(validU_M_UM_UC_OL_Pre_y,y_pred)
                se=pd.DataFrame([[param,precision,recall,f1,auc]],columns=["params","precision","recall","f1","auc"])
                ext_res=pd.concat([ext_res,se],axis=0,ignore_index=True)

ext_res.sort_values(by="auc",ascending=False)#(200,20,None,1)  auc:0.84752 #(200,26,sqrt,1) 0.7766


#(3)XGBoost:
#未调参:batch auc=0.7748
#调参:batch auc=0.7859  
#1 先固定learning_rate,调节最适估计器数n_estimators
#2 再调树参数（主要调节:max_depth,min_child_weight,gamma,正则化参）
#3 最后调迭代次数和学习率

#1 固定learning_rate 调n_estimators
n_estimators=[i for i in range(50,550,50)]
xgbc_res=pd.DataFrame(columns=["params","precision","recall","f1","auc"])
for n in n_estimators:
    param=("n_estimators:"+str(n))
    xgbc1=xgb.XGBClassifier(n_jobs=-1,random_state=42,n_estimators=n)
    xgbc1.fit(trainU_M_UM_UC_OL_Pre_x_undersam,trainU_M_UM_UC_OL_Pre_y_undersam)
    y_pred=xgbc1.predict(validU_M_UM_UC_OL_Pre_x)
    precision=precision_score(validU_M_UM_UC_OL_Pre_y,y_pred)
    recall=recall_score(validU_M_UM_UC_OL_Pre_y,y_pred)
    f1=f1_score(validU_M_UM_UC_OL_Pre_y,y_pred)
    auc=roc_auc_score(validU_M_UM_UC_OL_Pre_y,y_pred)
    se=pd.DataFrame([[param,precision,recall,f1,auc]],columns=["params","precision","recall","f1","auc"])
    xgbc_res=pd.concat([xgbc_res,se],axis=0,ignore_index=True)

xgbc_res.sort_values(by="auc",ascending=False #batch 250 auc:0.7807)

#2 调节树参
max_depth=[4,5,6,7,9]
min_child_weight=[1,3,5]
gamma=[0,0.1,0.2,0.3,0.4]
xgbc1_res=pd.DataFrame(columns=["params","precision","recall","f1","auc"])
for dep in max_depth:
    for cw in min_child_weight:
        for g in gamma:
            param=("max_depth:"+str(dep),"min_child_weight:"+str(cw),"gamma:"+str(g))
            xgbc1=xgb.XGBClassifier(n_jobs=-1,random_state=42,max_depth=dep,min_child_weight=cw,gamma=g,n_estimators=250)
            xgbc1.fit(trainU_M_UM_UC_OL_Pre_x_undersam,trainU_M_UM_UC_OL_Pre_y_undersam)
            y_pred=xgbc1.predict(validU_M_UM_UC_OL_Pre_x)
            precision=precision_score(validU_M_UM_UC_OL_Pre_y,y_pred)
            recall=recall_score(validU_M_UM_UC_OL_Pre_y,y_pred)
            f1=f1_score(validU_M_UM_UC_OL_Pre_y,y_pred)
            auc=roc_auc_score(validU_M_UM_UC_OL_Pre_y,y_pred)
            se=pd.DataFrame([[param,precision,recall,f1,auc]],columns=["params","precision","recall","f1","auc"])
            xgbc1_res=pd.concat([xgbc1_res,se],axis=0,ignore_index=True)

xgbc1_res.sort_values(by="auc",ascending=False)#(9,3,0.1) auc:0.8515  batch:(5,1,0,250) 0.7848


#调booster参
n_estimators=[50,100,150,200,250]
learning_rate=[0.04,0.06,0.08,0.1,0.12,0.14]
xgbc2_res=pd.DataFrame(columns=["params","precision","recall","f1","auc"])
for esNum in n_estimators:
    for lr in learning_rate:
        param=("n_estimators:"+str(esNum),"learning_rate:"+str(lr))
        xgbc2=xgb.XGBClassifier(n_jobs=-1,random_state=42,max_depth=5,min_child_weight=1,gamma=0,n_estimators=esNum,learning_rate=lr)
        xgbc2.fit(trainU_M_UM_UC_OL_Pre_x_undersam,trainU_M_UM_UC_OL_Pre_y_undersam)
        y_pred=xgbc2.predict(validU_M_UM_UC_OL_Pre_x)
        precision=precision_score(validU_M_UM_UC_OL_Pre_y,y_pred)
        recall=recall_score(validU_M_UM_UC_OL_Pre_y,y_pred)
        f1=f1_score(validU_M_UM_UC_OL_Pre_y,y_pred)
        auc=roc_auc_score(validU_M_UM_UC_OL_Pre_y,y_pred)
        se=pd.DataFrame([[param,precision,recall,f1,auc]],columns=["params","precision","recall","f1","auc"])
        xgbc2_res=pd.concat([xgbc2_res,se],axis=0,ignore_index=True)

xgbc2_res.sort_values(by="auc",ascending=False)#(100,0.1) auc=0.8515 #(150,0.1) 0.7859

#(4)GBM
#batch 未调参：auc:0.7747
#batch 调参： auc:0.7832

#固定learning_rate,调n_estimators
n_estimators=[i for i in range(50,550,50)]
gbm_res=pd.DataFrame(columns=["params","precision","recall","f1","auc"])
for esNum in n_estimators:
    param=("n_estimators:"+str(esNum))
    gbmc=GradientBoostingClassifier(random_state=42,n_estimators=esNum)
    gbmc.fit(trainU_M_UM_UC_OL_Pre_x,trainU_M_UM_UC_OL_Pre_y)
    y_pred=gbmc.predict(validU_M_UM_UC_OL_Pre_x)
    precision=precision_score(validU_M_UM_UC_OL_Pre_y,y_pred)
    recall=recall_score(validU_M_UM_UC_OL_Pre_y,y_pred)
    f1=f1_score(validU_M_UM_UC_OL_Pre_y,y_pred)
    auc=roc_auc_score(validU_M_UM_UC_OL_Pre_y,y_pred)
    se=pd.DataFrame([[param,precision,recall,f1,auc]],columns=["params","precision","recall","f1","auc"])
    gbm_res=pd.concat([gbm_res,se],axis=0,ignore_index=True)

gbm_res.sort_values(by="auc",ascending=False)#batch 400 auc=0.6389


#调树参：max_depth,min_samples_split,max_features
max_depth=[3,4,5,6,7,9]
min_samples_leaf=[1,3,5,7]
max_features=["sqrt"]
gbmc2_res=pd.DataFrame(columns=["params","precision","recall","f1","auc"])
for dep in max_depth:
    for leafNum in min_samples_leaf:
        for feNum in max_features:
            param=("max_depth:"+str(dep),"min_samples_leaf:"+str(leafNum),"max_features:"+str(feNum))
            gbmc2=GradientBoostingClassifier(random_state=42,max_depth=dep,min_samples_leaf=leafNum,max_features=feNum,n_estimators=400)
            gbmc2.fit(trainU_M_UM_UC_OL_Pre_x_undersam,trainU_M_UM_UC_OL_Pre_y_undersam)
            y_pred=gbmc2.predict(validU_M_UM_UC_OL_Pre_x)
            precision=precision_score(validU_M_UM_UC_OL_Pre_y,y_pred)
            recall=recall_score(validU_M_UM_UC_OL_Pre_y,y_pred)
            f1=f1_score(validU_M_UM_UC_OL_Pre_y,y_pred)
            auc=roc_auc_score(validU_M_UM_UC_OL_Pre_y,y_pred)
            se=pd.DataFrame([[param,precision,recall,f1,auc]],columns=["params","precision","recall","f1","auc"])
            gbmc2_res=pd.concat([gbmc2_res,se],axis=0,ignore_index=True)

gbmc2_res.sort_values(by="auc",ascending=False)#(11,3,sqrt) auc:0.8503 #batch (4,7,sqrt) auc:0.7828

#调n_estimators,learning_rate
n_estimators=[50,100,150]
learning_rate=[0.06,0.08,0.1,0.12,0.14]
gbm3_res=pd.DataFrame(columns=["params","precision","recall","f1","auc"])
for esNum in n_estimators:
    for lr in learning_rate:
        param=("n_estimators:"+str(esNum),"learning_rate:"+str(lr))
        gbm3=GradientBoostingClassifier(random_state=42,max_depth=4,min_samples_leaf=7,max_features="sqrt",n_estimators=esNum,learning_rate=lr)
        gbm3.fit(trainU_M_UM_UC_OL_Pre_x_undersam,trainU_M_UM_UC_OL_Pre_y_undersam)
        y_pred=gbm3.predict(validU_M_UM_UC_OL_Pre_x)
        precision=precision_score(validU_M_UM_UC_OL_Pre_y,y_pred)
        recall=recall_score(validU_M_UM_UC_OL_Pre_y,y_pred)
        f1=f1_score(validU_M_UM_UC_OL_Pre_y,y_pred)
        auc=roc_auc_score(validU_M_UM_UC_OL_Pre_y,y_pred)
        se=pd.DataFrame([[param,precision,recall,f1,auc]],columns=["params","precision","recall","f1","auc"])
        gbm3_res=pd.concat([gbm3_res,se],axis=0,ignore_index=True)

gbm3_res.sort_values(by="auc",ascending=False)#(150,0.08) auc:0.8508 #batch (100,0.1) auc:0.7832


#学习曲线:自定义，含训练集，交叉验证集，验证集，
#randomForest
size=[0.1,0.25,0.5,0.75,1]
samples_x=trainU_M_UM_UC_OL_Pre_x_undersam.copy()
samples_y=trainU_M_UM_UC_OL_Pre_y_undersam.copy()
rfr_undersam=RandomForestClassifier(n_jobs=-1,random_state=42,oob_score=True,n_estimators=200,max_depth=18,min_samples_leaf=1)
rfr_out=o2oFunc.learningCurve(rfr_undersam,samples_x,samples_y,validU_M_UM_UC_OL_Pre_x,validU_M_UM_UC_OL_Pre_y,size,"roc_auc",cv=5)
plt.plot(rfr_out[0],rfr_out[1],"b",color="green",label="train_undersam_auc")
plt.plot(rfr_out[0],rfr_out[2],"b",color="blue",label="CV_auc")
plt.plot(rfr_out[0],rfr_out[3],"b",color="red",label="valid_auc")
plt.legend()
plt.xlabel("TrainSize")
plt.ylabel("AUC")
plt.title("Undersam_RandomForest LearningCurve")
plt.savefig("Undersam_RandomForest-learningCurve.png")
plt.close()

#stacking模型
P1_output=o2oFunc.stackingP1(rfr,ext,gbmc,xgbc,trainU_M_UM_UC_OL_Pre_x_undersam,trainU_M_UM_UC_OL_Pre_y_undersam,validU_M_UM_UC_OL_Pre_x,cv=10)
stack_train_x,stack_train_y=P1_output['train_pred'].drop('realY',axis=1),P1_output['train_pred']['realY']
P2_test_x=P1_output['test_pred']

#logistic模型调参
paramdic={"C":[0.1,1,10]}
clf=LogisticRegression(random_state=42,max_iter=10000,penalty='l2',class_weight="balanced")
grid=GridSearchCV(clf,param_grid=paramdic,scoring='roc_auc',cv=10)#网格搜索
grid.fit(stack_train_x,stack_train_y)
grid.best_score_#0.8777 #0.9164
grid.best_params_#{'C': 10} #{"C":1}
#指标评估
valid_clf=LogisticRegression(random_state=42,max_iter=10000,penalty='l2',class_weight="balanced",C=10)
valid_clf.fit(stack_train_x,stack_train_y)

valid_pred=valid_clf.predict(P2_test_x)
logit_y_pro=valid_clf.predict_proba(P2_test_x)
confusion_matrix(validU_M_UM_UC_OL_Pre_y,valid_pred)
precision_score(validU_M_UM_UC_OL_Pre_y,valid_pred)#0.2851 #0.3426 #0.3679  #0.2235 batch
recall_score(validU_M_UM_UC_OL_Pre_y,valid_pred)#0.7940 #0.8421 #0.8529  #0.7486 batch
f1_score(validU_M_UM_UC_OL_Pre_y,valid_pred)#0.4195 #0.4870 #0.5140  #0.3443 batch
roc_auc_score(validU_M_UM_UC_OL_Pre_y,valid_pred)#0.7966 #0.8384 #0.8515  #0.7815 batch

#验证集stackingModel AUC计算
validPred_df=pd.DataFrame(columns=['User_id','Coupon_id','Probability','TrueLabel'])
validPred_df.User_id=merge_train_test.loc[validU_M_UM_UC_OL_Pre_x.index,:].User_id
validPred_df.Coupon_id=merge_train_test.loc[validU_M_UM_UC_OL_Pre_x.index,:].Coupon_id
validPred_df.Probability=logit_y_pro[:,1]
validPred_df.TrueLabel=validU_M_UM_UC_OL_Pre_y
validAuc=o2oFunc.av_auc(validPred_df,"Coupon_id","Probability","TrueLabel")
validAuc["AucMean"]#0.7282 #0.8351 #0.8478  #0.7251 batch

###################################################################################测试集预测
print("测试集添加用户画像，商户画像，交叉特征（UM,UC）")
test0=temp[temp.tag=="test"]
test0["cvTag"]="predict"
testU=o2oFunc.addUFeatures(offline,test0)
testU_M=o2oFunc.addMFeatures(offline,testU["train"][-1])
testU_M_C=o2oFunc.addCFeatures(offline,testU_M["train"][-1])
testU_M_UM=o2oFunc.addUMFeatures(offline,testU_M_C)
testU_M_UM_UC=o2oFunc.addUCFeatures(offline,testU_M_UM)
#增加线上特征UOL
test_UOL=o2oFunc.addUFeatures(online,test0)["train"][-1].rename(columns=lambda x:x+"_OL")
test_UOL.drop_duplicates(inplace=True)
testU_M_C_UM_UC_UOL=pd.merge(testU_M_UM_UC,test_UOL,left_on=list(test0.columns),right_on=[i+"_OL" for i in test0.columns],how="left").drop([i+"_OL" for i in test0.columns],axis=1)
#增加线上其他特征(Coupon,User X Coupon)
test0_OL=o2oFunc.addOLFeatures(online,test0)
test0_OL.drop_duplicates(inplace=True)#去重
#合并线下，线上特征
testU_M_C_UM_UC_OL=pd.merge(testU_M_C_UM_UC_UOL,test0_OL,on=list(test0.columns),how="left")
#增加预测区间特征
test_Pre=o2oFunc.addPreFeatures(test0)
test_Pre.drop_duplicates(inplace=True)#去重
testU_M_C_UM_UC_OL_Pre=pd.merge(testU_M_C_UM_UC_OL,test_Pre,on=list(test0.columns),how="left")

#测试集，训练集融合
mergeU_M_C_UM_UC_OL_Pre=pd.concat([merge_train_test,testU_M_C_UM_UC_OL_Pre],axis=0,ignore_index=True)
mergeU_M_C_UM_UC_OL_Pre.fillna({"Distance":-1},inplace=True)#Distance缺失值视作-1

#规范化
mergeU_M_C_UM_UC_OL_Pre_stdTemp=mergeU_M_C_UM_UC_OL_Pre.drop(cols,axis=1).apply(lambda x:o2oFunc.dataStand(x),axis=0)
mergeU_M_C_UM_UC_OL_Pre.loc[:,mergeU_M_C_UM_UC_OL_Pre.drop(cols,axis=1).columns]=mergeU_M_C_UM_UC_OL_Pre_stdTemp

#测试集，训练集分割
testU_M_C_UM_UC_OL_Pre_std=mergeU_M_C_UM_UC_OL_Pre[mergeU_M_C_UM_UC_OL_Pre.tag=="test"]
trainU_M_C_UM_UC_OL_Pre_std=mergeU_M_C_UM_UC_OL_Pre[mergeU_M_C_UM_UC_OL_Pre.tag=="offt"]

testU_M_C_UM_UC_OL_Pre_std_x,testU_M_C_UM_UC_OL_Pre_std_y=testU_M_C_UM_UC_OL_Pre_std.drop(["cvTag","User_id","Merchant_id","Coupon_id","Discount_rate","Date_received","tag","TN","Date"],axis=1),testU_M_C_UM_UC_OL_Pre_std.TN

trainU_M_C_UM_UC_OL_Pre_std_x,trainU_M_C_UM_UC_OL_Pre_std_y=trainU_M_C_UM_UC_OL_Pre_std.drop(["cvTag","User_id","Merchant_id","Coupon_id","Discount_rate","Date_received","tag","TN","Date"],axis=1),trainU_M_C_UM_UC_OL_Pre_std.TN

print("训练集欠抽样处理")
train_undersam=o2oFunc.underSampler(trainU_M_C_UM_UC_OL_Pre_std_x,trainU_M_C_UM_UC_OL_Pre_std_y)
trainU_M_C_UM_UC_OL_Pre_std_x_undersam=train_undersam["train_x"]
trainU_M_C_UM_UC_OL_Pre_std_y_undersam=train_undersam["train_y"]

#调参后的初级学习器
rfr=RandomForestClassifier(n_jobs=-1,random_state=42,oob_score=True,n_estimators=200,max_depth=14,max_features="sqrt",min_samples_leaf=1)
ext=ExtraTreesClassifier(n_estimators=200,max_depth=26,max_features="sqrt",min_samples_leaf=1,n_jobs=-1,random_state=42,oob_score=True,bootstrap=True)
xgbc=xgb.XGBClassifier(n_jobs=-1,random_state=42,max_depth=5,min_child_weight=1,gamma=0,n_estimators=150,learning_rate=0.1)
gbmc=GradientBoostingClassifier(random_state=42,max_depth=4,min_samples_leaf=7,max_features="sqrt",n_estimators=100,learning_rate=0.1)

print("stacking_p2：次级学习器")
P1_output=o2oFunc.stackingP1(rfr,ext,gbmc,xgbc,trainU_M_C_UM_UC_OL_Pre_std_x_undersam,trainU_M_C_UM_UC_OL_Pre_std_y_undersam,testU_M_C_UM_UC_OL_Pre_std_x,cv=10)
stack_train_x,stack_train_y=P1_output['train_pred'].drop('realY',axis=1),P1_output['train_pred']['realY']
P2_test_x=P1_output['test_pred']

print("logistic模型调参")
paramdic={"C":[0.1,1,10]}
clf=LogisticRegression(random_state=42,max_iter=10000,penalty='l2',class_weight="balanced")
grid=GridSearchCV(clf,param_grid=paramdic,scoring='roc_auc',cv=10)#网格搜索
grid.fit(stack_train_x,stack_train_y)
print(grid.best_score_)#0.8783 #0.9177 #0.9307
print(grid.best_params_)#{'C': 10}  #{"C":0.1}#{"C":1}

#模型预测
logitC=LogisticRegression(random_state=42,max_iter=10000,penalty='l2',class_weight="balanced",C=1)
logitC.fit(stack_train_x,stack_train_y)

logit_y=logitC.predict(P2_test_x)
logit_y_pro=logitC.predict_proba(P2_test_x)

#输出结果字段
res_df=pd.DataFrame(columns=['User_id','Coupon_id','Date_received','Probability'])
res_df.User_id=mergeU_M_C_UM_UC_OL_Pre.loc[testU_M_C_UM_UC_OL_Pre_std_x.index,:].User_id.values
res_df.Coupon_id=mergeU_M_C_UM_UC_OL_Pre.loc[testU_M_C_UM_UC_OL_Pre_std_x.index,:].Coupon_id.values
res_df.Coupon_id=res_df.Coupon_id.map(lambda x:int(x))
res_df.Date_received=mergeU_M_C_UM_UC_OL_Pre.loc[testU_M_C_UM_UC_OL_Pre_std_x.index,:].Date_received.values
res_df.Probability=logit_y_pro[:,1]
res_df.Date_received=res_df.Date_received.map(lambda x:dt.strftime(x,"%Y%m%d"))
res_df.to_csv("run_predict_result_OL_Pre_crossTimeValid.csv",index=False)

























