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

#欠抽样并洗牌
def underSampler(train_x,train_y):
	model_RandomUnderSampler = RandomUnderSampler()
	train_x_undersam_arr, train_y_undersam_arr =model_RandomUnderSampler.fit_sample(train_x,train_y)
	train_x_undersam=pd.DataFrame(train_x_undersam_arr,columns=train_x.columns)
	train_y_undersam=pd.Series(train_y_undersam_arr)
	#洗牌（不平衡样本处理后按标签顺序排列，需要随机打乱）
	np.random.seed(42)
	shuffle_index = np.random.permutation(train_x_undersam.shape[0])
	train_x_undersam,train_y_undersam=train_x_undersam.loc[shuffle_index,:],train_y_undersam.loc[shuffle_index]
	return {"train_x":train_x_undersam,"train_y":train_y_undersam}

#网格搜索返回最优参数和分数
def clfGrid(train_x,train_y,clf,paramdic,cv=10,scoring="f1"):
	grid=GridSearchCV(clf,param_grid=paramdic,scoring=scoring,cv=cv)#网格搜索	
	grid.fit(train_x,train_y)
	bestScore=grid.best_score_
	bestParam=grid.best_params_
	return {"bestScore":bestScore,"bestParam":bestParam}

#AUC计算
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
def av_auc(df,CouponId,predProba,TrueLabel):
	idArr=df[CouponId].unique()
	idAucLst=[]
	ids=[]
	for id in idArr:
		group_df=df[df[CouponId]==id]
		if group_df[TrueLabel].unique().shape[0]==2:
			ids.append(id)
			y_true=group_df[TrueLabel]
			y_score=group_df[predProba]
			fpr,tpr,thresh=roc_curve(y_true,y_score)
			idAucLst.append(auc(fpr,tpr))
		else:
			pass
	return {"AucLst":idAucLst,"idArr":idArr,"AucMean":np.mean(idAucLst)}

#分层抽样
def Stra_samples(size,samples_y):
    totalnum=samples_x.shape[0]
    values_class=samples_y.unique()
    yclassCount=values_class.shape[0]
    y_dic={}#y类别：类别数组
    for value in values_class:
        y_dic[value]=samples_y[samples_y==value]
    indexs_dic={}#size:index
    for s in size:
        num=int(s*totalnum/yclassCount)
        lst=[]
        for key,value in y_dic.items():
            new_index=sample(list(value.index),num)
            lst.extend(new_index)
        indexs_dic[s]=lst
    return dict(sorted(indexs_dic.items(),key=lambda x:x[0]))

#学习曲线:含训练集，交叉验证集，指定验证集
def learningCurve(clf,samples_x,samples_y,valid_x,valid_y,size=[0.1,0.25,0.5,0.75,1],score="f1",cv=10):
    sizeDic=Stra_samples(size,samples_y)
    trainsize=[]
    score_train=[]
    score_cv=[]
    score_valid=[]
    dic_items=sorted(sizeDic.items(),key=lambda x:x[0])#按key排序
    if score=="roc_auc":
        for ratio,indexs in dic_items:
            x=samples_x.loc[indexs,:]
            y=samples_y.loc[indexs]
            skf=StratifiedKFold(n_splits=cv,random_state=42,shuffle=True)#打乱顺序
            score_df=pd.DataFrame(columns=["train","cv","valid"],index=[i for i in range(1,cv+1)])
            i=1
            for train_index,cv_index in skf.split(x,y):
                train_x,train_y,cv_x,cv_y=x.iloc[train_index,:],y.iloc[train_index],x.iloc[cv_index,:],y.iloc[cv_index]
                clf.fit(train_x,train_y)
                trainscore=roc_auc_score(train_y,clf.predict(train_x))#训练集
                cvscore=roc_auc_score(cv_y,clf.predict(cv_x))#交叉验证集
                validscore=roc_auc_score(valid_y,clf.predict(valid_x))#指定验证集
                score_df.loc[i,"train"]=trainscore
                score_df.loc[i,"cv"]=cvscore
                score_df.loc[i,"valid"]=validscore
                i=i+1
            score_train.append(score_df.train.mean())
            score_cv.append(score_df.cv.mean())
            score_valid.append(score_df.valid.mean())
            trainsize.append(ratio)
        return (trainsize,score_train,score_cv,score_valid)
    elif score=="f1":
        for ratio,indexs in dic_items:
            x=samples_x.loc[indexs,:]
            y=samples_y.loc[indexs]
            skf=StratifiedKFold(n_splits=cv,random_state=42,shuffle=True)#打乱顺序
            score_df=pd.DataFrame(columns=["train","cv","valid"],index=[i for i in range(1,cv+1)])
            i=1
            for train_index,cv_index in skf.split(x,y):
                train_x,train_y,cv_x,cv_y=x.iloc[train_index,:],y.iloc[train_index],x.iloc[cv_index,:],y.iloc[cv_index]
                clf.fit(train_x,train_y)
                trainscore=f1_score(train_y,clf.predict(train_x))#训练集
                cvscore=f1_score(cv_y,clf.predict(cv_x))#交叉验证集
                validscore=f1_score(valid_y,clf.predict(valid_x))#指定验证集
                score_df.loc[i,"train"]=trainscore
                score_df.loc[i,"cv"]=cvscore
                score_df.loc[i,"valid"]=validscore
                i=i+1
            score_train.append(score_df.train.mean())
            score_cv.append(score_df.cv.mean())
            score_valid.append(score_df.valid.mean())
            trainsize.append(ratio)
        return (trainsize,score_train,score_cv,score_valid)
    elif score=="precision":
        for ratio,indexs in dic_items:
            x=samples_x.loc[indexs,:]
            y=samples_y.loc[indexs]
            skf=StratifiedKFold(n_splits=cv,random_state=42,shuffle=True)#打乱顺序
            score_df=pd.DataFrame(columns=["train","cv","valid"],index=[i for i in range(1,cv+1)])
            i=1
            for train_index,cv_index in skf.split(x,y):
                train_x,train_y,cv_x,cv_y=x.iloc[train_index,:],y.iloc[train_index],x.iloc[cv_index,:],y.iloc[cv_index]
                clf.fit(train_x,train_y)
                trainscore=precision_score(train_y,clf.predict(train_x))#训练集
                cvscore=precision_score(cv_y,clf.predict(cv_x))#交叉验证集
                validscore=precision_score(valid_y,clf.predict(valid_x))#指定验证集
                score_df.loc[i,"train"]=trainscore
                score_df.loc[i,"cv"]=cvscore
                score_df.loc[i,"valid"]=validscore
                i=i+1
            score_train.append(score_df.train.mean())
            score_cv.append(score_df.cv.mean())
            score_valid.append(score_df.valid.mean())
            trainsize.append(ratio)
        return (trainsize,score_train,score_cv,score_valid)
    elif score=="recall":
        for ratio,indexs in dic_items:
            x=samples_x.loc[indexs,:]
            y=samples_y.loc[indexs]
            skf=StratifiedKFold(n_splits=cv,random_state=42,shuffle=True)#打乱顺序
            score_df=pd.DataFrame(columns=["train","cv","valid"],index=[i for i in range(1,cv+1)])
            i=1
            for train_index,cv_index in skf.split(x,y):
                train_x,train_y,cv_x,cv_y=x.iloc[train_index,:],y.iloc[train_index],x.iloc[cv_index,:],y.iloc[cv_index]
                clf.fit(train_x,train_y)
                trainscore=recall_score(train_y,clf.predict(train_x))#训练集
                cvscore=recall_score(cv_y,clf.predict(cv_x))#交叉验证集
                validscore=recall_score(valid_y,clf.predict(valid_x))#指定验证集
                score_df.loc[i,"train"]=trainscore
                score_df.loc[i,"cv"]=cvscore
                score_df.loc[i,"valid"]=validscore
                i=i+1
            score_train.append(score_df.train.mean())
            score_cv.append(score_df.cv.mean())
            score_valid.append(score_df.valid.mean())
            trainsize.append(ratio)
        return (trainsize,score_train,score_cv,score_valid)
    elif score=="accuracy":
        for ratio,indexs in dic_items:
            x=samples_x.loc[indexs,:]
            y=samples_y.loc[indexs]
            skf=StratifiedKFold(n_splits=cv,random_state=42,shuffle=True)#打乱顺序
            score_df=pd.DataFrame(columns=["train","cv","valid"],index=[i for i in range(1,cv+1)])
            i=1
            for train_index,cv_index in skf.split(x,y):
                train_x,train_y,cv_x,cv_y=x.iloc[train_index,:],y.iloc[train_index],x.iloc[cv_index,:],y.iloc[cv_index]
                clf.fit(train_x,train_y)
                trainscore=accuracy_score(train_y,clf.predict(train_x))#训练集
                cvscore=accuracy_score(cv_y,clf.predict(cv_x))#交叉验证集
                validscore=accuracy_score(valid_y,clf.predict(valid_x))#指定验证集
                score_df.loc[i,"train"]=trainscore
                score_df.loc[i,"cv"]=cvscore
                score_df.loc[i,"valid"]=validscore
                i=i+1
            score_train.append(score_df.train.mean())
            score_cv.append(score_df.cv.mean())
            score_valid.append(score_df.valid.mean())
            trainsize.append(ratio)
        return (trainsize,score_train,score_cv,score_valid)
    else:
        print("The evaluation indicator doesn't exist!")

def stackingP1(rfr,ext,gbmc,xgbc,train_x,train_y,test_x,cv=10):
	skf=StratifiedKFold(n_splits=cv,random_state=42,shuffle=True)#分层抽样cv折
	train_p1=pd.DataFrame(columns=['rfr','extraRFR','gbm','xgboost','realY'])
	train_p1.realY=train_y
	test_rfr=pd.DataFrame(columns=["pred_"+str(i) for i in range(1,int(cv)+1)])
	test_ext=pd.DataFrame(columns=["pred_"+str(i) for i in range(1,int(cv)+1)])
	test_gbm=pd.DataFrame(columns=["pred_"+str(i) for i in range(1,int(cv)+1)])
	test_xgb=pd.DataFrame(columns=["pred_"+str(i) for i in range(1,int(cv)+1)])
	colnum=1
	for train_index,valid_index in skf.split(train_x,train_y):
		print("第{}折预测.....".format(colnum))
		tx,vx=train_x.iloc[train_index,:],train_x.iloc[valid_index,:]#划分训练集，验证集
		ty,vy=train_y.iloc[train_index],train_y.iloc[valid_index]	
		#rfr:随机森林预测
		#rfr=RandomForestClassifier(n_jobs=-1,random_state=42,oob_score=True,n_estimators=200,max_depth=18,max_features=None,min_samples_leaf=1)
		rfr.fit(tx,ty)
		rfr_y=pd.Series(rfr.predict_proba(vx)[:,1],index=vx.index)
		train_p1.loc[vx.index,'rfr']=rfr_y
		test_rfr["pred_"+str(colnum)]=pd.Series(rfr.predict_proba(test_x)[:,1],index=test_x.index)
		#extraRFR:极端树预测
		#ext=ExtraTreesClassifier(n_estimators=200,max_depth=20,max_features=None,min_samples_leaf=1,n_jobs=-1,random_state=42,oob_score=True,bootstrap=True)
		ext.fit(tx,ty)
		ext_y=pd.Series(ext.predict_proba(vx)[:,1],index=vx.index)
		train_p1.loc[vx.index,'extraRFR']=ext_y
		test_ext['pred_'+str(colnum)]=pd.Series(ext.predict_proba(test_x)[:,1],index=test_x.index)
		#gbm:梯度上升预测
		#gbmc=GradientBoostingClassifier(random_state=42,max_depth=11,max_features="sqrt",min_samples_leaf=3,n_estimators=150,learning_rate=0.08)
		gbmc.fit(tx,ty)
		gbmc_y=pd.Series(gbmc.predict_proba(vx)[:,1],index=vx.index)
		train_p1.loc[vx.index,'gbm']=gbmc_y
		test_gbm['pred_'+str(colnum)]=pd.Series(gbmc.predict_proba(test_x)[:,1],index=test_x.index)
		#xgboost:xgboost预测
		#xgbc=xgb.XGBClassifier(n_jobs=-1,random_state=42,max_depth=9,min_child_weight=3,gamma=0.1,n_estimators=100,learning_rate=0.1)
		xgbc.fit(tx,ty)
		xgbc_y=pd.Series(xgbc.predict_proba(vx)[:,1],index=vx.index)
		train_p1.loc[vx.index,'xgboost']=xgbc_y
		test_xgb['pred_'+str(colnum)]=pd.Series(xgbc.predict_proba(test_x)[:,1],index=test_x.index)
		colnum=colnum+1
	test_p1=pd.DataFrame(columns=['rfr','extraRFR','gbm','xgboost'])
	test_p1.rfr=np.mean(test_rfr,axis=1)
	test_p1.extraRFR=np.mean(test_ext,axis=1)
	test_p1.gbm=np.mean(test_gbm,axis=1)
	test_p1.xgboost=np.mean(test_xgb,axis=1)
	return {"train_pred":train_p1,"test_pred":test_p1,"test_rfr":test_rfr,"test_ext":test_ext,"test_gbm":test_gbm,"test_xgb":test_xgb}


#(1)前5个月领券消费比率：领券消费/领券数,0表示从不领券或者领券后从不使用
def addUFeatures(offline_all_pre,offline_train):#添加用户特征
	#用户重叠情况
	new_user=list(set(offline_train.User_id)-set(offline_all_pre.User_id))
	#添加lqxfRatio
	offline_pre=offline_all_pre.loc[(offline_all_pre.Date_received<dt.strptime("20160515","%Y%m%d")),:]#前五个月领券记录
	offline_lq=offline_pre.groupby('User_id').size()#领券用户
	offline_blq=list(set(offline_all_pre.User_id)-set(offline_lq.index))#
	offline_blq_df=pd.DataFrame(np.zeros((len(offline_blq),1)),index=offline_blq,columns=['lqxfRatio'])#不领券用户，视作lqxfRatio=0
	offline_lqxf=offline_pre[offline_pre.TN==1].groupby('User_id').size()#领券并消费用户
	lqxfRatio=offline_lqxf/offline_lq #nan领券不消费用户
	lqxfRatio_df=pd.DataFrame(lqxfRatio.fillna(0),columns=["lqxfRatio"])
	lqxfRatio_df=pd.concat([lqxfRatio_df,offline_blq_df],axis=0)
	offline_train1=pd.merge(offline_train,lqxfRatio_df,left_on="User_id",right_index=True,how="left")
	offline_train1.fillna({"lqxfRatio":-1},inplace=True)#新用户视作lqxfRatio=-1
	#offline_train1..corr()
	#添加是否新用户new_user
	newUser=pd.DataFrame(np.ones((len(new_user),1)),index=new_user,columns=["new_user"])
	offline_train2=pd.merge(offline_train1,newUser,left_on="User_id",right_index=True,how="left")
	offline_train2.fillna({"new_user":0},inplace=True)
	offline_train3=pd.get_dummies(offline_train2,columns=['new_user'])
	#offline_train3.corr()
	#添加线下消费力度xfFreq
	offline_xfFreq_df=pd.DataFrame(offline_all_pre.loc[offline_all_pre.Date.notnull(),:].groupby("User_id").size(),columns=["xfFreq"])#前五个月消费频率
	offline_0xf=list(set(offline_all_pre.User_id)-set(offline_xfFreq_df.index))
	offline_0xf_df=pd.DataFrame(np.zeros((len(offline_0xf),1)),index=offline_0xf,columns=['xfFreq'])#前五个月消费为0
	offline_xfFreq_df=pd.concat([offline_xfFreq_df,offline_0xf_df],axis=0)
	offline_train4=pd.merge(offline_train3,offline_xfFreq_df,left_on="User_id",right_index=True,how="left")
	offline_train4.fillna({"xfFreq":-1},inplace=True)
	#offline_train4.corr()
	#添加用户光顾的商家数目mCounts
	mCounts_df=pd.DataFrame(offline_all_pre.loc[(offline_all_pre.TN==1)|(offline_all_pre.TN==2),:].drop_duplicates(subset=["User_id","Merchant_id"]).groupby("User_id")[["Merchant_id"]].size().sort_values(ascending=False),columns=["mCounts"])
	offline_train5=pd.merge(offline_train4,mCounts_df,left_on="User_id",right_index=True,how="left")
	offline_train5.fillna({"mCounts":-1},inplace=True)
	#offline_train5.corr()
	return {"new_user":new_user,"train":[offline_train1,offline_train3,offline_train4,offline_train5]}

#商家：根据训练集得出的画像（不适用于测试集的新用户），商家发券力度，受欢迎程度，距离远近
#商家重叠情况

def addMFeatures(offline_all_pre,offline_train5):
	#用户重叠情况
	new_Merchant=list(set(offline_train5.Merchant_id)-set(offline_all_pre.Merchant_id))#提取特征数据集与训练集重叠情况
	#添加用户被领取的优惠券数：mCouponCounts
	mCouponCounts_df=pd.DataFrame(offline_all_pre.dropna(subset=["Coupon_id"]).groupby("Merchant_id")["Coupon_id"].size()).rename(columns={"Coupon_id":"mCouponCounts"})
	mCC0=list(set(offline_all_pre.Merchant_id)-set(mCouponCounts_df.index))
	mCouponCounts_0=pd.DataFrame(np.zeros((len(mCC0),1)),index=mCC0,columns=["mCouponCounts"])
	mCouponCounts_df=pd.concat([mCouponCounts_df,mCouponCounts_0],axis=0)
	offline_train6=pd.merge(offline_train5,mCouponCounts_df,left_on="Merchant_id",right_index=True,how="left")
	offline_train6.fillna({"mCouponCounts":-1},inplace=True)
	#offline_train6.corr()
	#添加商家被购买的次数：mBoughtCounts
	offline_pre_bought=offline_all_pre.loc[(offline_all_pre.TN==1)|(offline_all_pre.TN==2)]
	mBoughtCounts_df=pd.DataFrame(offline_pre_bought.groupby("Merchant_id").size(),columns=["mBoughtCounts"])
	mBoughtCounts0=list(set(offline_all_pre.Merchant_id)-set(mBoughtCounts_df.index))
	mBoughtCounts0_df=pd.DataFrame(np.zeros((len(mBoughtCounts0),1)),index=mBoughtCounts0,columns=["mBoughtCounts"])
	mBoughtCounts_all_df=pd.concat([mBoughtCounts_df,mBoughtCounts0_df],axis=0)#商家被购买的次数
	offline_train7=pd.merge(offline_train6,mBoughtCounts_all_df,left_on="Merchant_id",right_index=True,how="left")
	offline_train7.fillna({"mBoughtCounts":-1},inplace=True)
	#offline_train7.corr()
	#添加商户用户消费/商家发放优惠券的比例：Mer_lqxfRatio
	offline_train7["Mer_lqxfRatio"]=offline_train7.mBoughtCounts/offline_train7.mCouponCounts
	offline_train7.loc[offline_train7.Mer_lqxfRatio==np.inf,"Mer_lqxfRatio"]=0#从未领券的视作0
	offline_train7_temp=offline_train7.set_index("Merchant_id")
	offline_train7_temp.loc[new_Merchant,"Mer_lqxfRatio"]=-1#新商店Mer_lqxfRatio视作-1
	offline_train7.Mer_lqxfRatio=pd.Series(offline_train7_temp.Mer_lqxfRatio.values,index=offline_train7.index)
	#offline_train7.corr()
	#添加商户用户持有优惠券并消费次数：Mer_lqxfCount
	lqxfCount=pd.DataFrame(offline_all_pre[offline_all_pre.TN==1].groupby("Merchant_id").size(),columns=["Mer_lqxfCount"])
	lqxfCount0=list(set(offline_all_pre.Merchant_id)-set(lqxfCount.index))
	lqxfCount0_df=pd.DataFrame(np.zeros((len(lqxfCount0),1)),index=lqxfCount0,columns=["Mer_lqxfCount"])
	lqxfCount_df=pd.concat([lqxfCount,lqxfCount0_df],axis=0)
	offline_train8=pd.merge(offline_train7,lqxfCount_df,left_on="Merchant_id",right_index=True,how="left")
	offline_train8.fillna({"Mer_lqxfCount":-1},inplace=True)
	#offline_train8.corr()
	#添加商户用户持券消费/用户领券:Mer_lqAndxfRatio
	offline_train8["Mer_lqAndxfRatio"]=offline_train8.Mer_lqxfCount/offline_train8.mCouponCounts
	offline_train8.fillna({"Mer_lqAndxfRatio":0},inplace=True)
	offline_train8_temp=offline_train8.set_index("Merchant_id")
	offline_train8_temp.loc[new_Merchant,"Mer_lqAndxfRatio"]=-1
	offline_train8["Mer_lqAndxfRatio"]=pd.Series(offline_train8_temp.Mer_lqAndxfRatio.values,index=offline_train8.index)
	#offline_train8.corr()
	#商家：被领取的优惠券平均满减(发放的优惠券类型)
	return {"new_merchant":new_Merchant,"train":[offline_train6,offline_train7,offline_train8]}

#优惠券
def addCFeatures(offline_all_pre,offline_train):
    #以优惠券id作为标识
    new_couponId=list(set(offline_train.Coupon_id.unique())-set(offline_all_pre.Coupon_id.unique()))
    CouponId_df=pd.DataFrame(index=offline_train.Coupon_id.unique())
    #历史被领取次数
    CouponCount_lq=offline_all_pre[offline_all_pre.Coupon_id.notnull()].groupby("Coupon_id").size()
    CouponId_df.loc[new_couponId,"CouponCount_lq"]=-1
    CouponId_df.loc[CouponId_df.CouponCount_lq.isnull(),"CouponCount_lq"]=CouponCount_lq[CouponId_df[CouponId_df.CouponCount_lq.isnull()].index]
    CouponId_df.fillna({"CouponCount_lq":0},inplace=True)
    #历史被领券消费次数
    CouponCount_lqxf=offline_all_pre[(offline_all_pre.Date.notnull())&(offline_all_pre.Coupon_id.notnull())].groupby("Coupon_id").size()
    CouponId_df.loc[new_couponId,"CouponCount_lqxf"]=-1
    CouponId_df.loc[CouponId_df.CouponCount_lqxf.isnull(),"CouponCount_lqxf"]=CouponCount_lqxf[CouponId_df[CouponId_df.CouponCount_lqxf.isnull()].index]
    CouponId_df.fillna({"CouponCount_lqxf":0},inplace=True)
    #历史被领券消费率
    Coupon_xfRatio=CouponCount_lqxf/CouponCount_lq
    Coupon_xfRatio.fillna(0,inplace=True)
    CouponId_df.loc[new_couponId,"Coupon_xfRatio"]=-1
    CouponId_df.loc[CouponId_df.Coupon_xfRatio.isnull(),"Coupon_xfRatio"]=Coupon_xfRatio[CouponId_df[CouponId_df.Coupon_xfRatio.isnull()].index]
    CouponId_df.fillna({"Coupon_xfRatio":0},inplace=True)

    offline_train_1=pd.merge(offline_train,CouponId_df,left_on="Coupon_id",right_index=True,how="left")

    #以优惠券满减门槛为标识
    offline_all_pre["DisType"]=np.where(offline_all_pre.Discount_rate.isnull(),np.nan,np.where(offline_all_pre.Discount_rate=="fixed","fixed",np.where(offline_all_pre.Discount_rate.map(lambda x:":" in str(x)),"mj","ratio")))
    offline_all_pre.loc[offline_all_pre.DisType=="mj","mjThresh"]=offline_all_pre[offline_all_pre.DisType=="mj"]["Discount_rate"].map(lambda x:int(x.split(":")[0]))
    offline_all_pre.mjThresh.fillna(0,inplace=True)
    offline_all_pre.loc[offline_all_pre.DisType=="ratio","ratioIntense"]=offline_all_pre[offline_all_pre.DisType=="ratio"].Discount_rate.map(lambda x:float(x))
    offline_all_pre.fillna({"ratioIntense":0},inplace=True)
    
    new_mjThresh=list(set(offline_train.mjThresh.unique())-set(offline_all_pre.mjThresh.unique()))
    Coupon_mjThresh_df=pd.DataFrame(index=list(offline_train.mjThresh.unique()))
    #历史被领取次数
    mjThresh_lq=offline_all_pre[offline_all_pre.Coupon_id.notnull()].groupby("mjThresh").size()
    Coupon_mjThresh_df.loc[new_mjThresh,"mjThresh_lq"]=-1
    Coupon_mjThresh_df.loc[Coupon_mjThresh_df.mjThresh_lq.isnull(),"mjThresh_lq"]=mjThresh_lq[Coupon_mjThresh_df[Coupon_mjThresh_df.mjThresh_lq.isnull()].index]
    Coupon_mjThresh_df.fillna({"mjThresh_lq":0},inplace=True)
    #历史被消费次数
    mjThresh_lqxf=offline_all_pre[(offline_all_pre.Date.notnull())&(offline_all_pre.Coupon_id.notnull())].groupby("mjThresh").size()
    Coupon_mjThresh_df.loc[new_mjThresh,"mjThresh_lqxf"]=-1
    Coupon_mjThresh_df.loc[Coupon_mjThresh_df.mjThresh_lqxf.isnull(),"mjThresh_lqxf"]=mjThresh_lqxf[Coupon_mjThresh_df[Coupon_mjThresh_df.mjThresh_lqxf.isnull()].index]
    Coupon_mjThresh_df.fillna({"mjThresh_lqxf":0},inplace=True)
    #历史被领券消费率
    mjThresh_xfRatio=mjThresh_lqxf/mjThresh_lq
    mjThresh_xfRatio.fillna(0,inplace=True)
    Coupon_mjThresh_df.loc[new_mjThresh,"mjThresh_xfRatio"]=-1
    Coupon_mjThresh_df.loc[Coupon_mjThresh_df.mjThresh_xfRatio.isnull(),"mjThresh_xfRatio"]=mjThresh_xfRatio[Coupon_mjThresh_df[Coupon_mjThresh_df.mjThresh_xfRatio.isnull()].index]
    Coupon_mjThresh_df.fillna({"mjThresh_xfRatio":0},inplace=True)

    offline_train_2=pd.merge(offline_train_1,Coupon_mjThresh_df,left_on="mjThresh",right_index=True,how="left")

    #以优惠券折扣率为标识 
    new_ratioIntense=list(set(offline_train.ratioIntense)-set(offline_all_pre.ratioIntense))
    Coupon_ratioIntense_df=pd.DataFrame(index=list(offline_train.ratioIntense.unique()))
    #历史被领取次数
    ratioIntense_lq=offline_all_pre[offline_all_pre.Coupon_id.notnull()].groupby("ratioIntense").size()
    Coupon_ratioIntense_df.loc[new_ratioIntense,"ratioIntense_lq"]=-1
    Coupon_ratioIntense_df.loc[Coupon_ratioIntense_df.ratioIntense_lq.isnull(),"ratioIntense_lq"]=ratioIntense_lq[Coupon_ratioIntense_df[Coupon_ratioIntense_df.ratioIntense_lq.isnull()].index]
    Coupon_ratioIntense_df.fillna({"ratioIntense_lq":0},inplace=True)
    #历史被消费
    ratioIntense_lqxf=offline_all_pre[(offline_all_pre.Coupon_id.notnull())&(offline_all_pre.Date.notnull())].groupby("ratioIntense").size()
    Coupon_ratioIntense_df.loc[new_ratioIntense,"ratioIntense_lqxf"]=-1
    Coupon_ratioIntense_df.loc[Coupon_ratioIntense_df.ratioIntense_lqxf.isnull(),"ratioIntense_lqxf"]=ratioIntense_lqxf[Coupon_ratioIntense_df[Coupon_ratioIntense_df.ratioIntense_lqxf.isnull()].index]
    #历史领取消费率
    ratioIntense_xfRatio=ratioIntense_lqxf/ratioIntense_lq
    ratioIntense_xfRatio.fillna(0,inplace=True)
    Coupon_ratioIntense_df.loc[new_ratioIntense,"ratioIntense_xfRatio"]=-1
    Coupon_ratioIntense_df.loc[Coupon_ratioIntense_df.ratioIntense_xfRatio.isnull(),"ratioIntense_xfRatio"]=ratioIntense_xfRatio[Coupon_ratioIntense_df[Coupon_ratioIntense_df.ratioIntense_xfRatio.isnull()].index]
    Coupon_ratioIntense_df.fillna({"ratioIntense_xfRatio":0},inplace=True)

    offline_train_3=pd.merge(offline_train_2,Coupon_ratioIntense_df,left_on="ratioIntense",right_index=True,how="left")
    
    return offline_train_3


#特征交叉
#用户X商家
def addUMFeatures(offline_all_pre,offline_train8):
	#购买次数:UM_xfCount
	UM_xf=offline_all_pre[offline_all_pre.Date.notnull()].groupby(["User_id","Merchant_id"],as_index=False).size()
	UM_xf_df=UM_xf.reset_index().rename(columns={0:"UM_xfCount"})
	offline_train9=pd.merge(offline_train8,UM_xf_df,on=["User_id","Merchant_id"],how="left")
	offline_train9.fillna({"UM_xfCount":0},inplace=True)
	#领券次数:UM_lqCount
	UM_lq=offline_all_pre[offline_all_pre.Coupon_id.notnull()].groupby(["User_id","Merchant_id"]).size()
	UM_lq_df=UM_lq.reset_index().rename(columns={0:"UM_lqCount"})
	offline_train9=pd.merge(offline_train9,UM_lq_df,on=["User_id","Merchant_id"],how="left")
	offline_train9.fillna({"UM_lqCount":0},inplace=True)
	#持券消费次数:UM_lqxfCount
	UM_lqxf=offline_all_pre[(offline_all_pre.Coupon_id.notnull())&(offline_all_pre.Date.notnull())].groupby(["User_id","Merchant_id"]).size()
	UM_lqxf_df=UM_lqxf.reset_index().rename(columns={0:"UM_lqxfCount"})
	offline_train9=pd.merge(offline_train9,UM_lqxf_df,on=["User_id","Merchant_id"],how="left")
	offline_train9.fillna({"UM_lqxfCount":0},inplace=True)
	#领券消费/领券:UM_lqAndxfRatio
	UM_lqAndxfRatio=offline_train9.UM_lqxfCount/offline_train9.UM_lqCount
	offline_train9["UM_lqAndxfRatio"]=UM_lqAndxfRatio
	offline_train9.fillna({"UM_lqAndxfRatio":0},inplace=True)
	#offline_train9.corr()
	return offline_train9	

#用户X优惠券
def addUCFeatures(offline_all_pre,offline_train9):
	#领券数量:UC_lqCount
	UC_lqCount=offline_all_pre[offline_all_pre.Coupon_id.notnull()].groupby(["User_id","Coupon_id"]).size()
	UC_lqCount_df=UC_lqCount.reset_index().rename(columns={0:"UC_lqCount"})
	offline_train10=pd.merge(offline_train9,UC_lqCount_df,on=["User_id","Coupon_id"],how="left")
	offline_train10.fillna({"UC_lqCount":0},inplace=True)
	#持券购买数量:UC_lqAndxfCount
	UC_lqAndxfCount=offline_all_pre[(offline_all_pre.Date.notnull())&(offline_all_pre.Coupon_id.notnull())].groupby(["User_id","Coupon_id"]).size()	
	UC_lqAndxfCount_df=UC_lqAndxfCount.reset_index().rename(columns={0:"UC_lqAndxfCount"})
	offline_train10=pd.merge(offline_train10,UC_lqAndxfCount_df,on=["User_id","Coupon_id"],how="left")
	offline_train10.fillna({"UC_lqAndxfCount":0},inplace=True)
	#持券购买/购券:UC_lqAndxfRatio
	offline_train10["UC_lqAndxfRatio"]=offline_train10.UC_lqAndxfCount/offline_train10.UC_lqCount
	offline_train10.fillna({"UC_lqAndxfRatio":0},inplace=True)
	#offline_train10.corr()
	return offline_train10

def addOLFeatures(online_all_pre,offline_train):
    new_user=list(set(offline_train.User_id)-set(online_all_pre.User_id))
    #用户线上特征
    #添加用户线上购物次数
    xfCount=pd.Series(index=offline_train.User_id.unique())
    xfCount.loc[new_user]=-1
    xfCount_online_all=online_all_pre.groupby("User_id").size()
    xfCount[xfCount.isnull()]=xfCount_online_all[xfCount[xfCount.isnull()].index]
    xfCount_df=pd.DataFrame(xfCount.values,index=xfCount.index,columns=["xfCount_OL"])#线下训练集用户线上购物次数
    offline_train_1=pd.merge(offline_train,xfCount_df,left_on="User_id",right_index=True,how="left")
    #添加用户线上领券次数
    lqCount=pd.Series(index=offline_train.User_id.unique())
    lqCount[new_user]=-1
    lqCount_online_all=online_all_pre[online_all_pre.Date_received.notnull()].groupby("User_id").size()
    lqCount[lqCount.isnull()]=lqCount_online_all[lqCount[lqCount.isnull()].index]
    lqCount.fillna(0,inplace=True)
    lqCount_df=pd.DataFrame(lqCount.values,index=lqCount.index,columns=["lqCount_OL"])
    offline_train_2=pd.merge(offline_train_1,lqCount_df,left_on="User_id",right_index=True,how="left")
    #添加用户线上领券并消费次数
    lqxfCount=pd.Series(index=offline_train.User_id.unique())
    lqxfCount[new_user]=-1
    lqxf_online_all=online_all_pre.loc[(online_all_pre.Date.notnull())&(online_all_pre.Date_received.notnull()),:].groupby("User_id").size()
    lqxfCount[lqxfCount.isnull()]=lqxf_online_all[lqxfCount[lqxfCount.isnull()].index]
    lqxfCount.fillna(0,inplace=True)
    lqxfCount_df=pd.DataFrame(lqxfCount.values,index=lqxfCount.index,columns=["lqxfCount_OL"])
    offline_train_3=pd.merge(offline_train_2,lqxfCount_df,left_on="User_id",right_index=True,how="left")
    #优惠券特征
    #添加用户线上偏好优惠券类型
    online_all_pre["mjType"]=np.where(online_all_pre.Discount_rate.isnull(),np.nan,np.where(online_all_pre.Discount_rate=="fixed","fixed",np.where(online_all_pre.Discount_rate.map(lambda x:":" in str(x)),"mj","ratio")))
    online_all_pre.loc[online_all_pre.mjType=="mj","mjThresh"]=online_all_pre[online_all_pre.mjType=="mj"].Discount_rate.map(lambda x:float(x.split(":")[0]))
    online_all_pre.loc[online_all_pre.mjType!="mj","mjThresh"]=0#满减门槛
    online_all_pre.loc[online_all_pre.mjType=="mj","mjRatio"]=(online_all_pre[online_all_pre.mjType=="mj"].Discount_rate.map(lambda x:float(x.split(":")[0]))-online_all_pre[online_all_pre.mjType=="mj"].Discount_rate.map(lambda x:float(x.split(":")[1])))/online_all_pre[online_all_pre.mjType=="mj"].Discount_rate.map(lambda x:float(x.split(":")[0]))    
    online_all_pre.loc[online_all_pre.mjType!="mj","mjRatio"]=0#满减折扣率
    online_all_pre.loc[online_all_pre.mjType=="mj","jmnum"]=online_all_pre[online_all_pre.mjType=="mj"].Discount_rate.map(lambda x:float(x.split(":")[1]))
    online_all_pre.loc[online_all_pre.mjType!="mj","jmnum"]=0#减免额度
    online_all_pre.loc[online_all_pre.mjType=="ratio","ratioIntense"]=online_all_pre[online_all_pre.mjType=="ratio"].Discount_rate
    online_all_pre.loc[online_all_pre.mjType!="ratio","ratioIntense"]=0#折扣率
    
    #不同mjType优惠券被领取次数
    coupon_mjType=online_all_pre[online_all_pre.Coupon_id.notnull()].groupby("mjType").size()
    #不同mjType优惠券被领取均值
    user_lst=list(online_all_pre[online_all_pre.Coupon_id.notnull()].groupby("mjType")["User_id"])
    userCount_mjType=pd.Series([i[1].unique().shape[0] for i in user_lst],index=[i[0] for i in user_lst])
    lqCountmean_mjType=coupon_mjType/userCount_mjType
    #不同mjType优惠券被消费次数
    xfCount_mjType=online_all_pre[(online_all_pre.Coupon_id.notnull())&(online_all_pre.Date.notnull())].groupby("mjType").size()
    #不同mjType优惠券被消费比率
    lqxfRatio_mjType=xfCount_mjType/coupon_mjType
    
    mjType_df=pd.concat([coupon_mjType,xfCount_mjType,lqxfRatio_mjType],axis=1).rename(columns={0:"coupon_mjType_OL",1:"xfCount_mjType_OL",2:"lqxfRatio_mjType_OL"})
    offline_train_3_temp=offline_train_3.copy()
    offline_train_3_temp["DisType"]=np.where(offline_train_3_temp.Discount_rate.isnull(),np.nan,np.where(offline_train_3.Discount_rate=="fixed","fixed",np.where(offline_train_3.Discount_rate.map(lambda x:":" in str(x)),"mj","ratio")))
    offline_train_4_temp=pd.merge(offline_train_3_temp,mjType_df,left_on="DisType",right_index=True,how="left")
    offline_train_4=pd.concat([offline_train_3,offline_train_4_temp[["coupon_mjType_OL","xfCount_mjType_OL","lqxfRatio_mjType_OL"]]],axis=1)
    offline_train_4.fillna(pd.Series(0,index=offline_train_4.columns[-3:]),inplace=True)

    #不同mjThresh优惠券领取次数
    coupon_mjThresh=online_all_pre[online_all_pre.Coupon_id.notnull()].groupby("mjThresh").size()
    #不同mjThresh优惠券被消费次数
    xfCount_mjThresh=online_all_pre[(online_all_pre.Coupon_id.notnull())&(online_all_pre.Date.notnull())].groupby("mjThresh").size()
    #不同mjThres优惠券被消费比率
    lqxfRatio_mjThresh=xfCount_mjThresh/coupon_mjThresh
    
    mjThresh_df=pd.concat([coupon_mjThresh,xfCount_mjThresh,lqxfRatio_mjThresh],axis=1).rename(columns={0:"coupon_mjThresh_OL",1:"xfCount_mjThresh_OL",2:"lqxfRatio_mjThresh_OL"})
    offline_train_5=pd.merge(offline_train_4,mjThresh_df,left_on="mjThresh",right_index=True,how="left")
    offline_train_5.fillna(pd.Series(0,index=offline_train_5.columns[-3:]),inplace=True)
    
    #不同ratioIntense优惠券领取次数
    coupon_ratioIntense=online_all_pre[online_all_pre.Coupon_id.notnull()].groupby("ratioIntense").size()
    #不同ratioIntense优惠券消费次数
    xfCount_ratioIntense=online_all_pre[(online_all_pre.Coupon_id.notnull())&(online_all_pre.Date.notnull())].groupby("ratioIntense").size()
    #不同ratioIntense优惠券被消费比率
    lqxfRatio_ratioIntense=xfCount_ratioIntense/coupon_ratioIntense
    
    ri_df=pd.concat([coupon_ratioIntense,xfCount_ratioIntense,lqxfRatio_ratioIntense],axis=1).rename(columns={0:"coupon_ratioIntense_OL",1:"xfCount_ratioIntense_OL",2:"lqxfRatio_ratioIntense_OL"})
    offline_train_6=pd.merge(offline_train_5,ri_df,left_on="ratioIntense",right_index=True,how="left")
    offline_train_6.fillna(pd.Series(0,index=offline_train_6.columns[-3:]),inplace=True)

    #用户X优惠券
    #User_id X mjType
    UserMjType_lq=online_all_pre[online_all_pre.Coupon_id.notnull()].groupby(["User_id","mjType"]).size()
    UserMjType_lqxf=online_all_pre[(online_all_pre.Coupon_id.notnull())&(online_all_pre.Date.notnull())].groupby(["User_id","mjType"]).size()
    UserMjType_xfRatio=UserMjType_lqxf/UserMjType_lq
    UserMjType_xfRatio.fillna(0,inplace=True)

    UserMjType_df=pd.concat([UserMjType_lq,UserMjType_lqxf,UserMjType_xfRatio],axis=1).rename(columns={0:"UserMjType_lq_OL",1:"UserMjType_lqxf_OL",2:"UserMjType_xfRatio_OL"}).reset_index()
    UserMjType_df.fillna(0,inplace=True)
    offline_train_6_temp=offline_train_6.copy()
    offline_train_6_temp["DisType"]=np.where(offline_train_6_temp.Discount_rate.isnull(),np.nan,np.where(offline_train_3.Discount_rate=="fixed","fixed",np.where(offline_train_3.Discount_rate.map(lambda x:":" in str(x)),"mj","ratio")))
    offline_train_7_temp=pd.merge(offline_train_6_temp,UserMjType_df,left_on=["User_id","DisType"],right_on=["User_id","mjType"],how="left")
    offline_train_7_temp.index=offline_train_6_temp.index
    offline_train_7_temp.fillna(pd.Series(0,index=offline_train_7_temp.columns[-3:]),inplace=True)
    offline_train_7=pd.concat([offline_train_6,offline_train_7_temp[offline_train_7_temp.columns[-3:]]],axis=1)
    
    #User_id X mjThresh 
    UserMjThresh_lq=online_all_pre[online_all_pre.Coupon_id.notnull()].groupby(["User_id","mjThresh"]).size()
    UserMjThresh_lqxf=online_all_pre[(online_all_pre.Coupon_id.notnull())&(online_all_pre.Date.notnull())].groupby(["User_id","mjThresh"]).size()
    UserMjThresh_xfRatio=UserMjThresh_lqxf/UserMjThresh_lq
    UserMjThresh_xfRatio.fillna(0,inplace=True)

    UserMjThresh_df=pd.concat([UserMjThresh_lq,UserMjThresh_lqxf,UserMjThresh_xfRatio],axis=1).rename(columns={0:"UserThresh_lq_OL",1:"UserThresh_lqxf_OL",2:"UserThresh_xfRatio_OL"}).reset_index()
    UserMjThresh_df.fillna(0,inplace=True)
    offline_train_8=pd.merge(offline_train_7,UserMjThresh_df,on=["User_id","mjThresh"],how="left")
    offline_train_8.fillna(pd.Series(0,index=offline_train_8.columns[-3:]),inplace=True)

    #User_id X ratioIntense    
    UserRI_lq=online_all_pre[online_all_pre.Coupon_id.notnull()].groupby(["User_id","ratioIntense"]).size()
    UserRI_lqxf=online_all_pre[(online_all_pre.Coupon_id.notnull())&(online_all_pre.Date.notnull())].groupby(["User_id","ratioIntense"]).size()
    UserRI_xfRatio=UserRI_lqxf/UserRI_lq
    UserRI_xfRatio.fillna(0,inplace=True)

    UserRI_df=pd.concat([UserRI_lq,UserRI_lqxf,UserRI_xfRatio],axis=1).rename(columns={0:"UserRI_lq_OL",1:"UserRI_lqxf_OL",2:"UserRI_xfRatio_OL"}).reset_index()
    UserRI_df.fillna(0,inplace=True)
    offline_train_9=pd.merge(offline_train_8,UserRI_df,on=["User_id","ratioIntense"],how="left")
    offline_train_9.fillna(pd.Series(0,index=offline_train_9.columns[-3:]),inplace=True)
    
    return offline_train_9

def DateBeforeCount(s,df):
    lq=df[(df.User_id==s["User_id"])&(df.Date_received<s["Date_received"])].shape[0]#用户当天之前所有领券数
    lq_CouId=df[(df.User_id==s["User_id"])&(df.Date_received<s["Date_received"])&(df.Coupon_id==s["Coupon_id"])].shape[0]#用户当天之前领取指定CoupId数
    lq_mjThresh=df[(df.User_id==s["User_id"])&(df.mjThresh==s["mjThresh"])&(df.Date_received<s["Date_received"])].shape[0]#用户当天之前领取指定mjThresh数
    lq_ri=df[(df.User_id==s["User_id"])&(df.ratioIntense==s["ratioIntense"])&(df.Date_received<s["Date_received"])].shape[0]#用户当天之前领取指定ratioIntense数
    return pd.Series([lq,lq_CouId,lq_mjThresh,lq_ri],index=["lq_dtBefore_UserPre","lq_dtBeforeCouId_UserPre","lq_dtBeforemjThresh_UserPre","lq_dtBeforeri_UserPre"])

def DateAfterCount(s,df):
    lq=df[(df.User_id==s["User_id"])&(df.Date_received>s["Date_received"])].shape[0]#用户当天之前所有领券数
    lq_CouId=df[(df.User_id==s["User_id"])&(df.Date_received>s["Date_received"])&(df.Coupon_id==s["Coupon_id"])].shape[0]#用户当天之前领取指定CoupId数
    lq_mjThresh=df[(df.User_id==s["User_id"])&(df.mjThresh==s["mjThresh"])&(df.Date_received>s["Date_received"])].shape[0]#用户当天之前领取指定mjThresh数
    lq_ri=df[(df.User_id==s["User_id"])&(df.ratioIntense==s["ratioIntense"])&(df.Date_received>s["Date_received"])].shape[0]#用户当天之前领取指定ratioIntense数
    return pd.Series([lq,lq_CouId,lq_mjThresh,lq_ri],index=["lq_dtAfter_UserPre","lq_dtAfterCouId_UserPre","lq_dtAftermjThresh_UserPre","lq_dtAfterri_UserPre"])

#增加带预测区间特征：提取预测区间的特征信息
def addPreFeatures(offline_train):
    df=offline_train.copy()
    print("增加用户特征")
    #领取券数
    lq_UserPre=pd.DataFrame(df.groupby("User_id").size()).rename(columns={0:"lq_UserPre"})
    df1=pd.merge(df,lq_UserPre,left_on="User_id",right_index=True,how="left")
    #光顾的商家数
    merchants_User=list(df.groupby("User_id")["Merchant_id"])
    mCounts_User=pd.Series([i[1].unique().shape[0] for i in merchants_User],index=[i[0] for i in merchants_User])
    mCounts_UserPre=pd.DataFrame(mCounts_User).rename(columns={0:"mCounts_UserPre"})
    df2=pd.merge(df1,mCounts_UserPre,left_on="User_id",right_index=True,how="left")
    print("增加商户特征")
    #商户特征
    #被领取券数
    lq_MerPre=pd.DataFrame(df.groupby("Merchant_id").size()).rename(columns={0:"lq_MerPre"})
    df3=pd.merge(df2,lq_MerPre,left_on="Merchant_id",right_index=True,how="left")
    #被光顾的用户数
    users_Mer=list(df.groupby("Merchant_id")["User_id"])
    userCounts_Mer=pd.Series([i[1].unique().shape[0] for i in users_Mer],index=[i[0] for i in users_Mer])
    userCounts_MerPre=pd.DataFrame(userCounts_Mer).rename(columns={0:"userCounts_MerPre"})
    df4=pd.merge(df3,userCounts_MerPre,left_on="Merchant_id",right_index=True,how="left")
    print("增加优惠券特征")
    #优惠券特征
    #以优惠券id作为标识,被领取数
    lq_CouIdPre=pd.DataFrame(df.groupby("Coupon_id").size()).rename(columns={0:"lq_CouIdPre"})
    df5=pd.merge(df4,lq_CouIdPre,left_on="Coupon_id",right_index=True,how="left")
    #以mjThresh为标识,被领取数
    lq_mjThreshPre=pd.DataFrame(df.groupby("mjThresh").size()).rename(columns={0:"lq_mjThreshPre"})
    df6=pd.merge(df5,lq_mjThreshPre,left_on="mjThresh",right_index=True,how="left")
    #以ratioIntense为标识,被领取数
    lq_RIPre=pd.DataFrame(df.groupby("ratioIntense").size()).rename(columns={0:"lq_ratioIntensePre"})
    df7=pd.merge(df6,lq_RIPre,left_on="ratioIntense",right_index=True,how="left")
    print("增加用户-商户交叉特征")
    lq_UMPre=df.groupby(["User_id","Merchant_id"]).size().reset_index().rename(columns={0:"lq_UMPre"})
    df8=pd.merge(df7,lq_UMPre,on=["User_id","Merchant_id"],how="left")
    print("增加用户-优惠券交叉特征")
    lq_CouId_UCPre=df.groupby(["User_id","Coupon_id"]).size().reset_index().rename(columns={0:"lq_CouId_UCPre"})
    df9=pd.merge(df8,lq_CouId_UCPre,on=["User_id","Coupon_id"],how="left")
    lq_mjThresh_UCPre=df.groupby(["User_id","mjThresh"]).size().reset_index().rename(columns={0:"lq_mjThresh_UCPre"})
    df10=pd.merge(df9,lq_mjThresh_UCPre,on=["User_id","mjThresh"],how="left")
    lq_ratioIntense_UCPre=df.groupby(["User_id","ratioIntense"]).size().reset_index().rename(columns={0:"lq_ratioIntense_UCPre"})
    df11=pd.merge(df10,lq_ratioIntense_UCPre,on=["User_id","ratioIntense"],how="left")
    print("增加商户-优惠券交叉特征")
    lq_CouId_MCPre=df.groupby(["Merchant_id","Coupon_id"]).size().reset_index().rename(columns={0:"lq_CouId_MCPre"})
    df12=pd.merge(df11,lq_CouId_MCPre,on=["Merchant_id","Coupon_id"],how="left")
    lq_mjThresh_MCPre=df.groupby(["Merchant_id","mjThresh"]).size().reset_index().rename(columns={0:"lq_mjThresh_MCPre"})
    df13=pd.merge(df12,lq_mjThresh_MCPre,on=["Merchant_id","mjThresh"],how="left")
    lq_ratioIntense_MCPre=df.groupby(["Merchant_id","ratioIntense"]).size().reset_index().rename(columns={0:"lq_ratioIntense_MCPre"})
    df14=pd.merge(df13,lq_ratioIntense_MCPre,on=["Merchant_id","ratioIntense"],how="left")
    print("增加用户当天领取的券数")
    lq_dt_UserPre=df.groupby(["User_id","Date_received"]).size().reset_index().rename(columns={0:"lq_dt_UserPre"})
    df15=pd.merge(df14,lq_dt_UserPre,on=["User_id","Date_received"],how="left")
    lq_dt_CouId_UserPre=df.groupby(["User_id","Coupon_id","Date_received"]).size().reset_index().rename(columns={0:"lq_dt_CouId_UserPre"})
    df16=pd.merge(df15,lq_dt_CouId_UserPre,on=["User_id","Coupon_id","Date_received"],how="left")
    lq_dt_mjThresh_UserPre=df.groupby(["User_id","mjThresh","Date_received"]).size().reset_index().rename(columns={0:"lq_dt_mjThresh_UserPre"})
    df17=pd.merge(df16,lq_dt_mjThresh_UserPre,on=["User_id","mjThresh","Date_received"],how="left")
    lq_dt_ratioIntense_UserPre=df.groupby(["User_id","ratioIntense","Date_received"]).size().reset_index().rename(columns={0:"lq_dt_ri_UserPre"})
    df18=pd.merge(df17,lq_dt_ratioIntense_UserPre,on=["User_id","ratioIntense","Date_received"],how="left")
    print("用户当天之前领取的券数(Couon_id,mjThresh,ratioIntense)")
    #df19_temp=df17.apply(lambda x:DateBeforeCount(x,df17),axis=1)#耗时太久
    #CouId标识
    df19_temp=df.groupby(["User_id","Coupon_id","Date_received"]).size().reset_index().rename(columns={0:"lq_dt_CouId_UserPre"})
    df19_temp["lq_dtBefore_CouId_UserPre"]=df19_temp.groupby(["User_id","Coupon_id"])["lq_dt_CouId_UserPre"].apply(lambda x:x.cumsum())
    df19_temp.drop("lq_dt_CouId_UserPre",axis=1,inplace=True)
    df19=pd.merge(left=df18,right=df19_temp,on=["User_id","Coupon_id","Date_received"],how="left")
    #mjThresh标识
    df20_temp=df.groupby(["User_id","mjThresh","Date_received"]).size().reset_index().rename(columns={0:"lq_dt_mjThresh_UserPre"})
    df20_temp["lq_dtBefore_mjThresh_UserPre"]=df20_temp.groupby(["User_id","mjThresh"])["lq_dt_mjThresh_UserPre"].apply(lambda x:x.cumsum())
    df20_temp.drop("lq_dt_mjThresh_UserPre",axis=1,inplace=True)
    df20=pd.merge(left=df19,right=df20_temp,on=["User_id","mjThresh","Date_received"],how="left")
    #ratioIntense标识
    df21_temp=df.groupby(["User_id","ratioIntense","Date_received"]).size().reset_index().rename(columns={0:"lq_dt_ratioIntense_UserPre"})
    df21_temp["lq_dtBefore_ratioIntense_UserPre"]=df21_temp.groupby(["User_id","ratioIntense"])["lq_dt_ratioIntense_UserPre"].apply(lambda x:x.cumsum())
    df21_temp.drop("lq_dt_ratioIntense_UserPre",axis=1,inplace=True)
    df21=pd.merge(left=df20,right=df21_temp,on=["User_id","ratioIntense","Date_received"],how="left")
    print("用户当天之后领取的券数(Coupon_id,mjThresh,ratioIntense)")
    #CouId标识
    df22_temp=df.groupby(["User_id","Coupon_id","Date_received"]).size().reset_index().rename(columns={0:"lq_dt_CouId_UserPre"}).sort_values(by=["User_id","Coupon_id","Date_received"],ascending=[True,True,False])
    df22_temp["lq_dtAfter_CouId_UserPre"]=df22_temp.groupby(["User_id","Coupon_id"])["lq_dt_CouId_UserPre"].apply(lambda x:x.cumsum())
    df22_temp.drop("lq_dt_CouId_UserPre",axis=1,inplace=True)
    df22=pd.merge(left=df21,right=df22_temp,on=["User_id","Coupon_id","Date_received"],how="left")
    #mjThresh标识
    df23_temp=df.groupby(["User_id","mjThresh","Date_received"]).size().reset_index().rename(columns={0:"lq_dt_mjThresh_UserPre"}).sort_values(by=["User_id","mjThresh","Date_received"],ascending=[True,True,False])
    df23_temp["lq_dtAfter_mjThresh_UserPre"]=df23_temp.groupby(["User_id","mjThresh"])["lq_dt_mjThresh_UserPre"].apply(lambda x:x.cumsum())
    df23_temp.drop("lq_dt_mjThresh_UserPre",axis=1,inplace=True)
    df23=pd.merge(left=df22,right=df23_temp,on=["User_id","mjThresh","Date_received"],how="left")
    #ratioIntense标识
    df24_temp=df.groupby(["User_id","ratioIntense","Date_received"]).size().reset_index().rename(columns={0:"lq_dt_ratioIntense_UserPre"}).sort_values(by=["User_id","ratioIntense","Date_received"],ascending=[True,True,False])
    df24_temp["lq_dtAfter_ratioIntense_UserPre"]=df24_temp.groupby(["User_id","ratioIntense"])["lq_dt_ratioIntense_UserPre"].apply(lambda x:x.cumsum())
    df24_temp.drop("lq_dt_ratioIntense_UserPre",axis=1,inplace=True)
    df24=pd.merge(left=df23,right=df24_temp,on=["User_id","ratioIntense","Date_received"],how="left")
    #用户上一次领券时间间隔:略
    return df24

def dataStand(se):
        m=se[se!=-1].mean()
        v=se[se!=-1].std()
        nn=(se[se!=-1]-m)/v
        se[nn.index]=nn
        return se






















