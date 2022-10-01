# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 09:31:07 2022

@author: hp
"""



import pandas as pd              
import pickle                   
import streamlit  as st
import modulee 
from sklearn.metrics import confusion_matrix
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss


#loadig the saved model
loaded_model=pickle.load(open('trained_model_lg1.sav','rb'))
loaded_model1=pickle.load(open('pca11.sav','rb'))
nrows=100
data=pd.read_csv("newdata1.csv",nrows=nrows)
X=pd.read_csv("xdata1.csv")
y=pd.read_csv("ydata1.csv")
cal=pickle.load(open('cali1.sav','rb'))

#Creating a function for prediction
def telstranet_disruption_prediction(Id1,event_type1,log_feature1,volume1,resource_type1,severity_type1):
   id1=str(Id1)
   id=modulee.preproccesing(id1)
   if (id=='enter a valid value'):
     return 'enter a valid value'
   event_type= modulee.preproccesing(event_type1)
   if (event_type=='enter a valid value'):
     return 'enter a valid value'
   log_feature=modulee.preproccesing(log_feature1)
   if (log_feature=='enter a valid value'):
     return 'enter a valid value'
   resource1=modulee.resourcet(resource_type1)
   if (resource1=='enter a valid value'):
     return 'enter a valid value'
   severity1=modulee.severityt(severity_type1)
   if (severity1=='enter a valid value'):
     return 'enter a valid value'
   v1=str(volume1)
   volume=modulee.preproccesing(v1)
   if (volume=='enter a valid value'):
     return 'enter a valid value'
   
   d2=dict(enumerate(resource1))
   d1=dict(enumerate(severity1))
   severity_type1=d1[0]
   severity_type2=d1[1]
   severity_type3=d1[2]
   severity_type4=d1[3]
   severity_type5=d1[4]
   resource_type1=d2[0]
   resource_type10=d2[1]
   resource_type2=d2[2]
   resource_type3=d2[3]
   resource_type4=d2[4]
   resource_type5=d2[5]
   resource_type6=d2[6]
   resource_type7=d2[7]
   resource_type8=d2[8]
   resource_type9=d2[9]
   
   p1=[[id,event_type,log_feature,volume,resource_type1,resource_type10,resource_type2,resource_type3,resource_type4,resource_type5,resource_type6,resource_type7,resource_type8,resource_type9,severity_type1,severity_type2,severity_type3,severity_type4,severity_type5]]
   pr=[id,event_type,log_feature,volume,resource_type1,resource_type10,resource_type2,resource_type3,resource_type4,resource_type5,resource_type6,resource_type7,resource_type8,resource_type9,severity_type1,severity_type2,severity_type3,severity_type4,severity_type5]
   print(pr)
   p2=loaded_model1.transform(p1)
   pca1=p2[0][0]
   print(pca1)
   prediction = loaded_model.predict([[id,event_type,log_feature,volume,resource_type1,resource_type10,resource_type2,resource_type3,resource_type4,resource_type5,resource_type6,resource_type7,resource_type8,resource_type9,severity_type1,severity_type2,severity_type3,severity_type4,severity_type5,pca1]])
   print(prediction)
   if prediction[0]==0:
      return 'no fault in the telstra network'
   if prediction[0]==1:
      return 'few faults in the telstra network'
   if prediction[0]==2:
      return 'many faults in the telstra network'




def telstranet_disruption_prediction1(x,y1):
    d=x["event_type"].values
    c= modulee.dfeature(d)
    x["event_type"]=c
    X1=x
    d1=X1["log_feature"].values
    c1=modulee.dfeature(d1)
    X1["log_feature"]=c1
    dummies = pd.get_dummies(X1.resource_type)  
    dummies1 = pd.get_dummies(X1.severity_type)                                    #getting the dummie variables of severity_type                   #
    X2= pd.concat([X1,dummies,dummies1],axis=1) 
    X3=X2.drop(['resource_type','severity_type'],axis=1)
    pca=loaded_model1.transform(X3)
    dp=pd.DataFrame(data=pca ,columns=['pca1'])
    X4=pd.concat([X3,dp],axis = 1,join='inner')
    predict_y=cal.predict_proba(X4)
    ll=log_loss(y1, predict_y,labels=loaded_model.classes_, eps=1e-15)
    C=confusion_matrix(y1, cal.predict(X4))
    return ll,C

xy,c=telstranet_disruption_prediction1(X,y)
print("*"*20)  
print("logloss",xy)  
print("confusion_matrix",c)
vv=telstranet_disruption_prediction(1234,'event_type 2','feature 203',3,'resource_type 6','severity_type 2')
print(vv)
#streamlit part were we use user interface
def main():
    #giving a title
    st.title('Telstra Network Disruptions web App')
    
    #load 1000 rows of data into the dataframed
    st.write(data)
    

    #getting the input from user
    
   
    Id=st.number_input(label='id number',min_value=0)
    
    
   
    event_type=st.text_input(label='event type value',type='password')
    
   
    
    log_feature=st.text_input(label='log features value',type='password')
    
    
   
   
    volume=st.number_input(label='volume_value',min_value=0)
    
   
    
    resource_type=st.text_input(label='resource_type value',type='password')
    

   
    severity_type=st.text_input(label='severity_type value',type='password')
    
    
    #prediction code
    network_disruption=''
    
    #creating a button for prediction
    
    if st.button('network disruption result'):
       
        network_disruption= telstranet_disruption_prediction(Id,event_type,log_feature,volume,resource_type,severity_type)
       
    st.success(network_disruption)



    
    
    
if __name__=='__main__':
    main()