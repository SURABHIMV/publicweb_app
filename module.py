# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 13:29:19 2022

@author: hp
"""
import re
import pandas as pd

def preproccesing(i):
 if (len(i)!=0):
  q=re.findall(r'\d+',i)
  q1=''.join(q)
  q1=int(q1)
  return  q1
 else:
     return "enter a valid value"


def severityt(m):
  

  ss=['severity_type 1','severity_type 2', 'severity_type 3', 'severity_type 4','severity_type 5']
  x1=pd.get_dummies(ss)
  print(x1)
  h1=x1.to_dict(orient='index')
  h2=h1.values()
  h3=list(h2)
  print(h3)
  gg=[]
  for dicts in h3:
    jj=[]

    for keys in dicts:
        dicts[keys] = int(dicts[keys])
        jj.append(dicts[keys])
    gg.append(jj)

  p=list(dicts.keys())
  res = dict(zip(p,gg))

  rr2=type(m)
  print("*****************************",rr2)
  if (len(m)== 0):
    return "enter a valid value"
  else:
    v=res[m]
    return v
  
d3=severityt('')
print(d3)

def resourcet(n):
 
  ss1=['resource_type 1','resource_type 10','resource_type 2','resource_type 3','resource_type 4','resource_type 5','resource_type 6','resource_type 7','resource_type 8', 'resource_type 9']
  xl1=pd.get_dummies(ss1)
  
  hl1=xl1.to_dict(orient='index')

  hl2=hl1.values()
  hl3=list(hl2)

  ggl=[]

  for dicts in hl3:
    jjl=[]

    for keys in dicts:
        dicts[keys] = int(dicts[keys])
        jjl.append(dicts[keys])
    ggl.append(jjl)
  pl=list(dicts.keys())   

  resl= dict(zip(pl,ggl))

  if (len(n)!= 0):
    vv=resl[n]
    return vv
  else:
     return "enter a valid value"


def dfeature(d5):
 if (len(d5)!= 0):
  s5=[]
  for i in d5:
     q=re.findall(r'\d+',i)
     q1=''.join(q)
     s5.append(int(q1))
  return s5
 else:
     return "enter a valid value"