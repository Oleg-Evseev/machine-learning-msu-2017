#!/usr/bin/env/python

import numpy as np


D3=np.loadtxt("3fgl_full.dat", dtype='string')
tp=D3[:,29]
D3[np.where((tp=='bll')|(tp=='BLL')|(tp=='bcu')|(tp=='BCU')|(tp=='fsrq')|(tp=='FSRQ')),29]=-1
D3[np.where((tp=='psr')|(tp=='PSR')),29]=1

D_TRAIN=D3[np.where((D3[:,29]=='-1')|(D3[:,29]=='1'))][0::2]
# every second element for training set (must be random for production)
D_TRAIN[:,0]=1.0
# replace object name with weight
D_TRAIN=D_TRAIN.astype('float')
# convert to float

def splitbranch(D,column):
    col=D[:,column].astype('float')
    m=np.median(col)
# TODO: weighted median
    return [D[np.where(col<m)],D[np.where(col>=m)],[column,m]]

DT1=splitbranch(D_TRAIN,28)
len(np.where(DT1[0][:,29]==-1)[0])
#400
len(np.where(DT1[0][:,29]==1)[0])
#71
len(np.where(DT1[1][:,29]==-1)[0])
#450
len(np.where(DT1[1][:,29]==1)[0])
#21

def entr(T1,tcol):
    ent=0;
    for t in (-1, 1):
        n1=len(np.where(T1[0][:,tcol]==t)[0])
        n2=len(np.where(T1[1][:,tcol]==t)[0])
# TODO: weighted entropy
        p1 = float(n1)/float(n1+n2)
        p2 = float(n2)/float(n1+n2)
        ent+= -p1*np.log(p1)/np.log(2) - p2*np.log(p2)/np.log(2)
    return ent


entr(DT1,29)
#1.7724698136958414

DT1=splitbranch(D_TRAIN,7)
entr(DT1,29)
#1.996893765206637
DT1=splitbranch(D_TRAIN,8)
entr(DT1,29)
#1.6629811096380782


def buildtree(D,columns):
    if not columns:
        return D
    else:
        c = columns [ : ]
        col = c.pop(0)
        T = splitbranch(D,col)
        return [buildtree(T[0],c),buildtree(T[1],c),T[2]]

DT2 = buildtree(D_TRAIN,[28,8])
