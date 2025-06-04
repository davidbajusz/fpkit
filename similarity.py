''' FPKit. Module for bitvector similarity calculations.
    For the abbreviations and similarity definitions, see Todeschini et al, J Chem Inf Model 52(11):2884-2901, 2012.
'''

from math import sqrt, log, asin, pi
import numpy as np

metrics_redundant=['SM','RT','JT','Gle','RR','For','Sim','BB','DK','BUB','Kul','SS1','SS2','Ja','Fai','Mou','Mic','RG','HD','Yu1','Yu2','Fos','Den','Co1','Co2','dis','GK','SS3','SS4','Phi','Di1','Di2','Sor','Coh','Pe1','Pe2','MP','HL','CT1','CT2','CT3','CT4','CT5','AC','Ham','McC','GL','BU2','Joh','Sco','Maa']

metrics=['SM','RT','JT','Gle','RR','For','Sim','BB','DK','BUB','Kul','SS1','SS2','Ja','Fai','Mou','Mic','RG','HD','Yu1','Yu2','Fos','Den','Co1','Co2','dis','GK','SS3','SS4','Phi','Di1','Di2','Sor','Coh','Pe1','Pe2','MP','HL','CT1','CT2','CT3','CT4','CT5','AC']

class FPError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)
    
def GetLen(aFP):
    '''Collect getlength methods here and decide based on data type.
    '''

    # Get type
    datatype=str(type(aFP))
    
    # Calculate length by type
    if datatype=="<type 'list'>" or datatype=="<class 'list'>":
        length=len(aFP)
    elif datatype=="<class 'numpy.ndarray'>":
        length=len(aFP)
    elif datatype=="<class 'pandas.core.series.Series'>":
        length=len(aFP)
    elif datatype=="<class 'cinfony.cdk.Fingerprint'>":
        cdkFPlengths=np.array([881,1024,79,166,4860,307])
        length=min(cdkFPlengths[cdkFPlengths <= aFP.fp.size()],key=lambda x:abs(x-aFP.fp.size()))
        # Not the most elegant solution, but cdkMolecule.fp.length() gives highest occupied bit position, not length!
    elif datatype=="<class 'cinfony.rdk.Fingerprint'>":
        length=len(aFP.fp)
    elif datatype=="<class 'rdkit.DataStructs.cDataStructs.LongSparseIntVect'>" or datatype=="<class 'rdkit.DataStructs.cDataStructs.IntSparseIntVect'>":
        length=aFP.GetLength()
    elif datatype=="<class 'rdkit.DataStructs.cDataStructs.ExplicitBitVect'>" or datatype=="<class 'rdkit.DataStructs.cDataStructs.SparseBitVect'>":
        length=aFP.GetNumBits()
    elif datatype=="<class 'cinfony.pybel.Fingerprint'>":
        length=len(aFP.fp)
    
    return float(length)
    
def get_abcdp(aFP,bFP):
    ''' Get a,b,c,d and p (length) from two FP objects.
    '''
    
    #Get type
    if str(type(aFP))==str(type(bFP)):
        datatype=str(type(aFP))
    else:
        raise FPError('Fingerprints of different types supplied')
    
    # Get Length
    if GetLen(aFP)==GetLen(bFP):
        p=GetLen(aFP)
    else:
        raise FPError('Fingerprints with different lengths supplied')
            
    # Extract fingerprints to list of "on" bit positions
    if datatype=="<type 'list'>" or datatype=="<class 'list'>":
        aList=[i for i,j in enumerate(aFP) if aFP[i]!=0]
    elif datatype=="<class 'numpy.ndarray'>":
        aList=np.nonzero(aFP)
    elif datatype=="<class 'pandas.core.series.Series'>":
        aList=[i for i in aFP.index if aFP[i]!=0]
    elif datatype=="<class 'cinfony.cdk.Fingerprint'>":
        aList=aFP.bits
    elif datatype=="<class 'cinfony.rdk.Fingerprint'>":
        aList=aFP.bits
    elif datatype=="<class 'rdkit.DataStructs.cDataStructs.LongSparseIntVect'>" or datatype=="<class 'rdkit.DataStructs.cDataStructs.IntSparseIntVect'>":
        aList=aFP.GetNonzeroElements().keys()
    elif datatype=="<class 'cinfony.pybel.Fingerprint'>":
        aList=aFP.bits
    elif datatype=="<class 'rdkit.DataStructs.cDataStructs.ExplicitBitVect'>" or datatype=="<class 'rdkit.DataStructs.cDataStructs.SparseBitVect'>":
        aList=[i for i in aFP.GetOnBits()]

    if datatype=="<type 'list'>" or datatype=="<class 'list'>":
        bList=[i for i,j in enumerate(bFP) if bFP[i]!=0]
    elif datatype=="<class 'numpy.ndarray'>":
        bList=np.nonzero(bFP)
    elif datatype=="<class 'pandas.core.series.Series'>":
        bList=[i for i in bFP.index if bFP[i]!=0]
    elif datatype=="<class 'cinfony.cdk.Fingerprint'>":
        bList=bFP.bits
    elif datatype=="<class 'cinfony.rdk.Fingerprint'>":
        bList=bFP.bits
    elif datatype=="<class 'rdkit.DataStructs.cDataStructs.LongSparseIntVect'>" or datatype=="<class 'rdkit.DataStructs.cDataStructs.IntSparseIntVect'>":
        bList=bFP.GetNonzeroElements().keys()
    elif datatype=="<class 'cinfony.pybel.Fingerprint'>":
        bList=bFP.bits
    elif datatype=="<class 'rdkit.DataStructs.cDataStructs.ExplicitBitVect'>" or datatype=="<class 'rdkit.DataStructs.cDataStructs.SparseBitVect'>":
        bList=[i for i in bFP.GetOnBits()]

    # Calculate a,b,c by set operations
    a=float(np.sum(np.in1d(aList,bList)))
    b=float(len(np.setdiff1d(aList,bList)))
    c=float(len(np.setdiff1d(bList,aList)))
    
    # Calculate d from the rest
    d=p-a-b-c
    
    return [a,b,c,d,p]
    
def sim(a,b,c,d,p,metric='JT',scale=False):
    ''' Calculate similarities from precomputed parameters.
    '''
    
    oneConditions={
                'SM': False,
                'RT': False,
                'JT': False,
                'Gle': False,
                'RR': False,
                'For': False,
                'Sim': False,
                'BB': False,
                'DK': False,
                'BUB': d==p,
                'Kul': False,
                'SS1': False,
                'SS2': False,
                'Ja': False,
                'Fai': False,
                'Mou': False,
                'Mic': a==p or d==p or (b+c)==0,
                'RG': a==p or d==p,
                'HD': a==p or d==p,
                'Yu1': a==p or d==p or (b*c)==0,
                'Yu2': a==p or d==p or (b*c)==0,
                'Fos': False,
                'Den': a==p or d==p,
                'Co1': a==p or d==p,
                'Co2': a==p or d==p,
                'dis': a==p or d==p,
                'GK': a==p or d==p,
                'SS3': a==p or d==p,
                'SS4': a==p or d==p,
                'Phi': a==p or d==p,
                'Di1': False,
                'Di2': False,
                'Sor': False,
                'Coh': a==p or d==p,
                'Pe1': a==p or d==p,
                'Pe2': a==p or d==p,
                'MP': a==p or d==p,
                'HL': a==p or d==p,
                'CT1': False,
                'CT2': False,
                'CT3': False,
                'CT4': False,
                'CT5': False,
                'AC': False,
                'Ham': False,
                'McC': False,
                'GL': False,
                'BU2': d==p,
                'Joh': False,
                'Sco': a==p or d==p,
                'Maa': False,
                }
    
    zeroConditions={
                'SM': False,
                'RT': False,
                'JT': a==0,
                'Gle': a==0,
                'RR': False,
                'For': (a+b)*(a+c)==0 or a==0,
                'Sim': min((a+b),(a+c))==0 or a==0,
                'BB': a==0,
                'DK': sqrt((a+b)*(a+c))==0,
                'BUB': False,
                'Kul': a==0,
                'SS1': a==0,
                'SS2': False,
                'Ja': a==0,
                'Fai': False,
                'Mou': False,
                'Mic': False,
                'RG': False,
                'HD': False,
                'Yu1': False,
                'Yu2': False,
                'Fos': (a+b)*(a+c)==0,
                'Den': sqrt(p*(a+b)*(a+c))==0,
                'Co1': (a+c)*(c+d)==0,
                'Co2': (a+b)*(b+d)==0,
                'dis': False,
                'GK': False,
                'SS3': a==0 and d==0,
                'SS4': a==0 or d==0,
                'Phi': b==p or c==p or sqrt((a+b)*(a+c)*(b+d)*(c+d))==0,
                'Di1': a==0,
                'Di2': a==0,
                'Sor': a==0,
                'Coh': ((a+b)*(b+d)+(a+c)*(c+d))==0,
                'Pe1': b==p or c==p,
                'Pe2': b==p or c==p,
                'MP': ((a+b)*(c+d)+(a+c)*(b+d))==0,
                'HL': (2*(a+b+c))==0 or (2*(b+c+d))==0,
                'CT1': False,
                'CT2': False,
                'CT3': False,
                'CT4': False,
                'CT5': False,
                'AC': False,
                'Ham': False,
                'McC': a==0,
                'GL': False,
                'BU2': False,
                'Joh': a==0,
                'Sco': False,
                'Maa': a==0,
                }
    
    if metric=='Mou' and (a*b+a*c+2*b*c)==0:
        return a/p
    
    if oneConditions[metric]:
        return 1.0
    
    if zeroConditions[metric]:
        return 0.0
    
    int_result = {
                'SM': lambda a,b,c,d,p: (a+d)/p,
                'RT': lambda a,b,c,d,p: (a+d)/(b+c+p),
                'JT': lambda a,b,c,d,p: a/(a+b+c),
                'Gle': lambda a,b,c,d,p: 2*a/(2*a+b+c),
                'RR': lambda a,b,c,d,p: a/p,
                'For': lambda a,b,c,d,p: p*a/((a+b)*(a+c)),
                'Sim': lambda a,b,c,d,p: a/min((a+b),(a+c)),
                'BB': lambda a,b,c,d,p: a/max((a+b),(a+c)),
                'DK': lambda a,b,c,d,p: a/sqrt((a+b)*(a+c)),
                'BUB': lambda a,b,c,d,p: (sqrt(a*d)+a)/(sqrt(a*d)+a+b+c),
                'Kul': lambda a,b,c,d,p: 0.5*(a/(a+b)+a/(a+c)),
                'SS1': lambda a,b,c,d,p: a/(a+2*b+2*c),
                'SS2': lambda a,b,c,d,p: (2*a+2*d)/(p+a+d),
                'Ja': lambda a,b,c,d,p: 3*a/(3*a+b+c),
                'Fai': lambda a,b,c,d,p: (a+0.5*d)/p,
                'Mou': lambda a,b,c,d,p: (2*a)/(a*b+a*c+2*b*c),
                'Mic': lambda a,b,c,d,p: 4*(a*d-b*c)/((a+d)**2+(b+c)**2),
                'RG': lambda a,b,c,d,p: a/(2*a+b+c)+d/(2*d+b+c),
                'HD': lambda a,b,c,d,p: 0.5*(a/(a+b+c)+d/(b+c+d)),
                'Yu1': lambda a,b,c,d,p: (a*d-b*c)/(a*d+b*c),
                'Yu2': lambda a,b,c,d,p: (sqrt(a*d)-sqrt(b*c))/(sqrt(a*d)+sqrt(b*c)),
                'Fos': lambda a,b,c,d,p: p*(a-0.5)**2/((a+b)*(a+c)),
                'Den': lambda a,b,c,d,p: (a*d-b*c)/sqrt(p*(a+b)*(a+c)),
                'Co1': lambda a,b,c,d,p: (a*d-b*c)/((a+c)*(c+d)),
                'Co2': lambda a,b,c,d,p: (a*d-b*c)/((a+b)*(b+d)),
                'dis': lambda a,b,c,d,p: (a*d-b*c)/p**2,
                'GK': lambda a,b,c,d,p: (2*min(a,d)-b-c)/(2*min(a,d)+b+c),
                'SS3': lambda a,b,c,d,p: 0.25*( (a/(a+b) if not a==0 else 0) + (a/(a+c) if not a==0 else 0) + (d/(b+d) if not d==0 else 0) + (d/(c+d) if not d==0 else 0) ),
                'SS4': lambda a,b,c,d,p: a/sqrt((a+b)*(a+c))*d/sqrt((b+d)*(c+d)),
                'Phi': lambda a,b,c,d,p: (a*d-b*c)/sqrt((a+b)*(a+c)*(b+d)*(c+d)),
                'Di1': lambda a,b,c,d,p: a/(a+b),
                'Di2': lambda a,b,c,d,p: a/(a+c),
                'Sor': lambda a,b,c,d,p: a**2/((a+b)*(a+c)),
                'Coh': lambda a,b,c,d,p: 2*(a*d-b*c)/((a+b)*(b+d)+(a+c)*(c+d)),
                'Pe1': lambda a,b,c,d,p: ( (a*d-b*c)/((a+b)*(c+d)) if not ((a+b)*(c+d))==0 else 0 ),
                'Pe2': lambda a,b,c,d,p: ( (a*d-b*c)/((a+c)*(b+d)) if not ((a+c)*(b+d))==0 else 0 ),
                'MP': lambda a,b,c,d,p: 2*(a*d-b*c)/((a+b)*(c+d)+(a+c)*(b+d)),
                'HL': lambda a,b,c,d,p: (a*(2*d)+b+c)/(2*(a+b+c))+(d*(2*a+b+c))/(2*(b+c+d)),
                'CT1': lambda a,b,c,d,p: log(1+a+d)/log(1+p),
                'CT2': lambda a,b,c,d,p: (log(1+p)-log(1+b+c))/log(1+p),
                'CT3': lambda a,b,c,d,p: log(1+a)/log(1+p),
                'CT4': lambda a,b,c,d,p: (log(1+a)/log(1+a+b+c) if not (a+b+c)==0 else 0),
                'CT5': lambda a,b,c,d,p: (log(1+a*d)-log(1+b*c))/log(1+p**2/4),
                'AC': lambda a,b,c,d,p: (2/pi)*asin(sqrt((a+d)/p)),
                'Ham': lambda a,b,c,d,p: (a+d-b-c)/p,
                'McC': lambda a,b,c,d,p: (a**2-b*c)/((a+b)*(a+c)),
                'GL': lambda a,b,c,d,p: (a+d)/(a+0.5*(b+c)+d),
                'BU2': lambda a,b,c,d,p: (sqrt(a*d)+a-b-c)/((sqrt(a*d)+a+b+c)),
                'Joh': lambda a,b,c,d,p: a/(a+b)+a/(a+c),
                'Sco': lambda a,b,c,d,p: (4*a*d-(b+c)**2)/((2*a+b+c)*(2*d+b+c)),
                'Maa': lambda a,b,c,d,p: (2*a-b-c)/(2*a+b+c),
                }[metric]

    if not scale:
        return int_result(a,b,c,d,p)
    
    alpha = {
            'SM': 0,
            'RT': 0,
            'JT': 0,
            'Gle': 0,
            'RR': 0,
            'For': 0,
            'Sim': 0,
            'BB': 0,
            'DK': 0,
            'BUB': 0,
            'Kul': 0,
            'SS1': 0,
            'SS2': 0,
            'Ja': 0,
            'Fai': 0,
            'Mou': 0,
            'Mic': 1,
            'RG': 0,
            'HD': 0,
            'Yu1': 1,
            'Yu2': 1,
            'Fos': 0,
            'Den': sqrt(p)/2.0,
            'Co1': p-1,
            'Co2': p-1,
            'dis': 0.25,
            'GK': 1,
            'SS3': 0,
            'SS4': 0,
            'Phi': 1,
            'Di1': 0,
            'Di2': 0,
            'Sor': 0,
            'Coh': 1,
            'Pe1': 1,
            'Pe2': 1,
            'MP': 1,
            'HL': 0,
            'CT1': 0,
            'CT2': 0,
            'CT3': 0,
            'CT4': 0,
            'CT5': 1,
            'AC': 0,
            'Ham': 1,
            'McC': 1,
            'GL': 0,
            'BU2': 1,
            'Joh': 0,
            'Sco': 1,
            'Maa': 1,
        }[metric]
    
    beta = {
            'SM': 1,
            'RT': 1,
            'JT': 1,
            'Gle': 1,
            'RR': 1,
            'For': (p/a if not a==0 else 0.00001), # Should be ruled out by zeroCondition, but clearly visible if not
            'Sim': 1,
            'BB': 1,
            'DK': 1,
            'BUB': 1,
            'Kul': 1,
            'SS1': 1,
            'SS2': 1,
            'Ja': 1,
            'Fai': 1,
            'Mou': 2,
            'Mic': 2,
            'RG': 1,
            'HD': 1,
            'Yu1': 2,
            'Yu2': 2,
            'Fos': (p-0.5)**2/p,
            'Den': 3*sqrt(p)/2,
            'Co1': p,
            'Co2': p,
            'dis': 0.5,
            'GK': 2,
            'SS3': 1,
            'SS4': 1,
            'Phi': 2,
            'Di1': 1,
            'Di2': 1,
            'Sor': 1,
            'Coh': 2,
            'Pe1': 2,
            'Pe2': 2,
            'MP': 2,
            'HL': p,
            'CT1': 1,
            'CT2': 1,
            'CT3': 1,
            'CT4': 1,
            'CT5': 2,
            'AC': 1,
            'Ham': 2,
            'McC': 2,
            'GL': 1,
            'BU2': 2,
            'Joh': 2,
            'Sco': 2,
            'Maa': 2,
        }[metric]
    
    result = (int_result(a,b,c,d,p)+float(alpha))/float(beta)
    
    return result


def similarity(aFP,bFP,metric='JT',scale=False):
    ''' Shortcut for complete similarity calculation of two fingerprints.'''
    
    return sim(*get_abcdp(aFP,bFP),metric=metric,scale=scale)

def dissimilarity(aFP,bFP,metric='JT',scale=True):
    ''' Shortcut for complete dissimilarity calculation of two fingerprints.
    By default, dissimilarities are calculated from the [0,1]-scaled similarity values.'''
    
    return (1 - sim(*get_abcdp(aFP,bFP),metric=metric,scale=scale))

def distance(aFP,bFP,metric='JT',scale=True):
    ''' Shortcut for complete distance calculation of two fingerprints.
    By default, distances are calculated from the [0,1]-scaled similarity values.'''
    
    return (1 / sim(*get_abcdp(aFP,bFP),metric=metric,scale=scale) - 1)
