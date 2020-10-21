# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 18:15:41 2020

@author: Etienne Kischlin
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
import warnings
warnings.filterwarnings("ignore")

import six
import sys
sys.modules['sklearn.externals.six'] = six
import mlrose
import heapq
from math import sqrt
import scipy
import math


import matplotlib
matplotlib.use('TkAgg')


from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as Tk
from tkinter import Button
from matplotlib.figure import Figure




# Lager erstellen
grid = np.array([
    
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    
    [0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0],
    
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    
    [0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0],
    
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    
    [0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0],
    
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    
    [0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0],
    
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    
    [0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0],
    
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    
    [0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0],
    
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],])



###########################Apriori#############################################
# Lagerplätze 

a1_max = 1000
a2_max = 2000
a3_max = 2000
a4_max = 2000
a4_max = 2000
a5_max = 2000
a6_max = 2000

b1_max = 1000
b2_max = 2000
b3_max = 2000
b4_max = 2000
b4_max = 2000
b5_max = 2000
b6_max = 2000

c1_max = 1000
c2_max = 2000
c3_max = 2000
c4_max = 2000
c4_max = 2000
c5_max = 2000
c6_max = 2000

d1_max = 1000
d2_max = 2000
d3_max = 2000
d4_max = 2000
d4_max = 2000
d5_max = 2000
d6_max = 2000

###########################Apriori#############################################
fig = Figure(figsize=(18,8), dpi=100, facecolor='grey')
a = fig.add_subplot(111)

a.imshow(grid, cmap=plt.cm.Dark2, aspect='auto')

#axes = plt.gca()
a.set_xlim([0,len(grid[0])-1])
a.set_ylim([0,len(grid)-1])
a.axis('off')

t_fi1 = a.text(1.5, 11.3,'', horizontalalignment='center', weight='bold', fontsize=12)
t_a1_max = a.text(0.5, 10.85,'', horizontalalignment='left', fontsize=10)
t_size_fi1 = a.text(0.5, 10.6,'', horizontalalignment='left', fontsize=10)
n_fi1 = a.text(2.5, 10.85,'', horizontalalignment='right', fontsize=10)
nr_fi1 = a.text(2.5, 10.85,'', horizontalalignment='right', weight='bold', fontsize=10, color='red')

t_fi2 = a.text(1.5, 9.3,'', horizontalalignment='center', weight='bold', fontsize=12)
t_a2_max = a.text(0.5, 8.85,'', horizontalalignment='left', fontsize=10)
t_size_fi2 = a.text(0.5, 8.6,'', horizontalalignment='left', fontsize=10)
n_fi2 = a.text(2.5, 8.85,'', horizontalalignment='right', fontsize=10)
nr_fi2 = a.text(2.5, 8.85,'', horizontalalignment='right', weight='bold', fontsize=10, color='red')

t_fi3 = a.text(1.5, 7.3,'', horizontalalignment='center', weight='bold', fontsize=12)
t_a3_max = a.text(0.5, 6.85,'', horizontalalignment='left', fontsize=10)
t_size_fi3 = a.text(0.5, 6.6,'', horizontalalignment='left', fontsize=10)
n_fi3 = a.text(2.5, 6.85,'', horizontalalignment='right', fontsize=10)
nr_fi3 = a.text(2.5, 6.85,'', horizontalalignment='right', weight='bold', fontsize=10, color='red')

t_fi4 = a.text(1.5, 5.3,'', horizontalalignment='center', weight='bold', fontsize=12)
t_a4_max = a.text(0.5, 4.85,'', horizontalalignment='left', fontsize=10)
t_size_fi4 = a.text(0.5, 4.6,'', horizontalalignment='left', fontsize=10)
n_fi4 = a.text(2.5, 4.85,'', horizontalalignment='right', fontsize=10)
nr_fi4 = a.text(2.5, 4.85,'', horizontalalignment='right', weight='bold', fontsize=10, color='red')

t_fi5 = a.text(1.5, 3.3,'', horizontalalignment='center', weight='bold', fontsize=12)
t_a5_max = a.text(0.5, 2.85,'', horizontalalignment='left', fontsize=10)
t_size_fi5 = a.text(0.5, 2.6,'', horizontalalignment='left', fontsize=10)
n_fi5 = a.text(2.5, 2.85,'', horizontalalignment='right', fontsize=10)
nr_fi5 = a.text(2.5, 2.85,'', horizontalalignment='right', weight='bold', fontsize=10, color='red')

t_fi6 = a.text(1.5, 1.3,'', horizontalalignment='center', weight='bold', fontsize=12)
t_a6_max = a.text(0.5, 0.85,'', horizontalalignment='left', fontsize=10)
t_size_fi6 = a.text(0.5, 0.6,'', horizontalalignment='left', fontsize=10)
n_fi6 = a.text(2.5, 0.85,'', horizontalalignment='right', fontsize=10)
nr_fi6 = a.text(2.5, 0.85,'', horizontalalignment='right', weight='bold', fontsize=10, color='red')
#============================================================================================================
t_fi7 = a.text(4.5, 11.3,'', horizontalalignment='center', weight='bold', fontsize=12)
t_b1_max = a.text(3.5, 10.85,'', horizontalalignment='left', fontsize=10)
t_size_fi7 = a.text(3.5, 10.6,'', horizontalalignment='left', fontsize=10)
n_fi7 = a.text(5.5, 10.85,'', horizontalalignment='right', fontsize=10)
nr_fi7 = a.text(5.5, 10.85,'', horizontalalignment='right', weight='bold', fontsize=10, color='red')

t_fi8 = a.text(4.5, 9.3,'', horizontalalignment='center', weight='bold', fontsize=12)
t_b2_max = a.text(3.5, 8.85,'', horizontalalignment='left', fontsize=10)
t_size_fi8 = a.text(3.5, 8.6,'', horizontalalignment='left', fontsize=10)
n_fi8 = a.text(5.5, 8.85,'', horizontalalignment='right', fontsize=10)
nr_fi8 = a.text(5.5, 8.85,'', horizontalalignment='right', weight='bold', fontsize=10, color='red')

t_fi9 = a.text(4.5, 7.3,'', horizontalalignment='center', weight='bold', fontsize=12)
t_b3_max = a.text(3.5, 6.85,'', horizontalalignment='left', fontsize=10)
t_size_fi9 = a.text(3.5, 6.6,'', horizontalalignment='left', fontsize=10)
n_fi9 = a.text(5.5, 6.85,'', horizontalalignment='right', fontsize=10)
nr_fi9 = a.text(5.5, 6.85,'', horizontalalignment='right', weight='bold', fontsize=10, color='red')

t_fi10 = a.text(4.5, 5.3,'', horizontalalignment='center', weight='bold', fontsize=12)
t_b4_max = a.text(3.5, 4.85,'', horizontalalignment='left', fontsize=10)
t_size_fi10 = a.text(3.5, 4.6,'', horizontalalignment='left', fontsize=10)
n_fi10 = a.text(5.5, 4.85,'', horizontalalignment='right', fontsize=10)
nr_fi10 = a.text(5.5, 4.85,'', horizontalalignment='right', weight='bold', fontsize=10, color='red')

t_fi11 = a.text(4.5, 3.3,'', horizontalalignment='center', weight='bold', fontsize=12)
t_b5_max = a.text(3.5, 2.85,'', horizontalalignment='left', fontsize=10)
t_size_fi11 = a.text(3.5, 2.6,'', horizontalalignment='left', fontsize=10)
n_fi11 = a.text(5.5, 2.85,'', horizontalalignment='right', fontsize=10)
nr_fi11 = a.text(5.5, 2.85,'', horizontalalignment='right', weight='bold', fontsize=10, color='red')

t_fi12 = a.text(4.5, 1.3,'', horizontalalignment='center', weight='bold', fontsize=12)
t_b6_max = a.text(3.5, 0.85,'', horizontalalignment='left', fontsize=10)
t_size_fi12 = a.text(3.5, 0.6,'', horizontalalignment='left', fontsize=10)
n_fi12 = a.text(5.5, 0.85,'', horizontalalignment='right', fontsize=10)
nr_fi12 = a.text(5.5, 0.85,'', horizontalalignment='right', weight='bold', fontsize=10, color='red')
#========================================================================================0
t_fi13 = a.text(7.5, 11.3,'', horizontalalignment='center', weight='bold', fontsize=12)
t_c1_max = a.text(6.5, 10.85,'', horizontalalignment='left', fontsize=10)
t_size_fi13 = a.text(6.5, 10.6,'', horizontalalignment='left', fontsize=10)
n_fi13 = a.text(8.5, 10.85,'', horizontalalignment='right', fontsize=10)
nr_fi13 = a.text(8.5, 10.85,'', horizontalalignment='right', weight='bold', fontsize=10, color='red')

t_fi14 = a.text(7.5, 9.3,'', horizontalalignment='center', weight='bold', fontsize=12)
t_c2_max = a.text(6.5, 8.85,'', horizontalalignment='left', fontsize=10)
t_size_fi14 = a.text(6.5, 8.6,'', horizontalalignment='left', fontsize=10)
n_fi14 = a.text(8.5, 8.85,'', horizontalalignment='right', fontsize=10)
nr_fi14 = a.text(8.5, 8.85,'', horizontalalignment='right', weight='bold', fontsize=10, color='red')

t_fi15 = a.text(7.5, 7.3,'', horizontalalignment='center', weight='bold', fontsize=12)
t_c3_max = a.text(6.5, 6.85,'', horizontalalignment='left', fontsize=10)
t_size_fi15 = a.text(6.5, 6.6,'', horizontalalignment='left', fontsize=10)
n_fi15 = a.text(8.5, 6.85,'', horizontalalignment='right', fontsize=10)
nr_fi15 = a.text(8.5, 6.85,'', horizontalalignment='right', weight='bold', fontsize=10, color='red')

t_fi16 = a.text(7.5, 5.3,'', horizontalalignment='center', weight='bold', fontsize=12)
t_c4_max = a.text(6.5, 4.85,'', horizontalalignment='left', fontsize=10)
t_size_fi16 = a.text(6.5, 4.6,'', horizontalalignment='left', fontsize=10)
n_fi16 = a.text(8.5, 4.85,'', horizontalalignment='right', fontsize=10)
nr_fi16 = a.text(8.5, 4.85,'', horizontalalignment='right', weight='bold', fontsize=10, color='red')

t_fi17 = a.text(7.5, 3.3,'', horizontalalignment='center', weight='bold', fontsize=12)
t_c5_max = a.text(6.5, 2.85,'', horizontalalignment='left', fontsize=10)
t_size_fi17 = a.text(6.5, 2.6,'', horizontalalignment='left', fontsize=10)
n_fi17 = a.text(8.5, 2.85,'', horizontalalignment='right', fontsize=10)
nr_fi17 = a.text(8.5, 2.85,'', horizontalalignment='right', weight='bold', fontsize=10, color='red')

t_fi18 = a.text(7.5, 1.3,'', horizontalalignment='center', weight='bold', fontsize=12)
t_c6_max = a.text(6.5, 0.85,'', horizontalalignment='left', fontsize=10)
t_size_fi18 = a.text(6.5, 0.6,'', horizontalalignment='left', fontsize=10)
n_fi18 = a.text(8.5, 0.85,'', horizontalalignment='right', fontsize=10)
nr_fi18 = a.text(8.5, 0.85,'', horizontalalignment='right', weight='bold', fontsize=10, color='red')
#========================================================================================0
t_fi19 = a.text(10.5, 11.3,'', horizontalalignment='center', weight='bold', fontsize=12)
t_d1_max = a.text(9.5, 10.85,'', horizontalalignment='left', fontsize=10)
t_size_fi19= a.text(9.5, 10.6,'', horizontalalignment='left', fontsize=10)
n_fi19 = a.text(11.5, 10.85,'', horizontalalignment='right', fontsize=10)
nr_fi19 = a.text(11.5, 10.85,'', horizontalalignment='right', weight='bold', fontsize=10, color='red')

t_fi20 = a.text(10.5, 9.3,'', horizontalalignment='center', weight='bold', fontsize=12)
t_d2_max = a.text(9.5, 8.85,'', horizontalalignment='left', fontsize=10)
t_size_fi20 = a.text(9.5, 8.6,'', horizontalalignment='left', fontsize=10)
n_fi20 = a.text(11.5, 8.85,'', horizontalalignment='right', fontsize=10)
nr_fi20 = a.text(11.5, 8.85,'', horizontalalignment='right', weight='bold', fontsize=10, color='red')

t_fi21 = a.text(10.5, 7.3,'', horizontalalignment='center', weight='bold', fontsize=12)
t_d3_max = a.text(9.5, 6.85,'', horizontalalignment='left', fontsize=10)
t_size_fi21 = a.text(9.5, 6.6,'', horizontalalignment='left', fontsize=10)
n_fi21 = a.text(11.5, 6.85,'', horizontalalignment='right', fontsize=10)
nr_fi21 = a.text(11.5, 6.85,'', horizontalalignment='right', weight='bold', fontsize=10, color='red')

t_fi22 = a.text(10.5, 5.3,'', horizontalalignment='center', weight='bold', fontsize=12)
t_d4_max = a.text(9.5, 4.85,'', horizontalalignment='left', fontsize=10)
t_size_fi22 = a.text(9.5, 4.6,'', horizontalalignment='left', fontsize=10)
n_fi22 = a.text(11.5, 4.85,'', horizontalalignment='right', fontsize=10)
nr_fi22 = a.text(11.5, 4.85,'', horizontalalignment='right', weight='bold', fontsize=10, color='red')

t_fi23 = a.text(10.5, 3.3,'', horizontalalignment='center', weight='bold', fontsize=12)
t_d5_max = a.text(9.5, 2.85,'', horizontalalignment='left', fontsize=10)
t_size_fi23 = a.text(9.5, 2.6,'', horizontalalignment='left', fontsize=10)
n_fi23 = a.text(11.5, 2.85,'', horizontalalignment='right', fontsize=10)
nr_fi23 = a.text(11.5, 2.85,'', horizontalalignment='right', weight='bold', fontsize=10, color='red')

t_fi24 = a.text(10.5, 1.3,'', horizontalalignment='center', weight='bold', fontsize=12)
t_d6_max = a.text(9.5, 0.85,'', horizontalalignment='left', fontsize=10)
t_size_fi24 = a.text(9.5, 0.6,'', horizontalalignment='left', fontsize=10)
n_fi24 = a.text(11.5, 0.85,'', horizontalalignment='right', fontsize=10)
nr_fi24 = a.text(11.5, 0.85,'', horizontalalignment='right', weight='bold', fontsize=10, color='red')

def init():
    t_fi1.set_text('')
    t_a1_max.set_text('')
    t_size_fi1.set_text('')
    n_fi1.set_text('')
    nr_fi1.set_text('')
    
    t_fi2.set_text('')
    t_a2_max.set_text('')
    t_size_fi2.set_text('')
    n_fi2.set_text('')
    nr_fi2.set_text('')
    
    t_fi3.set_text('')
    t_a3_max.set_text('')
    t_size_fi3.set_text('')
    n_fi3.set_text('')
    nr_fi3.set_text('')
    
    t_fi4.set_text('')
    t_a4_max.set_text('')
    t_size_fi4.set_text('')
    n_fi4.set_text('')
    nr_fi4.set_text('')
    
    t_fi5.set_text('')
    t_a5_max.set_text('')
    t_size_fi5.set_text('')
    n_fi5.set_text('')
    nr_fi5.set_text('')

    t_fi6.set_text('')
    t_a6_max.set_text('')
    t_size_fi6.set_text('')
    n_fi6.set_text('')
    nr_fi6.set_text('')
    #==============================
    t_fi7.set_text('')
    t_b1_max.set_text('')
    t_size_fi7.set_text('')
    n_fi7.set_text('')
    nr_fi7.set_text('')
    
    t_fi8.set_text('')
    t_b2_max.set_text('')
    t_size_fi8.set_text('')
    n_fi8.set_text('')
    nr_fi8.set_text('')
    
    t_fi9.set_text('')
    t_b3_max.set_text('')
    t_size_fi9.set_text('')
    n_fi9.set_text('')
    nr_fi9.set_text('')
    
    t_fi10.set_text('')
    t_b4_max.set_text('')
    t_size_fi10.set_text('')
    n_fi10.set_text('')
    nr_fi10.set_text('')
    
    t_fi11.set_text('')
    t_b5_max.set_text('')
    t_size_fi11.set_text('')
    n_fi11.set_text('')
    nr_fi11.set_text('')

    t_fi12.set_text('')
    t_b6_max.set_text('')
    t_size_fi12.set_text('')
    n_fi12.set_text('')
    nr_fi12.set_text('')
    #==================================
    t_fi13.set_text('')
    t_c1_max.set_text('')
    t_size_fi13.set_text('')
    n_fi13.set_text('')
    nr_fi13.set_text('')
    
    t_fi14.set_text('')
    t_c2_max.set_text('')
    t_size_fi14.set_text('')
    n_fi14.set_text('')
    nr_fi14.set_text('')
    
    t_fi15.set_text('')
    t_c3_max.set_text('')
    t_size_fi15.set_text('')
    n_fi15.set_text('')
    nr_fi15.set_text('')
    
    t_fi16.set_text('')
    t_c4_max.set_text('')
    t_size_fi16.set_text('')
    n_fi16.set_text('')
    nr_fi16.set_text('')
    
    t_fi17.set_text('')
    t_c5_max.set_text('')
    t_size_fi17.set_text('')
    n_fi17.set_text('')
    nr_fi17.set_text('')

    t_fi18.set_text('')
    t_c6_max.set_text('')
    t_size_fi18.set_text('')
    n_fi18.set_text('')
    nr_fi18.set_text('')
    
    #===================================
    t_fi19.set_text('')
    t_d1_max.set_text('')
    t_size_fi19.set_text('')
    n_fi19.set_text('')
    nr_fi19.set_text('')
    
    t_fi20.set_text('')
    t_d2_max.set_text('')
    t_size_fi20.set_text('')
    n_fi20.set_text('')
    nr_fi20.set_text('')
    
    t_fi21.set_text('')
    t_d3_max.set_text('')
    t_size_fi21.set_text('')
    n_fi21.set_text('')
    nr_fi21.set_text('')
    
    t_fi22.set_text('')
    t_d4_max.set_text('')
    t_size_fi22.set_text('')
    n_fi22.set_text('')
    nr_fi22.set_text('')
    
    t_fi23.set_text('')
    t_d5_max.set_text('')
    t_size_fi23.set_text('')
    n_fi23.set_text('')
    nr_fi23.set_text('')

    t_fi24.set_text('')
    t_d6_max.set_text('')
    t_size_fi24.set_text('')
    n_fi24.set_text('')
    nr_fi24.set_text('')

    return (t_fi1, t_a1_max, t_size_fi1, n_fi1, 
            t_fi2, t_a2_max, t_size_fi2, n_fi2, 
            t_fi3, t_a3_max, t_size_fi3, n_fi3, 
            t_fi4, t_a4_max, t_size_fi4, n_fi4,
            t_fi5, t_a5_max, t_size_fi5, n_fi5,
            t_fi6, t_a6_max, t_size_fi6, n_fi6,
            
            t_fi7, t_b1_max, t_size_fi7, n_fi7, 
            t_fi8, t_b2_max, t_size_fi8, n_fi8, 
            t_fi9, t_b3_max, t_size_fi9, n_fi9, 
            t_fi10, t_b4_max, t_size_fi10, n_fi10,
            t_fi11, t_b5_max, t_size_fi11, n_fi11,
            t_fi12, t_b6_max, t_size_fi12, n_fi12,
            
            t_fi13, t_c1_max, t_size_fi13, n_fi13, 
            t_fi14, t_c2_max, t_size_fi14, n_fi14, 
            t_fi15, t_c3_max, t_size_fi15, n_fi15, 
            t_fi16, t_c4_max, t_size_fi16, n_fi16,
            t_fi17, t_c5_max, t_size_fi17, n_fi17,
            t_fi18, t_c6_max, t_size_fi18, n_fi18,
            
            t_fi19, t_d1_max, t_size_fi19, n_fi19, 
            t_fi20, t_d2_max, t_size_fi20, n_fi20, 
            t_fi21, t_d3_max, t_size_fi21, n_fi21, 
            t_fi22, t_d4_max, t_size_fi22, n_fi22,
            t_fi23, t_d5_max, t_size_fi23, n_fi23,
            t_fi24, t_d6_max, t_size_fi24, n_fi24,)

# Nur nach support und laenge sortiern
def update(i):
    
    # Datenset aufstellen
    dataset = pd.read_excel('apriori.xlsx', header=None)
    dataset = dataset.drop(dataset.index[0])
    size = list(dataset.columns)



    trans = []
    for i in range(0, len(dataset)):
        trans.append([str(dataset.values[i,j]) for j in range(0, 20)])

    trans = np.array(trans)
    te = TransactionEncoder()
    data = te.fit_transform(trans)
    data = pd.DataFrame(data, columns = te.columns_)
    del data['nan']

# Nur nach support und laenge sortiern
    frequent_itemsets = apriori(data, min_support = 0.01, use_colnames=True)
    frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
    s_frequent_itemsets = frequent_itemsets.sort_values(by ='support', ascending = False, inplace = True)
    fi = frequent_itemsets[ (frequent_itemsets['length'] == 1)]
    #print(fi.head(24))
    
    fi1 = (fi['itemsets']).head(1).to_string(index=False)
    fi1 = fi1.replace('(', "").lstrip()
    fi1 = fi1.replace(')', "").lstrip()
    size_fi1 = sum(dataset[size].eq(fi1).sum())
    num_fi1 = a1_max - size_fi1
    
    fi2 = fi['itemsets'][1:2].to_string(index=False)
    fi2 = fi2.replace('(', "").lstrip()
    fi2 = fi2.replace(')', "").lstrip()
    size_fi2 = sum(dataset[size].eq(fi2).sum())
    num_fi2 = a2_max - size_fi2
    
    fi3 = fi['itemsets'][2:3].to_string(index=False)
    fi3 = fi3.replace('(', "").lstrip()
    fi3 = fi3.replace(')', "").lstrip()
    size_fi3 = sum(dataset[size].eq(fi3).sum())
    num_fi3 = a3_max - size_fi3

    fi4 = fi['itemsets'][3:4].to_string(index=False)
    fi4 = fi4.replace('(', "").lstrip()
    fi4 = fi4.replace(')', "").lstrip()
    size_fi4 = sum(dataset[size].eq(fi4).sum())
    num_fi4 = a4_max - size_fi4
    
    fi5 = fi['itemsets'][4:5].to_string(index=False)
    fi5 = fi5.replace('(', "").lstrip()
    fi5 = fi5.replace(')', "").lstrip()
    size_fi5 = sum(dataset[size].eq(fi5).sum())
    num_fi5 = a5_max - size_fi5
    
    fi6 = fi['itemsets'][5:6].to_string(index=False)
    fi6 = fi6.replace('(', "").lstrip()
    fi6 = fi6.replace(')', "").lstrip()
    size_fi6 = sum(dataset[size].eq(fi6).sum())
    num_fi6 = a6_max - size_fi6
    #===============================================================
    fi7 = (fi['itemsets'])[6:7].to_string(index=False)
    fi7 = fi7.replace('(', "").lstrip()
    fi7 = fi7.replace(')', "").lstrip()
    size_fi7 = sum(dataset[size].eq(fi7).sum())
    num_fi7 = b1_max - size_fi7
    
    fi8 = fi['itemsets'][7:8].to_string(index=False)
    fi8 = fi8.replace('(', "").lstrip()
    fi8 = fi8.replace(')', "").lstrip()
    size_fi8 = sum(dataset[size].eq(fi8).sum())
    num_fi8 = b2_max - size_fi8
    
    fi9 = fi['itemsets'][8:9].to_string(index=False)
    fi9 = fi9.replace('(', "").lstrip()
    fi9 = fi9.replace(')', "").lstrip()
    size_fi9 = sum(dataset[size].eq(fi9).sum())
    num_fi9 = b3_max - size_fi9

    fi10 = fi['itemsets'][9:10].to_string(index=False)
    fi10 = fi10.replace('(', "").lstrip()
    fi10 = fi10.replace(')', "").lstrip()
    size_fi10 = sum(dataset[size].eq(fi10).sum())
    num_fi10 = b4_max - size_fi10
    
    fi11 = fi['itemsets'][10:11].to_string(index=False)
    fi11 = fi11.replace('(', "").lstrip()
    fi11 = fi11.replace(')', "").lstrip()
    size_fi11 = sum(dataset[size].eq(fi11).sum())
    num_fi11 = b5_max - size_fi11
    
    fi12 = fi['itemsets'][11:12].to_string(index=False)
    fi12 = fi12.replace('(', "").lstrip()
    fi12 = fi12.replace(')', "").lstrip()
    size_fi12 = sum(dataset[size].eq(fi12).sum())
    num_fi12 = b6_max - size_fi12
    #======================================================
    fi13 = (fi['itemsets'])[12:13].to_string(index=False)
    fi13 = fi13.replace('(', "").lstrip()
    fi13 = fi13.replace(')', "").lstrip()
    size_fi13 = sum(dataset[size].eq(fi13).sum())
    num_fi13 = c1_max - size_fi13
    
    fi14 = fi['itemsets'][13:14].to_string(index=False)
    fi14 = fi14.replace('(', "").lstrip()
    fi14 = fi14.replace(')', "").lstrip()
    size_fi14 = sum(dataset[size].eq(fi14).sum())
    num_fi14 = c2_max - size_fi14
    
    fi15 = fi['itemsets'][14:15].to_string(index=False)
    fi15 = fi15.replace('(', "").lstrip()
    fi15 = fi15.replace(')', "").lstrip()
    size_fi15 = sum(dataset[size].eq(fi15).sum())
    num_fi15 = c3_max - size_fi15

    fi16 = fi['itemsets'][15:16].to_string(index=False)
    fi16 = fi16.replace('(', "").lstrip()
    fi16 = fi16.replace(')', "").lstrip()
    size_fi16 = sum(dataset[size].eq(fi16).sum())
    num_fi16 = c4_max - size_fi16
    
    fi17 = fi['itemsets'][16:17].to_string(index=False)
    fi17 = fi17.replace('(', "").lstrip()
    fi17 = fi17.replace(')', "").lstrip()
    size_fi17 = sum(dataset[size].eq(fi17).sum())
    num_fi17 = c5_max - size_fi17
    
    fi18 = fi['itemsets'][17:18].to_string(index=False)
    fi18 = fi18.replace('(', "").lstrip()
    fi18 = fi18.replace(')', "").lstrip()
    size_fi18 = sum(dataset[size].eq(fi18).sum())
    num_fi18 = c6_max - size_fi18
    
    #======================================================
    fi19 = (fi['itemsets'])[18:19].to_string(index=False)
    fi19 = fi19.replace('(', "").lstrip()
    fi19 = fi19.replace(')', "").lstrip()
    size_fi19 = sum(dataset[size].eq(fi19).sum())
    num_fi19 = d1_max - size_fi19
    
    fi20 = fi['itemsets'][19:20].to_string(index=False)
    fi20 = fi20.replace('(', "").lstrip()
    fi20 = fi20.replace(')', "").lstrip()
    size_fi20 = sum(dataset[size].eq(fi14).sum())
    num_fi20 = d2_max - size_fi20
    
    fi21 = fi['itemsets'][20:21].to_string(index=False)
    fi21 = fi21.replace('(', "").lstrip()
    fi21 = fi21.replace(')', "").lstrip()
    size_fi21 = sum(dataset[size].eq(fi21).sum())
    num_fi21 = d3_max - size_fi21

    fi22 = fi['itemsets'][21:22].to_string(index=False)
    fi22 = fi22.replace('(', "").lstrip()
    fi22 = fi22.replace(')', "").lstrip()
    size_fi22 = sum(dataset[size].eq(fi22).sum())
    num_fi22 = d4_max - size_fi22
    
    fi23 = fi['itemsets'][22:23].to_string(index=False)
    fi23 = fi23.replace('(', "").lstrip()
    fi23 = fi23.replace(')', "").lstrip()
    size_fi23 = sum(dataset[size].eq(fi23).sum())
    num_fi23 = d5_max - size_fi23
    
    fi24 = fi['itemsets'][23:24].to_string(index=False)
    fi24 = fi24.replace('(', "").lstrip()
    fi24 = fi24.replace(')', "").lstrip()
    size_fi24 = sum(dataset[size].eq(fi24).sum())
    num_fi24 = d6_max - size_fi24
######################## Vizualisiere Lagerhalle #############################
    #a.set_title("Optmierung der Logistik")
    
# Für A1:
    t_fi1.set_text(fi1)
    t_a1_max.set_text(a1_max)
    t_size_fi1.set_text(size_fi1)
    if num_fi1 > 0:
        n_fi1.set_text(num_fi1)
    elif num_fi1 < 0:
        nr_fi1.set_text(num_fi1)

# # Für A2:
    t_fi2.set_text(fi2)
    t_a2_max.set_text(a2_max)
    t_size_fi2.set_text(size_fi2)
    if num_fi2 > 0:
        n_fi2.set_text(num_fi2)
    elif num_fi2 < 0:
        nr_fi2.set_text(num_fi2)

# # Für A3:
    t_fi3.set_text(fi3)
    t_a3_max.set_text(a3_max)
    t_size_fi3.set_text(size_fi3)
    if num_fi3 > 0:
        n_fi3.set_text(num_fi3)
    elif num_fi3 < 0:
        nr_fi3.set_text(num_fi3)

# # Für A4:
    t_fi4.set_text(fi4)
    t_a4_max.set_text(a4_max)
    t_size_fi4.set_text(size_fi4)
    if num_fi4 > 0:
        n_fi4.set_text(num_fi4)
    elif num_fi4 < 0:
        nr_fi4.set_text(num_fi4)
        
# # Für A5:
    t_fi5.set_text(fi5)
    t_a5_max.set_text(a5_max)
    t_size_fi5.set_text(size_fi5)
    if num_fi5 > 0:
        n_fi5.set_text(num_fi5)
    elif num_fi5 < 0:
        nr_fi5.set_text(num_fi5)
        
# # Für A6:
    t_fi6.set_text(fi6)
    t_a6_max.set_text(a6_max)
    t_size_fi6.set_text(size_fi6)
    if num_fi6 > 0:
        n_fi6.set_text(num_fi6)
    elif num_fi6 < 0:
        nr_fi6.set_text(num_fi6)
        
# Für B1:
    t_fi7.set_text(fi7)
    t_b1_max.set_text(b1_max)
    t_size_fi7.set_text(size_fi7)
    if num_fi7 > 0:
        n_fi7.set_text(num_fi7)
    elif num_fi7 < 0:
        nr_fi7.set_text(num_fi7)

# # Für B2:
    t_fi8.set_text(fi8)
    t_b2_max.set_text(b2_max)
    t_size_fi8.set_text(size_fi8)
    if num_fi8 > 0:
        n_fi8.set_text(num_fi8)
    elif num_fi8 < 0:
        nr_fi8.set_text(num_fi8)

# # Für B3:
    t_fi9.set_text(fi9)
    t_b3_max.set_text(b3_max)
    t_size_fi9.set_text(size_fi9)
    if num_fi9 > 0:
        n_fi9.set_text(num_fi9)
    elif num_fi9 < 0:
        nr_fi9.set_text(num_fi9)

# # Für B4:
    t_fi10.set_text(fi10)
    t_b4_max.set_text(b4_max)
    t_size_fi10.set_text(size_fi10)
    if num_fi10 > 0:
        n_fi10.set_text(num_fi10)
    elif num_fi10 < 0:
        nr_fi10.set_text(num_fi10)
        
# # Für B5:
    t_fi11.set_text(fi11)
    t_b5_max.set_text(b5_max)
    t_size_fi11.set_text(size_fi11)
    if num_fi11 > 0:
        n_fi11.set_text(num_fi11)
    elif num_fi11 < 0:
        nr_fi5.set_text(num_fi11)
        
# # Für B6:
    t_fi12.set_text(fi12)
    t_b6_max.set_text(b6_max)
    t_size_fi12.set_text(size_fi12)
    if num_fi12 > 0:
        n_fi12.set_text(num_fi12)
    elif num_fi12 < 0:
        nr_fi12.set_text(num_fi12)
        
# Für C1:
    t_fi13.set_text(fi13)
    t_c1_max.set_text(c1_max)
    t_size_fi13.set_text(size_fi13)
    if num_fi13 > 0:
        n_fi13.set_text(num_fi13)
    elif num_fi13 < 0:
        nr_fi13.set_text(num_fi13)

# # Für C2:
    t_fi14.set_text(fi14)
    t_c2_max.set_text(c2_max)
    t_size_fi14.set_text(size_fi14)
    if num_fi14 > 0:
        n_fi14.set_text(num_fi14)
    elif num_fi14 < 0:
        nr_fi14.set_text(num_fi14)

# # Für C3:
    t_fi15.set_text(fi15)
    t_c3_max.set_text(c3_max)
    t_size_fi15.set_text(size_fi15)
    if num_fi15 > 0:
        n_fi15.set_text(num_fi15)
    elif num_fi15 < 0:
        nr_fi15.set_text(num_fi15)

# # Für C4:
    t_fi16.set_text(fi16)
    t_c4_max.set_text(c4_max)
    t_size_fi16.set_text(size_fi16)
    if num_fi16 > 0:
        n_fi16.set_text(num_fi16)
    elif num_fi16 < 0:
        nr_fi16.set_text(num_fi16)
        
# # Für C5:
    t_fi17.set_text(fi17)
    t_c5_max.set_text(c5_max)
    t_size_fi17.set_text(size_fi17)
    if num_fi17 > 0:
        n_fi17.set_text(num_fi17)
    elif num_fi17 < 0:
        nr_fi17.set_text(num_fi17)
        
# # Für C6:
    t_fi18.set_text(fi18)
    t_c6_max.set_text(c6_max)
    t_size_fi18.set_text(size_fi18)
    if num_fi18 > 0:
        n_fi18.set_text(num_fi18)
    elif num_fi18 < 0:
        nr_fi18.set_text(num_fi18)
        
# Für D1:
    t_fi19.set_text(fi19)
    t_d1_max.set_text(d1_max)
    t_size_fi19.set_text(size_fi19)
    if num_fi19 > 0:
        n_fi19.set_text(num_fi19)
    elif num_fi19 < 0:
        nr_fi19.set_text(num_fi19)

# # Für D2:
    t_fi20.set_text(fi20)
    t_d2_max.set_text(d2_max)
    t_size_fi20.set_text(size_fi20)
    if num_fi20 > 0:
        n_fi20.set_text(num_fi20)
    elif num_fi20 < 0:
        nr_fi20.set_text(num_fi20)

# # Für D3:
    t_fi21.set_text(fi21)
    t_d3_max.set_text(d3_max)
    t_size_fi21.set_text(size_fi21)
    if num_fi21 > 0:
        n_fi21.set_text(num_fi21)
    elif num_fi21 < 0:
        nr_fi21.set_text(num_fi21)

# # Für D4:
    t_fi22.set_text(fi22)
    t_d4_max.set_text(d4_max)
    t_size_fi22.set_text(size_fi22)
    if num_fi22 > 0:
        n_fi22.set_text(num_fi22)
    elif num_fi22 < 0:
        nr_fi22.set_text(num_fi22)
        
# # Für D5:
    t_fi23.set_text(fi23)
    t_d5_max.set_text(d5_max)
    t_size_fi23.set_text(size_fi23)
    if num_fi23 > 0:
        n_fi23.set_text(num_fi23)
    elif num_fi23 < 0:
        nr_fi23.set_text(num_fi23)
        
# # Für D6:
    t_fi24.set_text(fi24)
    t_d6_max.set_text(d6_max)
    t_size_fi24.set_text(size_fi24)
    if num_fi24 > 0:
        n_fi24.set_text(num_fi24)
    elif num_fi24 < 0:
        nr_fi24.set_text(num_fi24)
        

    return (t_fi1, t_a1_max, t_size_fi1, n_fi1, 
            t_fi2, t_a2_max, t_size_fi2, n_fi2,
            t_fi3, t_a3_max, t_size_fi3, n_fi3, 
            t_fi4, t_a4_max, t_size_fi4, n_fi4,
            t_fi5, t_a5_max, t_size_fi5, n_fi5,
            t_fi6, t_a6_max, t_size_fi6, n_fi6,
            
            t_fi7, t_b1_max, t_size_fi7, n_fi7, 
            t_fi8, t_b2_max, t_size_fi8, n_fi8, 
            t_fi9, t_b3_max, t_size_fi9, n_fi9, 
            t_fi10, t_b4_max, t_size_fi10, n_fi10,
            t_fi11, t_b5_max, t_size_fi11, n_fi11,
            t_fi12, t_b6_max, t_size_fi12, n_fi12,

            t_fi13, t_c1_max, t_size_fi13, n_fi13, 
            t_fi14, t_c2_max, t_size_fi14, n_fi14, 
            t_fi15, t_c3_max, t_size_fi15, n_fi15, 
            t_fi16, t_c4_max, t_size_fi16, n_fi16,
            t_fi17, t_c5_max, t_size_fi17, n_fi17,
            t_fi18, t_c6_max, t_size_fi18, n_fi18,
            
            t_fi19, t_d1_max, t_size_fi19, n_fi19, 
            t_fi20, t_d2_max, t_size_fi20, n_fi20, 
            t_fi21, t_d3_max, t_size_fi21, n_fi21, 
            t_fi22, t_d4_max, t_size_fi22, n_fi22,
            t_fi23, t_d5_max, t_size_fi23, n_fi23,
            t_fi24, t_d6_max, t_size_fi24, n_fi24,)


# ###########################Apriori#############################################

def get_selection():
    loc = [coords_loc[int(item)] for item in choose_loc.curselection()]
    lst = []
    lst1 = []
# TSP über loc laufen lassen:
    fitness_coords = mlrose.TravellingSales(coords = loc)
    problem_fit = mlrose.TSPOpt(length = len(loc), fitness_fn = fitness_coords, maximize=False)
    best_state, best_fitness = mlrose.genetic_alg(problem_fit, mutation_prob = 0.2, 
                                                  max_attempts = 100, random_state = 2)
    a_list = [loc[i] for i in best_state]

# TSP soll immer bei Start (0,0) beginnen und enden
    start_coord = (0, 0)
    startIndex = a_list.index( start_coord )
    rotated_list = a_list[ startIndex: ] + a_list [ :startIndex ]
    
    for start, goal in zip(rotated_list[::1], rotated_list[1::1]):
        route = astar(grid, start, goal)
        route = route + [start]
        route = route[::-1]

        x_coords = []
        y_coords = []
        
        for i in (range(0,len(route))):
            x = route[i][0]
            y = route[i][1]
            x_coords.append(x)
            y_coords.append(y)
            
        a.plot(y_coords,x_coords, color = "black")
        
        for x, y in zip(route[0:len(route):1], route[1:len(route):1]):
            dist = (sqrt( (x[0] - y[0])**2 + (x[1] - y[1])**2 ))
            lst.append(dist)
            d1 = np.sum(lst)
    start = start_coord
    goal = goal[-2::]

    route1 = astar(grid, start, goal)
    route1 = route1 + [start]
    route1 = route1[::-1]

    x_coords = []
    y_coords = []
    
    for i in (range(0,len(route1))):
        x = route1[i][0]
        y = route1[i][1]
        x_coords.append(x)
        y_coords.append(y)
        
        a.plot(y_coords,x_coords, color = "black")
    
    for x1, y1 in zip(route1[0::1], route1[1::1]):
        dist1 = (sqrt( (x1[0] - y1[0])**2 + (x1[1] - y1[1])**2 ))
        lst1.append(dist1)
        d2 =np.sum(lst1)
        
    if len(loc) > 2:
        tot = d1+d2
    elif len(loc) == 2:
        tot = d1
    tot_dis = round(tot, 2)
    
    x, y = zip(*rotated_list)
    a.scatter(y, x, s=200, color='red')
    
    for i, txt in enumerate(list(range(len(loc)))):
        a.annotate(txt, (y[i], x[i]), fontsize = 16)
        
    box = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    a.text(0.9, 1.05, tot_dis, transform=a.transAxes, fontsize=14,
           verticalalignment='top', bbox=box)
    a.text(0.82, 1.05, 'Distance:', transform=a.transAxes, fontsize=14,
           verticalalignment='top', bbox=box)


# A* zwischen Punkte aus TSP:
def heuristic(a, b):
    return np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)


def astar(array, start, goal):
    neighbors = [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]
    close_set = set()
    came_from = {}
    gscore = {start:0}
    fscore = {start:heuristic(start, goal)}
    oheap = []
    heapq.heappush(oheap, (fscore[start], start))

    while oheap:
        current = heapq.heappop(oheap)[1]
        if current == goal:
            data = []
            while current in came_from:
                data.append(current)
                current = came_from[current]
            return data

        close_set.add(current)

        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j
            tentative_g_score = gscore[current] + heuristic(current, neighbor)
            if 0 <= neighbor[0] < array.shape[0]:
                if 0 <= neighbor[1] < array.shape[1]:                
                    if array[neighbor[0]][neighbor[1]] == 1:
                        continue
            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                continue
            if  tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1]for i in oheap]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(oheap, (fscore[neighbor], neighbor))
    return False

from tkinter.ttk import Progressbar
def plot():
    if window.counter <= 1:
        if not window.counter:
            canvas = FigureCanvasTkAgg(fig, master = window)
            canvas.get_tk_widget().pack()
            ani = animation.FuncAnimation(fig, update, frames=200, interval = 20, 
                                          init_func = init, blit = False, repeat=False)
            canvas.draw()
            plt.show(ani)
            plt.show()
            plot_button['text'] = 'Exit'
        else:
            window.destroy()
    window.counter += 1
    
from tkinter import *
from tkinter import ttk

window = Tk()
window.configure(background='grey')
window.title('Virtual warehouse')
window.counter = 0



w, h = window.winfo_screenwidth(), window.winfo_screenheight()
window.geometry("%dx%d+0+0" % (w, h))

nb = ttk.Notebook(window, width=200, height=580)

page1 = ttk.Frame(nb)
nb.add(page1, text='Shortest Path'
                       '\n'
                       'Start with (0,0)!')

choose_loc = Listbox(page1)
choose_loc.configure(selectmode=MULTIPLE, width=9, height=20)
choose_loc.pack()

btnGet = Button(page1,text="Get Selection",command=get_selection)
btnGet.pack(side=TOP)

a0 = (0,0)
a1 = (0,2)
a2 = (0,4)
a3 = (0,8)
a4 = (0,10)

b0 = (2,0)
b1 = (2,2)
b2 = (2,4)
b3 = (2,8)
b4 = (2,10)

c0 = (4,0)
c1 = (4,2)
c2 = (4,4)
c3 = (4,8)
c4 = (4,10)

d0 = (6,0)
d1 = (6,2)
d2 = (6,4)
d3 = (6,8)
d4 = (6,10)

e0 = (8,0)
e1 = (8,2)
e2 = (8,4)
e3 = (8,8)
e4 = (8,10)

f0 = (10,0)
f1 = (10,2)
f2 = (10,4)
f3 = (10,8)
f4 = (10,10)

g0 = (12,0)
g1 = (12,2)
g2 = (12,4)
g3 = (12,8)
g4 = (12,10)

coords_loc = [a0, a1, a2, a3, a4,
              b0, b1, b2, b3, b4,
              c0, c1, c2, c3, c4,
              d0, d1, d2, d3, d4,
              e0, e1, e2, e3, e4,
              f0, f1, f2, f3, f4,
              g0, g1, g2, g3, g4]

for lang in coords_loc:
    choose_loc.insert(END, lang)
  
# button that displays the plot 
plot_button = Button(master = window,  
                     command = plot, 
                     height = 2,  
                     width = 10, 
                     text = "START")

# def restart_program():
#     for widget in window.winfo_children():
#         widget.destroy()
    
#     plot_button = Button(master = window,  
#                          command = plot, 
#                          height = 2,  
#                          width = 10, 
#                          text = "START")
#     plot_button.pack()




# reset_button = Button(master = window,  
#                      command = restart_program, 
#                      height = 2,  
#                      width = 10, 
#                      text = "Restart")

from tkinter import messagebox
messagebox.showinfo("showinfo", 'Information:' 
                    '\n'
                    '1. Choose storage points.'
                    '\n'
                    "2. Press Get Selection."
                    '\n'
                    '3. Press Start!')


nb.pack(side=LEFT, anchor=NW)
plot_button.pack()
# reset_button.pack(side=TOP)

window.mainloop()