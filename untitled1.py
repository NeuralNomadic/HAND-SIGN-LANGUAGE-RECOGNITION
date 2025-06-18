# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 08:49:18 2024

@author: pande
"""

n = int(input("len : "))
num = []
for i in range(n):
    a = int(input(" : "))
    num.append(a)
for i in range(n-1):
    if (num[i]&num[i+1])*2 < (num[i] | num[])