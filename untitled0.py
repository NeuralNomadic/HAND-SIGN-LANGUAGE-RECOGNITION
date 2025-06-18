# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 22:44:14 2024

@author: pande
"""

n = int(input("NO. OF MONSTERS : "))
e = int(input("INTIAL EXPERIENCE : "))
mon = []
bon = []

c = 0

for i in range(n):
    pi = int(input("MON :"))
    mon.append(pi)
    b = int(input("BON :"))
    bon.append(b)
#print(mon,bon)
a = mon.sort()
           
if a != None:
    for i in a:
        if i <= e:
            x = mon.index(i)
            e += bon[x]
            c += 1
else:
    for i in mon:
        if i <= e:
            x = mon.index(i)
            e += bon[x]
            c += 1
    
print(c)
        
    
    