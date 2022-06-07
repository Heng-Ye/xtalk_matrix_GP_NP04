import pandas as pd
import xlrd as xl 
from pandas import ExcelWriter
from pandas import ExcelFile

import math
import numpy as np
import matplotlib.pyplot as plt
#from scipy.fftpack import fft,ifft
#import seaborn
#from scipy import fftpack
#from scipy import signal
import scipy.fftpack as sg
import scipy.signal as signal

#scipy.fftpack

import operator
from numpy import diff

import copy

import sys
file_name = sys.argv[1]
figName = sys.argv[2]
outName = sys.argv[3]

#Time bin
dt_s=1./(20.*1000.)
dt_us=dt_s*1000000. 

# to read just one sheet to dataframe:
#file_name='2018_08_03_00H45mn31.162996s_NP04_DSS_FASTACQ.xlsx'
data = pd.read_excel(file_name, sheet_name="Acquired Channels Group")

#output file name
#figName = "pulser_3.pdf"

print("Column headings:")
print(data.columns)

n_row=data.shape[0] #gives number of row count
n_col=data.shape[1] #gives number of col count

print("number of rows: ", n_row)
print("number of columns: ", n_col)
#ncol = data.ncols

UP_DS_BR = data['UP-DS-BR']
UP_DS_BL = data['UP-DS-BL']
UP_MS_BR = data['UP-MS-BR']
UP_MS_BL = data['UP-MS-BL']
UP_US_BR = data['UP-US-BR']
UP_US_BL = data['UP-US-BL']

DN_DS_BR = data['DN-DS-BR']
DN_DS_BL = data['DN-DS-BL']
DN_MS_BR = data['DN-MS-BR']
DN_MS_BL = data['DN-MS-BL']
DN_US_BR = data['DN-US-BR']
DN_US_BL = data['DN-US-BL']

TOR_1 = data['GND012']
BP = data['GND013']
TOR_2 = data['GND014']
LM = data['LAr_Lvl']

'''
up_ds_br=copy.deepcopy(UP_DS_BR)
up_ds_bl=copy.deepcopy(UP_DS_BL)
up_ms_br=copy.deepcopy(UP_MS_BR)
up_ms_bl=copy.deepcopy(UP_MS_BL)
up_us_br=copy.deepcopy(UP_US_BR)
up_us_bl=copy.deepcopy(UP_US_BL)

dn_ds_br=copy.deepcopy(DN_DS_BR)
dn_ds_bl=copy.deepcopy(DN_DS_BL)
dn_ms_br=copy.deepcopy(DN_MS_BR)
dn_ms_bl=copy.deepcopy(DN_MS_BL)
dn_us_br=copy.deepcopy(DN_US_BR)
dn_us_bl=copy.deepcopy(DN_US_BL)

tor_1=copy.deepcopy(TOR_1)
bp=copy.deepcopy(BP)
tor_2=copy.deepcopy(TOR_2)
lm=copy.deepcopy(LM)
'''

dUP_DS_BR = diff(UP_DS_BR)/dt_us
dUP_DS_BL = diff(UP_DS_BL)/dt_us
dUP_MS_BR = diff(UP_MS_BR)/dt_us
dUP_MS_BL = diff(UP_MS_BL)/dt_us
dUP_US_BR = diff(UP_US_BR)/dt_us
dUP_US_BL = diff(UP_US_BL)/dt_us

dDN_DS_BR = diff(DN_DS_BR)/dt_us
dDN_DS_BL = diff(DN_DS_BL)/dt_us
dDN_MS_BR = diff(DN_MS_BR)/dt_us
dDN_MS_BL = diff(DN_MS_BL)/dt_us
dDN_US_BR = diff(DN_US_BR)/dt_us
#dDN_US_BL = diff(dn_us_bl)/dt_us
dDN_US_BL = diff(DN_US_BL)/dt_us

dTOR_1 = diff(TOR_1)/dt_us
dBP = diff(BP)/dt_us
dTOR_2 = diff(TOR_2)/dt_us
dLM = diff(LM)/dt_us

print(UP_DS_BR[0])
#print(UP_DS_BR[199999])
print(UP_DS_BR[n_row-1])
#print(UP_DS_BR[n_row])
#print(UP_DS_BR[n_row+1])
#print(UP_DS_BR[data.max_column])

#print("Max. of UP_DS_BL:",max(UP_DS_BL[1:]))
#Object[] amax = new Object[12]
n_ch = 16
#amax = [None] * n_ch  
#amax[0]=max(UP_DS_BR[1:])
#amax[1]=max(UP_DS_BL[1:])
#amax[2]=max(UP_MS_BR[1:])
#amax[3]=max(UP_MS_BL[1:])
#amax[4]=max(UP_US_BR[1:])
#amax[5]=max(UP_US_BL[1:])

#amax[6]=max(DN_DS_BR[1:])
#amax[7]=max(DN_DS_BL[1:])
#amax[8]=max(DN_MS_BR[1:])
#amax[9]=max(DN_MS_BL[1:])
#amax[10]=max(DN_US_BR[1:])
#amax[11]=max(DN_US_BL[1:])

#get maximum amp and its key
tmp_max=1.9
for i in range(len(UP_DS_BR)):
#for i in range(0,25552):
#for i in range(99116,len(UP_DS_BR)):
#for i in range(62826,len(UP_DS_BR)):
    amax = [None] * n_ch  
    amax[0]=UP_DS_BR[i]
    amax[1]=UP_DS_BL[i]
    amax[2]=UP_MS_BR[i]
    amax[3]=UP_MS_BL[i]
    amax[4]=UP_US_BR[i]
    amax[5]=UP_US_BL[i]

    amax[6]=DN_DS_BR[i]
    amax[7]=DN_DS_BL[i]
    amax[8]=DN_MS_BR[i]
    amax[9]=DN_MS_BL[i]
    amax[10]=DN_US_BR[i]
    amax[11]=DN_US_BL[i]

    amax[12]=TOR_1[i]
    amax[13]=BP[i]
    amax[14]=TOR_2[i]
    amax[15]=LM[i]

    #tmp_amax_max=max(amax)
    ch_amax_max, tmp_amax_max = max(enumerate(amax), key=operator.itemgetter(1))

    if tmp_amax_max >= tmp_max:
        amax_max =  tmp_amax_max
        tmp_tmaxIndex = i
        break

print("index_tMax.:", tmp_tmaxIndex)
print("tmp_tMax.:", tmp_tmaxIndex*dt_us, " us")
print("pulser signal on ch:", ch_amax_max+1)
print("Max. amp.:", amax_max, " volt")

#y-axis limit
ymin = [None] * n_ch  
ymax = [None] * n_ch  

for i in range(0,n_ch):
    if (i==ch_amax_max):
        ymin[i]=-0.2
        ymax[i]=3.4
    else: 
        #ymin[i]=-0.005
        #ymax[i]=0.08
        ymin[i]=-3.4
        #ymax[i]=0.08
        ymax[i]=3.4

#if key==0, pick up amax on UP_DS_BR
if (tmp_tmaxIndex==0):
      tmp_tmaxIndex, tmp_amax_max = max(enumerate(UP_DS_BR), key=operator.itemgetter(1))
      #tmp_tmaxIndex, tmp_amax_max = max(enumerate(UP_DS_BR[0:2552]), key=operator.itemgetter(1))
      #tmp_tmaxIndex, tmp_amax_max = max(enumerate(UP_DS_BR[99116:len(UP_DS_BR)]), key=operator.itemgetter(1))
      #tmp_tmaxIndex, tmp_amax_max = max(enumerate(UP_DS_BR[62826:len(UP_DS_BR)]), key=operator.itemgetter(1))
      amax = [None] * n_ch  
      amax[0]=UP_DS_BR[tmp_tmaxIndex]
      amax[1]=UP_DS_BL[tmp_tmaxIndex]
      amax[2]=UP_MS_BR[tmp_tmaxIndex]
      amax[3]=UP_MS_BL[tmp_tmaxIndex]
      amax[4]=UP_US_BR[tmp_tmaxIndex]
      amax[5]=UP_US_BL[tmp_tmaxIndex]

      amax[6]=DN_DS_BR[tmp_tmaxIndex]
      amax[7]=DN_DS_BL[tmp_tmaxIndex]
      amax[8]=DN_MS_BR[tmp_tmaxIndex]
      amax[9]=DN_MS_BL[tmp_tmaxIndex]
      amax[10]=DN_US_BR[tmp_tmaxIndex]
      amax[11]=DN_US_BL[tmp_tmaxIndex]

      amax[12]=TOR_1[tmp_tmaxIndex]
      amax[13]=BP[tmp_tmaxIndex]
      amax[14]=TOR_2[tmp_tmaxIndex]
      amax[15]=LM[tmp_tmaxIndex]

      print("index_tMax.:", tmp_tmaxIndex)
      print("tmp_tMax.:", tmp_tmaxIndex*dt_us, " us")

#ch1
time = [n_row]
time[0] = 0

t0 = tmp_tmaxIndex*dt_us
t0 = t0-700

for i in range(n_row-1):
    time.append(0+i*dt_us-t0)
    #print("[",i,"]:", 0+i*dt_us, "(us)")

#set range in x & y
xmin = 0
xmax = 1400 

#calculate pedestal
av = [None] * n_ch
n_ped=100
#if (tmp_tmaxIndex!=0):
av[0]=sum(UP_DS_BR[tmp_tmaxIndex-n_ped:tmp_tmaxIndex]) / float(len(UP_DS_BR[tmp_tmaxIndex-n_ped:tmp_tmaxIndex]))
av[1]=sum(UP_DS_BL[tmp_tmaxIndex-n_ped:tmp_tmaxIndex]) / float(len(UP_DS_BL[tmp_tmaxIndex-n_ped:tmp_tmaxIndex]))
av[2]=sum(UP_MS_BR[tmp_tmaxIndex-n_ped:tmp_tmaxIndex]) / float(len(UP_MS_BR[tmp_tmaxIndex-n_ped:tmp_tmaxIndex]))
av[3]=sum(UP_MS_BL[tmp_tmaxIndex-n_ped:tmp_tmaxIndex]) / float(len(UP_MS_BL[tmp_tmaxIndex-n_ped:tmp_tmaxIndex]))
av[4]=sum(UP_US_BR[tmp_tmaxIndex-n_ped:tmp_tmaxIndex]) / float(len(UP_US_BR[tmp_tmaxIndex-n_ped:tmp_tmaxIndex]))
av[5]=sum(UP_US_BL[tmp_tmaxIndex-n_ped:tmp_tmaxIndex]) / float(len(UP_US_BL[tmp_tmaxIndex-n_ped:tmp_tmaxIndex]))

av[6]=sum(DN_DS_BR[tmp_tmaxIndex-n_ped:tmp_tmaxIndex]) / float(len(DN_DS_BR[tmp_tmaxIndex-n_ped:tmp_tmaxIndex]))
av[7]=sum(DN_DS_BL[tmp_tmaxIndex-n_ped:tmp_tmaxIndex]) / float(len(DN_DS_BL[tmp_tmaxIndex-n_ped:tmp_tmaxIndex]))
av[8]=sum(DN_MS_BR[tmp_tmaxIndex-n_ped:tmp_tmaxIndex]) / float(len(DN_MS_BR[tmp_tmaxIndex-n_ped:tmp_tmaxIndex]))
av[9]=sum(DN_MS_BL[tmp_tmaxIndex-n_ped:tmp_tmaxIndex]) / float(len(DN_MS_BL[tmp_tmaxIndex-n_ped:tmp_tmaxIndex]))
av[10]=sum(DN_US_BR[tmp_tmaxIndex-n_ped:tmp_tmaxIndex]) / float(len(DN_US_BR[tmp_tmaxIndex-n_ped:tmp_tmaxIndex]))
av[11]=sum(DN_US_BL[tmp_tmaxIndex-n_ped:tmp_tmaxIndex]) / float(len(DN_US_BL[tmp_tmaxIndex-n_ped:tmp_tmaxIndex]))

av[12]=sum(TOR_1[tmp_tmaxIndex-n_ped:tmp_tmaxIndex]) / float(len(TOR_1[tmp_tmaxIndex-n_ped:tmp_tmaxIndex]))
av[13]=sum(BP[tmp_tmaxIndex-n_ped:tmp_tmaxIndex]) / float(len(BP[tmp_tmaxIndex-n_ped:tmp_tmaxIndex]))
av[14]=sum(TOR_2[tmp_tmaxIndex-n_ped:tmp_tmaxIndex]) / float(len(TOR_2[tmp_tmaxIndex-n_ped:tmp_tmaxIndex]))
av[15]=sum(LM[tmp_tmaxIndex-n_ped:tmp_tmaxIndex]) / float(len(LM[tmp_tmaxIndex-n_ped:tmp_tmaxIndex]))

#if (tmp_tmaxIndex==0):
#  av[0]=sum(UP_DS_BR[len(UP_DS_BR)-n_ped:len(UP_DS_BR)]) / float(len(UP_DS_BR[len(UP_DS_BR)-n_ped:len(UP_DS_BR)]))
#  av[1]=sum(UP_DS_BL[len(UP_DS_BL)-n_ped:len(UP_DS_BL)]) / float(len(UP_DS_BL[len(UP_DS_BR)-n_ped:len(UP_DS_BL)]))
#  av[2]=sum(UP_MS_BR[len(UP_MS_BR)-n_ped:len(UP_MS_BR)]) / float(len(UP_MS_BR[len(UP_MS_BR)-n_ped:len(UP_MS_BR)]))
#  av[3]=sum(UP_MS_BL[len(UP_MS_BL)-n_ped:len(UP_MS_BL)]) / float(len(UP_MS_BL[len(UP_MS_BL)-n_ped:len(UP_MS_BL)]))
#  av[4]=sum(UP_US_BR[len(UP_US_BR)-n_ped:len(UP_US_BR)]) / float(len(UP_US_BR[len(UP_MS_BR)-n_ped:len(UP_US_BR)]))
#av[5]=sum(UP_US_BL[len(UP_US_BL)-n_ped:len(UP_US_BL)]) / float(len(UP_US_BL[len(UP_MS_BL)-n_ped:len(UP_US_BL)]))

#  av[6]=sum(DN_DS_BR[len(DN_DS_BR)-n_ped:len(DN_DS_BR)]) / float(len(DN_DS_BR[len(DN_DS_BR)-n_ped:len(DN_DS_BR)]))
#av[7]=sum(DN_DS_BL[len(DN_DS_BL)-n_ped:len(DN_DS_BL)]) / float(len(DN_DS_BL[len(DN_DS_BR)-n_ped:len(DN_DS_BL)]))
#  av[8]=sum(DN_MS_BR[len(DN_MS_BR)-n_ped:len(DN_MS_BR)]) / float(len(DN_MS_BR[len(DN_MS_BR)-n_ped:len(DN_MS_BR)]))
#  av[9]=sum(DN_MS_BL[len(DN_MS_BL)-n_ped:len(DN_MS_BL)]) / float(len(DN_MS_BL[len(DN_MS_BL)-n_ped:len(DN_MS_BL)]))
#  av[10]=sum(DN_US_BR[len(DN_US_BR)-n_ped:len(DN_US_BR)]) / float(len(DN_US_BR[len(DN_MS_BR)-n_ped:len(DN_US_BR)]))
#  av[11]=sum(DN_US_BL[len(DN_US_BL)-n_ped:len(DN_US_BL)]) / float(len(DN_US_BL[len(DN_MS_BL)-n_ped:len(DN_US_BL)]))

ped_amax_max=av[ch_amax_max]

ch_name = ["UP_DS_BR", "UP_DS_BL", "UP_MS_BR", "UP_MS_BL", "UP_US_BR", "UP_US_BL", "DN_DS_BR", "DN_DS_BL", "DN_MS_BR", "DN_MS_BL", "DN_US_BR", "DN_US_BL", "TOR1", "BP", "TOR2", "LEVEL_METER"]

#plt.figure(1)
plt.figure(figsize=(20,15))
plt.xlabel('time [us]')
plt.ylabel('voltage [volt]')

#plt.grid(True)
#plt.legend()
plt.rcParams['axes.grid'] = True
#plt.subplots(3,4,sharex=True,sharey=True)

#subplot(nrows, ncols, plot_number)
x_txt=0.8
y_txt=0.85
plt.subplot(4,4,1)
plt.plot(time,UP_DS_BR, label='Ch1 UP-DS-BR', color='black', marker='o')
plt.legend(loc="upper left")
plt.ylabel("Amplitude [V]")
plt.xlim([xmin,xmax])
plt.ylim([ymin[0],ymax[0]])
print("av of UP-DS-BR:",av[0])
print("Max. of UP-DS-BR:",(amax[0]-av[0]), " frac:", 100.*(amax[0]-av[0])/(amax_max-ped_amax_max))
plt.annotate('{:.2} {}'.format(100.*amax[0]/amax_max,'%'), xy=(x_txt, y_txt), xycoords='axes fraction')

plt.subplot(4,4,2)
plt.plot(time,UP_DS_BL, label='Ch2 UP-DS-BL', color='red', marker='o')
plt.legend(loc="upper left")
plt.xlim([xmin,xmax])
plt.ylim([ymin[1],ymax[1]])
print("av of UP-DS-BL:",av[1])
print("Max. of UP-DS-BL:",(amax[1]-av[1]), " frac:", 100.*(amax[1]-av[1])/(amax_max-ped_amax_max))
plt.annotate('{:.2} {}'.format(100.*amax[1]/amax_max,'%'), xy=(x_txt, y_txt), xycoords='axes fraction')

plt.subplot(4,4,3)
plt.plot(time,UP_MS_BR, label='Ch3 UP-MS-BR', color='orange', marker='o')
plt.legend(loc="upper left")
plt.xlim([xmin,xmax])
plt.ylim([ymin[2],ymax[2]])
print("av of UP-MS-BR:",av[2])
print("Max. of UP-MS-BR:",(amax[2]-av[2]), " frac:", 100.*(amax[2]-av[2])/(amax_max-ped_amax_max))
plt.annotate('{:.2} {}'.format(100.*amax[2]/amax_max,'%'), xy=(x_txt, y_txt), xycoords='axes fraction')

plt.subplot(4,4,4)
plt.plot(time,UP_MS_BL, label='Ch4 UP-MS-BL', color='seagreen', marker='o')
plt.legend(loc="upper left")
plt.xlim([xmin,xmax])
plt.ylim([ymin[3],ymax[3]])
print("av of UP-MS-BL:",av[3])
print("Max. of UP-MS-BL:",(amax[3]-av[3]), " frac:", 100.*(amax[3]-av[3])/(amax_max-ped_amax_max))
plt.annotate('{:.2} {}'.format(100.*amax[3]/amax_max,'%'), xy=(x_txt, y_txt), xycoords='axes fraction')

plt.subplot(4,4,5)
plt.plot(time,UP_US_BR, label='Ch5 UP-US-BR', color='blue', marker='o')
plt.legend(loc="upper left")
plt.ylabel("Amplitude [V]")
plt.xlim([xmin,xmax])
plt.ylim([ymin[4],ymax[4]])
print("av of UP-US-BR:",av[4])
print("Max. of UP-US-BR:",(amax[4]-av[4]), " frac:", 100.*(amax[4]-av[4])/(amax_max-ped_amax_max))
plt.annotate('{:.2} {}'.format(100.*amax[4]/amax_max,'%'), xy=(x_txt, y_txt), xycoords='axes fraction')

plt.subplot(4,4,6)
plt.plot(time,UP_US_BL, label='Ch6 UP-US-BL', color='blueviolet', marker='o')
plt.legend(loc="upper left")
plt.xlim([xmin,xmax])
plt.ylim([ymin[5],ymax[5]])
print("av of UP-US-BL:",av[5])
print("Max. of UP-US-BL:",(amax[5]-av[5]), " frac:", 100.*(amax[5]-av[5])/(amax_max-ped_amax_max))
plt.annotate('{:.2} {}'.format(100.*amax[5]/amax_max,'%'), xy=(x_txt, y_txt), xycoords='axes fraction')

plt.subplot(4,4,7)
plt.plot(time,DN_DS_BR, label='Ch7 DN-DS-BR', color='gray', marker='o')
plt.legend(loc="upper left")
plt.xlim([xmin,xmax])
plt.ylim([ymin[6],ymax[6]])
print("av of DN-DS-BR:",av[6])
print("Max. of DN-DS-BR:",(amax[6]-av[6]), " frac:", 100.*(amax[6]-av[6])/(amax_max-ped_amax_max))
plt.annotate('{:.2} {}'.format(100.*amax[6]/amax_max,'%'), xy=(x_txt, y_txt), xycoords='axes fraction')

plt.subplot(4,4,8)
plt.plot(time,DN_DS_BL, label='Ch8 DN-DS-BL', color='deepskyblue', marker='o')
plt.legend(loc="upper left")
plt.xlim([xmin,xmax])
plt.ylim([ymin[7],ymax[7]])
print("av of DN-DS-BL:",av[7])
print("Max. of DN-DS-BL:",(amax[7]-av[7]), " frac:", 100.*(amax[7]-av[7])/(amax_max-ped_amax_max))
plt.annotate('{:.2} {}'.format(100.*amax[7]/amax_max,'%'), xy=(x_txt, y_txt), xycoords='axes fraction')

plt.subplot(4,4,9)
plt.plot(time,DN_MS_BR, label='Ch9 DN-MS-BR', color='fuchsia', marker='o')
plt.legend(loc="upper left")
plt.xlim([xmin,xmax])
plt.ylim([ymin[8],ymax[8]])
plt.ylabel("Amplitude [V]")
plt.xlabel("Time [us]")
print("av of DN-MS-BR:",av[8])
print("Max. of DN-MS-BR:",(amax[8]-av[8]), " frac:", 100.*(amax[8]-av[8])/(amax_max-ped_amax_max))
plt.annotate('{:.2} {}'.format(100.*amax[8]/amax_max,'%'), xy=(x_txt, y_txt), xycoords='axes fraction')

plt.subplot(4,4,10)
plt.plot(time,DN_MS_BL, label='Ch10 DN-MS-BL', color='turquoise', marker='o')
plt.legend(loc="upper left")
plt.xlim([xmin,xmax])
plt.ylim([ymin[9],ymax[9]])
plt.xlabel("Time [us]")
print("av of DN-MS-BL:",av[9])
print("Max. of DN-MS-BL:",(amax[9]-av[9]), " frac:", 100.*(amax[9]-av[9])/(amax_max-ped_amax_max))
plt.annotate('{:.2} {}'.format(100.*amax[9]/amax_max,'%'), xy=(x_txt, y_txt), xycoords='axes fraction')

plt.subplot(4,4,11)
plt.plot(time,DN_US_BR, label='Ch11 DN-US-BR', color='sienna', marker='o')
plt.legend(loc="upper left")
plt.xlim([xmin,xmax])
plt.ylim([ymin[10],ymax[10]])
plt.xlabel("Time [us]")
print("av of DN-US-BR:",av[10])
print("Max. of DN-US-BR:",(amax[10]-av[10]), " frac:", 100.*(amax[10]-av[10])/(amax_max-ped_amax_max))
plt.annotate('{:.2} {}'.format(100.*amax[10]/amax_max,'%'), xy=(x_txt, y_txt), xycoords='axes fraction')

plt.subplot(4,4,12)
#plt.plot(time[:-1], DN_US_BL, label='Ch12 DN-US-BL', color='greenyellow', marker='o')
plt.plot(time, DN_US_BL, label='Ch12 DN-US-BL', color='greenyellow', marker='o')
plt.legend(loc="upper left")
plt.xlim([xmin,xmax])
#plt.ylim([ymin[11],ymax[11]])
plt.xlabel("Time [us]")
print("av of DN-US-BL:",av[11])
print("Max. of DN-US-BL:",(amax[11]-av[11]), " frac:", 100.*(amax[11]-av[11])/(amax_max-ped_amax_max))
plt.annotate('{:.2} {}'.format(100.*amax[11]/amax_max,'%'), xy=(x_txt, y_txt), xycoords='axes fraction')

plt.subplot(4,4,13)
#plt.plot(time,BP, label='Ch13 BEAM PLUG', color='grey', marker='o')
plt.plot(time,TOR_1, label='Ch13 EMPTY', color='black', marker='o')
plt.legend(loc="upper left")
plt.xlim([xmin,xmax])
plt.ylim([ymin[12],ymax[12]])
plt.xlabel("Time [us]")
plt.ylabel("Amplitude [V]")
print("av of TOR1:",av[12])
print("Max. of TOR1:",(amax[12]-av[12]), " frac:", 100.*(amax[12]-av[12])/(amax_max-ped_amax_max))
plt.annotate('{:.2} {}'.format(100.*amax[12]/amax_max,'%'), xy=(x_txt, y_txt), xycoords='axes fraction')

plt.subplot(4,4,14)
plt.plot(time,BP, label='Ch14 BEAM PLUG', color='grey', marker='o')
plt.legend(loc="upper left")
plt.xlim([xmin,xmax])
plt.ylim([ymin[13],ymax[13]])
plt.xlabel("Time [us]")
print("av of BP:",av[13])
print("Max. of BP:",(amax[13]-av[13]), " frac:", 100.*(amax[13]-av[13])/(amax_max-ped_amax_max))
plt.annotate('{:.2} {}'.format(100.*amax[13]/amax_max,'%'), xy=(x_txt, y_txt), xycoords='axes fraction')

plt.subplot(4,4,15)
plt.plot(time,TOR_2, label='Ch15 EMPTY', color='black', marker='o')
plt.legend(loc="upper left")
plt.xlim([xmin,xmax])
plt.ylim([ymin[14],ymax[14]])
plt.xlabel("Time [us]")
print("av of TOR2:",av[14])
print("Max. of TOR2:",(amax[14]-av[14]), " frac:", 100.*(amax[14]-av[14])/(amax_max-ped_amax_max))
plt.annotate('{:.2} {}'.format(100.*amax[14]/amax_max,'%'), xy=(x_txt, y_txt), xycoords='axes fraction')

plt.subplot(4,4,16)
plt.plot(time,LM, label='Ch16 Levl Meter', color='black', marker='o')
plt.legend(loc="upper left")
plt.xlim([xmin,xmax])
plt.ylim([ymin[15],ymax[15]])
plt.xlabel("Time [us]")
print("av of LM:",av[15])
print("Max. of LM:",(amax[15]-av[15]), " frac:", 100.*(amax[15]-av[15])/(amax_max-ped_amax_max))
plt.annotate('{:.2} {}'.format(100.*amax[14]/amax_max,'%'), xy=(x_txt, y_txt), xycoords='axes fraction')

plt.savefig(figName)




#FFT!
#Mag_S = sg.fft(dDN_US_BL[tmp_tmaxIndex-700:tmp_tmaxIndex+700])
#y=dDN_US_BR[tmp_tmaxIndex-700:tmp_tmaxIndex+700]
#x=time[tmp_tmaxIndex-700:tmp_tmaxIndex+700]
y=dDN_US_BL[tmp_tmaxIndex-20:tmp_tmaxIndex+20]
s=DN_US_BL[tmp_tmaxIndex-20:tmp_tmaxIndex+20]
response_func=dDN_US_BL[tmp_tmaxIndex-3:tmp_tmaxIndex+3]
#HY::Response func acts as a 'filter', the filter size needs to be smaller than the size of original signal 
x=time[tmp_tmaxIndex-20:tmp_tmaxIndex+20]
n = len(y) # length of the signal
print('length of pulser signal:',n)
print(response_func)

k = np.arange(n)
Fs=1./dt_s #sampling rate [Hz]
T = n/Fs
frq = k/T # two sides frequency range
frq = frq[range(n // 2)] # one side frequency range
#// flor division operator, which returns an integer

Y = np.fft.fft(y)/float(n) # fft computing and normalization
Y = Y[range(n // 2)]
print('length of pulser signal (in freq domain):',len(Y))
print('length of freq:',len(frq))

#Deconvolution
deconv,  _ = signal.deconvolve( s, response_func)
n_n = len(s)-len(response_func)+1
#the deconvolution has n = len(s) - len(response_func) + 1 points
print('length of observed signal',len(s))
print('length of response_func',len(response_func))

# so we need to expand it by 
n_s = (len(s)-n_n)//2
print('length of signal expansion',n_s)
#on both sides.
deconv_res = np.zeros(len(s))
deconv_res[n_s:len(s)-n_s-1] = deconv
deconv = deconv_res

# now deconv contains the deconvolution 
# expanded to the original shape (filled with zeros) 
print('length of deconv',len(deconv))

#filtered = np.convolve(s, gauss, mode='same') 

#normalization of reconv
amax_deconv=-999
for i in range(len(deconv)):
    tmp_amax=deconv[i]
    if tmp_amax >= amax_deconv:
        amax_deconv = tmp_amax
        tmaxIndex_deconv = i

norm_deconv=amax_max/amax_deconv
ndeconv=deconv*norm_deconv

print("tmaxIndex_deconv:", tmaxIndex_deconv)
#print("tmp_tMax.:", tmp_tmaxIndex*dt_us, " us")
print("Max. amp. of deconv:", amax_deconv, " volt")
print("Max. amp. :", amax_max, " volt")



#plot deconv results!
ax = plt.figure(figsize=(6,10))
plt.subplot(311)
plt.plot(x, s, label='Measured Signal', color='black', marker='o', linewidth=2)
plt.legend(loc="upper left")
plt.xlabel("Time [us]")
plt.ylabel("Amplitude [V]")

plt.subplot(312)
plt.plot(x, y, label='Response Function (Time Domain)', color='blue', marker='o', linewidth=2)
plt.legend(loc="upper left")
#plt.title('Pulser Channel')
#plt.xlim([xmin,xmax])
plt.xlabel("Time [us]")
plt.ylabel("Amplitude [V]")

'''
#ax = plt.subplots(211)
plt.subplot(312)
plt.plot(frq, abs(Y), label='Response Function (Frequency Domain)', color='blue', marker='o', linewidth=2)
#plt.title("Magitude")
plt.legend(loc="upper left")

plt.xlabel("Frequency [Hz]")
plt.ylabel("Amplitude [a.u.] ")
#plt.xlim([0,1240])
#plt.savefig(figName+'_resp_zoom.png')
'''

plt.subplot(313)
plt.plot(x, ndeconv, label='Deconvoluted Signal', color='red', marker='s', linewidth=2)
plt.legend(loc="upper left")
plt.xlabel("Time [us]")
plt.ylabel("Amplitude [V]")
plt.savefig(figName+'_resp.png')


#ax.subplot(212)
#ax.title("Phase")
#ax.plot(np.angle(Mag_S))
#ax.show()

#plt.subplot(222)
#plt.plot(xf,yf,'r')
#plt.title('FFT of Mixed wave(two sides frequency range)',fontsize=7) 

#plt.savefig(figName+'_resp.png')
#plt.show()
#show will clear everything

#output data
#o_file = open(outName, "w")
#for i in range(0,n_ch-4):
  #o_file.write("{:.3}\n".format((amax[i]-av[i])/(amax_max-ped_amax_max)))
  #o_file.write("{:.3}\n".format((amax[i])/(amax_max)))

#o_file.close()
