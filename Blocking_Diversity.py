#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 00:25:58 2024

@author: liu3315
"""

#%%
###### This code is to study the blocking diversity (separation of 3 types of blocks) ######
from math import pi
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import datetime as dt
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import pandas as pd
import cv2
import copy
import matplotlib.path as mpath
import pickle
import glob
from netCDF4 import Dataset
import scipy.stats as stats
import cartopy

### A function to calculate distance between two grid points on earth ###
from math import radians, cos, sin, asin, sqrt
def haversine(lon1, lat1, lon2, lat2): # 经度1，纬度1，经度2，纬度2 （十进制度数）
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # 将十进制度数转化为弧度
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
 
    # haversine公式
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # earth radius
    return c * r * 1000

#%%
### read basic data ###
path_LWA_AC = glob.glob(r"/depot/wanglei/data/Reanalysis/MERRA2/LWA_Z_AC/*.nc")
path_LWA_AC.sort()
N=len(path_LWA_AC)   

path_LWA = glob.glob(r"/depot/wanglei/data/Reanalysis/MERRA2/LWA_Z/*.nc")
path_LWA.sort()
N=len(path_LWA)  

path_LWA_C = glob.glob(r"/depot/wanglei/data/Reanalysis/MERRA2/LWA_Z_C/*.nc")
path_LWA_C.sort()
N=len(path_LWA_C)  

path_Z = glob.glob(r"/depot/wanglei/data/Reanalysis/MERRA2/Z500/*.nc")
path_Z.sort()
N=len(path_Z)

path_dT = glob.glob(r"/depot/wanglei/data/Reanalysis/MERRA2/tdt_moist/*.nc4")
path_dT.sort()
N=len(path_dT)

path_T = glob.glob(r"/depot/wanglei/data/Reanalysis/MERRA2/T/*.nc4")
path_T.sort()
N=len(path_T)

path_dA = glob.glob(r"/depot/wanglei/data/Reanalysis/MERRA2/LWA/*.nc")
path_dA.sort()
N=len(path_dA)


### Read basic variables ###
file0 = Dataset(path_dT[0],'r')
lon = file0.variables['lon'][:]
lat = file0.variables['lat'][:]
plev = file0.variables['lev'][:]
lat_SH = lat[0:180]
lat_NH = lat[180:]
nplev = len(plev)
nlon = len(lon)
nlat = len(lat)
nlat_SH = len(lat_SH)
nlat_NH =len(lat_NH)
dlat = (lat[1]-lat[0])
dlon = (lon[1] - lon[0])
file0.close()

### read blocking data ###
with open("/depot/wanglei/data/Reanalysis/MERRA2/Blocking/Blocking_date", "rb") as fp:
    Blocking_date = pickle.load(fp)
with open("/depot/wanglei/data/Reanalysis/MERRA2/Blocking/Blocking_lat", "rb") as fp:
    Blocking_lat = pickle.load(fp)
with open("/depot/wanglei/data/Reanalysis/MERRA2/Blocking/Blocking_lon", "rb") as fp:
    Blocking_lon = pickle.load(fp)    
with open("/depot/wanglei/data/Reanalysis/MERRA2/Blocking/Blocking_lon_wide", "rb") as fp:
    Blocking_lon_wide = pickle.load(fp) 
with open("/depot/wanglei/data/Reanalysis/MERRA2/Blocking/Blocking_area", "rb") as fp:
    Blocking_area = pickle.load(fp) 
    
B_freq = np.load("/depot/wanglei/data/Reanalysis/MERRA2/Blocking/B_freq.npy")
Thresh = 71362440.0
### Time Management ###
Datestamp = pd.date_range(start="1980-01-01",end="2023-05-01")
Date0 = pd.DataFrame({'date': pd.to_datetime(Datestamp)})
Month = Date0['date'].dt.month 
Year = Date0['date'].dt.year
Day = Date0['date'].dt.day
Date = list(Date0['date'])
nday = len(Date)

#%%
### get the blocking peaking date and location and wave activity ###

with open("/depot/wanglei/data/Reanalysis/MERRA2/Blocking/Blocking_peaking_date", "rb") as fp:
    Blocking_peaking_date = pickle.load(fp)
with open("/depot/wanglei/data/Reanalysis/MERRA2/Blocking/Blocking_peaking_lat", "rb") as fp:
    Blocking_peaking_lat = pickle.load(fp)
with open("/depot/wanglei/data/Reanalysis/MERRA2/Blocking/Blocking_peaking_lon", "rb") as fp:
    Blocking_peaking_lon = pickle.load(fp)    
with open("/depot/wanglei/data/Reanalysis/MERRA2/Blocking/Blocking_peaking_lon_wide", "rb") as fp:
    Blocking_peaking_lon_wide = pickle.load(fp)    
with open("/depot/wanglei/data/Reanalysis/MERRA2/Blocking/Blocking_peaking_area", "rb") as fp:
    Blocking_peaking_area = pickle.load(fp) 
with open("/depot/wanglei/data/Reanalysis/MERRA2/Blocking/Blocking_peaking_LWA", "rb") as fp:
    Blocking_peaking_LWA = pickle.load(fp) 
with open("/depot/wanglei/data/Reanalysis/MERRA2/Blocking/Blocking_velocity", "rb") as fp:
    Blocking_velocity = pickle.load(fp)
with open("/depot/wanglei/data/Reanalysis/MERRA2/Blocking/Blocking_duration", "rb") as fp:
    Blocking_duration = pickle.load(fp)
    

#%%
###### Now we separate 3 types of blocks (ridge, trough, dipole)
Blocking_ridge_date = [];  Blocking_ridge_lon = []; Blocking_ridge_lat=[];  Blocking_ridge_peaking_date = [];   Blocking_ridge_peaking_lon = []; Blocking_ridge_peaking_lat=[];  Blocking_ridge_duration = [];  Blocking_ridge_velocity = [];    Blocking_ridge_area = [];   Blocking_ridge_peaking_LWA = []
Blocking_trough_date = []; Blocking_trough_lon =[]; Blocking_trough_lat=[]; Blocking_trough_peaking_date = [];  Blocking_trough_peaking_lon =[]; Blocking_trough_peaking_lat=[]; Blocking_trough_duration = []; Blocking_trough_velocity = [];   Blocking_trough_area = [];  Blocking_trough_peaking_LWA = []
Blocking_dipole_date = []; Blocking_dipole_lon =[]; Blocking_dipole_lat=[]; Blocking_dipole_peaking_date = [];  Blocking_dipole_peaking_lon =[]; Blocking_dipole_peaking_lat=[]; Blocking_dipole_duration= []; Blocking_dipole_velocity =[];    Blocking_dipole_area = [];   Blocking_dipole_peaking_LWA = []
Blocking_AW_date = [];     Blocking_AW_lon =[];     Blocking_AW_lat=[];     Blocking_AW_peaking_date = [];      Blocking_AW_peaking_lon =[];     Blocking_AW_peaking_lat=[];     Blocking_AW_duration = [];     Blocking_AW_velocity=[];         Blocking_AW_area = [];      Blocking_AW_peaking_LWA = []
Blocking_CW_date = [];     Blocking_CW_lon =[];     Blocking_CW_lat=[];     Blocking_CW_peaking_date = [];      Blocking_CW_peaking_lon =[];     Blocking_CW_peaking_lat=[];     Blocking_CW_duration = [];     Blocking_CW_velocity=[];         Blocking_CW_area = [];      Blocking_CW_peaking_LWA = []
Blocking_diff=[]
for n in np.arange(len(Blocking_lon)):
    
    ### peaking date information ###
    peaking_date_index = Date.index(Blocking_peaking_date[n])
    peaking_lon_index = np.squeeze(np.array(np.where( lon[:]==Blocking_peaking_lon[n])))
    peaking_lat_index = np.squeeze(np.array(np.where( lat[:]==Blocking_peaking_lat[n])))
    
    ### peaking date maximum LWA ###
    file_LWA = Dataset(path_LWA[peaking_date_index],'r')
    LWA_max  = file_LWA.variables['LWA_Z500'][0,0,peaking_lat_index,peaking_lon_index]
    file_LWA.close()
    ### peaking date maximum LWA_AC ###
    file_LWA_AC = Dataset(path_LWA_AC[peaking_date_index],'r')
    LWA_AC_max  = file_LWA.variables['LWA_Z500'][0,0,peaking_lat_index,peaking_lon_index]
    file_LWA_AC.close()
    ### peaking date maximum LWA_C ###
    file_LWA_C = Dataset(path_LWA_C[peaking_date_index],'r')
    LWA_C_max  = file_LWA.variables['LWA_Z500'][0,0,peaking_lat_index,peaking_lon_index]
    file_LWA_C.close()

    ### A dipole requries both LWA_AC and LWA_C at the peaking date are lagrer than the threshold ###
    if LWA_AC_max>0 and LWA_C_max == 0:
        Blocking_ridge_date.append(Blocking_date[n]);                 Blocking_ridge_lon.append(Blocking_lon[n]);                  Blocking_ridge_lat.append(Blocking_lat[n])
        Blocking_ridge_peaking_date.append(Blocking_peaking_date[n]); Blocking_ridge_peaking_lon.append(Blocking_peaking_lon[n]);  Blocking_ridge_peaking_lat.append(Blocking_peaking_lat[n]); Blocking_ridge_peaking_LWA.append(LWA_max)
        Blocking_ridge_duration.append(len(Blocking_date[n]));        Blocking_ridge_velocity.append(Blocking_velocity[n]);        Blocking_ridge_area.append(Blocking_peaking_area[n])
    elif LWA_AC_max == 0 and LWA_C_max >0:
        Blocking_trough_date.append(Blocking_date[n]);                 Blocking_trough_lon.append(Blocking_lon[n]);                 Blocking_trough_lat.append(Blocking_lat[n])
        Blocking_trough_peaking_date.append(Blocking_peaking_date[n]); Blocking_trough_peaking_lon.append(Blocking_peaking_lon[n]); Blocking_trough_peaking_lat.append(Blocking_peaking_lat[n]); Blocking_trough_peaking_LWA.append(LWA_max)
        Blocking_trough_duration.append(len(Blocking_date[n]));        Blocking_trough_velocity.append(Blocking_velocity[n]);       Blocking_trough_area.append(Blocking_peaking_area[n])
    elif LWA_AC_max>(Thresh/2) and LWA_C_max>(Thresh/2):
        Blocking_dipole_date.append(Blocking_date[n]);                 Blocking_dipole_lon.append(Blocking_lon[n]);                 Blocking_dipole_lat.append(Blocking_lat[n])
        Blocking_dipole_peaking_date.append(Blocking_peaking_date[n]); Blocking_dipole_peaking_lon.append(Blocking_peaking_lon[n]); Blocking_dipole_peaking_lat.append(Blocking_peaking_lat[n]); Blocking_dipole_peaking_LWA.append(LWA_max)
        Blocking_dipole_duration.append(len(Blocking_date[n]));       Blocking_dipole_velocity.append(Blocking_velocity[n]);       Blocking_dipole_area.append(Blocking_peaking_area[n])
    elif LWA_AC_max>(Thresh/2) and LWA_C_max<(Thresh/2) and LWA_C_max>0:
        Blocking_AW_date.append(Blocking_date[n]);                     Blocking_AW_lon.append(Blocking_lon[n]);                     Blocking_AW_lat.append(Blocking_lat[n])
        Blocking_AW_peaking_date.append(Blocking_peaking_date[n]);     Blocking_AW_peaking_lon.append(Blocking_peaking_lon[n]);     Blocking_AW_peaking_lat.append(Blocking_peaking_lat[n]);  Blocking_AW_peaking_LWA.append(LWA_max)
        Blocking_AW_duration.append(len(Blocking_date[n]));            Blocking_AW_velocity.append(Blocking_velocity[n]);           Blocking_AW_area.append(Blocking_peaking_area[n])
    elif LWA_C_max>(Thresh/2) and LWA_AC_max<(Thresh/2) and LWA_AC_max>0:
        Blocking_CW_date.append(Blocking_date[n]);                     Blocking_CW_lon.append(Blocking_lon[n]);                     Blocking_CW_lat.append(Blocking_lat[n])
        Blocking_CW_peaking_date.append(Blocking_peaking_date[n]);     Blocking_CW_peaking_lon.append(Blocking_peaking_lon[n]);     Blocking_CW_peaking_lat.append(Blocking_peaking_lat[n]);  Blocking_CW_peaking_LWA.append(LWA_max)
        Blocking_CW_duration.append(len(Blocking_date[n]));        

Blocking_diversity_date= [];   Blocking_diversity_lon= []; Blocking_diversity_lat= []; Blocking_diversity_date= []; Blocking_diversity_peaking_date= []; Blocking_diversity_peaking_lon= [];  Blocking_diversity_peaking_lat=[]; Blocking_diversity_peaking_LWA=[]; Blocking_diversity_duration=[]; Blocking_diversity_area=[]; Blocking_diversity_velocity=[]
Blocking_diversity_date.append(Blocking_ridge_date);   Blocking_diversity_lon.append(Blocking_ridge_lon); Blocking_diversity_lat.append(Blocking_ridge_lat); Blocking_diversity_peaking_date.append(Blocking_ridge_peaking_date); Blocking_diversity_peaking_lat.append(Blocking_ridge_peaking_lat); Blocking_diversity_peaking_lon.append(Blocking_ridge_peaking_lon); Blocking_diversity_peaking_LWA.append(Blocking_ridge_peaking_LWA); Blocking_diversity_velocity.append(Blocking_ridge_velocity); Blocking_diversity_duration.append(Blocking_ridge_duration); Blocking_diversity_area.append(Blocking_ridge_area)
Blocking_diversity_date.append(Blocking_trough_date);  Blocking_diversity_lon.append(Blocking_trough_lon); Blocking_diversity_lat.append(Blocking_trough_lat); Blocking_diversity_peaking_date.append(Blocking_trough_peaking_date); Blocking_diversity_peaking_lat.append(Blocking_trough_peaking_lat); Blocking_diversity_peaking_lon.append(Blocking_trough_peaking_lon);Blocking_diversity_peaking_LWA.append(Blocking_trough_peaking_LWA); Blocking_diversity_velocity.append(Blocking_trough_velocity); Blocking_diversity_duration.append(Blocking_trough_duration); Blocking_diversity_area.append(Blocking_trough_area)  
Blocking_diversity_date.append(Blocking_dipole_date);  Blocking_diversity_lon.append(Blocking_dipole_lon); Blocking_diversity_lat.append(Blocking_dipole_lat); Blocking_diversity_peaking_date.append(Blocking_dipole_peaking_date); Blocking_diversity_peaking_lat.append(Blocking_dipole_peaking_lat); Blocking_diversity_peaking_lon.append(Blocking_dipole_peaking_lon); Blocking_diversity_peaking_LWA.append(Blocking_dipole_peaking_LWA); Blocking_diversity_velocity.append(Blocking_dipole_velocity); Blocking_diversity_duration.append(Blocking_dipole_duration); Blocking_diversity_area.append(Blocking_dipole_area)   
Blocking_diversity_date.append(Blocking_AW_date);      Blocking_diversity_lon.append(Blocking_AW_lon); Blocking_diversity_lat.append(Blocking_AW_lat); Blocking_diversity_peaking_date.append(Blocking_AW_peaking_date); Blocking_diversity_peaking_lat.append(Blocking_AW_peaking_lat); Blocking_diversity_peaking_lon.append(Blocking_AW_peaking_lon); Blocking_diversity_peaking_LWA.append(Blocking_AW_peaking_LWA); Blocking_diversity_velocity.append(Blocking_AW_velocity); Blocking_diversity_duration.append(Blocking_AW_duration); Blocking_diversity_area.append(Blocking_AW_area) 
Blocking_diversity_date.append(Blocking_CW_date);      Blocking_diversity_lon.append(Blocking_CW_lon); Blocking_diversity_lat.append(Blocking_CW_lat); Blocking_diversity_peaking_date.append(Blocking_CW_peaking_date); Blocking_diversity_peaking_lat.append(Blocking_CW_peaking_lat); Blocking_diversity_peaking_lon.append(Blocking_CW_peaking_lon); Blocking_diversity_peaking_LWA.append(Blocking_CW_peaking_LWA); Blocking_diversity_velocity.append(Blocking_CW_velocity); Blocking_diversity_duration.append(Blocking_CW_duration); Blocking_diversity_area.append(Blocking_CW_area) 
 




######################################################################################################################################################################################################################################
#%%
###### Figure 1 of the manuscript ######
with open("/depot/wanglei/data/Reanalysis/MERRA2/Blocking/dTdt_Blocking_diversity_m", "rb") as fp:
    dTdt_Blocking_diversity_m = pickle.load(fp)
with open("/depot/wanglei/data/Reanalysis/MERRA2/Blocking/Z_Blocking_diversity_m", "rb") as fp:
    Z_Blocking_diversity_m = pickle.load(fp)
with open("/depot/wanglei/data/Reanalysis/MERRA2/Blocking/dTdt_diversity_cros_com", "rb") as fp:
    dTdt_diversity_cros_com = pickle.load(fp)    
with open("/depot/wanglei/data/Reanalysis/MERRA2/Blocking/LWA_diversity_cros_com", "rb") as fp:
    LWA_diversity_cros_com = pickle.load(fp)
with open("/depot/wanglei/data/Reanalysis/MERRA2/Blocking/dTdt_ridge_list", "rb") as fp:
    dTdt_ridge_list = pickle.load(fp) 
with open("/depot/wanglei/data/Reanalysis/MERRA2/Blocking/dTdt_dipole_list", "rb") as fp:
    dTdt_dipole_list = pickle.load(fp) 
           
minlev = Z_Blocking_diversity_m[2].min()
maxlev = Z_Blocking_diversity_m[2].max()
levs_Z = np.linspace(5270,5780,11)
minlev = dTdt_Blocking_diversity_m[2].min()
maxlev = dTdt_Blocking_diversity_m[2].max()
levs_dTdt = np.linspace(0, 2.7e-5,15)
lon_range=int(60/dlon)+1 
  
fig = plt.figure(figsize=[10,12])
ax = fig.add_subplot(3,2,1)
a = plt.contourf(np.arange(0,lon_range),np.arange(0,len(dTdt_Blocking_diversity_m[0])), dTdt_Blocking_diversity_m[0], levs_dTdt, cmap='hot_r',extend='both')  
ax.contour(np.arange(0,lon_range),np.arange(0,len(dTdt_Blocking_diversity_m[0])), Z_Blocking_diversity_m[0], levs_Z, colors='k')  
ax.set_yticks([0,21,42, 63,84])
ax.set_yticklabels(['-20','-10','lat_c','+10','+20'])
ax.set_xticks([0,16,32,48,64,80,96])
ax.set_xticklabels(['-30','-20','-10','lon_c','+10','+20','+30'])
ax.set_title("Composite Ridge Blocks \n(a)", pad=5, fontsize=12)
# ax.set_xlabel('relative longitude',fontsize=12)
ax.set_ylabel('relative latitude',fontsize=12)     

bx = fig.add_subplot(3,2,2)
b = plt.contourf(np.arange(0,lon_range),np.arange(0,len(dTdt_Blocking_diversity_m[2])), dTdt_Blocking_diversity_m[2], levs_dTdt, cmap='hot_r',extend='both')  
bx.contour(np.arange(0,lon_range),np.arange(0,len(Z_Blocking_diversity_m[2])), Z_Blocking_diversity_m[2], levs_Z, colors='k')  
bx.set_yticks([0,19.5,39, 58.5,78])
bx.set_yticklabels(['-20','-10','lat_c','+10','+20'])
bx.set_xticks([0,16,32,48,64,80,96])
bx.set_xticklabels(['-30','-20','-10','lon_c','+10','+20','+30'])
bx.set_title("Composite Dipole Blocks \n(b)" , pad=5, fontsize=12)
# bx.set_xlabel('relative longitude ',fontsize=12)
# bx.set_ylabel('relative latitude',fontsize=12)    
cbar = fig.add_axes([0.93,0.64,0.01,0.25])
cb = plt.colorbar(a, cax=cbar, ticks=[0,1e-5,2e-5,3e-5]) 
cb.set_ticklabels(['0','1','2','3'])
cb.set_label('moist-induced diabatic heating ($10^{-5}$K/s)',fontsize=10)

maxlevel = np.max(dTdt_diversity_cros_com[2])
minlevel = np.min(dTdt_diversity_cros_com[2]) 
levs_dTdt_cross = np.linspace(-3e-5, 3e-5, 19)

maxlevel = np.max(LWA_diversity_cros_com[2])
minlevel = np.min(LWA_diversity_cros_com[2]) 
levs_LWA_cross = np.linspace(0, 200, 15)

## Here we interp the LWA_cross to the pressure level ###
H=7000
plev_z = np.flipud(np.array([1000*np.exp(-z/H) for z in np.arange(0,48001,1000)]))
zlev = np.flipud(np.arange(0,48001,1000)) 
plev_z = plev_z[1:-1]

LWA_ridge_cros_com_plev = np.zeros([nplev,nlon])
for loo in np.arange(nlon):
    LWA_ridge_cros_com_plev[:,loo] = np.interp(np.flipud(plev[:]), plev_z[:], np.flipud(LWA_diversity_cros_com[0][:,loo]))
LWA_trough_cros_com_plev = np.zeros([nplev,nlon])
for loo in np.arange(nlon):
    LWA_trough_cros_com_plev[:,loo] = np.interp(np.flipud(plev[:]), plev_z[:], np.flipud(LWA_diversity_cros_com[1][:,loo]))    
LWA_dipole_cros_com_plev = np.zeros([nplev,nlon])
for loo in np.arange(nlon):
    LWA_dipole_cros_com_plev[:,loo] = np.interp(np.flipud(plev[:]), plev_z[:], np.flipud(LWA_diversity_cros_com[2][:,loo]))

LWA_diversity_cros_com_plev = []
LWA_diversity_cros_com_plev.append(np.flipud(LWA_ridge_cros_com_plev))
LWA_diversity_cros_com_plev.append(np.flipud(LWA_trough_cros_com_plev))
LWA_diversity_cros_com_plev.append(np.flipud(LWA_dipole_cros_com_plev))

cx = fig.add_subplot(3,2,3)
c=plt.contourf(np.arange(0,lon_range), plev, dTdt_diversity_cros_com[0][:,int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1], levs_dTdt_cross,cmap='RdBu_r',extend ='both')
plt.contour(np.arange(0,lon_range), plev,   LWA_diversity_cros_com_plev[0][:,int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1], levs_LWA_cross, colors="k", linewidths=0.5)
plt.xlabel('relative longitude',fontsize=12)
plt.ylabel('pressure level (hPa)',fontsize=12)
cx.set_ylim(1000,200)
cx.set_yticks([1000,850,700,600,500,300,200])
cx.set_xticks([0,16,32,48,64,80,96])
cx.set_xticklabels(['-30','-20','-10','lon_c','+10','+20','+30'])
plt.title("(c)", pad=5, fontdict={'family':'Times New Roman', 'size':12})

dx = fig.add_subplot(3,2,4)
c=plt.contourf(np.arange(0,lon_range), plev, dTdt_diversity_cros_com[2][:,int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1], levs_dTdt_cross,cmap='RdBu_r',extend ='both')
plt.contour(np.arange(0,lon_range), plev,   LWA_diversity_cros_com_plev[2][:,int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1], levs_LWA_cross, colors="k", linewidths=0.5)
plt.xlabel('relative longitude',fontsize=12)
# plt.ylabel('pressure level',fontsize=12)
dx.set_ylim(1000,200)
dx.set_yticks([1000,850,700,600,500,300,200])
dx.set_xticks([0,16,32,48,64,80,96])
dx.set_xticklabels(['-30','-20','-10','lon_c','+10','+20','+30'])
plt.title("(d)", pad=5, fontdict={'family':'Times New Roman', 'size':12})
cbar = fig.add_axes([0.93,0.37,0.01,0.25])
cb = plt.colorbar(c, cax=cbar, ticks=[-3e-5,-2e-5,-1e-5,0,1e-5,2e-5,3e-5]) 
cb.set_ticklabels(['-3','-2','-1','0','1','2','3'])
cb.set_label('moist-induced diabatic heating  ($10^{-5}$K/s)',fontsize=10)

proj=ccrs.PlateCarree(central_longitude=180)
ex = fig.add_subplot(3,1,3, projection=proj)
ex.scatter(Blocking_diversity_peaking_lon[0][:],Blocking_diversity_peaking_lat[0][:], transform=ccrs.PlateCarree(), cmap='RdBu_r' ,s= 5, label="ridges", color='r', alpha=0.5)  
ex.scatter(Blocking_diversity_peaking_lon[2][:],Blocking_diversity_peaking_lat[2][:], transform=ccrs.PlateCarree(), cmap='RdBu_r' ,s= 5, label="dipoles",color='b',alpha=0.5)  
plt.xlabel('longitude',fontsize=12)
plt.ylabel('latitude',fontsize=12)     
plt.title("(e) Distribution of Ridges and Dipoles", pad=5, fontdict={'family':'Times New Roman', 'size':12})
ex.coastlines()
ex.gridlines(linestyle="--", alpha=0.7)
ex.set_extent([-180,180,0,90],crs=ccrs.PlateCarree())
plt.legend(loc='lower left')
ex.set_xticks([0,60,120,180,240,300,358.5], crs=ccrs.PlateCarree())
ex.set_yticks([0,30,60,90], crs=ccrs.PlateCarree())
lon_formatter = LongitudeFormatter(zero_direction_label='FALSE')
lat_formatter = LatitudeFormatter()
ex.xaxis.set_major_formatter(lon_formatter)
ex.yaxis.set_major_formatter(lat_formatter) 


plt.savefig("/home/liu3315/Research/Blocking_Diversity/Figure1_new.png",dpi=600)
################################################################################################################
#%%
###### Figure 3 of the manuscript ######
with open("/depot/wanglei/data/Reanalysis/MERRA2/Blocking/dAdt_Blocking_diversity_m", "rb") as fp:
    dAdt_Blocking_diversity_m = pickle.load(fp)
with open("/depot/wanglei/data/Reanalysis/MERRA2/Blocking/Z_Blocking_diversity_m", "rb") as fp:
    Z_Blocking_diversity_m = pickle.load(fp)
with open("/depot/wanglei/data/Reanalysis/MERRA2/Blocking/dAdt_diversity_cros_com", "rb") as fp:
    dAdt_diversity_cros_com = pickle.load(fp)    
with open("/depot/wanglei/data/Reanalysis/MERRA2/Blocking/LWA_diversity_cros_com", "rb") as fp:
    LWA_diversity_cros_com = pickle.load(fp)
with open("/depot/wanglei/data/Reanalysis/MERRA2/Blocking/LWA_Hov", "rb") as fp:
    LWA_Blocking_diversity_com = pickle.load(fp)
with open("/depot/wanglei/data/Reanalysis/MERRA2/Blocking/dAdt_Hov", "rb") as fp:
    dAdt_Blocking_diversity_com = pickle.load(fp)
        
minlev = Z_Blocking_diversity_m[2].min()
maxlev = Z_Blocking_diversity_m[2].max()
levs_Z = np.linspace(5270,5780,11)
minlev = dAdt_Blocking_diversity_m[2].min()
maxlev = dAdt_Blocking_diversity_m[2].max()
levs_dAdt = np.linspace(-5e-5, 5e-5,21)
lat_range=int(10/dlat)+1
lon_range=int(60/dlon)+1
  
fig = plt.figure(figsize=[10,12])
ax = fig.add_subplot(3,2,1)
a = plt.contourf(np.arange(0,lon_range),np.arange(0,len(dAdt_Blocking_diversity_m[0])), dAdt_Blocking_diversity_m[0], levs_dAdt, cmap='RdBu_r',extend='both')  
ax.contour(np.arange(0,lon_range),np.arange(0,len(dAdt_Blocking_diversity_m[0])), Z_Blocking_diversity_m[0], levs_Z, colors='k')  
ax.set_yticks([0,21,42, 63,84])
ax.set_yticklabels(['-20','-10','lat_c','+10','+20'])
ax.set_xticks([0,16,32,48,64,80,96])
ax.set_xticklabels(['-30','-20','-10','lon_c','+10','+20','+30'])
ax.set_title("Composite Ridge Blocks \n(a)", pad=5, fontsize=12)
# ax.set_xlabel('relative longitude',fontsize=12)
ax.set_ylabel('relative latitude',fontsize=12)     

bx = fig.add_subplot(3,2,2)
b = plt.contourf(np.arange(0,lon_range),np.arange(0,len(dAdt_Blocking_diversity_m[2])), dAdt_Blocking_diversity_m[2], levs_dAdt, cmap='RdBu_r',extend='both')  
bx.contour(np.arange(0,lon_range),np.arange(0,len(Z_Blocking_diversity_m[2])), Z_Blocking_diversity_m[2], levs_Z, colors='k')  
bx.set_yticks([0,19.5,39, 58.5,78])
bx.set_yticklabels(['-20','-10','lat_c','+10','+20'])
bx.set_xticks([0,16,32,48,64,80,96])
bx.set_xticklabels(['-30','-20','-10','lon_c','+10','+20','+30'])
bx.set_title("Composite Dipole Blocks \n(b)" , pad=5, fontsize=12)
# bx.set_xlabel('relative longitude ',fontsize=12)
# bx.set_ylabel('relative latitude',fontsize=12)    
cbar = fig.add_axes([0.93,0.64,0.01,0.25])
cb = plt.colorbar(a, cax=cbar, ticks=[-5e-5,-4e-5,-3e-5,-2e-5,-1e-5,0,1e-5,2e-5,3e-5,4e-5,5e-5]) 
cb.set_ticklabels(['-5.0','-4.0','-3.0','-2.0','-1.0','0','1.0','2.0','3.0','4.0','5.0'])
cb.set_label('moist-induced LWA tendency ($10^{-5}$m/$s^2$)',fontsize=10)

maxlevel = np.max(dAdt_diversity_cros_com[0])
minlevel = np.min(dAdt_diversity_cros_com[0]) 
levs_dAdt_cross = np.linspace(-2e-4, 2e-4, 24)

maxlevel = np.max(LWA_diversity_cros_com[0])
minlevel = np.min(LWA_diversity_cros_com[0]) 
levs_LWA_cross = np.linspace(0, 200, 15)

zlev = np.arange(1,31001,1000)
zlev1=0; zlev2=12

cx = fig.add_subplot(3,2,3)
c=plt.contourf(lon[int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1], zlev[zlev1:zlev2], dAdt_diversity_cros_com[0][zlev1:zlev2,int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1], levs_dAdt_cross,cmap='RdBu_r',extend ='both')
plt.contour(lon[int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1], zlev[zlev1:zlev2], LWA_diversity_cros_com[0][zlev1:zlev2,int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1], levs_LWA_cross, colors="k", linewidths=0.5)
# plt.xlabel('relative longitude',fontsize=12)
plt.ylabel('height (m)',fontsize=12)
cx.set_ylim(1000,10000)
cx.set_yticks([2000,4000,6000,8000,10000])
# cx.set_xticks([0,16,32,48,64,80,96])
cx.set_xticklabels(['-30','-20','-10','lon_c','+10','+20','+30'])
plt.title("(c)", pad=5, fontdict={'family':'Times New Roman', 'size':12})

dx = fig.add_subplot(3,2,4)
d = plt.contourf(lon[int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1], zlev[zlev1:zlev2], dAdt_diversity_cros_com[2][zlev1:zlev2,int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1], levs_dAdt_cross,cmap='RdBu_r',extend ='both')
plt.contour(lon[int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1], zlev[zlev1:zlev2], LWA_diversity_cros_com[2][zlev1:zlev2,int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1], levs_LWA_cross, colors="k", linewidths=0.5)
# plt.xlabel('relative longitude',fontsize=12)
# plt.ylabel('pressure level',fontsize=12)
dx.set_ylim(1000,10000)
dx.set_yticks([2000,4000,6000,8000,10000])
# dx.set_xticks([0,16,32,48,64,80,96])
dx.set_xticklabels(['-30','-20','-10','lon_c','+10','+20','+30'])
plt.title("(d)", pad=5, fontdict={'family':'Times New Roman', 'size':12})
cbar = fig.add_axes([0.93,0.37,0.01,0.25])
cb = plt.colorbar(c, cax=cbar, ticks=[-2e-4,-1.5e-4,-1e-4,-0.5e-4, 0, 0.5e-4, 1e-4, 1.5e-4, 2e-4]) 
cb.set_ticklabels(['-2','-1.5','-1','0.5','0','0.5','1','1.5','2'])
cb.set_label('moist-induced LWA tendency ($10^{-4}$m/$s^2$)',fontsize=10)


lat_range=int(10/dlat)+1
lon_range=int(180/dlon)+1
t_range=6
duration = 6
maxlevel = LWA_Blocking_diversity_com[2][:,:].max()
minlevel = LWA_Blocking_diversity_com[2][:,:].min()  
levs_LWA_Hov = np.linspace(40, 80, 11)
maxlevel = dAdt_Blocking_diversity_com[2][:,:].max()
minlevel = dAdt_Blocking_diversity_com[2][:,:].min()  
levs_dAdt_Hov = np.linspace(-5.5e-5, 5.5e-5, 11)
        
ex = fig.add_subplot(3,2,5)
ex.contour(lon[int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1], np.arange(13),LWA_Blocking_diversity_com[0][int((2*duration+1)/2)-t_range:int((2*duration+1)/2)+t_range+1,int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1], levs_LWA_Hov, colors="k", linewidths=0.5)
ex.contourf(lon[int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1], np.arange(13),dAdt_Blocking_diversity_com[0][int((2*duration+1)/2)-t_range:int((2*duration+1)/2)+t_range+1,int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1], levs_dAdt_Hov, cmap='RdBu_r',extend ='both')
ex.set_ylabel('time (days)',fontsize=12)
ex.set_yticks([0,3,6,9,12])
ex.set_yticklabels([-6,-3,0,3,6])
ex.set_xticks([-90,-60,-30,0,30,60,90])
ex.set_xticklabels(['-90','-60','-30','lon_c','+30','+60','+90'])
plt.xlabel('relative longitude',fontsize=12)
plt.title("(e)", pad=5, fontdict={'family':'Times New Roman', 'size':12})

fx = fig.add_subplot(3,2,6)
fx.contour(lon[int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1], np.arange(13),LWA_Blocking_diversity_com[2][int((2*duration+1)/2)-t_range:int((2*duration+1)/2)+t_range+1,int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1], levs_LWA_Hov, colors="k", linewidths=0.5)
f = fx.contourf(lon[int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1], np.arange(13),dAdt_Blocking_diversity_com[2][int((2*duration+1)/2)-t_range:int((2*duration+1)/2)+t_range+1,int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1], levs_dAdt_Hov, cmap='RdBu_r',extend ='both')
# bx.ylabel('time (days)',fontsize=12)
fx.set_yticks([0,3,6,9,12])
fx.set_yticklabels([-6,-3,0,3,6])
fx.set_xticks([-90,-60,-30,0,30,60,90])
fx.set_xticklabels(['-90','-60','-30','lon_c','+30','+60','+90'])
plt.xlabel('relative longitude',fontsize=12)
plt.title("(f)", pad=5, fontdict={'family':'Times New Roman', 'size':12})
cbar = fig.add_axes([0.93,0.1,0.01,0.25])
cb = plt.colorbar(f, cax=cbar, ticks=[-5e-5, -4e-5, -3e-5,-2e-5,-1e-5, 0 ,1e-5,2e-5,3e-5,4e-5,5e-5]) 
cb.set_ticklabels(['-5.0','-4.0','-3.0','-2.0','-1.0','0','1.0','2.0','3.0','4.0','5.0'])
cb.set_label('moist-induced LWA tendency ($10^{-5}$m/$s^2$)',fontsize=10)
plt.savefig("/home/liu3315/Research/Blocking_Diversity/Figure3_new.png",dpi=600)
#################################################################################################################
#%%
####### Figure 4 ##########
with open("/depot/wanglei/data/Reanalysis/MERRA2/Blocking/dAdt_ridge_list", "rb") as fp:
    dAdt_ridge_list= pickle.load(fp) 
with open("/depot/wanglei/data/Reanalysis/MERRA2/Blocking/dTdt_ridge_list", "rb") as fp:
    dTdt_ridge_list = pickle.load(fp) 
with open("/depot/wanglei/data/Reanalysis/MERRA2/Blocking/dAdt_dipole_list", "rb") as fp:
    dAdt_dipole_list = pickle.load(fp) 
with open("/depot/wanglei/data/Reanalysis/MERRA2/Blocking/dTdt_dipole_list", "rb") as fp:
    dTdt_dipole_list = pickle.load(fp) 


 
proj=ccrs.PlateCarree(central_longitude=180)
fig = plt.figure(figsize=[12,14])
ax = fig.add_subplot(4,1,1, projection=proj)
dT = ax.scatter(Blocking_diversity_peaking_lon[0][:],Blocking_diversity_peaking_lat[0][:], c=np.array(dTdt_ridge_list), transform=ccrs.PlateCarree(), vmin=0, vmax=2e-5, cmap='hot_r' ,s= 50, label="dTdt_moist")    
ax.set_title("(a) Ridge Blocks", pad=5, fontdict={'family':'Times New Roman', 'size':12})
ax.add_feature(cartopy.feature.LAND, facecolor='lightgray',alpha = 0.2)
# ax.set_ylabel("dTdt_moist", fontsize=12)
ax.coastlines()
ax.gridlines(linestyle="--", alpha=0.7)
ax.set_extent([-180,180,0,90],crs=ccrs.PlateCarree())
ax.set_xticks([0,60,120,180,240,300,358.5], crs=ccrs.PlateCarree())
ax.set_yticks([0,30,60,90], crs=ccrs.PlateCarree())
lon_formatter = LongitudeFormatter(zero_direction_label='FALSE')
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter) 


bx = fig.add_subplot(4,1,2, projection=proj)
dT = bx.scatter(Blocking_diversity_peaking_lon[2][:],Blocking_diversity_peaking_lat[2][:], c=np.array(dTdt_dipole_list), transform=ccrs.PlateCarree(), vmin=0, vmax=2e-5, cmap='hot_r', s=50, label='dTdt_moist')   
bx.set_title("(b) Dipole Blocks", pad=5, fontdict={'family':'Times New Roman', 'size':12})
bx.add_feature(cartopy.feature.LAND, facecolor='lightgray',alpha = 0.2)
bx.coastlines()
bx.gridlines(linestyle="--", alpha=0.7)
bx.set_extent([-180,180,0,90],crs=ccrs.PlateCarree())
bx.set_xticks([0,60,120,180,240,300,358.5], crs=ccrs.PlateCarree())
bx.set_yticks([0,30,60,90], crs=ccrs.PlateCarree())
lon_formatter = LongitudeFormatter(zero_direction_label='FALSE')
lat_formatter = LatitudeFormatter()
bx.xaxis.set_major_formatter(lon_formatter)
bx.yaxis.set_major_formatter(lat_formatter) 
cbar = fig.add_axes([0.93,0.5,0.01,0.38])
cb = plt.colorbar(dT, cax=cbar, ticks=[0,0.5e-5,1e-5,1.5e-5,2e-5]) 
cb.set_ticklabels(['0','0.5','1','1.5','2'])
cb.set_label('moist-induced diabatic heating ($10^{-5}$K/s)',fontsize=10)


cx = fig.add_subplot(4,1,3, projection=proj)
cx.scatter(Blocking_diversity_peaking_lon[0][:],Blocking_diversity_peaking_lat[0][:], c=np.array(dAdt_ridge_list), transform=ccrs.PlateCarree(), vmin=-1.5e-4, vmax=1.5e-4, cmap='RdBu_r' ,s= 50, label="dAdt_moist")    
cx.set_title("(c) Ridge Blocks", pad=5, fontdict={'family':'Times New Roman', 'size':12})
cx.add_feature(cartopy.feature.LAND, facecolor='lightgray',alpha = 0.2)
# cx.set_ylabel("dAdt_moist", fontsize=12)
# cx.set_xlabel("longitude", fontsize=12)
cx.coastlines()
cx.gridlines(linestyle="--", alpha=0.7)
cx.set_extent([-180,180,0,90],crs=ccrs.PlateCarree())
cx.set_extent([-180,180,0,90],crs=ccrs.PlateCarree())
cx.set_xticks([0,60,120,180,240,300,358.5], crs=ccrs.PlateCarree())
cx.set_yticks([0,30,60,90], crs=ccrs.PlateCarree())
lon_formatter = LongitudeFormatter(zero_direction_label='FALSE')
lat_formatter = LatitudeFormatter()
cx.xaxis.set_major_formatter(lon_formatter)
cx.yaxis.set_major_formatter(lat_formatter) 


dx = fig.add_subplot(4,1,4, projection=proj)
dA = dx.scatter(Blocking_diversity_peaking_lon[2][:],Blocking_diversity_peaking_lat[2][:], c=np.array(dAdt_dipole_list), transform=ccrs.PlateCarree(), vmin=-1.5e-4, vmax=1.5e-4, cmap='RdBu_r', s=50, label='dAdt_moist')   
dx.set_title("(d) Dipole Blocks", pad=5, fontdict={'family':'Times New Roman', 'size':12})
dx.add_feature(cartopy.feature.LAND, facecolor='lightgray',alpha = 0.2)
# dx.set_xlabel("longitude", fontsize=12)
dx.coastlines()
dx.gridlines(linestyle="--", alpha=0.7)
dx.set_extent([-180,180,0,90],crs=ccrs.PlateCarree())
dx.set_extent([-180,180,0,90],crs=ccrs.PlateCarree())
dx.set_xticks([0,60,120,180,240,300,358.5], crs=ccrs.PlateCarree())
dx.set_yticks([0,30,60,90], crs=ccrs.PlateCarree())
lon_formatter = LongitudeFormatter(zero_direction_label='FALSE')
lat_formatter = LatitudeFormatter()
dx.xaxis.set_major_formatter(lon_formatter)
dx.yaxis.set_major_formatter(lat_formatter) 
cbar = fig.add_axes([0.93,0.1,0.01,0.38])
cb = plt.colorbar(dA, cax=cbar, ticks=[-1.5e-4,-1e-4,-0.5e-4,0,0.5e-4, 1e-4,1.5e-4]) 
cb.set_ticklabels(['-1.5','-1','-0.5','0','0.5','1','1.5'])
cb.set_label('moist-induced LWA tendency ($10^{-4}$m/$s^2$)',fontsize=10)


# fig.tight_layout()
plt.savefig("/home/liu3315/Research/Blocking_Diversity/Figure4_new.png",dpi=600)

