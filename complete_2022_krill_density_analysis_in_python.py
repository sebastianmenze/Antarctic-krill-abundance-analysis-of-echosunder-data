# -*- coding: utf-8 -*-
"""
Created on Tue May 10 13:03:21 2022

@author: Administrator
"""


from echolab2.instruments import EK80, EK60

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import glob 
import os

from scipy.ndimage.filters import uniform_filter1d

from scipy.signal import convolve2d
from scipy.interpolate import interp1d

from echopy import transform as tf
from echopy import resample as rs
from echopy import mask_impulse as mIN
from echopy import mask_seabed as mSB
from echopy import get_background as gBN
from echopy import mask_signal2noise as pip
from echopy import mask_range as mRG
from echopy import mask_shoals as mSH
from echopy import mask_signal2noise as mSN

from skimage.transform import rescale, resize 

from pyproj import Geod
geod = Geod(ellps="WGS84")


#%%


os.chdir(r'C:\Users\a5278\Documents\postdoc_krill\2022_cruise\echosounder')

rawfiles= np.sort( glob.glob(r'I:\2022_krill_cruise_EK80\*.raw'  ) )    

rawfile=rawfiles[80]


window_length_sec=60*10
current_ek_length_sec=0

sv=pd.DataFrame([])
track=pd.DataFrame([])


for rawfile in rawfiles:  
        try:
                
           raw_obj = EK80.EK80()
           raw_obj.read_raw(rawfile)
            
           print(raw_obj)
            
           raw_data = raw_obj.raw_data['WBT 998487-15 ES120-7C_ES'][0]
            
           cal_obj = raw_data.get_calibration()
           
           cal_obj.gain=26.88 
           cal_obj.sa_correction=-0.0233
           cal_obj.beam_width_alongship=6.96
           cal_obj.beam_width_athartship=6.63
           cal_obj.angle_offset_alongship=0.04
           cal_obj.angle_offset_athartship=0.06

           
            # Get sv values
           sv_obj = raw_data.get_sv(calibration = cal_obj)
         
           positions = raw_obj.nmea_data.interpolate(sv_obj, 'GGA')[1]
           freq = sv_obj.frequency
            
            # Expand sv values into a 3d object
           # data3d = np.expand_dims(sv_obj.data, axis=0)
           svr = np.transpose( 10*np.log10( sv_obj.data ) )

           r120=np.arange( sv_obj.range.min() , sv_obj.range.max() , 0.5 )
           
           print([sv_obj.range.min() , sv_obj.range.max()])
           
           Sv120=  resize(svr,[ len(r120) , svr.shape[1] ] )
      
           t120=sv_obj.ping_time
     
           Sv120in=Sv120.copy()  
        # -------------------------------------------------------------------------
         # estimate and correct background noise       
           p120           = np.arange(len(t120))                
           s120           = np.arange(len(r120))          
           bn120, m120bn_ = gBN.derobertis(Sv120, s120, p120, 5, 20, r120, np.mean(cal_obj.absorption_coefficient) ) # whats correct absoprtion?
           b=pd.DataFrame(bn120)
           bn120=  b.interpolate(axis=1).interpolate(axis=0).values 
           
           Sv120clean     = tf.log(tf.lin(Sv120in) - tf.lin(bn120))
  
        # plt.figure(0)
        # plt.clf()
        
        # plt.imshow(  Sv120clean ,aspect='auto')
        # plt.clim([-90,-30])
        # plt.colorbar()
           
         # -------------------------------------------------------------------------
         # mask low signal-to-noise 
           m120sn             = mSN.derobertis(Sv120clean, bn120, thr=12)
           Sv120clean[m120sn] = np.nan

        # get mask for seabed
           m120sb = mSB.ariza(Sv120, r120, r0=20, r1=1000, roff=0,
                              thr=-38, ec=1, ek=(3,3), dc=10, dk=(5,15))
           Sv120clean[m120sb]=-999
           
           sv_new=pd.DataFrame( np.transpose(Sv120clean) )
           sv_new.index=pd.to_datetime( t120 )
           sv_new.columns=r120
           sv=pd.concat([ sv , sv_new])
           
           pos=pd.DataFrame(positions)
           pos.index= pos['ping_time']           
           track=pd.concat([ track,pos] )
           
           current_ek_length_sec= (sv.index.max() -  sv.index.min() ).seconds
           if  current_ek_length_sec < window_length_sec:
               continue
           else:
               t120 =sv.index
               r120 =sv.columns.values

               Sv120=  np.transpose( sv.values )
               # get swarms mask
               k = np.ones((3, 3))/3**2
               Sv120cvv = tf.log(convolve2d(tf.lin( Sv120 ), k,'same',boundary='symm'))   
 
               p120           = np.arange(np.shape(Sv120cvv)[1]+1 )                 
               s120           = np.arange(np.shape(Sv120cvv)[0]+1 )           
               m120sh, m120sh_ = mSH.echoview(Sv120cvv, s120, p120, thr=-70,
                                          mincan=(3,10), maxlink=(3,15), minsho=(3,15))

              # -------------------------------------------------------------------------
              # get Sv with only swarms
               Sv120sw =  Sv120.copy()
               Sv120sw[~m120sh] = np.nan
    
               ixdepthvalid= (r120>=20) & (r120<=500)
               Sv120sw[~ixdepthvalid,:]=np.nan
        
               
               cell_thickness=np.abs(np.mean(np.diff( r120) ))               
               nasc_swarm=4*np.pi*1852**2 * np.nansum( np.power(10, Sv120sw /10)*cell_thickness ,axis=0)    
               track['nasc_swarm'] = nasc_swarm
               
               # cell_thickness=np.abs(np.mean(np.diff( r120) ))               
               nasc=4*np.pi*1852**2 * np.nansum( np.power(10, Sv120[ixdepthvalid,:] /10)*cell_thickness ,axis=0)    
               track['nasc'] = nasc               
               track=track.resample('10s').mean()
                
               df_sv=pd.DataFrame( np.rot90(Sv120sw) )
               df_sv.index=t120
               df_sv.columns=r120
               df_sv.to_hdf( rawfile.split('.')[0].split('\\')[-1] +'_swarm_sv.h5', key='df')
                
         
               # plt.figure(num=1)
               # plt.clf()
               # plt.subplot(311)
               # plt.title(rawfile.split('.')[0].split('\\')[-1])
        
               # plt.imshow((Sv120),aspect='auto',extent=[ 0, (t120.max()-t120.min()).seconds , -r120.max(),-r120.min() ] )
               # plt.clim([-82,-50])
               # plt.colorbar()
               
               # plt.subplot(312)
               # plt.imshow((Sv120sw),aspect='auto',extent=[ 0, (t120.max()-t120.min()).seconds , -r120.max(),-r120.min() ] )
               # plt.clim([-82,-50])
               # plt.colorbar()
               
               # plt.subplot(313)
               # # plt.plot(track['nasc'],'-k' )
               # plt.plot(track['nasc_swarm'],'-b')
               # plt.colorbar()
               # plt.xlim([ track.index.min(),  track.index.max() ])
               # plt.tight_layout()
                 
               # plt.savefig(rawfile.split('.')[0].split('\\')[-1] +'_img.jpg' )
 
               track.to_hdf( rawfile.split('.')[0].split('\\')[-1] +'_track_and_nasc.h5', key='df')
               
               sv=pd.DataFrame([])
               track=pd.DataFrame([])


            
           
           del raw_obj
        except:
            print('error')
#%%

files= np.sort( glob.glob('*_track_and_nasc.h5'  ) )    
file=files[0]

df=   pd.DataFrame( [] )

for file in files:  
    
   df_sv=pd.read_hdf(file,index_col=0)
   
   df = pd.concat( [df,  df_sv  ])
   print(df.shape)

df['lon']=df['longitude']
df['lat']=df['latitude']
# # df.to_csv("df_krill_2022_calibrated.csv")              
# df.to_hdf( 'df_krill_2022_calibrated.h5', key='df')

#%%


import numpy as np
from matplotlib import pyplot as plt
import glob
import pandas as pd
import datetime as dt

from scipy.stats import bootstrap

from pyproj import Geod
geod = Geod("+ellps=WGS84")

#%%
df_krill_2022=pd.read_csv(r"df_krill_2022_calibrated.csv",index_col=0)      
df_krill_2022.index=pd.to_datetime(df_krill_2022.index)
df_krill_2022['datetime']=pd.to_datetime(df_krill_2022.index)

nasc_cutoff=20000
ix=(df_krill_2022['nasc_swarm']>nasc_cutoff) 
ix.sum()
df_krill_2022.loc[ix,'nasc_swarm']=np.nan

df_krill_2022['density'] = df_krill_2022['nasc_swarm']*0.3313


from pyproj import Geod
geod = Geod("+ellps=WGS84")
ll=geod.line_lengths(df_krill_2022['lon'], df_krill_2022['lat'])
ll= np.concatenate([np.array([0]), ll ])
timedelta =np.concatenate([np.array([0]), np.diff( df_krill_2022.index.values ) /1e9 ])

df_krill_2022['speed_ms'] = ll / timedelta.astype(float)
df_krill_2022['speed_knots']=df_krill_2022['speed_ms']*1.94384449

ixspeed= (df_krill_2022['speed_knots']>=5) & (df_krill_2022['speed_knots']<=15) 
# ixspeed= (df_krill_2022['speed_knots']>=5) & (df_krill_2022['speed_knots']<=15) & (df_krill_2022.index.hour>=6) & (df_krill_2022.index.hour<=18)


#%%
ixtransetime= np.zeros(len( df_krill_2022))

tlims=[pd.Timestamp('2022-02-12 17:21:40'), pd.Timestamp('2022-02-13 17:56:00')]
ixtransetime[ (df_krill_2022.index >=tlims[0]) & (df_krill_2022.index <=tlims[1]) ] =1

tlims=[pd.Timestamp('2022-02-14 00:00:30'), pd.Timestamp('2022-02-14 09:58:50')]
ixtransetime[ (df_krill_2022.index >=tlims[0]) & (df_krill_2022.index <=tlims[1]) ] =1

tlims=[pd.Timestamp('2022-02-14 17:50:20'), pd.Timestamp('2022-02-15 02:43:10')]
ixtransetime[ (df_krill_2022.index >=tlims[0]) & (df_krill_2022.index <=tlims[1]) ] =1

tlims=[pd.Timestamp('2022-02-15 04:38:10'), pd.Timestamp('2022-02-15 13:36:30')]
ixtransetime[ (df_krill_2022.index >=tlims[0]) & (df_krill_2022.index <=tlims[1]) ] =1

tlims=[pd.Timestamp('2022-02-15 15:44:30'), pd.Timestamp('2022-02-15 20:36:10')]
ixtransetime[ (df_krill_2022.index >=tlims[0]) & (df_krill_2022.index <=tlims[1]) ] =1

tlims=[pd.Timestamp('2022-02-15 23:00:30'), pd.Timestamp('2022-02-16 14:44:10')]
ixtransetime[ (df_krill_2022.index >=tlims[0]) & (df_krill_2022.index <=tlims[1]) ] =1

tlims=[pd.Timestamp('2022-02-16 17:48:40'), pd.Timestamp('2022-02-17 11:32:30')]
ixtransetime[ (df_krill_2022.index >=tlims[0]) & (df_krill_2022.index <=tlims[1]) ] =1



fname=r"C:\Users\a5278\Documents\postdoc_krill\2022_cruise\mmobs\transects_north_south.xlsx"
tran=pd.read_excel(fname)
tran['id']=np.repeat(np.arange(len(tran)/2), 2)

ix_transect=np.ones(len(df_krill_2022)) * np.nan

for tid in np.unique(tran['id']):
      
    i,j = np.where(tran['id']==tid)[0]
    ix_mm =  ( np.abs(df_krill_2022['longitude']-tran.loc[i,'Longitude'])<0.1 ) &   ( df_krill_2022['latitude'] >=tran.loc[i,'Latitude'] ) & \
        ( df_krill_2022['latitude'] <=tran.loc[j,'Latitude'] ) & ixspeed & ixtransetime &  df_krill_2022['nasc_swarm'].notna()
    ix_transect[ix_mm] = tid

#%%

    
area_all, poly_perimeter = geod.polygon_area_perimeter([-44.00,-47.50,-47.50,-44.00], [-62.00,-62.00,-59.666667,-59.666667])
area_north, poly_perimeter = geod.polygon_area_perimeter([-44.00,-47.50,-47.50,-44.00], [ -60.600, -60.600,-59.666667,-59.666667])
area_south, poly_perimeter = geod.polygon_area_perimeter([-44.00,-47.50,-47.50,-44.00], [-62.00,-62.00,-60.600, -60.600])


#%% study area

rho_j=[]
length=[]

ll=geod.line_lengths(df_krill_2022['lon'], df_krill_2022['lat'])
ll= np.concatenate([np.array([0]), ll ])
rho_j=[]
length=[]
x=np.array([])
for tid in np.unique(tran['id']):
    ix= ix_transect==tid
    data= df_krill_2022.loc[ix,'nasc_swarm']
    x=np.concatenate([x,data.values])
    rho_j.append( data.mean() )
    print( data.mean()) 
    length.append( np.sum( ll[ix] ) )

w=   length / np.mean(length)

diravg=np.mean( x ) 

res =  bootstrap((x,), np.mean, confidence_level=0.95)
val=np.array([res.confidence_interval[0],res.confidence_interval[1]])

rho_k=np.average( rho_j,weights=w ) 
np.mean( rho_j * w )

var_k= np.sum(  w**2 *(rho_j - rho_k)**2 ) / (len(rho_j) * (len(rho_j)-1))

cv_k = 100 * np.sqrt(var_k) / rho_k



tbl=np.concatenate( [np.array([rho_k , var_k ,cv_k,diravg]),   val]) 
print(tbl.round(1) )
print((tbl*0.3313).round(1) )
    

biomass_tons=rho_k*0.3313*np.abs( area_all ) / (1e6)

biomass_tons.round(0)

print( val*0.3313*np.abs( area_all ) / (1e6) )
#%% south
x=np.array([])
rho_j=[]
length=[]
for tid in np.arange(0,10,2):
    ix= ix_transect==tid
    data= df_krill_2022.loc[ix,'nasc_swarm']
    rho_j.append( data.mean() )
    # print( data.mean()) 
    length.append( np.sum( ll[ix] ) )
    x=np.concatenate([x,data.values])

w=   length / np.mean(length)

rho_k=np.average( rho_j,weights=w ) 
print( rho_k )

var_k= np.sum(  w**2 *(rho_j - rho_k)**2 ) / (len(rho_j) * (len(rho_j)-1))

cv_k = 100 * np.sqrt(var_k) / rho_k

np.mean( rho_j ) 
diravg=np.mean( x ) 

res =  bootstrap((x,), np.mean, confidence_level=0.95)
val=np.array([res.confidence_interval[0],res.confidence_interval[1]])

tbl=np.concatenate( [np.array([rho_k , var_k ,cv_k,diravg]),   val]) 
print(tbl.round(1) )
print((tbl*0.3313).round(1) )

biomass_tons=rho_k*0.3313*np.abs( area_south ) / (1e6)

print( val*0.3313*np.abs( area_south ) / (1e6) )

#%% north
x=np.array([])
rho_j=[]
length=[]
for tid in np.arange(1,10,2):
    ix= ix_transect==tid
    data= df_krill_2022.loc[ix,'nasc_swarm']
    rho_j.append( data.mean() )
    # print( data.mean()) 
    length.append( np.sum( ll[ix] ) )
    x=np.concatenate([x,data.values])

w=   length / np.mean(length)

rho_k=np.average( rho_j,weights=w ) 
print( rho_k )


var_k= np.sum(  w**2 *(rho_j - rho_k)**2 ) / (len(rho_j) * (len(rho_j)-1))

cv_k = 100 * np.sqrt(var_k) / rho_k

np.mean( rho_j ) 

res =  bootstrap((x,), np.mean, confidence_level=0.95)
val=np.array([res.confidence_interval[0],res.confidence_interval[1]])

diravg=np.mean( x ) 
tbl=np.concatenate( [np.array([rho_k , var_k ,cv_k,diravg]),   val]) 
print(tbl.round(1) )
print((tbl*0.3313).round(1) )

biomass_tons=rho_k*0.3313*np.abs( area_north ) / (1e6)
print( val*0.3313*np.abs( area_north ) / (1e6) )


from netCDF4 import Dataset
import pandas as pd
import cartopy
import cartopy.crs as ccrs
import numpy as np
import matplotlib.pyplot as plt
import utm

# load data and slice out region of interest

# read mapdata



latlim=[-62.1,-59.5]
lonlim=[-48.5,-43.5]

spacer=1
gebcofile=r"C:\Users\a5278\Documents\gebco_2020_netcdf\GEBCO_2020.nc"
gebco = Dataset(gebcofile, mode='r')
g_lons = gebco.variables['lon'][:]
g_lon_inds = np.where((g_lons>=lonlim[0]) & (g_lons<=lonlim[1]))[0]
# jump over entries to reduce data
g_lon_inds=g_lon_inds[::spacer]

g_lons = g_lons[g_lon_inds].data
g_lats = gebco.variables['lat'][:]
g_lat_inds = np.where((g_lats>=latlim[0]) & (g_lats<=latlim[1]))[0]
# jump over entries to reduce data
g_lat_inds=g_lat_inds[::spacer]

g_lats = g_lats[g_lat_inds].data
d = gebco.variables['elevation'][g_lat_inds, g_lon_inds].data
gebco.close()

def scale_bar(ax, length=None, location=(0.5, 0.05), linewidth=3,scalecolor='k'):
    """
    EXAMPLE: scale_bar(ax, length=100,location=(0.8, 0.05))

    ax is the axes to draw the scalebar on.
    length is the length of the scalebar in km.
    location is center of the scalebar in axis coordinates.
    (ie. 0.5 is the middle of the plot)
    linewidth is the thickness of the scalebar.
    """
    #Get the limits of the axis in lat long
    llx0, llx1, lly0, lly1 = ax.get_extent(ccrs.PlateCarree())
    #Make tmc horizontally centred on the middle of the map,
    #vertically at scale bar location
    sbllx = (llx1 + llx0) / 2
    sblly = lly0 + (lly1 - lly0) * location[1]
    tmc = ccrs.TransverseMercator(sbllx, sblly)
    #Get the extent of the plotted area in coordinates in metres
    x0, x1, y0, y1 = ax.get_extent(tmc)
    #Turn the specified scalebar location into coordinates in metres
    sbx = x0 + (x1 - x0) * location[0]
    sby = y0 + (y1 - y0) * location[1]

    #Calculate a scale bar length if none has been given
    #(Theres probably a more pythonic way of rounding the number but this works)
    if not length: 
        length = (x1 - x0) / 5000 #in km
        ndim = int(np.floor(np.log10(length))) #number of digits in number
        length = round(length, -ndim) #round to 1sf
        #Returns numbers starting with the list
        def scale_number(x):
            if str(x)[0] in ['1', '2', '5']: return int(x)        
            else: return scale_number(x - 10 ** ndim)
        length = scale_number(length) 

    #Generate the x coordinate for the ends of the scalebar
    bar_xs = [sbx - length * 500, sbx + length * 500]
    #Plot the scalebar
    ax.plot(bar_xs, [sby, sby], transform=tmc, color=scalecolor, linewidth=linewidth)
    #Plot the scalebar label
    ax.text(sbx, sby, str(length) + ' km', transform=tmc, color=scalecolor,
            horizontalalignment='center', verticalalignment='bottom')


#%%

fig=plt.figure(num=5)
plt.clf()
fig.set_size_inches(7,7)

central_lon= lonlim[0]+(lonlim[1]-lonlim[0])/2
central_lat = latlim[0]+(latlim[1]-latlim[0])/2
extent = [lonlim[0],lonlim[1], latlim[0],latlim[1]]
#ax = plt.axes(projection=ccrs.PlateCarree(central_longitude= lonlim[0]+(lonlim[1]-lonlim[0])/2 ))

ax = plt.axes(projection=ccrs.Orthographic(central_lon, central_lat),aspect='auto')

ax.set_extent(extent)
  
ax.gridlines(draw_labels=True)
#ax.coastlines(resolution='50m')
#ax.add_feature(cartopy.feature.LAND)


d_plot=d
d_plot[d<-4000]=-4000

plt.contourf(g_lons, g_lats, d_plot, np.arange(-4000,0,100),cmap='Blues_r',
                  linestyles=None, transform=ccrs.PlateCarree())
CS=plt.contour(g_lons, g_lats, d, [-2000,-1000,-500,-200,-100,-50],colors='k',linewidth=.1,
                  linestyles='-', transform=ccrs.PlateCarree())
plt.clabel(CS, inline=True, fontsize=10, fmt='%i')


CS=plt.contourf(g_lons, g_lats, d, [0,8000],colors='silver',linewidth=1,
                  linestyles='-', transform=ccrs.PlateCarree())

CS=plt.contour(g_lons, g_lats, d, [0],colors='k',linewidth=1,
                  linestyles='-', transform=ccrs.PlateCarree())


scale_bar(ax, length=100,location=(0.5, 0.05),scalecolor='silver')

kkk=~np.isnan(ix_transect)
plt.plot(df_krill_2022.loc[kkk,'longitude'],df_krill_2022.loc[kkk,'latitude'],'.',color='orange', transform=ccrs.PlateCarree())

kkk=~np.isnan(ix_transect)

# plt.scatter(df_krill_2022.loc[kkk,'longitude'],df_krill_2022.loc[kkk,'latitude'],s=df_krill_2022.loc[kkk,'nasc_swarm']/100,c=ix_transect[kkk],edgecolor='k',cmap='rainbow',zorder=200, transform=ccrs.PlateCarree())

sc=plt.scatter(df_krill_2022.loc[kkk,'longitude'],df_krill_2022.loc[kkk,'latitude'],s=df_krill_2022.loc[kkk,'nasc_swarm']/50,c='r',edgecolor='k',cmap='rainbow',zorder=200, transform=ccrs.PlateCarree())
plt.legend(*sc.legend_elements("sizes",color='r',markeredgecolor='k',func=lambda x:x*50, num=5))

plt.tight_layout()
# plt.savefig('krill_nasc_transect_data_2022.jpg',dpi=200)
