"""
A script to process input data and create time series graphics for the AntClimNow Antarctic Climate Indicators project (see https://scar.org/science/research-programmes/antclimnow/climate-indicators).
Authors: Tom Bracegirdle, Tania Glazkova ...
Email: tjbra@bas.ac.uk
Date: 12 December 2024

The code has been developed and tested in a conda environment running Python v3.10 with the following packages installed:
ipython, ipykernel, iris (v3.11), nc-time-axis, netcdf4, pandas, scipy, seaborn and xarray

The script requires three arguments: 
diag_name [currently accepted options are: JSI_sh_ERA5', 'JLI_sh_ERA5', 'SMB_MARv3.12', 'SMB_HIRHAM5', 'SMB_all', 'zw3index_SH_mag_ERA5', 'zw3index_SH_phase_ERA5', 'U_10hPa_55-65S', 'Ross_gyre', 'Weddell_gyre', 'ACC_transport', 'blockingP90_150–90W', 'AR_eant_IWV', 'AR_eant_IVT', 'AR_want_IWV', 'AR_want_IVT' and 'SAM_marshall']
mean_period [currently accepted options are 'monthly' or 'seasonal']
update_data ['1' to update and '0' to not update]

An example of running the script in ipython: 
In [1]: run aci_plot.py 'SAM_marshall' 'monthly' 0

"""
import iris  
import iris.plot as iplt
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from matplotlib.offsetbox import (TextArea, DrawingArea, OffsetImage,
                                  AnnotationBbox)
from matplotlib.cbook import get_sample_data
import iris.coord_categorisation
import os, sys
import netCDF4 as nc
from netCDF4 import Dataset  
import datetime as dt
from datetime import datetime
from netCDF4 import date2num,num2date
import cftime
from matplotlib.dates import DateFormatter, YearLocator
import nc_time_axis
import pandas as pd
import xarray as xr
import seaborn as sns

### --------- Definitions ------------

def yrange_calc(cube):
  d_max = np.max(cube.data)
  d_min = np.min(cube.data)
  ylim_buffer = .1*(d_max-d_min)
  y_range = [d_min - ylim_buffer, d_max+ylim_buffer]
  return y_range

def season_agg(cube, seasons=None, sea_agg_type=None):
    ### Create seasonal means (unique and specified 3-month seasons)
  if (seasons == None): 
        seasons=['mam', 'jja', 'son', 'djf']
    
  if len(cube.coords('clim_season')) == 0:
        iris.coord_categorisation.add_season(cube, 'time', name='clim_season', seasons=seasons)

  if len(cube.coords('season_year')) == 0:
        iris.coord_categorisation.add_season_year(cube, 'time', name='season_year', seasons=seasons)

    # Keep only those times where we can produce seasonal means using exactly 3 months
    # (i.e., remove times from cubelist where only 1 or 2 times exist for that season)
  clim_seasons = cube.coords('clim_season')[0].points
  season_years = cube.coords('season_year')[0].points
  ntimes = len(cube.coords('time')[0].points)    
  keep_ind = np.zeros((0), dtype=int)
  for i in range(0,ntimes):
    ind = np.where( (clim_seasons == clim_seasons[i]) & (season_years == season_years[i]) )[0]
    n_months_in_season = len(clim_seasons[i]) # length of string, usually 3 (e.g., 'djfm' = 4)
    if (len(ind) == n_months_in_season): keep_ind = np.append(keep_ind,i)

  cube = cube[keep_ind]
  if (sea_agg_type == 'mean'): cube_out = cube.aggregated_by(['clim_season', 'season_year'], iris.analysis.MEAN)
  if (sea_agg_type == 'total'): cube_out = cube.aggregated_by(['clim_season', 'season_year'], iris.analysis.SUM)
  return cube_out


def blocking_days_count(cube, diag_name, fname_in, nctitle, ncvarname, ncvar_standard_name):
  print('fname = '+fname)
  ymin = min(cube.coord('year').points)
  ymax = max(cube.coord('year').points)
  yr_arr = ymin + np.arange(1+ymax-ymin)
  nyrs = len(yr_arr)
  mon_arr = 1+np.arange(12)
  try: ncfile.close()  # just to be safe, make sure dataset is not already open.
  except: pass
  ncfile = Dataset('t_srs/blocking/'+fname,mode='w',format='NETCDF4_CLASSIC') 
  #print(ncfile)
  time_dim = ncfile.createDimension('time', None)
  ncfile.title = nctitle
  time = ncfile.createVariable('time', np.float64, ('time',))
  time.units = 'hours since 1800-01-01'
  time.long_name = 'time'
# Define a 3D variable to hold the data
  var = ncfile.createVariable(ncvarname,np.float64,('time')) # note: unlimited dimension is leftmost
  var.units = 'hours' # no units
  var.standard_name = ncvar_standard_name # this is a CF standard name
  calc_var = np.array([])
  dates = np.array([])
  for i in range(0,nyrs):
    for j in range(0,12):      
      dates = np.append(dates,dt.datetime(yr_arr[i],mon_arr[j],15,0)) # pick middle month and ay of SON (month 10, day 15)  
      ii_yr = np.where(cube.coord('year').points == yr_arr[i])      
      if len(ii_yr[0]) > 0: 
        ii_yr_mon = np.where(cube[ii_yr].coord('month_number').points == mon_arr[j])
        calc_var = np.append(calc_var, len(ii_yr_mon[0]))
      else:
        calc_var = np.append(calc_var, 0)
      
  times = date2num(dates, time.units)
  time[:] = times
  var[:] = calc_var
  ncfile.close(); print('Dataset is closed!')  


def low_pass_weights(window, cutoff):
  #"""Calculate weights for a low pass Lanczos filter.
  #Args:
  #window: int
      #The length of the filter window.
  #cutoff: float
      #The cutoff frequency in inverse time steps.
  #"""
  order = ((window - 1) // 2 ) + 1
  nwts = 2 * order + 1
  w = np.zeros([nwts])
  n = nwts // 2
  w[n] = 2 * cutoff
  k = np.arange(1., n)
  sigma = np.sin(np.pi * k / n) * n / (np.pi * k)
  firstfactor = np.sin(2. * np.pi * cutoff * k) / (np.pi * k)
  w[n-1:0:-1] = firstfactor * sigma
  w[n+1:-1] = firstfactor * sigma
  return w[1:-1]


#---Defininhs the y-axxis of the plots

### Arguments (diagnostics name, season months)
diag_name = sys.argv[1]
mean_period = sys.argv[2]
update_data = float(sys.argv[3])

aci_version_num = '1.0'
aci_version = 'ACI beta v'+aci_version_num
nsource = 1 ## default number of data sources is one
plt_date = '20240612'

if diag_name == 't2m_ERA5':
  sea_agg_calc = 1
  sea_agg_type_in = 'mean'
  vn='20250121'
  fname = diag_name+'_monthly_'+vn+'.nc'
  diag_in = iris.load_cube('t_srs/t2m/'+fname,'area_weighted_time_series')
  diag_plt = iris.cube.CubeList()
  diag_plt.append(diag_in)
  nmon_clim = 12
  p_title = 'Antarctic continent surface air temperature'
  source_lab = ['ERA5']
  ylab_units = '$\degree$C'
  iris.coord_categorisation.add_year(diag_plt[0],'time', 'year')
  iris.coord_categorisation.add_month_number(diag_plt[0], 'time', 'month_number')
  yrs_max = max(diag_plt[0].coord('year').points)
  diag_plt[0]=diag_plt[0].extract(iris.Constraint(year = range(1979,yrs_max+1)))

if diag_name == 'JSI_sh_ERA5': 
  sea_agg_calc = 1
  sea_agg_type_in = 'mean'
  vn='20240607'
  fname = diag_name+'_monthly_'+vn+'.nc'
  diag_plt = iris.cube.CubeList()
  diag_plt.append(iris.load_cube('t_srs/jetdiags/'+fname))
  nmon_clim = 12
  p_title = 'Southern Hemisphere westerly jet speed index (JSI)'
  source_lab = ['ERA5']
  ylab_units = 'm s$^{-1}$'
  yrs_max = max(diag_plt[0].coord('year').points)
  diag_plt[0]=diag_plt[0].extract(iris.Constraint(year = range(1979,yrs_max+1)))

if diag_name == 'JLI_sh_ERA5': 
  sea_agg_calc = 1
  sea_agg_type_in = 'mean'
  vn='20240607'
  fname = diag_name+'_monthly_'+vn+'.nc'
  diag_plt = iris.cube.CubeList()  
  diag_plt.append(iris.load_cube('t_srs/jetdiags/'+fname))
  nmon_clim = 12
  p_title = 'Southern Hemisphere westerly jet latitude index (JLI)'
  source_lab = ['ERA5']
  ylab_units = 'degrees latitude'  
  yrs_max = max(diag_plt[0].coord('year').points)  
  diag_plt[0]=diag_plt[0].extract(iris.Constraint(year = range(1979,yrs_max+1)))

if diag_name == 'SMB_MARv3.12':
  sea_agg_calc=1
  sea_agg_type_in = 'mean'
  vn='20230113'
  #fname = diag_name+'_'+season+'_'+vn+'.nc'
  fname = diag_name+'_mon_'+vn+'.nc'
  diag_plt = iris.cube.CubeList()
  diag_plt.append(iris.load_cube('t_srs/SMB/MAR/'+fname))
  diag_plt[0].coord('TIME').var_name = 'time' # Change the name of the time coordinate
  iris.coord_categorisation.add_month_number(diag_plt[0], 'time', 'month_number')
  iris.coord_categorisation.add_year(diag_plt[0],'time', 'year')
  nmon_clim = 12
  p_title = 'Antarctic SMB'
  source_lab = ['MARv3.12']
  ylab_units = 'SMB (Gt)'
  
if diag_name == 'SMB_HIRHAM5':
  sea_agg_calc=1
  sea_agg_type_in = 'mean'
  vn='20240306'
  if update_data == 1:
    datadirModel=os.path.join('t_srs/SMB/HIRHAM5/SMBdata2Tom_MM/')
    xlim_min=1960
    xlim_max=2023
    xval=np.arange(xlim_min,xlim_max)
    gt=1000000 #convert to GT
    surface_data_smb = nc.MFDataset(datadirModel+'ATF-TRANS_SMB_*_MM.nc','r')
    cubelist_in = iris.load(datadirModel+'ATF-TRANS_SMB_*_MM.nc')
    for i in range(0, 64): del cubelist_in[i].attributes['history'] 
    cube_in = cubelist_in.concatenate_cube()
    smb = cube_in.data 
  #  smb = surface_data_smb.variables['smb'][:]
    smb = np.squeeze(smb)
    Agrid_load = nc.MFDataset(datadirModel+'Agrid.nc','r') 
    GAIS_load = xr.open_dataset(datadirModel+'Basins_HIRHAM5H_ANT_IMBIE2_Full.nc') #Grounded ice sheet
    # Variable from multiple files.
    Agrid = Agrid_load.variables['Agrid'][:]
    Agrid = np.squeeze(Agrid)
    GAIS = GAIS_load['GAIS'][:].data
    GAIS = np.squeeze(GAIS)
    GAIS[GAIS < -1 ]= np.NaN  
    smb_GT = smb * Agrid /gt
    smb_GT_sum= np.nansum(smb_GT,axis=(1, 2))
    smb_GT_GAIS = (smb * Agrid /gt)*GAIS
    smb_GT_sum_GAIS= np.nansum(smb_GT_GAIS,axis=(1, 2))
    cube_t = cube_in.collapsed(['longitude'], iris.analysis.MEAN)
  #  cube_t = cube_t.collapsed(['latitude'], iris.analysis.MEAN)
    cube_t = iris.util.squeeze(cube_t)
    cube_t.data = smb_GT_sum_GAIS
    cube_t.rename('GAIS')
    cube_t.units = 'Gt'
    iris.save(cube_t, 't_srs/SMB/HIRHAM5/GAIS_HIRHAM5.nc')
    
  diag_plt = iris.cube.CubeList()
  diag_plt.append(iris.load_cube('t_srs/SMB/HIRHAM5/GAIS_HIRHAM5.nc'))
  iris.coord_categorisation.add_month_number(diag_plt[0], 'time', 'month_number')
  iris.coord_categorisation.add_year(diag_plt[0],'time', 'year')
  nmon_clim = 12
  p_title = 'Antarctic SMB'
  source_lab = ['HIRHAM5']
  ylab_units = 'SMB (Gt)'
  yrs_max = max(diag_plt[0].coord('year').points)
  diag_plt[0]=diag_plt[0].extract(iris.Constraint(year = range(1979,yrs_max+1)))

if diag_name == 'SMB_all':
  sea_agg_calc=1
  sea_agg_type_in = 'mean'
  nsource = 2
  vn_HIRHAM = '20240306'
  vn_MAR = '20230113'
  vn = '20240310'
  diag_plt = iris.cube.CubeList()
  fname = diag_name+'_mon_'+vn+'.nc'
  diag_plt.append(iris.load_cube('t_srs/SMB/MAR/SMB_MARv3.12_mon_'+vn_MAR+'.nc'))
  diag_plt.append(iris.load_cube('t_srs/SMB/HIRHAM5/GAIS_HIRHAM5.nc'))
  diag_plt[0].coord('TIME').var_name = 'time' # Change the name of the time coordinate
  for i in range(0,nsource):
    iris.coord_categorisation.add_month_number(diag_plt[i], 'time', 'month_number')
    iris.coord_categorisation.add_year(diag_plt[i],'time', 'year')
    yrs_max = max(diag_plt[i].coord('year').points)
    diag_plt[i]=diag_plt[i].extract(iris.Constraint(year = range(1979,yrs_max+1)))
  
  nmon_clim = 12
  p_title = 'Antarctic SMB'
  source_lab = ['MARv3.12', 'HIRHAM5']
  ylab_units = 'SMB (Gt)'

if diag_name == 'zw3index_SH_mag_ERA5':
  sea_agg_type_in = 'mean'
  vn='20230216'
  nsource = 1
  sea_agg_calc = 2
  #if season == 'djf': month = 'Jan'
  #if season == 'jja': month = 'Jul'
  #fname = diag_name+'_'+month+'_'+vn+'.nc'
  if mean_period == 'monthly': 
    fname = 'zw3index_SH_ERA5_197901-202212_months_20230216.nc'
    diag_plt = iris.load('t_srs/ZW3/'+fname,'zw3index_magnitude')
    iris.coord_categorisation.add_month_number(diag_plt[0], 'time', 'month_number')
    iris.coord_categorisation.add_year(diag_plt[0],'time', 'year')
    
  if mean_period == 'seasonal': 
    fname = 'zw3index_SH_ERA5_*_20230627.nc'
    diag_plt = iris.load('t_srs/ZW3/'+fname, 'zw3index_magnitude')
    for i in range(0, 4): iris.coord_categorisation.add_month_number(diag_plt[i], 'time', 'month_number')
    for i in range(0, 4): iris.coord_categorisation.add_year(diag_plt[i],'time', 'year')
  
  nmon_clim = 12
  p_title = 'SH Zonal wavenumber 3'
  source_lab = ['ERA5']
  ylab_units = 'ZW3 Magnitude'
  
if diag_name == 'zw3index_SH_phase_ERA5':
  sea_agg_type_in = 'mean'
  vn='20230216'
  nsource = 1
  sea_agg_calc = 2
  #if season == 'djf': month = 'Jan'
  #if season == 'jja': month = 'Jul'
  #fname = diag_name+'_'+month+'_'+vn+'.nc'
  if mean_period == 'monthly': 
    fname = 'zw3index_SH_ERA5_197901-202212_months_20230216.nc'
    diag_plt = iris.load('t_srs/ZW3/'+fname,'zw3index_phase')
    iris.coord_categorisation.add_month_number(diag_plt[0], 'time', 'month_number')
    iris.coord_categorisation.add_year(diag_plt[0],'time', 'year')
    
  if mean_period == 'seasonal': 
    fname = 'zw3index_SH_ERA5_*_20230627.nc'
    diag_plt = iris.load('t_srs/ZW3/'+fname, 'zw3index_phase')
    for i in range(0, 4): iris.coord_categorisation.add_month_number(diag_plt[i], 'time', 'month_number')
    for i in range(0, 4): iris.coord_categorisation.add_year(diag_plt[i],'time', 'year')
  
  nmon_clim = 12
  p_title = 'SH Zonal wavenumber 3'
  source_lab = ['ERA5']
  ylab_units = 'ZW3 Phase'
  

if diag_name == 'U_10hPa_55-65S':
  vn = '20230113'
  nsource = 1
  sea_agg_calc = 0
  fname_csv = 'PolarVortexStrength_son_SH_AntClimNow_post1979.csv'
  fname = diag_name+'_son_'+vn+'.nc'
  pvtx_in=np.genfromtxt('t_srs/pvortex/'+fname_csv, delimiter=',',skip_header=1)
  pvtx_yrs = pvtx_in[:,0]
  pvtx_dat_tsrs = pvtx_in[:,1]
  ntims = len(pvtx_dat_tsrs)
  yr_arr_tsrs = pvtx_yrs.astype(int)  
  try: ncfile.close()  # just to be safe, make sure dataset is not already open.
  except: pass
  ncfile = Dataset('t_srs/pvortex/'+fname,mode='w',format='NETCDF4_CLASSIC') 
  #print(ncfile)
  time_dim = ncfile.createDimension('time', None)
  ncfile.title='Southern polar vortex strength at 10 hPa'
  time = ncfile.createVariable('time', np.float64, ('time',))
  time.units = 'hours since 1800-01-01'
  time.long_name = 'time'
# Define a 3D variable to hold the data
  var = ncfile.createVariable('pvtx',np.float64,('time')) # note: unlimited dimension is leftmost
  var.units = '' # no units
  var.standard_name = 'PVTX' # this is a CF standard name
  var[:] = pvtx_dat_tsrs
  dates = np.array([])
  for i in range(0,ntims): 
    dates=np.append(dates,dt.datetime(yr_arr_tsrs[i],10,15,0)) # pick middle month and ay of SON (month 10, day 15)  
  times = date2num(dates, time.units)
  time[:] = times
  ncfile.close(); print('Dataset is closed!')  
  diag_plt = iris.cube.CubeList()  
  diag_plt.append(iris.load_cube('t_srs/pvortex/'+fname))
  iris.coord_categorisation.add_month_number(diag_plt[0], 'time', 'month_number')
  iris.coord_categorisation.add_year(diag_plt[0],'time', 'year')
  nmon_clim = 1
  p_title = 'Southern polar vortex strength at 10 hPa (SON)'
  source_lab = ['ERA5']
  ylab_units = 'm/s'

if (diag_name == 'Ross_gyre') | (diag_name == 'Weddell_gyre') | (diag_name == 'ACC_transport'):
  vn = '20240322'
  nsource = 1
  sea_agg_calc = 1
  sea_agg_type_in = 'mean'
  fname_csv = 'AntClimNow_Ocean_ACI.csv'
  fname = diag_name+'_monthly_'+vn+'.nc'  
  if update_data == 1:
    OCN_df = pd.read_csv('t_srs/Ocean/'+fname_csv, parse_dates=['time'])
    OCN_df['time'] = pd.to_datetime(OCN_df['time'], format = '%Y-%m-%d')  
    OCN_ds = OCN_df.set_index(['time']).to_xarray()
    OCN_ds.to_netcdf('t_srs/Ocean/AntClimNow_Ocean_ACI.nc')
  
  diag_plt = iris.load('t_srs/Ocean/AntClimNow_Ocean_ACI.nc',diag_name)
  iris.coord_categorisation.add_month_number(diag_plt[0], 'time', 'month_number')
  iris.coord_categorisation.add_year(diag_plt[0],'time', 'year')
  nmon_clim = 12
  p_title = diag_name
  source_lab = ['ORAS5']
  ylab_units = 'Sv'

if diag_name == 'blockingP90_150–90W':
  fname_in = diag_name+'_50–70S_ERA5_annual_20240404_vf.nc'
  diag_in = iris.load('t_srs/blocking/'+fname_in)
  nmon_clim = 12
  vn = '20240404'
  nsource = 1
  iris.coord_categorisation.add_year(diag_in[0], 'time', 'year')
  iris.coord_categorisation.add_month_number(diag_in[0], 'time', 'month_number')
  sea_agg_calc = 1
  sea_agg_type_in = 'total'
  fname = diag_name+'_50–70S_ERA5_anndays_'+vn+'.nc'  
  ncvarname = 'blocking_P90_150-90W'
  nctitle = 'blocking_days_P90_150-90W'
  ncvar_standard_name = 'blocking_days'
  blocking_days_count(diag_in[0], diag_name, fname, nctitle, ncvarname, ncvar_standard_name) 
  diag_plt = iris.cube.CubeList()
  diag_plt.append(iris.load_cube('t_srs/blocking/'+fname))  
  iris.coord_categorisation.add_month_number(diag_plt[0], 'time', 'month_number')
  iris.coord_categorisation.add_year(diag_plt[0], 'time', 'year')
  iris.coord_categorisation.add_season(diag_plt[0], 'time', 'clim_season')
  nmon_clim = 12
  p_title = 'Blocking days 150-90W P90'
  source_lab = ['ERA5']
  if mean_period == 'monthly': ylab_units = 'Days per month'
  if mean_period == 'seasonal': ylab_units = 'Days per season'

if diag_name == 'AR_eant_IWV':
  vn = '20230716'
  nsource = 1
  sea_agg_calc = 1
  sea_agg_type_in = 'total'
  fname_csv = 'east_ant_ARs_lons_IWV_80-22.csv'
  fname = diag_name+'_ann_'+vn+'.nc'
  #AR_df = pd.read_csv('t_srs/ARs/'+fname_csv, dtype = np.float64, parse_dates=['time'])
  AR_df = pd.read_csv('t_srs/ARs/'+fname_csv, parse_dates=['time'])
  AR_df['time'] = pd.to_datetime(AR_df['time'], format = '%d/%m/%Y %H:%M')
  AR_df = AR_df.rename(columns = {'Median_Longitude_(66-72S)': 'Longitude'})
  AR_ds = AR_df.set_index(['time']).to_xarray()
  AR_ds.to_netcdf('t_srs/ARs/'+fname)
  AR_cubelist = iris.load('t_srs/ARs/'+fname)
  #AR_df=pd.read_csv('t_srs/ARs/'+fname_csv, parse_dates=['time'])
  #AR_ds = AR_df.set_index(['time']).to_xarray()
  #AR_ds.to_netcdf('t_srs/ARs/'+fname)
  #AR_cubelist = iris.load('t_srs/ARs/'+fname)
  IVT_cube = AR_cubelist[0]
  iris.coord_categorisation.add_month(IVT_cube, 'time', 'month')
  iris.coord_categorisation.add_month_number(IVT_cube, 'time', 'month_number')
  iris.coord_categorisation.add_year(IVT_cube, 'time', 'year')
  nyrs=44
  yr_arr = 1979+np.arange(44)
  mon_arr = 1+np.arange(12)
  nmon = len(yr_arr)*12
  try: ncfile.close()  # just to be safe, make sure dataset is not already open.
  except: pass
  ncfile = Dataset('t_srs/ARs/'+fname,mode='w',format='NETCDF4_CLASSIC') 
  #print(ncfile)
  time_dim = ncfile.createDimension('time', None)
  ncfile.title='West_Antarctic_AR_residence_time'
  time = ncfile.createVariable('time', np.float64, ('time',))
  time.units = 'hours since 1800-01-01'
  time.long_name = 'time'
# Define a 3D variable to hold the data
  var = ncfile.createVariable('WA_AR_residence',np.float64,('time')) # note: unlimited dimension is leftmost
  var.units = 'hours' # no units
  var.standard_name = 'AR_accumulated_time' # this is a CF standard name
  calc_var = np.array([])
  dates = np.array([])
  for i in range(0,nyrs):
    for j in range(0,12):      
      dates = np.append(dates,dt.datetime(yr_arr[i],mon_arr[j],15,0)) # pick middle month and ay of SON (month 10, day 15)  
      ii_yr = np.where(IVT_cube.coord('year').points == yr_arr[i])      
      if len(ii_yr[0]) > 0: 
        ii_yr_mon = np.where(IVT_cube[ii_yr].coord('month_number').points == mon_arr[j])
        calc_var = np.append(calc_var, (3/24)*len(ii_yr_mon[0]))
      else:
        calc_var = np.append(calc_var, 0)
      
  times = date2num(dates, time.units)
  time[:] = times
  var[:] = calc_var
  ncfile.close(); print('Dataset is closed!')  
  diag_plt = iris.cube.CubeList()
  diag_plt.append(iris.load_cube('t_srs/ARs/'+fname))
  iris.coord_categorisation.add_month_number(diag_plt[0], 'time', 'month_number')
  iris.coord_categorisation.add_year(diag_plt[0],'time', 'year')
  nmon_clim = 12
  p_title = 'Accumulated East Antarctic AR residence time (IWV method)'
  source_lab = ['ERA5']
  if mean_period == 'monthly': ylab_units = 'Days per month'
  if mean_period == 'seasonal': ylab_units = 'Days per season'

if diag_name == 'AR_eant_IVT':
  vn = '20230716'
  nsource = 1
  sea_agg_calc = 1
  sea_agg_type_in = 'total'
  fname_csv = 'east_ant_ARs_lons_vIVT_80-22.csv'
  fname = diag_name+'_ann_'+vn+'.nc'
  AR_df = pd.read_csv('t_srs/ARs/'+fname_csv, dtype = np.float64, parse_dates=['time'])
  AR_df['time'] = pd.to_datetime(AR_df['time'], format = '%d/%m/%Y %H:%M')
  AR_df = AR_df.rename(columns = {'Median_Longitude_(66-72S)': 'Longitude'})
  AR_ds = AR_df.set_index(['time']).to_xarray()
  AR_ds.to_netcdf('t_srs/ARs/'+fname)
  AR_cubelist = iris.load('t_srs/ARs/'+fname)
  IVT_cube = AR_cubelist[0]
  iris.coord_categorisation.add_month(IVT_cube, 'time', 'month')
  iris.coord_categorisation.add_month_number(IVT_cube, 'time', 'month_number')
  iris.coord_categorisation.add_year(IVT_cube, 'time', 'year')
  nyrs=44
  yr_arr = 1979+np.arange(44)
  mon_arr = 1+np.arange(12)
  nmon = len(yr_arr)*12
  try: ncfile.close()  # just to be safe, make sure dataset is not already open.
  except: pass
  ncfile = Dataset('t_srs/ARs/'+fname,mode='w',format='NETCDF4_CLASSIC') 
  #print(ncfile)
  time_dim = ncfile.createDimension('time', None)
  ncfile.title='East_Antarctic_AR_residence_time'
  time = ncfile.createVariable('time', np.float64, ('time',))
  time.units = 'hours since 1800-01-01'
  time.long_name = 'time'
# Define a 3D variable to hold the data
  var = ncfile.createVariable('EA_AR_residence',np.float64,('time')) # note: unlimited dimension is leftmost
  var.units = 'hours' # no units
  var.standard_name = 'AR_accumulated_time' # this is a CF standard name
  calc_var = np.array([])
  dates = np.array([])
  for i in range(0,nyrs):
    for j in range(0,12):      
      dates = np.append(dates,dt.datetime(yr_arr[i],mon_arr[j],15,0)) # pick middle month and ay of SON (month 10, day 15)  
      ii_yr = np.where(IVT_cube.coord('year').points == yr_arr[i])      
      if len(ii_yr[0]) > 0: 
        ii_yr_mon = np.where(IVT_cube[ii_yr].coord('month_number').points == mon_arr[j])
        calc_var = np.append(calc_var, (3/24)*len(ii_yr_mon[0]))
      else:
        calc_var = np.append(calc_var, 0)
      
  times = date2num(dates, time.units)
  time[:] = times
  var[:] = calc_var
  ncfile.close(); print('Dataset is closed!') 
  diag_plt = iris.cube.CubeList()
  diag_plt.append(iris.load_cube('t_srs/ARs/'+fname))
  iris.coord_categorisation.add_month_number(diag_plt[0], 'time', 'month_number')
  iris.coord_categorisation.add_year(diag_plt[0],'time', 'year')
  nmon_clim = 12
  p_title = 'Accumulated east Antarctic AR residence time (IVT method)'
  source_lab = ['ERA5']
  if mean_period == 'monthly': ylab_units = 'Days per month'
  if mean_period == 'seasonal': ylab_units = 'Days per season'    
    
if diag_name == 'AR_want_IWV':
  vn = '20230716'
  nsource = 1
  sea_agg_calc = 1
  sea_agg_type_in = 'total'
  fname_csv = 'west_ant_ARs_lons_IWV_80-22.csv'
  fname = diag_name+'_ann_'+vn+'.nc'
  AR_df=pd.read_csv('t_srs/ARs/'+fname_csv, parse_dates=['time'])
  AR_ds = AR_df.set_index(['time']).to_xarray()
  AR_ds.to_netcdf('t_srs/ARs/'+fname)
  AR_cubelist = iris.load('t_srs/ARs/'+fname)
  IVT_cube = AR_cubelist[0]
  iris.coord_categorisation.add_month(IVT_cube, 'time', 'month')
  iris.coord_categorisation.add_month_number(IVT_cube, 'time', 'month_number')
  iris.coord_categorisation.add_year(IVT_cube, 'time', 'year')
  nyrs=44
  yr_arr = 1979+np.arange(44)
  mon_arr = 1+np.arange(12)
  nmon = len(yr_arr)*12
  try: ncfile.close()  # just to be safe, make sure dataset is not already open.
  except: pass
  ncfile = Dataset('t_srs/ARs/'+fname,mode='w',format='NETCDF4_CLASSIC') 
  #print(ncfile)
  time_dim = ncfile.createDimension('time', None)
  ncfile.title='West_Antarctic_AR_residence_time'
  time = ncfile.createVariable('time', np.float64, ('time',))
  time.units = 'hours since 1800-01-01'
  time.long_name = 'time'
# Define a 3D variable to hold the data
  var = ncfile.createVariable('WA_AR_residence',np.float64,('time')) # note: unlimited dimension is leftmost
  var.units = 'hours' # no units
  var.standard_name = 'AR_accumulated_time' # this is a CF standard name
  calc_var = np.array([])
  dates = np.array([])
  for i in range(0,nyrs):
    for j in range(0,12):      
      dates = np.append(dates,dt.datetime(yr_arr[i],mon_arr[j],15,0)) # pick middle month and ay of SON (month 10, day 15)  
      ii_yr = np.where(IVT_cube.coord('year').points == yr_arr[i])      
      if len(ii_yr[0]) > 0: 
        ii_yr_mon = np.where(IVT_cube[ii_yr].coord('month_number').points == mon_arr[j])
        calc_var = np.append(calc_var, (3/24)*len(ii_yr_mon[0]))
      else:
        calc_var = np.append(calc_var, 0)
      
  times = date2num(dates, time.units)
  time[:] = times
  var[:] = calc_var
  ncfile.close(); print('Dataset is closed!')  
  diag_plt = iris.cube.CubeList()
  diag_plt.append(iris.load_cube('t_srs/ARs/'+fname))
  iris.coord_categorisation.add_month_number(diag_plt[0], 'time', 'month_number')
  iris.coord_categorisation.add_year(diag_plt[0],'time', 'year')
  nmon_clim = 12
  p_title = 'Accumulated West Antarctic AR residence time (IWV method)'
  source_lab = ['ERA5']
  if mean_period == 'monthly': ylab_units = 'Days per month'
  if mean_period == 'seasonal': ylab_units = 'Days per season'

if diag_name == 'AR_want_IVT':
  vn = '20230716'
  nsource = 1
  sea_agg_calc = 1
  sea_agg_type_in = 'total'
  fname_csv = 'west_ant_ARs_lons_vIVT_80-22.csv'
  fname = diag_name+'_ann_'+vn+'.nc'
  AR_df=pd.read_csv('t_srs/ARs/'+fname_csv, parse_dates=['time'])
  AR_df['time'] = pd.to_datetime(AR_df['time'], format = '%d/%m/%Y %H:%M')
  AR_df = AR_df.rename(columns = {'Median_Longitude_(72-78S)': 'Longitude'})
  AR_ds = AR_df.set_index(['time']).to_xarray()
  AR_ds.to_netcdf('t_srs/ARs/'+fname)
  AR_cubelist = iris.load('t_srs/ARs/'+fname)
  IVT_cube = AR_cubelist[0]
  iris.coord_categorisation.add_month(IVT_cube, 'time', 'month')
  iris.coord_categorisation.add_month_number(IVT_cube, 'time', 'month_number')
  iris.coord_categorisation.add_year(IVT_cube, 'time', 'year')
  nyrs=44
  yr_arr = 1979+np.arange(44)
  mon_arr = 1+np.arange(12)
  nmon = len(yr_arr)*12
  try: ncfile.close()  # just to be safe, make sure dataset is not already open.
  except: pass
  ncfile = Dataset('t_srs/ARs/'+fname,mode='w',format='NETCDF4_CLASSIC') 
  #print(ncfile)
  time_dim = ncfile.createDimension('time', None)
  ncfile.title='West_Antarctic_AR_residence_time'
  time = ncfile.createVariable('time', np.float64, ('time',))
  time.units = 'hours since 1800-01-01'
  time.long_name = 'time'
# Define a 3D variable to hold the data
  var = ncfile.createVariable('EA_AR_residence',np.float64,('time')) # note: unlimited dimension is leftmost
  var.units = 'hours' # no units
  var.standard_name = 'AR_accumulated_time' # this is a CF standard name
  calc_var = np.array([])
  dates = np.array([])
  for i in range(0,nyrs):
    for j in range(0,12):      
      dates = np.append(dates,dt.datetime(yr_arr[i],mon_arr[j],15,0)) # pick middle month and ay of SON (month 10, day 15)  
      ii_yr = np.where(IVT_cube.coord('year').points == yr_arr[i])      
      if len(ii_yr[0]) > 0: 
        ii_yr_mon = np.where(IVT_cube[ii_yr].coord('month_number').points == mon_arr[j])
        calc_var = np.append(calc_var, (3/24)*len(ii_yr_mon[0])) # Data are in 3 hour intervals, convert to days. 
      else:
        calc_var = np.append(calc_var, 0)
      
  times = date2num(dates, time.units)
  time[:] = times
  var[:] = calc_var
  ncfile.close(); print('Dataset is closed!')  
  diag_plt = iris.cube.CubeList()
  diag_plt.append(iris.load_cube('t_srs/ARs/'+fname))
  iris.coord_categorisation.add_month_number(diag_plt[0], 'time', 'month_number')
  iris.coord_categorisation.add_year(diag_plt[0],'time', 'year')
  nmon_clim = 12
  p_title = 'West Antarctic AR residence time (IVT method)'
  source_lab = ['ERA5']
  if mean_period == 'monthly': ylab_units = 'Days per month'
  if mean_period == 'seasonal': ylab_units = 'Days per season'


if diag_name == 'SAM_marshall':
  vn = '20240224'
  nsource = 1
  sea_agg_calc = 0
  if mean_period == 'seasonal': 
    fname_txt = 't_srs/SAM/newsam.1957.2007.seas.txt'    
    col_arr=np.array(['ann', 'mam','jja', 'son', 'djf']) #array containing  column headings of input text file.  
    fname = 'aci_'+diag_name+'_'+mean_period+'_'+vn+'.nc'
    if update_data == 1:
      ncols = len(col_arr)+1
      sam_in = np.loadtxt(fname_txt, skiprows=2)
      sam_dat = sam_in[:,2:ncols]
      sam_yrs = sam_in[:,0]  
      sam_yrs_int = sam_yrs.astype(int)
      sam_yrs_arr = np.copy(sam_dat)
      sam_sea_mon_arr = np.copy(sam_dat)
      nyrs = len(sam_yrs)
      for i in range(0,nyrs): 
        ### Add one year to djf years to adhere to convention of year of DJF being defined as JF
        sam_yrs_arr[i,:] = ([sam_yrs_int[i], sam_yrs_int[i], sam_yrs_int[i], sam_yrs_int[i]+1])
        print('sam_yrs_arr[i,:]',sam_yrs_arr[i,:])
        sam_sea_mon_arr[i,:] = [4, 7, 10, 1] # central month of seasons
      
      sam_tsrs = np.ndarray.flatten(sam_dat)
      yr_tsrs = np.ndarray.flatten(sam_yrs_arr)
      mon_sea_tsrs = np.ndarray.flatten(sam_sea_mon_arr)  
      ntims = len(sam_tsrs)
    #  ii=np.where(col_arr == season) # select the season of interest  
      try: ncfile.close()  # just to be safe, make sure dataset is not already open.
      except: pass
      ncfile = Dataset('t_srs/SAM/'+fname,mode='w',format='NETCDF4_CLASSIC') 
      #print(ncfile)
      time_dim = ncfile.createDimension('time', None)
      ncfile.title='Marshall SAM index'
      time = ncfile.createVariable('time', np.float64, ('time',))
      time.units = 'hours since 1800-01-01'
      time.long_name = 'time'
    # Define a 3D variable to hold the data
      var = ncfile.createVariable('SAM',np.float64,('time')) # note: unlimited dimension is leftmost
      var.units = '' # no units
      var.standard_name = 'SAM' # this is a CF standard name
      var[:] = sam_tsrs
      dates = np.array([])
      for i in range(0,ntims): 
        dates=np.append(dates,dt.datetime(int(yr_tsrs[i]),int(mon_sea_tsrs[i]),15,0))  
        
      times = date2num(dates, time.units)
      time[:] = times
      ncfile.close(); print('Dataset is closed!')  

  #if mean_period == 'annual': 
    #fname_txt = 't_srs/SAM/newsam.1957.2007.seas.txt'    
    #col_arr=np.array(['ann', 'mam','jja', 'son', 'djf']) #array containing  column headings of input text file.  
    #fname = 'aci_'+diag_name+'_'+mean_period+'_'+vn+'.nc'
    #if update_data == 1:
      #ncols = len(col_arr)+1
      #sam_in = np.loadtxt(fname_txt, skiprows=2)
      #sam_dat = sam_in[:,1]
      #sam_yrs = sam_in[:,0]  
      #sam_mon_yrs_arr = np.copy(sam_dat)      
      #nyrs = len(sam_yrs)
      #for i in range(0,nyrs): 
        #sam_yrs_arr[i,:] = int(sam_yrs[i])
        #sam_mon_yrs_ann[i,1] = [6] # central month of seasons
      
      #sam_tsrs = np.ndarray.flatten(sam_dat)
      #yr_tsrs = np.ndarray.flatten(sam_yrs)
      #ntims = len(sam_tsrs)
    ##  ii=np.where(col_arr == season) # select the season of interest  
      #try: ncfile.close()  # just to be safe, make sure dataset is not already open.
      #except: pass
      #ncfile = Dataset('t_srs/SAM/'+fname+'.nc',mode='w',format='NETCDF4_CLASSIC') 
      ##print(ncfile)
      #time_dim = ncfile.createDimension('time', None)
      #ncfile.title='Marshall SAM index'
      #time = ncfile.createVariable('time', np.float64, ('time',))
      #time.units = 'hours since 1800-01-01'
      #time.long_name = 'time'
    ## Define a 3D variable to hold the data
      #var = ncfile.createVariable('SAM',np.float64,('time')) # note: unlimited dimension is leftmost
      #var.units = '' # no units
      #var.standard_name = 'SAM' # this is a CF standard name
      #var[:] = sam_tsrs
      #dates = np.array([])
      #for i in range(0,ntims): 
        #dates=np.append(dates,dt.datetime(int(yr_tsrs[i]),int(mon_sea_tsrs[i]),15,0))  
        
      #times = date2num(dates, time.units)
      #time[:] = times
      #ncfile.close(); print('Dataset is closed!')  

  if mean_period == 'monthly':     
    fname_txt = 't_srs/SAM/newsam.1957.2007_202302.txt'    
    col_arr=np.array(['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']) #array containing  column headings of input text file.    
    fname = 'aci_'+diag_name+'_'+mean_period+'_'+vn+'.nc'
    if update_data == 1: 
      ncols = len(col_arr)+1
      sam_in = np.loadtxt(fname_txt, skiprows=2)
      sam_dat = sam_in[:,1:ncols]
      sam_yrs = sam_in[:,0]  
      sam_yrs_arr = np.copy(sam_dat)
      sam_mons_arr = np.copy(sam_dat)
      nyrs = len(sam_yrs)
      for i in range(0,nyrs): 
        sam_yrs_arr[i,:] = int(sam_yrs[i])
        sam_mons_arr[i,:] = np.arange(1,ncols)
      
      sam_tsrs = np.ndarray.flatten(sam_dat)
      yr_tsrs = np.ndarray.flatten(sam_yrs_arr)
      mon_tsrs = np.ndarray.flatten(sam_mons_arr)  
      ntims = len(sam_tsrs)
    #  ii=np.where(col_arr == season) # select the season of interest  
      try: ncfile.close()  # just to be safe, make sure dataset is not already open.
      except: pass
      ncfile = Dataset('t_srs/SAM/'+fname,mode='w',format='NETCDF4_CLASSIC') 
      #print(ncfile)
      time_dim = ncfile.createDimension('time', None)
      ncfile.title='Marshall SAM index'
      time = ncfile.createVariable('time', np.float64, ('time',))
      time.units = 'hours since 1800-01-01'
      time.long_name = 'time'
    # Define a 3D variable to hold the data
      var = ncfile.createVariable('SAM',np.float64,('time')) # note: unlimited dimension is leftmost
      var.units = '' # no units
      var.standard_name = 'SAM' # this is a CF standard name
      var[:] = sam_tsrs
      dates = np.array([])
      for i in range(0,ntims): 
        dates=np.append(dates,dt.datetime(int(yr_tsrs[i]),int(mon_tsrs[i]),15,0))  
        
      times = date2num(dates, time.units)
      time[:] = times
      ncfile.close(); print('Dataset is closed!')  

  #print('diag_plt', diag_plt)
  diag_plt = iris.cube.CubeList()  
  diag_plt.append(iris.load_cube('t_srs/SAM/'+fname))
  nmon_clim = 12
  p_title = 'Marshall SAM index'
  source_lab = ' '
  ylab_units = 'SAM'
  iris.coord_categorisation.add_year(diag_plt[0], 'time', 'year')
  if mean_period == 'monthly': iris.coord_categorisation.add_month_number(diag_plt[0], 'time', 'month_number')
  if mean_period == 'seasonal': iris.coord_categorisation.add_season(diag_plt[0], 'time', 'clim_season')
#  if mean_period == 'seasonal': diag_plt_season = iris.coord_categorisation.add_season(diag_plt[0], 'time', 'clim_season')


####################### pre-process data for plotting #############
diag_plt_season = iris.cube.CubeList()

for i in range(0,nsource):
### Where necessary create seasonal aggregation
  if sea_agg_calc == 2: diag_plt_season = diag_plt
  if sea_agg_calc == 1: diag_plt_season.append(season_agg(diag_plt[i],sea_agg_type = sea_agg_type_in))        
  if sea_agg_calc == 0: diag_plt_season.append(diag_plt[i])

  ### Calculate monthly anomalies from climatology

  #De-seasonalise the data for available complete years
  #Monthly
  if mean_period == 'monthly': 
    diag_plt_monclim=diag_plt[i].aggregated_by('month_number', iris.analysis.MEAN)
    lastyr = max(diag_plt[i].coord('year').points)
    firstyr = min(diag_plt[i].coord('year').points)
    nyrs = lastyr-firstyr +1
    years = firstyr+np.arange(nyrs)
    for j in range(0, nmon_clim):
      #print(i)
      ii=np.where(diag_plt[i].coord('month_number').points == diag_plt_monclim.coord('month_number').points[j]) 
      diag_plt[i].data[ii] = diag_plt[i].data[ii] - diag_plt_monclim.data[j] 


############################ ----- Create time series plots -----
color_dset_arr = ['lightblue', 'lightgreen']
color_dset_arr_sm = ['darkblue', 'darkgreen']
linestyle_dset_arr = ['solid', 'dashed']
window = 25
w_5yr = low_pass_weights(window, 1.0 / 72.0)

############################ Monthly #####################
if mean_period == 'monthly':    
  fig, ax = plt.subplots(figsize = (7,4), dpi = 300)
  ax.axhline(0, color = '0.8', linewidth = 0.7, alpha=1)
  for i in range(0,nsource):
    cube_plt=diag_plt
    diag_plt_monclim=diag_plt[i].aggregated_by('month_number', iris.analysis.MEAN)
    lastyr = max(diag_plt[i].coord('year').points)
    firstyr = min(diag_plt[i].coord('year').points)
    nyrs = lastyr-firstyr +1
    years = firstyr+np.arange(nyrs)
    for j in range(0, nmon_clim):
      #print(i)
      ii=np.where(diag_plt[i].coord('month_number').points == diag_plt_monclim.coord('month_number').points[j]) 
      diag_plt[i].data[ii] = diag_plt[i].data[ii] - diag_plt_monclim.data[j] 

    ylim_diag = yrange_calc(diag_plt[i])
    if nmon_clim == 12:
      #diag_plt_sm = diag_plt[i].rolling_window('time', iris.analysis.SUM, len(w_5yr), weights=w_5yr)
      diag_plt_sm = diag_plt[i].rolling_window('time', iris.analysis.MEAN, 12)
      #linestyle='-', color = '#296e31', label = 'ERA5', linewidth = 0.8, alpha =0.8)
      #line1, = iplt.plot(diag_plt, color='#296e31', linewidth = 0.8, alpha = 0.8, label = source_lab)
    #  iplt.plot(diag_plt, color='#296e31', linewidth = 0.8, label = source_lab)
      iplt.plot(diag_plt[i], color=color_dset_arr[i], linestyle = linestyle_dset_arr[i], linewidth = 0.8)
      iplt.plot(diag_plt_sm, color=color_dset_arr_sm[i], linestyle = linestyle_dset_arr[i], linewidth = 1.6, label = source_lab[i]+' (2-year smoothing)')
      iris.save(diag_plt[i], 't_srs/data4figs/aci_'+diag_name+'_monthly_anom_v'+aci_version_num+'.nc')
      iris.save(diag_plt_sm, 't_srs/data4figs/aci_'+diag_name+'_sm_monthly_anom_v'+aci_version_num+'.nc')
    else:
      iplt.plot(diag_plt[i], color=color_dset_arr_sm[i], linestyle = linestyle_dset_arr[i], linewidth = 1.6, label = source_lab[i])
      iris.save(diag_plt_xsea, 't_srs/data4figs/aci_'+diag_name+'_monthly_anom_v'+aci_version_num+'.nc')
    
  #  iplt.plot(diag_plt, linestyle='-', color = 'blue', linewidth = 0.8, alpha = 0.6)
    

  xy = (.9, .1) # Annotation position (in axis fraction coordinates). 
  ax.plot(xy[0], xy[1])
  # Annotate the 1st position with a text box ('Test 1')
  antclimnow_logo = plt.imread('plotting_info/AntClimNow_logo_15cm_300dpi.png')
  scar_logo = plt.imread('plotting_info/SCAR_logo_2018_white_background.png')
  #nerc_logo = plt.imread('plotting_info/NERC_logo_black_no_background.png')
  bas_logo = plt.imread('plotting_info/bas-logo-default-transparent-256.png')
  #acn_logobox = OffsetImage(antclimnow_logo, zoom = 0.02)
  #scar_logobox = OffsetImage(scar_logo, zoom =0.027)
  ##nerc_logobox = OffsetImage(nerc_logo, zoom = 0.05)
  #bas_logobox = OffsetImage(bas_logo, zoom = 0.09)
  #ab1 = AnnotationBbox(acn_logobox, (.85,1), xycoords = 'axes fraction', frameon= False)
  #ab2 = AnnotationBbox(scar_logobox, (0.1,1), xycoords = 'axes fraction', frameon= False)
  ##ab3 = AnnotationBbox(nerc_logobox, (.93,0.05), xycoords = 'axes fraction', frameon= False)
  #ab4 = AnnotationBbox(bas_logobox, (.9,0.07), xycoords = 'axes fraction', frameon= False)
  acn_logobox = OffsetImage(antclimnow_logo, zoom = 0.015)
  scar_logobox = OffsetImage(scar_logo, zoom =0.016)
  #nerc_logobox = OffsetImage(nerc_logo, zoom = 0.05)
  bas_logobox = OffsetImage(bas_logo, zoom = 0.07)
  ab1 = AnnotationBbox(acn_logobox, (.73,0.985), xycoords = 'axes fraction', frameon= False)
  ab2 = AnnotationBbox(scar_logobox, (.83,0.98), xycoords = 'axes fraction', frameon= False)
  #ab3 = AnnotationBbox(nerc_logobox, (.93,0.05), xycoords = 'axes fraction', frameon= False)
  ab4 = AnnotationBbox(bas_logobox, (1,0.985), xycoords = 'axes fraction', frameon= False)
 
  ax.add_artist(ab1)
  ax.add_artist(ab2)
  #ax.add_artist(ab3)
  ax.add_artist(ab4)
  ax.set_ylim(ylim_diag)
  ax.set_xlabel('Year', size=10)
  ax.xaxis.set_major_locator(YearLocator(5, month=1, day=2)) # select day 2, otherwise the formatting can give the wrong year at at the year boundary (e.g. 1989 instead of 1990). 
  formatter = nc_time_axis.CFTimeFormatter("%Y", "gregorian")
  ax.xaxis.set_major_formatter(formatter)
  ax.set_ylabel('Monthly anomaly ('+ylab_units+')', size=10)
  ax.set_title(p_title+'\n ('+aci_version+')', size=10)
  #ax.text(.1,.9, season, transform=ax.transAxes)
  ax.xaxis.set_tick_params(labelsize=8)
  ax.yaxis.set_tick_params(labelsize=8)
  ax.legend(fontsize=8, loc='lower left', frameon=False)
  sns.despine() # remove plot borders 
  lyr = ax.get_children()
  countl=0
  for k in range(0, nsource+1, 2):
    print(k, countl)
    lyr[k].set_zorder(2*nsource-1-countl)
    lyr[k+1].set_zorder(0+countl)
    countl=countl+1
  
  #lyr[0].set_zorder(3)
  #lyr[1].set_zorder(0)
  #lyr[2].set_zorder(2)
  #lyr[3].set_zorder(1)
  #lyr[4]
  #lyr[5]
  #plt.show() 
  fname_plt = 'aci_'+diag_name+'_monthly_anom_v'+aci_version_num
  plt.savefig('plots/'+fname_plt+'.png')
  plt.close()

######################### Seasonal ################################ 
if mean_period == 'seasonal': 
#  diag_plt_seaclim=diag_plt_season.aggregated_by('clim_season', iris.analysis.MEAN)
  seasons_arr = ['djf','mam','jja','son']
  seasons_arr_plt = ['DJF','MAM','JJA','SON']
  seasons_colr = ['darkorange', '#008a82', '#1f5b74', '#67b050']
  plt.figure(figsize=(8, 6), dpi=600)  
  #fig, ax = plt.subplots(figsize = (5,12), dpi = 300)
  seas_names = {'djf':'Summer', 'mam':'Autumn', 'jja':'Winter', 'son':'Spring'}
  ax=plt.gca()
  antclimnow_logo = plt.imread('plotting_info/AntClimNow_logo_15cm_300dpi.png')
  scar_logo = plt.imread('plotting_info/SCAR_logo_2018_white_background.png')
  #nerc_logo = plt.imread('plotting_info/NERC_logo_black_no_background.png')
  bas_logo = plt.imread('plotting_info/bas-logo-default-transparent-256.png')
  acn_logobox = OffsetImage(antclimnow_logo, zoom = 0.019)
  scar_logobox = OffsetImage(scar_logo, zoom =0.022)
  #nerc_logobox = OffsetImage(nerc_logo, zoom = 0.05)
  bas_logobox = OffsetImage(bas_logo, zoom = 0.09)
  ab1 = AnnotationBbox(acn_logobox, (.73,1.055), xycoords = 'axes fraction', frameon= False)
  ab2 = AnnotationBbox(scar_logobox, (.83,1.05), xycoords = 'axes fraction', frameon= False)
  #ab3 = AnnotationBbox(nerc_logobox, (.93,0.05), xycoords = 'axes fraction', frameon= False)
  ab4 = AnnotationBbox(bas_logobox, (1,1.055), xycoords = 'axes fraction', frameon= False)
  ax.add_artist(ab1)
  ax.add_artist(ab2)
  #ax.add_artist(ab3)
  ax.add_artist(ab4)
  plt.axis('off')
  for i in range(0, nsource):
    for k in range(0,4):      
      if sea_agg_calc == 2: 
        diag_plt_xsea = diag_plt_season[k]
      else: 
        diag_plt_xsea = diag_plt_season[i].extract(iris.Constraint(clim_season=seasons_arr[k]))
        
      ### Save files of the time series shown in the seasonal plots
      iris.save(diag_plt_xsea, 't_srs/data4figs/aci_'+diag_name+'_'+seasons_arr[k]+'_v'+aci_version_num+'.nc')
      ax=plt.subplot(2,2,k+1)
    #  ax.axhline(0, color = '0.8', linewidth = 0.7)
      ylim_diag = yrange_calc(diag_plt_xsea)
      if (diag_name == 'SMB_all'): ylim_diag[0] = 0
      if diag_name == 'SMB_all': ylim_diag[1] = ylim_diag[1]+100
      #linestyle='-', color = '#296e31', label = 'ERA5', linewidth = 0.8, alpha =0.8)
      #line1, = iplt.plot(diag_plt, color='#296e31', linewidth = 0.8, alpha = 0.8, label = source_lab)
      iplt.plot(diag_plt_xsea, color=seasons_colr[k], linewidth = 1.5, alpha = 0.8, label = source_lab[i], linestyle = linestyle_dset_arr[i])
    #  iplt.plot(diag_plt_xsea, linestyle='-', color = seasons_colr[k], linewidth = 0.8, alpha = 0.6)
    #  xy = (.9, .1) # Annotation position (in axis fraction coordinates). 
    #  ax.plot(xy[0], xy[1])
      #ax.plot(jsi_sh_djf.data)
      # Annotate the 1st position with a text box ('Test 1')
      #arr_img = plt.imread('plotting_info/AntClimNow_logo_15cm_300dpi.png')
      ##imagebox = OffsetImage(arr_img, zoom=0.02)
      #imagebox.image.axes = ax
      #ab = AnnotationBbox(imagebox, xy, xycoords='axes fraction', frameon=False)
      #ax.add_artist(ab)
      ax.set_ylim(ylim_diag)
      if (k == 2) or (k == 3): ax.set_xlabel('Year', size=14)
      if (k == 0) or (k == 2): ax.set_ylabel(ylab_units, size=14)
      #ax.set_title(p_title, size=4)
      ax.text(.1,.9, seasons_arr_plt[k]+': '+seas_names.get(seasons_arr[k]), transform=ax.transAxes, color=seasons_colr[k], size = 14)
      #if (k == 0): ax.text(.95, 0.95, 'ERA5', transform=ax.transAxes, fontweight = 'book', color='silver', fontsize=16,
	    #bbox=dict(fc="none", ec='silver', linewidth = 0.5))
      ax.xaxis.set_tick_params(labelsize=10)
      ax.yaxis.set_tick_params(labelsize=10)
      ax.xaxis.set_major_locator(YearLocator(10, month=1, day=15)) # select day 2, otherwise the formatting can give the wrong year at at the year boundary (e.g. 1989 instead of 1990). 
      formatter = nc_time_axis.CFTimeFormatter("%Y", "gregorian")
      ax.xaxis.set_major_formatter(formatter)
      ax.legend(fontsize=8, loc='lower left', frameon=True)
      sns.despine() # remove plot borders 
      plt.suptitle(p_title+'\n ('+aci_version+')')
    
  #plt.show() 
  fname_plt = fname_plt = 'aci_'+diag_name+'_seasonal_v'+aci_version_num  
  plt.savefig('plots/'+fname_plt+'.png')
  plt.close()

########################## Annual ################################ 
#if mean_period == 'annual': 
##  diag_plt_seaclim=diag_plt_season.aggregated_by('clim_season', iris.analysis.MEAN)
  #plt.figure(figsize=(7, 4), dpi=300)  
##  ax.axhline(0, color = '0.8', linewidth = 0.7, alpha=1)
  #for i in range(0,nsource):
    #cube_plt=diag_plt
    #diag_plt_ann=diag_plt[i].aggregated_by('year', iris.analysis.MEAN)
    #lastyr = max(diag_plt[i].coord('year').points)
    #firstyr = min(diag_plt[i].coord('year').points)
    #nyrs = lastyr-firstyr +1
    #years = firstyr+np.arange(nyrs)
    #iplt.plot(diag_plt[i], color=color_dset_arr_sm[i], linestyle = linestyle_dset_arr[i], linewidth = 1.6, label = source_lab[i])
    
  ##  iplt.plot(diag_plt, linestyle='-', color = 'blue', linewidth = 0.8, alpha = 0.6)
    
  #xy = (.9, .1) # Annotation position (in axis fraction coordinates). 
  #ax.plot(xy[0], xy[1])
  ## Annotate the 1st position with a text box ('Test 1')
  #antclimnow_logo = plt.imread('plotting_info/AntClimNow_logo_15cm_300dpi.png')
  #scar_logo = plt.imread('plotting_info/SCAR_logo_2018_white_background.png')
  ##nerc_logo = plt.imread('plotting_info/NERC_logo_black_no_background.png')
  #bas_logo = plt.imread('plotting_info/bas-logo-default-transparent-256.png')
  ##acn_logobox = OffsetImage(antclimnow_logo, zoom = 0.02)
  ##scar_logobox = OffsetImage(scar_logo, zoom =0.027)
  ###nerc_logobox = OffsetImage(nerc_logo, zoom = 0.05)
  ##bas_logobox = OffsetImage(bas_logo, zoom = 0.09)
  ##ab1 = AnnotationBbox(acn_logobox, (.85,1), xycoords = 'axes fraction', frameon= False)
  ##ab2 = AnnotationBbox(scar_logobox, (0.1,1), xycoords = 'axes fraction', frameon= False)
  ###ab3 = AnnotationBbox(nerc_logobox, (.93,0.05), xycoords = 'axes fraction', frameon= False)
  ##ab4 = AnnotationBbox(bas_logobox, (.9,0.07), xycoords = 'axes fraction', frameon= False)
  #acn_logobox = OffsetImage(antclimnow_logo, zoom = 0.015)
  #scar_logobox = OffsetImage(scar_logo, zoom =0.016)
  ##nerc_logobox = OffsetImage(nerc_logo, zoom = 0.05)
  #bas_logobox = OffsetImage(bas_logo, zoom = 0.07)
  #ab1 = AnnotationBbox(acn_logobox, (.73,1.055), xycoords = 'axes fraction', frameon= False)
  #ab2 = AnnotationBbox(scar_logobox, (.83,1.05), xycoords = 'axes fraction', frameon= False)
  ##ab3 = AnnotationBbox(nerc_logobox, (.93,0.05), xycoords = 'axes fraction', frameon= False)
  #ab4 = AnnotationBbox(bas_logobox, (1,1.055), xycoords = 'axes fraction', frameon= False)
 
  #ax.add_artist(ab1)
  #ax.add_artist(ab2)
  ##ax.add_artist(ab3)
  #ax.add_artist(ab4)
  #ax.set_ylim(ylim_diag)
  #ax.set_xlabel('Year', size=10)
  #ax.xaxis.set_major_locator(YearLocator(5, month=1, day=2)) # select day 2, otherwise the formatting can give the wrong year at at the year boundary (e.g. 1989 instead of 1990). 
  #formatter = nc_time_axis.CFTimeFormatter("%Y", "gregorian")
  #ax.xaxis.set_major_formatter(formatter)
  #ax.set_ylabel('Monthly anomaly ('+ylab_units+')', size=10)
  #ax.set_title(p_title+'\n ('+aci_version+')', size=10)
  ##ax.text(.1,.9, season, transform=ax.transAxes)
  #ax.xaxis.set_tick_params(labelsize=8)
  #ax.yaxis.set_tick_params(labelsize=8)
  #ax.legend(fontsize=8, loc='lower left', frameon=False)
  #sns.despine() # remove plot borders 
  #lyr = ax.get_children()
  #countl=0
  #for k in range(0, nsource+1, 2):
    #print(k, countl)
    #lyr[k].set_zorder(2*nsource-1-countl)
    #lyr[k+1].set_zorder(0+countl)
    #countl=countl+1
  
  ##lyr[0].set_zorder(3)
  ##lyr[1].set_zorder(0)
  ##lyr[2].set_zorder(2)
  ##lyr[3].set_zorder(1)
  ##lyr[4]
  ##lyr[5]
  ###plt.show) 
  #fname_plt = diag_name+'_full_tsrs_'+vn
  #plt.savefig('plots/'+fname_plt+'.png')
  #plt.close()

