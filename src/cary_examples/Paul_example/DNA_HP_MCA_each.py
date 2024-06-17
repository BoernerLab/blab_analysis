# -*- coding: utf-8 -*-
"""
Created on Tue May 14 12:43:46 2024

@author: Paul Lehmann
"""

#Import everything from main
from main import  *

#Meta
EXTRA_META = {'2023_08_01_DNA_K_MOPS_pH65_PL.csv': {'Buffer': 'MOPS', 'pH': 6.5, 'bg (mM)': 2.29, '[Buffer] in mM': 10, "Divalent": "Mg"},  # 10
              '2023_08_01_DNA_MOPS_65_K_2.csv': {'Buffer': 'MOPS', 'pH': 6.5, 'bg (mM)': 2.29, '[Buffer] in mM': 10},  # 10
              '2023_08_02_DNA_MOPS_65_Na.csv': {'Buffer': 'MOPS', 'pH': 6.5, 'bg (mM)': 2.29, '[Buffer] in mM': 10},  # 10
              # 10
              '2023_08_03_DNA_MOPS_75_Na_salz.csv': {'Buffer': 'MOPS', 'pH': 7.5, 'bg (mM)': 22.91, '[Buffer] in mM': 10},
              # 10
              '2023_08_03_DNA_MOPS_75_Na_salz_2.csv': {'Buffer': 'MOPS', 'pH': 7.5, 'bg (mM)': 22.91, '[Buffer] in mM': 10},
              '2023_08_08_DNA_MOPS_75_Na.csv': {'Buffer': 'MOPS', 'pH': 7.5, 'bg (mM)': 22.91, '[Buffer] in mM': 10},  # 10
              '2023_08_09_DNA_MOPS_75_K.csv': {'Buffer': 'MOPS', 'pH': 7.5, 'bg (mM)': 22.91, '[Buffer] in mM': 10},  # 10
              # 10
              '2023_08_10_DNA_TRIS-HCl_75_K.csv': {'Buffer': 'TRIS-HCl', 'pH': 7.5, 'bg (mM)': 2.75, '[Buffer] in mM': 10},
              # 10
              '2023_08_29_DNA_TRIS-HCl_75_Na.csv': {'Buffer': 'TRIS-HCl', 'pH': 7.5, 'bg (mM)': 2.75, '[Buffer] in mM': 10},
              # 1
              '2023_08_30_DNA_TRIS-HCl_85_Na.csv': {'Buffer': 'TRIS-HCl', 'pH': 8.5, 'bg (mM)': 2.75, '[Buffer] in mM': 1},
              '2023_08_31_DNA_TRIS-HCl_85_K.csv': {'Buffer': 'TRIS-HCl', 'pH': 8.5, 'bg (mM)': 2.75, '[Buffer] in mM': 1},  # 1
              '2023_09_13_DNA_HEPES_75_Na.csv': {'Buffer': 'HEPES', 'pH': 7.5, 'bg (mM)': 2.62, '[Buffer] in mM': 2.5},  # 2.5
              '2023_09_12_DNA_MOPS_75_K_Na.csv': {'Buffer': 'MOPS', 'pH': 7.5, 'bg (mM)': 2.29, '[Buffer] in mM': 1},  # 1
              '2023_09_14_HEPES_75_K.csv': {'Buffer': 'HEPES', 'pH': 7.5, 'bg (mM)': 2.62, '[Buffer] in mM': 2.5},  # 2.5

              # 3.3
              '2023_03_30_DNA_HEPES_K_74_verdunnt_Schmelzkurve_PL.csv': {'Buffer': 'HEPES', 'pH': 7.4, 'bg (mM)': 2.77, '[Buffer] in mM': 3.3},
              # 3.3
              '2023_03_31_DNA_HEPES_K_74_verdunnt_Schmel_PL.csv': {'Buffer': 'HEPES', 'pH': 7.4, 'bg (mM)': 2.77, '[Buffer] in mM': 3.3},
              # 3.3
              '2023_04_03_DNA_HEPES_K_74_verdunnt_PL.csv': {'Buffer': 'HEPES', 'pH': 7.4, 'bg (mM)': 2.77, '[Buffer] in mM': 3.3},
              # 3.3
              '2023_04_03_DNA_HEPES_K_74_verdunnt_Schmelzkurve_PL.csv': {'Buffer': 'HEPES', 'pH': 7.4, 'bg (mM)': 2.77, '[Buffer] in mM': 3.3},

              '2023_05_08_DNA_HEPES_74_Na_2.csv': {'Buffer': 'HEPES', 'pH': 7.4, 'bg (mM)': 8.31, '[Buffer] in mM': 10},  # 10
              '2023_05_08_DNA_HEPES_Na_74_PL.csv': {'Buffer': 'HEPES', 'pH': 7.4, 'bg (mM)': 8.31, '[Buffer] in mM': 10},  # 10
              # 10
              '2023_05_09_Schmelzkurve_DNA_HEPES_74_Na_PL.csv': {'Buffer': 'HEPES', 'pH': 7.4, 'bg (mM)': 8.31, '[Buffer] in mM': 10},
              # 10
              '2023_05_10_Schmelzkurve_DNA_HEPES_74_Na_2_PL.csv': {'Buffer': 'HEPES', 'pH': 7.4, 'bg (mM)': 8.31, '[Buffer] in mM': 10},
              # 10
              '2023_05_10_Schmelzkurve_DNA_HEPES_74_Na_PL.csv': {'Buffer': 'HEPES', 'pH': 7.4, 'bg (mM)': 8.31, '[Buffer] in mM': 10},
              '2023_09_26_HEPES_75_K_Na.csv': {'Buffer': 'HEPES', 'pH': 7.5, 'bg (mM)': 10.47, '[Buffer] in mM': 10}  # 10
              }

names_all = ['2023_08_01_DNA_K_MOPS_pH65_PL.csv', '2023_08_01_DNA_MOPS_65_K_2.csv', '2023_08_02_DNA_MOPS_65_Na.csv',
             '2023_08_03_DNA_MOPS_75_Na_salz.csv', '2023_08_03_DNA_MOPS_75_Na_salz_2.csv', '2023_08_08_DNA_MOPS_75_Na.csv',
             '2023_08_09_DNA_MOPS_75_K.csv', '2023_08_10_DNA_TRIS-HCl_75_K.csv', '2023_08_29_DNA_TRIS-HCl_75_Na.csv',
             '2023_08_30_DNA_TRIS-HCl_85_Na.csv', '2023_08_31_DNA_TRIS-HCl_85_K.csv', '2023_09_13_DNA_HEPES_75_Na.csv',
             '2023_09_12_DNA_MOPS_75_K_Na.csv', '2023_09_14_HEPES_75_K.csv', '2023_03_30_DNA_HEPES_K_74_verdunnt_Schmelzkurve_PL.csv',
             '2023_03_31_DNA_HEPES_K_74_verdunnt_Schmel_PL.csv', '2023_04_03_DNA_HEPES_K_74_verdunnt_PL.csv', '2023_04_03_DNA_HEPES_K_74_verdunnt_Schmelzkurve_PL.csv',
             '2023_05_08_DNA_HEPES_74_Na_2.csv', '2023_05_08_DNA_HEPES_Na_74_PL.csv', '2023_05_09_Schmelzkurve_DNA_HEPES_74_Na_PL.csv',
             '2023_05_10_Schmelzkurve_DNA_HEPES_74_Na_2_PL.csv', '2023_05_10_Schmelzkurve_DNA_HEPES_74_Na_PL.csv', '2023_09_26_HEPES_75_K_Na.csv'
             ]

#Create class object
data_obj = MeltingCurveAnalysis(EXTRA_META)
#Load all files from 'data_raw' folder
data_obj.fill_data_dict()


#Generation of meta data, from the given sample name and add them to meta, all file names in '[]' 
data_obj.meta_from_name_Paul(names_all)
#Ceate an overview of all curves, this function takes metadata from all curves and puts them in one dataframe
data_obj._create_meta_overview()
data_obj.meta_overview
'''
Ausnahmen: 22.05, 24.05, 25.05, <- schon raus?  01.08, 08.08
'''
#%% Speichern/Laden
'''
#index
idx = 1 #Increase index after each save manually
#Save
file_name = f'example_{idx}'
data_obj.save(filename = file_name)

file_name = f'example_{idx+1}'
'''
    
#%% Fitten
#Input T/x-values
T_norm_min = 20
T_norm_max = 70

deribounds1=(0.0015, 0.0008)
deribounds2=(0.00025, 0.00025)

model = MeltingCurveAnalysis.create_model_e1()

#for index, group in grouped:
file_name = '2023_08_01_DNA_K_MOPS_pH65_PL.csv'
mi = 1
temp_curve = data_obj.data[file_name][f'Measurement_{mi}']['data']
MeltingCurveAnalysis.normalize_data(temp_curve, normalize_to=(T_norm_min, T_norm_max))
data_obj.derivative_fit(file_name, mi, num_peaks = 1, filter_bounds = [{'y_min': 0.002, 'column_y': 'dAbsorbance / dTemperature (K)'}])
data_obj.direct_melt_fit(file_name, mi, model_pars = model)
data_obj.linear_baseline_fit(file_name, mi, deribounds1 = deribounds1, deribounds2 = deribounds2)
    
df = data_obj.data[file_name][f'Measurement_{mi}']['data']
fig, ax = plt.subplots(2,2, sharex = True)
fig.set_size_inches(12,8)
ax[0,0].plot(df['Temperature (°C)'], df['Absorbance'], label = f'Curve {mi}')
ax[0,1].plot(df['Temperature (°C)'], df['Normalized Absorbance'], label = f'Curve {mi}')
ax[1,0].plot(df['Temperature (°C)'], df['dAbsorbance / dTemperature (K)'], label = f'Curve {mi}')
ax[1,1].plot(df['Temperature (°C)'], df['ddAbsorbance / dTemperature (K) / dTemperature (K)'], label = f'Curve {mi}')
#%%
mi = 2

#Input T/x-values
T_norm_min = 20
T_norm_max = 60

xmin = 295
xmax = 335

deribounds1=(0.0015, 0.0015)
deribounds2=(0.0003, 0.0003)

temp_curve = data_obj.data[file_name][f'Measurement_{mi}']['data']

MeltingCurveAnalysis.normalize_data(temp_curve, normalize_to=(T_norm_min, T_norm_max))
data_obj.derivative_fit(file_name, mi, num_peaks = 1, filter_bounds = [{'column_x': 'Temperature (K)', 'column_y': 'dAbsorbance / dTemperature (K)', 'y_min': 0.0025, 'x_max': 330}])
data_obj.direct_melt_fit(file_name, mi, model_pars = model, filter_bounds = [{'column_x': 'Temperature (K)', 'x_max' : xmax, 'x_min': xmin}])
data_obj.linear_baseline_fit(file_name, mi, deribounds1 = deribounds1, deribounds2 = deribounds2, filter_bounds = [{'column_x': 'Temperature (K)', 'x_max' : xmax, 'x_min': xmin}])
    
df = data_obj.data[file_name][f'Measurement_{mi}']['data']
fig, ax = plt.subplots(2,2, sharex = True)
fig.set_size_inches(12,8)
ax[0,0].plot(df['Temperature (°C)'], df['Absorbance'], label = f'Curve {mi}')
ax[0,1].plot(df['Temperature (°C)'], df['Normalized Absorbance'], label = f'Curve {mi}')
ax[1,0].plot(df['Temperature (°C)'], df['dAbsorbance / dTemperature (K)'], label = f'Curve {mi}')
ax[1,1].plot(df['Temperature (°C)'], df['ddAbsorbance / dTemperature (K) / dTemperature (K)'], label = f'Curve {mi}')

#%%
from main import  *
mi = 3

#Input T/x-values
T_norm_min = 20
T_norm_max = 60

xmin = 250
xmax = 400

deribounds1=(0.0003, 0.001)
deribounds2=(0.0003, 0.0003)

temp_curve = data_obj.data[file_name][f'Measurement_{mi}']['data']

MeltingCurveAnalysis.normalize_data(temp_curve, normalize_to=(T_norm_min, T_norm_max))
data_obj.derivative_fit(file_name, mi, num_peaks = 1, filter_bounds = [{'column_x': 'Temperature (K)', 'column_y': 'dAbsorbance / dTemperature (K)', 'y_min': 0.0025}])
data_obj.direct_melt_fit(file_name, mi, model_pars = model, filter_bounds = [{'column_x': 'Temperature (K)', 'x_max' : xmax, 'x_min': xmin}])
data_obj.linear_baseline_fit(file_name, mi, deribounds1 = deribounds1, deribounds2 = deribounds2, filter_bounds = [{'column_x': 'Temperature (K)', 'x_max' : xmax, 'x_min': xmin}])
    
df = data_obj.data[file_name][f'Measurement_{mi}']['data']
fig, ax = plt.subplots(2,2, sharex = True)
fig.set_size_inches(12,8)
ax[0,0].plot(df['Temperature (°C)'], df['Absorbance'], label = f'Curve {mi}')
ax[0,1].plot(df['Temperature (°C)'], df['Normalized Absorbance'], label = f'Curve {mi}')
ax[1,0].plot(df['Temperature (°C)'], df['dAbsorbance / dTemperature (K)'], label = f'Curve {mi}')
ax[1,1].plot(df['Temperature (°C)'], df['ddAbsorbance / dTemperature (K) / dTemperature (K)'], label = f'Curve {mi}')









































