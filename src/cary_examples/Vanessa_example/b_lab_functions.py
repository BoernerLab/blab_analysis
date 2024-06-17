# function script
# to import all functions, use: import b_lab_functions
# to import specific function, use: from b_lab_functions import NAME-OF-SPECIFIC-FUNCTION

import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt


# ID5 Functions
def read_ID5_data(filepath) -> dict:
    letter = ["A","B","C","D","E","F","G","H"]
    plate = []
    for l in letter:
        for i in range(1,13):
            plate.append(l+str(i))

    meta = []
    meta_list = []
    data = []
    cols = []
    data_dict = {}
    iterator = 1
    ex_wl_data = None
    contains_wavelength = False

    # input wavelength_fret händisch. BSP 595 660 660
    em_wl_values = list(map(int, input("Please enter Emission Wavelengths (3) without comma:\n").split()))
    ex_wl_values = list(map(int, input("Please enter Excitation Wavelengths (3) without comma:\n").split()))


    with open(filepath, 'r', encoding='UTF-8') as file:
        print('Using ID5-function to read ID5-files...')
        next(file)
        lines = file.readlines()
        
        for line in lines:
            if not line.isspace():
                line = line.strip('\n')
                if line.startswith("Group"):
                # hier conzentrationen auslesen
                # evtl als zweiten dict/df
                    break
                if line.startswith("Plate"):
                    meta = line.split("\t")
                    if 'Absorbance' in meta: # if "absorbance" is in meta data list
                        abs_meta = meta # write meta data into variable abs_meta (absorbance meta data)
                        print(f"Experiment '{abs_meta[1]}': emission wavelength {abs_meta[11]}nm - {abs_meta[12]}nm in steps of {abs_meta[13]}nm") 
                        abs_meta_list = list(filter(None, abs_meta)) # delete empty strings from list abs_meta
                        meta_list.append(abs_meta_list) # add the absorbance meta data to big meta data list
                    elif 'Fluorescence' in meta and meta[16] == "":
                        fluo_meta = meta # write meta data into variable fluo_meta (fluorescence meta data)
                        print(f"Experiment '{fluo_meta[1]}': emission wavelength {fluo_meta[12]}nm - {fluo_meta[13]}nm in steps of {fluo_meta[14]}nm, {fluo_meta[23]}: {fluo_meta[24]}nm")
                        fluo_meta_list = list(filter(None, fluo_meta))  # delete empty strings from list fluo_meta
                        meta_list.append(fluo_meta_list) # add the fluorescence meta data to big meta data list
                    elif 'FRET' in meta or meta[16] != "": 
                        fret_meta = meta  # write meta data into variable fret_meta (fluorescence meta data)
                        fret_meta_list = list(filter(None, fret_meta)) # delete empty strings from list fret_meta

                        wavelength_fret = fret_meta[16].split(' ')
                        if wavelength_fret[1] == '':
                            wavelength_fret = fret_meta[20].split(' ')
                            print(f"Experiment '{fret_meta[1]}': emission wavelengths {fret_meta[16]}nm and {wavelength_fret[2]}nm, excitation wavelength: {wavelength_fret[0]}nm")
                        else:
                            print(f"Experiment '{fret_meta[1]}': emission wavelengths {wavelength_fret[0]}nm and {wavelength_fret[1]}nm, excitation wavelength: {fret_meta[20]}nm")
                        meta_list.append(fret_meta_list) # add the fret meta data to big meta data list
                        ex_wl_data = fret_meta[20]

                elif line.startswith("Wavelength"):
                    cols = line.split('\t')
                    contains_wavelength = True
            
                elif line.startswith("\tTemperature(¡C)"):
                    contains_wavelength = False
                    cols = line.split('\t')
                    cols = cols[:-1]
                    del cols[0]
                    cols.insert(0, "ex Wavelength")
                    cols.insert(1, "em Wavelength")
                    #specific_colnames = ["wavelength (nm)","temperature (°C)", "wellnumber", "value (x)"]
                    
                elif line.startswith("~End"):
                    if contains_wavelength == True:
                        df = pd.DataFrame(data, columns=cols)
                        df = df.iloc[1: , :]
                        df = df.replace("", np.nan)
                        df = df.dropna(axis=1, how="any")
                        df = df.melt(id_vars = df.columns[:2], value_vars= list(set(plate).intersection(df.columns)))
                        print(df)
                        df.columns =  ["wavelength (nm)", "temperature (°C)", "wellnumber", "value (x)"]
                        df["wavelength (nm)"] = df["wavelength (nm)"].astype(float)
                        df["temperature (°C)"] = df["temperature (°C)"].astype(float)
                        df["value (x)"] = df["value (x)"].astype(float)
                        df_sort = df.sort_values(["temperature (°C)", 'wellnumber'], ignore_index = True)
                        
                        
                        
                        
                    elif contains_wavelength == False:
                        for i, x in enumerate(data):
                            del x[0]
                            del x[len(x)-1]
                            x.insert(0, ex_wl_values[i])
                            x.insert(1, em_wl_values[i])
                            
                        df = pd.DataFrame(data, columns=cols)
                        df = df.replace("", np.nan)
                        df = df.dropna(axis=1, how="any")
                        df = df.melt(id_vars = df.columns[:3], value_vars= list(set(plate).intersection(df.columns)))
                        
                        df.columns =  ["excitation wavelength (nm)", "emission wavelength (nm)", "temperature (°C)", "wellnumber", "value (x)"]
                        #df["value (x)"] = df["value (x)"].replace('#SAT', np.nan)
                        df[["excitation wavelength (nm)", "emission wavelength (nm)", "temperature (°C)", "value (x)"]] = df[["excitation wavelength (nm)", "emission wavelength (nm)", "temperature (°C)", "value (x)"]].apply(pd.to_numeric)
                        df_sort = df.sort_values(["excitation wavelength (nm)", "emission wavelength (nm)", 'wellnumber'], ignore_index = True)
                    
                    #if contains_wavelength:
                    #    df.columns =  ["excitation wavelength (nm)", "emission wavelength (nm)", "temperature (°C)", "wellnumber", "value (x)"]
                    #    df["value (x)"] = df["value (x)"].replace('#SAT', np.nan)
                    #    df[["excitation wavelength (nm)", "emission wavelength (nm)", "temperature (°C)", "value (x)"]] = df[["excitation wavelength (nm)", "emission wavelength (nm)", "temperature (°C)", "value (x)"]].apply(pd.to_numeric)
                    #    df_sort = df.sort_values(["excitation wavelength (nm)", "emission wavelength (nm)", 'wellnumber'], ignore_index = True)
                        # liste = []
                        # for wavelength in pd.unique(df_sort['wavelength (nm)']):
                        #     sub_df = df_sort[df_sort['wavelength (nm)'] == wavelength]
                        #     for well in pd.unique(sub_df['wellnumber'].str[0]):
                        #         char_df = sub_df[sub_df['wellnumber'].str[0] == well[0]]
                        #         val1 = char_df['value (x)'].iloc[0]
                        #         corr_vals = char_df['value (x)'] - val1
                        #         liste.append(corr_vals)
                        #         #df_sort['corrected_value'] = pd.concat(liste)
                   # else:
                        #df.columns =  ["excitation wavelength (nm)", "emission wavelength (nm)", "temperature (°C)", "wellnumber", "value (x)"]
                        #df["value (x)"] = df["value (x)"].replace('#SAT', np.nan)
                        #df[["excitation wavelength (nm)", "emission wavelength (nm)", "temperature (°C)", "value (x)"]] = df[["excitation wavelength (nm)", "emission wavelength (nm)", "temperature (°C)", "value (x)"]].apply(pd.to_numeric)
                        #df_sort = df.sort_values(["excitation wavelength (nm)", "emission wavelength (nm)", 'wellnumber'], ignore_index = True)
                    
                    data_dict[f"Measurement{iterator}_{meta[1]}"] = df_sort
                    meta_clean = list(filter(None, meta))
                    data_dict[f"Metadata{iterator}_{meta[1]}"] = meta_clean
                    if ex_wl_data:
                        data_dict[f"FRET_ExWl_{fret_meta[20]}"] = ex_wl_data
                    meta = list()
                    data = list()
                    cols = list()
                    iterator += 1
                    df.empty

                else:
                    if len(line.split('\t')) > 0:
                        hlp = line.split('\t')
                        data.append(hlp)
    print(f"\nTo access the data dictionary (measurements and meta data), use the following keys:")
    print(''.join(str(key) + '\n' for key in data_dict.keys()))
    return data_dict


# def read_ID5_data(filepath) -> dict:
#     letter=["A","B","C","D","E","F","G","H"]
#     plate = []
#     for l in letter:
#         for i in range(1,13):
#             plate.append(l+str(i))

#     meta = []
#     meta_list = []
#     data = []
#     cols = []
#     data_dict = {}
#     iterator = 1
#     ex_wl_data = None
#     contains_wavelength = False

#     with open(filepath, 'r', encoding='UTF-8') as file:
#         print('Using ID5-function to read ID5-files...')
#         next(file)
#         lines = file.readlines()
        
#         for line in lines:
#             if not line.isspace():
#                 line = line.strip('\n')
#                 if line.startswith("Group"):
#                 # hier conzentrationen auslesen
#                 # evtl als zweiten dict/df
#                     break
#                 if line.startswith("Plate"):
#                     meta = line.split("\t")
#                     if 'Absorbance' in meta: # if "absorbance" is in meta data list
#                         abs_meta = meta # write meta data into variable abs_meta (absorbance meta data)
#                         print(f"Absorbance {abs_meta[4]} Experiment '{abs_meta[1]}': emission wavelength {abs_meta[11]}nm - {abs_meta[12]}nm in steps of {abs_meta[13]}nm") 
#                         abs_meta_list = list(filter(None, abs_meta)) # delete empty strings from list abs_meta
#                         meta_list.append(abs_meta_list) # add the absorbance meta data to big meta data list
#                     elif 'Fluorescence' in meta and meta[16] == "":
#                         fluo_meta = meta # write meta data into variable fluo_meta (fluorescence meta data)
#                         print(f"Fluorescence {fluo_meta[4]} Experiment '{fluo_meta[1]}': emission wavelength {fluo_meta[12]}nm - {fluo_meta[13]}nm in steps of {fluo_meta[14]}nm, {fluo_meta[23]}: {fluo_meta[24]}nm")
#                         fluo_meta_list = list(filter(None, fluo_meta))  # delete empty strings from list fluo_meta
#                         meta_list.append(fluo_meta_list) # add the fluorescence meta data to big meta data list
#                     elif meta[16] != "" or 'FRET' in meta: 
#                         fret_meta = meta  # write meta data into variable fret_meta (fluorescence meta data)
#                         print(f"FRET {fret_meta[4]} Experiment '{fret_meta[1]}': emission wavelengths {fret_meta[16].split(' ')[0]}nm and {fret_meta[16].split(' ')[1]}nm, excitation wavelength: {fret_meta[20]}nm")
#                         fret_meta_list = list(filter(None, fret_meta)) # delete empty strings from list fret_meta
#                         wavelength_fret = [fret_meta[16].split(' ')[0], fret_meta[16].split(' ')[1]]
#                         meta_list.append(fret_meta_list) # add the fret meta data to big meta data list
#                         ex_wl_data = fret_meta[20]

#                 elif line.startswith("Wavelength"):
#                     cols = line.split('\t')
#                     contains_wavelength = True
            
#                 elif line.startswith("	Temperature(¡C)"):
#                     contains_wavelength = False
#                     cols = line.split('\t')
#                     cols = cols[:-1]
#                     del cols[0]
#                     cols.insert(0,"Wavelength")
#                     specific_colnames = ["wavelength (nm)","temperature (°C)", "wellnumber", "value (x)"]
                    
#                 elif line.startswith("~End"):
#                     if contains_wavelength == True:
#                         df = pd.DataFrame(data, columns=cols)
#                         df = df.iloc[1: , :]
#                         df = df.replace("", np.nan)
#                         df = df.dropna(axis=1, how="any")
#                         df = df.melt(id_vars = df.columns[:2],value_vars= list(set(plate).intersection(df.columns)))
#                     else:
#                         for i, x in enumerate(data):
#                             del x[0]
#                             del x[len(x)-1]
#                             x.insert(0,wavelength_fret[i])
#                         df =  pd.DataFrame(data, columns=cols)
#                         df = df.replace("", np.nan)
#                         df = df.dropna(axis=1, how="any")
#                         df = df.melt(id_vars = df.columns[:2],value_vars= list(set(plate).intersection(df.columns)))
                    
#                     if contains_wavelength:
#                         df.columns =  ["wavelength (nm)", "temperature (°C)", "wellnumber", "value (x)"]
#                         df[["wavelength (nm)", "temperature (°C)", "value (x)"]] = df[["wavelength (nm)", "temperature (°C)", "value (x)"]].apply(pd.to_numeric)
#                         df_sort = df.sort_values(['wavelength (nm)', 'wellnumber'], ignore_index = True)
#                         liste = []
#                         for wavelength in pd.unique(df_sort['wavelength (nm)']):
#                             sub_df = df_sort[df_sort['wavelength (nm)'] == wavelength]
#                             for well in pd.unique(sub_df['wellnumber'].str[0]):
#                                 char_df = sub_df[sub_df['wellnumber'].str[0] == well[0]]
#                                 val1 = char_df['value (x)'].iloc[0]
#                                 corr_vals = char_df['value (x)'] - val1
#                                 liste.append(corr_vals)
#                                 #df_sort['corrected_value'] = pd.concat(liste)
#                     else:
#                         df.columns =  ["wavelength (nm)","temperature (°C)", "wellnumber", "value (x)"]
#                         df[["wavelength (nm)", "temperature (°C)", "value (x)"]] = df[["wavelength (nm)", "temperature (°C)", "value (x)"]].apply(pd.to_numeric)
#                         df_sort = df.sort_values(['wavelength (nm)', 'wellnumber'], ignore_index = True)
                    
#                     data_dict[f"Measurement{iterator}_{meta[1]}"] = df_sort
#                     meta_clean = list(filter(None, meta))
#                     data_dict[f"Metadata{iterator}_{meta[1]}"] = meta_clean
#                     if ex_wl_data:
#                         data_dict[f"FRET_ExWl_{fret_meta[20]}"] = ex_wl_data
#                     meta = list()
#                     data = list()
#                     cols = list()
#                     iterator += 1
#                     df.empty

#                 else:
#                     if len(line.split('\t')) > 0:
#                         hlp = line.split('\t')
#                         data.append(hlp)
#     print(f"\nTo access the data dictionary (measurements and meta data), use the following keys:")
#     print(''.join(str(key) + '\n' for key in data_dict.keys()))
#     #print(f"\nThe excitation wavelengths are: ")
#     return data_dict

def get_well(dataframe: DataFrame, wellnumber: str, measurement: str = None) -> DataFrame:
    """
    Function to get the desired well of a dataframe

    Parameters
    ----------

    dataframe: DataFrame
        the dataframe or dictionary
    measurement: str
        desired measurement (eg. "Measurement1_Emission Cy5")
    wellnumber: str
        number of the well (eg. "A2")
    
    returns
    -------
    DataFrame
        a dataframe with all data of specified measurement and specified wellnumber.
        Columns: wavelength, temperature, well, value, (corrected value)
    """
    
    if isinstance(dataframe, pd.DataFrame):
        spec_df = dataframe[dataframe["wellnumber"] == wellnumber]
        
        if spec_df.empty:
            print("Wellnumber does not exist.")
        else:
            return spec_df

    else:    
        try: 
            meas_x = dataframe[measurement]
        except KeyError:
            print("Measurement does not exist.")
        else:
            meas_well = meas_x[(meas_x.loc[:,'wellnumber'] == wellnumber)]
            if meas_well.empty:
                print("Wellnumber does not exist.")
            else:
                return meas_well


def correction_matrix(dataframe: DataFrame, measurement_cy3: str, measurement_cy5: str, wellnumber: str, ex1 = 'default', ex2 = 'default') -> DataFrame:
    """
    Function to generate a correction matrix (a dataframe really) from the measurements of Cy3 and Cy5 (ID5)
    
    needed for calculation bleedthrough and direct Excitation 
    (see functions calculate_bleedthrough() and calculate_directExcitation())
    
    parameters
    -----------
    dataframe: DataFrame
        the dataframe/dictionary, that is generated after reading the ID5-file with the "read_ID5_data" function. 
    measurement_cy3: str
        the specific Cy3 measurement of that aforementioned dataframe for the correction matrix 
        (eg. "Measurement1_Corr. Matrix Cy3")
    measurement_cy5: str
        the specific Cy5 measurement of that aforementioned dataframe for the correction matrix 
        (eg. "Measurement1_Corr. Matrix Cy5")
    wellnumber: str
        number of the well (eg. "A2")
    ex1: int
        first excitation wavelength (eg. 595) - default is 'default'
    ex2: int
        second excitation wavelegth (eg. 666) - default is 'default'

    example use: 
    correct_mat = correction_matrix(data, "Measurement1_Corr. Matrix Cy3", "Measurement1_Corr. Matrix Cy5", "A2", 595, 666)

    returns a dataframe with the desired values
    """
    meas_cy3 = get_well(dataframe=dataframe, measurement=measurement_cy3, wellnumber=wellnumber)
    meas_cy5 = get_well(dataframe=dataframe, measurement=measurement_cy5, wellnumber=wellnumber)
    rows = [[f"ex_{ex1}", list(meas_cy3["value (x)"])[0], list(meas_cy3["value (x)"])[1]], [f"ex_{ex2}", 0.0, list(meas_cy5["value (x)"])[0]]]
    corrmat_cols = pd.unique(dataframe[measurement_cy3]["emission wavelength (nm)"])
    cols = ["Em/Ex", f"em_{corrmat_cols[0]}", f"em_{corrmat_cols[1]}"]
    corrMat = pd.DataFrame(rows, columns=cols)

    return corrMat

def calculate_bleedthrough(dataframe: DataFrame, don_acc_type: str) -> float:
    """
    Function to calculate bleedthrough for the acceptor or the donor.
    
    parameters
    ----------
    dataframe: DataFrame
            the correction matrix dataframe 
    don_acc_type: str
            either "A" for acceptor or "D" for donor

    example use: 
    bleedthrough_acceptor = calculate_bleedthrough(corrMat_cy5, "A")

    returns
    -------
    the calculated bleedthrough value
    """
    valid_type = {"A", "D"}
    if don_acc_type not in valid_type:
        print("Type unknown. Must be one of %r (A: acceptor, D: donor)" % valid_type)
    else:
        if don_acc_type == "D":
            print("You chose donor bleedthrough calculations.")
            if dataframe.iloc[0][2] == 0.0 or dataframe.iloc[0][1] == 0.0:
                print("Division with zero encountered. That is unfortunate.")
                print("Bleedthrough of donor set to zero.")
            else:
                bt_D = dataframe.iloc[0][2] / dataframe.iloc[0][1] # [row][col]
                print(f"Bleedthrough of donor: {round(bt_D, 4)} ({format(round(bt_D*100, 2),'.2f')}%).")
                return bt_D
        elif don_acc_type == "A":
            print("You chose acceptor bleedthrough calculations.")
            if dataframe.iloc[1][1] == 0.0 or dataframe.iloc[1][2] == 0.0:
                print("Division with zero encountered. It is what it is.")
                print("Bleedthrough of Acceptor set to zero.")
                return 0.0
            else:
                bt_A = dataframe.iloc[1][1] / dataframe.iloc[1][2] # [row][col]
                print(f"Bleedthrough of Acceptor: {round(bt_A, 4)} ({format(round(bt_A*100,2),'.2f')}%).")
                return bt_A

def calculate_directExcitation(dataframe: DataFrame, don_acc_type: str) -> float:
    """
    Function to calculate direct excitation for the acceptor or the donor.
    
    parameters
    ----------
    dataframe: DataFrame
            the correction matrix dataframe 
    don_acc_type: str
            either "A" for acceptor or "D" for donor

    example use: 
    directExcitation_acceptor = calculate_directExcitation(corrMat_cy5, "A")

    returns
    -------
    the calculated direct excitation value
    """
    valid_type = {"A", "D"}
    if don_acc_type not in valid_type:
        print("Type unknown. Must be one of %r (A: Acceptor, D: Donor)" % valid_type)
    else:
        if don_acc_type == "D":
            print("You chose donor direct excitation calculations.")
            if dataframe.iloc[1][1] == 0.0 or dataframe.iloc[0][1] == 0.0:
                print("Division with zero encountered. It is what it is.")
                print("Direct excitation of donor set to zero.")
                return 0.0
            else:
                dE_D = dataframe.iloc[1][1] / dataframe.iloc[0][1] # row, col
                print(f"Direct Excitation of Donor: {round(dE_D, 4)} ({format(round(dE_D*100,2), '.2f')}%).")
                return dE_D
        
        elif don_acc_type == "A":
            print("You chose acceptor bleedthrough calculations.")
            if dataframe.iloc[0][2] == 0.0 or dataframe.iloc[1][2] == 0.0:
                print("Encountered a zero in the calculation.")
                print("Direct Excitation of Acceptor set to zero.")
                return 0.0
            else:
                dE_A = dataframe.iloc[0][2] / dataframe.iloc[1][2] # row, col
                print(f"Direct Excitation of Acceptor: {round(dE_A, 4)} ({format(round(dE_A*100,2), '.2f')}%).")
                return dE_A

def calculate_bt_correction(dataframe: pd.DataFrame, measurement: str, wellnumber: str, bt_var, don_acc_type: str) -> float:
    """
    Function to calculate bleedthrough correction for the acceptor or the donor.
    
    parameters
    ----------
    dataframe: DataFrame
            the dataframe/dictionary, that is generated after reading the ID5-file with the "read_ID5_data" function. 
    measurement: str
            desired measurement (eg. "Measurement5_20 °C")
    wellnumber: str
            number of the well (eg. "A3")
    bt_var: a variable
            the beforehand calculated bleedthrough value variable
    don_acc_type: str
            either "A" for acceptor or "D" for donor

    example use: 
    bt_corr_D = calculate_bt_correction(data, "Measurement5_20 °C", "A3", bt_a, "D")

    returns
    -------
    the calculated bleedthrough correction value

    citation: 
    Börner R, Kowerko D, Hadzic MCAS, König SLB, Ritter M, et al. (2018) 
    Simulations of camera-based single-molecule fluorescence experiments. PLOS ONE 13(4): e0195277. 
    https://doi.org/10.1371/journal.pone.0195277
    """
    valid_type = {"A", "D"}
    if don_acc_type not in valid_type:
        print("Type unknown. Must be one of %r (A: Acceptor, D: Donor)" % valid_type)
    else:
        if don_acc_type == "D":
            print("You chose donor direct excitation calculations. \
                The input bleedthrough variable (bt_var) must be of bt acceptor.")
            data = get_well(dataframe=dataframe, measurement=measurement, wellnumber=wellnumber)
            bt_corr_D = data.iloc[0][3] - bt_var * data.iloc[1][3]
            print(f"I'^[D.em]_[D.ex] = {bt_corr_D}")
            return bt_corr_D
        
        elif don_acc_type == "A":
            print(f"You chose acceptor bleedthrough calculations. \
                The input bleedthrough variable (bt_var) must be of bt donor.")
            data = get_well(dataframe=dataframe, measurement=measurement, wellnumber=wellnumber)
            bt_corr_A = data.iloc[1][3] - bt_var * data.iloc[0][3]
            print(f"I'^[A.em]_[D.ex] = {bt_corr_A}")
            return bt_corr_A


def calculate_dED_correction(dataframe: DataFrame, measurement: str, wellnumber: str, dE_var) -> float:
    '''
    Function to calculate direct Excitation correction for the donor.

    parameters
    ----------
    dataframe: DataFrame
            the dataframe/dictionary, that is generated after reading the ID5-file with the "read_ID5_data" function. 
    measurement: str
            desired measurement (eg. "Measurement5_20 °C")
    wellnumber: str
            number of the well (eg. "A3")
    dE_var: a variable
            the beforehand calculated direct Excitation value variable

    example use: 
    dE_correction_donor = calculate_dED_correction(data, "Measurement5_20 °C", "A3", dE_A)

    returns
    -------
    the calculated direct Excitation correction value

    citation:
    Börner R, Kowerko D, Hadzic MCAS, König SLB, Ritter M, et al. (2018) 
    Simulations of camera-based single-molecule fluorescence experiments. PLOS ONE 13(4): e0195277. 
    https://doi.org/10.1371/journal.pone.0195277
    '''
    data = get_well(dataframe=dataframe, measurement=measurement, wellnumber=wellnumber)
    dE_corr = data.iloc[0, 3] - dE_var * data.iloc[1, 3]
    print(f"I''_[D.ex]^[D.em] = {dE_corr}")
    return dE_corr

def calculate_dEA_correction(dataframe: DataFrame, bt_corr_A, dE_var) -> float:
    """
    Function to calculate direct Excitation correction for the acceptor.
    
    parameters
    ----------
    dataframe: DataFrame
            the correction matrix dataframe of Cy5
    bt_corr_A: a variable
            the beforehand calculated corrected bleedthrough value variable
    dE_var: a variable
            the beforehand calculated direct Excitation value variable

    example use: 
    dE_correction_acceptor = calculate_dEA_correction(corrMat_cy5, bt_corr_A, dE_A)

    returns a value
    -------
    the calculated direct Excitation correction value

    citation:
    Börner R, Kowerko D, Hadzic MCAS, König SLB, Ritter M, et al. (2018) 
    Simulations of camera-based single-molecule fluorescence experiments. PLOS ONE 13(4): e0195277. 
    https://doi.org/10.1371/journal.pone.0195277
    """
    dE_corr = bt_corr_A - dE_var * dataframe.iloc[1, 2]
    print(f"I''_[D.ex]^[A.em] = {dE_corr}")
    return dE_corr

def calculate_FRET(dE_corr_A, dE_corr_D) -> float:
    '''
    Function to calculate FRET.

    parameters
    ----------
    dE_corr_A: a variable
            direct Excitation correction of acceptor
    dE_corr_D: a variable
            direct Excitation correction of donor
    example use: 
    fret_value = calculate_FRET(dE_corr_A, dE_corr_D)

    returns 
    -------
    the calculated FRET value

    citation:
    Börner R, Kowerko D, Hadzic MCAS, König SLB, Ritter M, et al. (2018) 
    Simulations of camera-based single-molecule fluorescence experiments. PLOS ONE 13(4): e0195277. 
    https://doi.org/10.1371/journal.pone.0195277
    '''
    FRET = dE_corr_A / (dE_corr_D + dE_corr_A)
    print(round(FRET, 3))
    return FRET


def calculate_bundleFRET(dataframe: DataFrame, corrMat_cy5, bt_d, dE_A) -> DataFrame:
    '''
    Function to calculate FRET in bundle.

    parameters
    ----------
    dataframe: DataFrame
            the dataframe with all temperatures (temperatur_data)
    corrMat_cy5: DataFrame
            correction matrix of Cy5
    bt_d: a variable
            the beforehand calculated bleedthrough (donor) value variable
    dE_A: a variable
            the beforehand calculated direct Excitation (acceptor) value variable

    example use: 
    fret_bundle = calculate_bundleFRET(temperature_data, bt_d=bt_d, dE_A=dE_A)#

    returns 
    -------
    the calculated FRET values in a dataframe
    '''
    well_list = dataframe["wellnumber"].unique()
    listfret = []

    for i in well_list:
        well_i = get_well(dataframe=dataframe, wellnumber=i)
        for temp in dataframe["temperature (°C)"].unique():
            
            temp_i = well_i[well_i["temperature (°C)"] == temp]

            #btcorrD = temp_i.iloc[0][3] - (bt_a * temp_i.iloc[1][3])
            btcorrA = temp_i.iloc[1, 3] - (bt_d * temp_i.iloc[0, 3])

            dEcorrD = temp_i.iloc[0, 3] - (dE_A * temp_i.iloc[1, 3])
            dEcorrA = btcorrA - (dE_A * corrMat_cy5.iloc[1, 2])

            fret_value = dEcorrA / (dEcorrD + dEcorrA)

            liste = [i, temp, dEcorrA, dEcorrD, fret_value]
            listfret.append(liste)
        
    fret_df = pd.DataFrame(listfret, columns=["wellnumber", "temperature (°C)", "dE_corr_A", "dE_corr_D", "FRET"])

    return fret_df



# Cary functions
def read_cary_abs_data(filename): # für absorptionsspektrum
    meta = []
    meta_list = []
    names_line = []
    data_list = []
    templist = []

    cols = []
    data = []
    data_dict = {}

    iterator = 1
    meth_to_samp_lines = False
    name_to_wl_lines = False

    with open(filename, 'r', encoding='UTF-8') as file:
        lines = file.readlines()
        for line in lines:
            if not line.isspace():
                line = line.strip('\n')

                if line.startswith('METHOD'):
                    meth_to_samp_lines = True
                    continue
                elif line.startswith('SAMPLES'):
                    meth_to_samp_lines = False
                    pass
                if meth_to_samp_lines:
                    line = line.strip('\n')
                    meta = line.split(',')
                    meta_list.append(meta)

                if line.startswith('Name'):
                    names_line = line.split(',')
                    names_line = list(filter(None, names_line))
                    name_to_wl_lines = True
                    continue
                elif line.startswith(',Wavelength'):
                    cols = line.split(',')
                    name_to_wl_lines = False
                    pass 
                if name_to_wl_lines:
                    line = line.strip('\n')
                    templist = line.split(',')
                    templist[:] = [item for item in templist if item != '']
                    data_list.append(templist)
            
                else:
                    if len(line.split(',')) > 0:
                        hlp = line.split(',')
                        data.append(hlp)


    data_fin = pd.DataFrame(data)
    data_fin = data_fin.iloc[16:, 1:]
    data_fin.columns = cols[1:]
    data_fin = data_fin[["Wavelength (nm)", "Abs"]]
    data_fin = data_fin.iloc[:, 3:]
    names_line.insert(1, "wavelength (nm)")
    del names_line[0]
    data_fin.columns = names_line
    data_fin.iloc[:] = data_fin.iloc[:].apply(pd.to_numeric)
    data_fin = data_fin.melt(id_vars=["wavelength (nm)"], value_vars=['KL 1.1', 'KL 1.2', 'TL 2.1', 'TL 2.2'], \
        var_name='sample', value_name='value (RFU)')

    dct = dict((item[0], item[1:]) for item in data_list)
    dctdat = dict((item[0], item[1:]) for item in meta_list)

    data_dict[f"Metadata_Method_{iterator}"] = dctdat
    data_dict[f"Metadata_Samples_{iterator}"] = dct
    data_dict[f"Measurement_{iterator}"] = data_fin

    print(f"\nTo access the data dictionary (measurements and meta data), use the following keys:")
    print(''.join(str(key) + '\n' for key in data_dict.keys()))
    return data_dict

def get_mess(dataframe: DataFrame, measurement: str, sample: str):
    """
    Function to get the desired contruct of a dataframe

    Parameters
    ----------
    dataframe: DataFrame
        the dataframe/dictionary, that is generated after reading the Cary 3500-file with the "read_caren_abs_data()" function. 
    measurement: str
        desired measurement (eg. "Measurement1")
    Construct: str
        name of sample (eg. "KL 1.1")
    
    returns
    -------
    DataFrame
        a dataframe with all data of specified measurement and specified construct.
        Columns: Wavelength (nm), Construct, Value
    """
    
    try: 
        meas_x = dataframe[measurement]
    except KeyError:
        print("Measurement does not exist.")
    else:
        meas_well = meas_x[(meas_x.loc[:,'sample'] == sample)]
        if meas_well.empty:
            print("Sample does not exist.")
        else:
            return meas_well
        


# NanoDrop functions
