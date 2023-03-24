import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize
from scipy.signal import find_peaks
from pathlib import Path


class Carry:
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path

    @staticmethod
    def add_column_data(df: pd.DataFrame, coloumn_name: str, values: list) -> pd.DataFrame:
        """
        Function to add data like concentration to the measurements.
        :param df: Dataframe to which the data should be appended.
        :param coloumn_name: Name of the column.Example: cK (mM)
        :param values: List of values per measurement (cuvette). Example: ["0","1","10","100"]
        :return: Dataframe with appended Date
        """
        pass

    def parse_metling_curve(self):
        print('Reading Cary3500-data...')
        data = pd.read_csv(self.file_path, header=1)
        # initialisation of variables
        data_dict = {}
        # setting iterator and i to value needed for first loop
        iterator = 1
        i = 0

        with open(self.file_path, 'r', encoding='UTF-8') as file:
            firstline = file.readlines()[0]
            firstline = firstline.strip('\n')
            # Here the metadata is loaded in for the very first line only
            # line.split() creates a list with all elements between the separator
            meta = firstline.split(',')
            meta = list(filter(None, meta))
            # create short elements in llist for naming keys in data_dict
            meta_list = []
            for element in meta:
                meta_split = element.split('_')
                meta_list.append(meta_split)

        # if dataframe has even number of columns
            # for every list element in list called "meta"

        list_of_data = []
        for number_of_measurement, element in enumerate(meta_list):
            # key = list element with numbering (iterator), value = every 2 columns
            measure = data.iloc[:, i:i + 2]
            measure.columns.values[0:2] = ["Temperature (C)", "Abs"]
            #measure["name"] = element[0]
            measure.insert(0, "measure_number", number_of_measurement)
            measure.insert(1, "name", element[0])
            measure.insert(2, "wavelength_nm", element[1])
            measure.insert(3, "temperature_range_celcius", element[2])
            #measure.assign(name=element[0])
            #measure.assign(wavelength=element[1])
            #measure.assign(temperature_range=element[2])
            #measure["wavelength (nm)"] = element[1]
            #measure["temperature_range (°C)"] = element[2]
            measure = measure.dropna()
            # adding metadata
            list_of_data.append(measure)
            # setting i and iterators to value needed in next loop
            i = i + 2
            iterator += 1

        df = pd.concat(list_of_data)

        print(
            f"\nTo access the data dictionary (measurements and meta data), use the following keys: \n{', '.join(str(key) for key in data_dict.keys())}\n")
        return df


    def parse_data_with_meta(self):
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

        with open(self.file_path, 'r', encoding='UTF-8') as file:
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
        data_fin = data_fin.melt(id_vars=["wavelength (nm)"],
                                 value_vars=['KL 1.1', 'KL 1.2', 'TL 2.1', 'TL 2.2'],
                                 var_name='sample',
                                 value_name='value (RFU)')

        dct = dict((item[0], item[1:]) for item in data_list)
        dctdat = dict((item[0], item[1:]) for item in meta_list)

        data_dict[f"Metadata_Method_{iterator}"] = dctdat
        data_dict[f"Metadata_Samples_{iterator}"] = dct
        data_dict[f"Measurement_{iterator}"] = data_fin

        print(f"\nTo access the data dictionary (measurements and meta data), use the following keys:")
        print(''.join(str(key) + '\n' for key in data_dict.keys()))
        return data_dict


if __name__ == '__main__':

    test_carry = Carry("carry_data/Export Data 2023_01_26_KOH_Schmelzkurve_3_PL.csv")
    test = test_carry.parse_metling_curve()
    print(test)