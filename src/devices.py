import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from dateutil.parser import parse
from scipy import optimize
from scipy.signal import find_peaks
from pathlib import Path


class Carry:
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self.x_mode = None
        self.x_mode = None
        self.collection_mode = None
        self.scan_range_start = None
        self.scan_range_ende = None
        self.data_interval = None
        self.scan_rate = None
        self.averanging_time = None
        self.spectral_bandwidth = None
        self.detector_modul = None
        self.baseline = None
        self.experiment_zones = None
        self.data = None
        self.read_data()

    def add_names_from_file(self, df):
        names = []
        with open(self.file_path, 'r', encoding='UTF-8') as file:
            firstline = file.readlines()[0]
            firstline = firstline.strip('\n')
            meta = firstline.split(',')
            meta = list(filter(None, meta))
            for name in meta:
                names.append(name)

        measure = 1
        for i, name in enumerate(names):
            df.loc[df['Measurement'] == measure, "Name"] = name
            measure += 1

        return df

    def add_column_data(self, coloumn_name: str, values: list) -> None:
        """
        Function to add data like concentration to the measurements.
        :param df: Dataframe to which the data should be appended.
        :param coloumn_name: Name of the column.Example: cK (mM)
        :param values: List of values per measurement (cuvette). Example: ["0","1","10","100"]
        :return: Dataframe with appended Date
        """

        measures = sorted(set(self.data["Measurement"]))
        for i, value in enumerate(values):
            self.data.loc[self.data['Measurement'] == measures[i], coloumn_name] = value

    def parse_melting_curve_data(self):
        data = pd.read_csv(self.file_path, header=1)

        new_df = pd.concat([pd.DataFrame({"Temperature (C)": data[t_col], "Abs": data[abs_col], "Measurement": i + 1})
                            for i, (t_col, abs_col) in
                            enumerate(zip(data.filter(like="Temperature"), data.filter(like="Abs")))],
                            ignore_index=True)


        self.data = self.add_names_from_file(new_df.dropna())

    def parse_data_absorbtion_spectra(self):
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
            print(lines[0])
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

    def read_data(self):
        with open(self.file_path, 'r', encoding='UTF-8') as file:
            lines = file.readlines()
            print(lines[0])
            firstline = lines[0]
            if "CSV Report" in firstline:
                print("Carrydata with meta")
                self.parse_data_absorbtion_spectra()
            else:
                print("Carrydata without meta")
                self.parse_melting_curve_data()


class ID5MeasureAbsorbance:
    def __init__(self, metadata: list, data: list) -> None:
        self.section_type = metadata[0]
        self.section_name = metadata[1]
        self.export_version = metadata[2]
        self.export_format = metadata[3]
        self.read_type = metadata[4]
        self.type_read_mode = metadata[5]
        self.data_type = metadata[6]
        self.pre_read = metadata[7]
        self.kinetic_point = metadata[8]
        self.read_time_pattern = metadata[9]
        self.kinetic_interval_well_scan_density = metadata[10]
        self.start_wavelength = metadata[11]
        self.end_wavelength = metadata[12]
        self.wavelength_step = metadata[13]
        self.number_of_wavelength = metadata[14]
        self.wavelengths = metadata[15]
        self.first_column = metadata[16]
        self.number_of_columns = metadata[17]
        self.number_of_wells = metadata[18]
        self.first_row = metadata[19]
        self.number_of_rows = metadata[20]
        self.time_tags = metadata[21]
        self.data = data
        self.restructure_data()

    @staticmethod
    def create_plate_id_list() -> list:
        letter = ["A", "B", "C", "D", "E", "F", "G", "H"]
        plate = []
        for l in letter:
            for i in range(1, 13):
                plate.append(l + str(i))
        return plate

    def get_well(self, wellnumber: str) -> pd.DataFrame:
        """
        Function to get the desired well of a dataframe

        Parameters
        ---------
        wellnumber: str
            number of the well (eg. "A2")

        returns
        -------
        DataFrame
            a dataframe with all data of specified measurement and specified wellnumber.
            Columns: wavelength, temperature, well, value, (corrected value)
        """
        wellnums_unique = self.working_df["wellnumber"].unique()
        spec_df = self.working_df[self.working_df["wellnumber"] == wellnumber]

        if spec_df.empty:
            print("ERROR: Wellnumber does not exist.")
            print(f"Accessible wellnumbers: {wellnums_unique}")
        else:
            return spec_df

    def print_meta_data(self):
        if self.type_read_mode == 'Absorbance':
            print(
                f"Experiment '{self.section_name}': emission wavelength {self.start_wavelength} nm - {self.end_wavelength} nm in steps of {self.wavelength_step} nm.")

    def restructure_data(self):
        if self.type_read_mode == "Absorbance":
            df = pd.DataFrame(self.data[1:], columns=self.data[0])
            df = df.replace("", np.nan)
            df = df.dropna(axis=1, how="any")
            df = df.melt(id_vars=df.columns[:2],
                        value_vars=list(set(self.create_plate_id_list()).intersection(df.columns)))
            df.columns = ["wavelength (nm)", "temperature (°C)", "wellnumber", "Abs"]
            df["Abs"] = df["Abs"].replace('#SAT', np.nan)
            df["wavelength (nm)"] = df["wavelength (nm)"].astype(float)
            df["temperature (°C)"] = df["temperature (°C)"].astype(float)
            df["Abs"] = df["Abs"].astype(float)
            df_sort = df.sort_values(["temperature (°C)", "wellnumber"], ignore_index=True)
            self.working_df = df_sort
            df.empty
            df_sort.empty
        else:
            print(
                f"Current read type = {self.type_read_mode}. If you can read this, implementation not done or file content faulty.")


class ID5MeasureFluorescence:
    def __init__(self, metadata: list, data: list, emission_wavelength) -> None:
        self.section_type = metadata[0]
        self.section_name = metadata[1]
        self.export_version = metadata[2]
        self.export_format = metadata[3]
        self.read_type = metadata[4]
        self.type_read_mode = metadata[5]
        self.bottom_read = metadata[6]
        self.data_type = metadata[7]
        self.pre_read = metadata[8]
        self.kinetic_point = metadata[9]
        self.read_time_pattern = metadata[10]
        self.kinetic_interval_well_scan_density = metadata[11]
        self.start_wavelength = metadata[12]
        self.end_wavelength = metadata[13]
        self.wavelength_step = metadata[14]
        self.number_of_wavelength = pd.to_numeric(metadata[15])
        self.wavelengths = metadata[16]
        self.first_column = metadata[17]
        self.number_of_columns = metadata[18]
        self.number_of_wells = metadata[19]
        self.excitation_wavelength = metadata[20]
        self.cutoff = metadata[21]
        self.cutoff_filters = metadata[22]
        self.sweep_waves = metadata[23]
        self.sweep_fixed_wavelength = metadata[24]
        self.reads_per_well = metadata[25]
        self.pmt_gain = metadata[26]
        self.start_integration_time = metadata[27]
        self.end_integration_time = metadata[28]
        self.first_row = metadata[29]
        self.number_of_rows = metadata[30]
        self.time_tags = metadata[31]
        self.data = data
        self.emission_wavelength = emission_wavelength
        self.restructure_data()

    @staticmethod
    def create_plate_id_list() -> list:
        letter = ["A", "B", "C", "D", "E", "F", "G", "H"]
        plate = []
        for l in letter:
            for i in range(1, 13):
                plate.append(l + str(i))
        return plate

    def get_well(self, wellnumber: str) -> pd.DataFrame:
        """
        Function to get the desired well of a dataframe

        Parameters
        ---------
        wellnumber: str
            number of the well (eg. "A2")

        returns
        -------
        DataFrame
            a dataframe with all data of specified measurement and specified wellnumber.
            Columns: wavelength, temperature, well, value, (corrected value)
        """
        wellnums_unique = self.working_df["wellnumber"].unique()
        spec_df = self.working_df[self.working_df["wellnumber"] == wellnumber]

        if spec_df.empty:
            print("ERROR: Wellnumber does not exist.")
            print(f"Accessible wellnumbers: {wellnums_unique}")
        else:
            return spec_df

    def print_meta_data(self):
        if self.type_read_mode == 'Fluorescence' and self.wavelengths is None:
            print(
                f"Experiment '{self.section_name}': emission wavelength {self.start_wavelength} nm - {self.end_wavelength} nm in steps of {self.wavelength_step}nm, {self.sweep_waves}: {self.sweep_fixed_wavelength}nm")
        elif self.emission_wavelength is not None:
            print_excitation_wavelength = self.excitation_wavelength.split()
            print_emission_wavelength = self.emission_wavelength.split()
            print(
                f"Experiment '{self.section_name}': \n"
                f"excitation    emission wavelengths \n"
                f"{print_excitation_wavelength[0]} nm        {print_emission_wavelength[0]} nm \n"
                f"{print_excitation_wavelength[1]} nm        {print_emission_wavelength[1]} nm \n"
                f"{print_excitation_wavelength[2]} nm        {print_emission_wavelength[2]} nm \n")

    def restructure_data(self):
        if self.read_type == "Spectrum":
            df = pd.DataFrame(self.data[1:], columns=self.data[0])
            df = df.replace("", np.nan)
            df = df.dropna(axis=1, how="any")
            df = df.melt(id_vars=df.columns[:2],
                        value_vars=list(set(self.create_plate_id_list()).intersection(df.columns)))
            df.columns = ["wavelength (nm)", "temperature (°C)", "wellnumber", "RFU"]
            df["RFU"] = df["RFU"].replace('#SAT', np.nan)
            df["wavelength (nm)"] = df["wavelength (nm)"].astype(float)
            df["temperature (°C)"] = df["temperature (°C)"].astype(float)
            df["RFU"] = df["RFU"].astype(float)
            df_sort = df.sort_values(["temperature (°C)", "wellnumber"], ignore_index=True)
            self.working_df = df_sort
            df.empty
            df_sort.empty

        elif self.read_type == "Endpoint":
            if self.emission_wavelength is None:
                if self.number_of_wavelength != len(self.excitation_wavelength.split()):
                    self.excitation_wavelength = input(
                        f"Please enter Excitation Wavelength ({self.number_of_wavelength}) for measurement {self.section_name} separated by space: \n")
                    self.emission_wavelength = input(
                        f"Please enter Emission Wavelength ({self.number_of_wavelength}) for measurement {self.section_name} separated by space: \n")
                    excitation_wavelength = f"ex_wl {self.excitation_wavelength}".split()
                    emission_wavelength = f"em_wl {self.emission_wavelength}".split()

                else:
                    self.emission_wavelength = input(
                        f"Please enter Emission Wavelength ({self.number_of_wavelength}) for measurement {self.section_name} separated by space: \n")
                    excitation_wavelength = f"ex_wl {self.excitation_wavelength}".split()
                    emission_wavelength = f"em_wl {self.emission_wavelength}".split()

            else:
                if len(self.emission_wavelength.split()) != len(self.excitation_wavelength.split()):
                    if self.number_of_wavelength != len(self.excitation_wavelength.split()):
                        self.excitation_wavelength = input(
                            f"Please re-enter Excitation Wavelength ({self.number_of_wavelength}) for measurement {self.section_name} separated by space: \n")
                        self.emission_wavelength = input(
                            f"Please re-enter Emission Wavelength ({self.number_of_wavelength}) for measurement {self.section_name} separated by space: \n")
                        excitation_wavelength = f"ex_wl {self.excitation_wavelength}".split()
                        emission_wavelength = f"em_wl {self.emission_wavelength}".split()

                    else:
                        self.emission_wavelength = input(
                            f"Please re-enter Emission Wavelength ({len(self.excitation_wavelength.split())}) for measurement {self.section_name} separated by space: \n")
                        excitation_wavelength = f"ex_wl {self.excitation_wavelength}".split()
                        emission_wavelength = f"em_wl {self.emission_wavelength}".split()

                else:
                    excitation_wavelength = f"ex_wl {self.excitation_wavelength}".split()
                    emission_wavelength = f"em_wl {self.emission_wavelength}".split()

            for i, data_line in enumerate(self.data):
                del data_line[0]
                del data_line[len(data_line) - 1]
                data_line.insert(0, excitation_wavelength[i])
                data_line.insert(1, emission_wavelength[i])

            df = pd.DataFrame(self.data[1:], columns=self.data[0])
            df = df.replace("", np.nan)
            df = df.dropna(axis=1, how="any")
            df = df.melt(id_vars=df.columns[:3],
                        value_vars=list(set(self.create_plate_id_list()).intersection(df.columns)))
            df.columns = ["excitation wavelength (nm)", "emission wavelength (nm)", "temperature (°C)", "wellnumber",
                        "RFU"]
            df["RFU"] = df["RFU"].replace('#SAT', np.nan)
            df[["excitation wavelength (nm)", "emission wavelength (nm)", "temperature (°C)", "RFU"]] = df[
                ["excitation wavelength (nm)", "emission wavelength (nm)", "temperature (°C)", "RFU"]].apply(
                pd.to_numeric)
            df_sort = df.sort_values(["excitation wavelength (nm)", "emission wavelength (nm)", 'wellnumber'],
                                    ignore_index=True)
            self.working_df = df_sort
            df.empty
            df_sort.empty

        else:
            print(f"Current read type = {self.read_type}. If you can read this, implementation not done.")



class ID5:
    def __init__(self, file_path: str, emission_wavelength=None):
        self.file_path = file_path
        self.measurements = {}
        self.emission_wavelength = emission_wavelength
        self.working_df = self.read_id5_data() 
        self.correct_matrix = self.correction_matrix()

        #self.well_acquired = self.get_well()

    def read_id5_data(self) -> None:
        number_of_measurements = 0
        meta = []
        data = []
        iterator = 0

        # ex_wl_data = None
        # contains_wavelength = False

        with open(self.file_path, 'r', encoding='UTF-8') as file:
            lines = file.readlines()

            for line in lines:
                if iterator <= int(number_of_measurements):
                    if not line.isspace():
                        line = line.strip('\n')
                        if line.startswith("##BLOCKS="):
                            number_of_measurements = line.split(" ")[1]
                        elif line.startswith("Group") or line.startswith("Original Filename") or line.startswith(
                                "Workflow"):
                            break
                        elif line.startswith("Plate"):
                            meta = line.split("\t")
                            meta = [None if x == "" else x for x in meta]
                        elif line.startswith("~End"):
                            iterator += 1
                            read_type = meta[4]
                            read_mode = meta[5]
                            if read_mode == "Absorbance":
                                self.measurements[f"Measurement_{iterator}"] = ID5MeasureAbsorbance(meta, data)
                            elif read_mode == "Fluorescence":
                                self.measurements[f"Measurement_{iterator}"] = ID5MeasureFluorescence(meta, data,
                                                                                                    self.emission_wavelength)
                            elif read_mode == "Luminescence" or "Time Resolved" or "Imaging":
                                print("Data reading routine is not implemented for these types of experiments.")
                            else:
                                print("Unknown type of experiment. Check your file or implement new data reading routine.")
                            print(f"{meta[1]}: {read_type}, {read_mode}") # print accessible keys of dict
                            data = []
                            meta = []
                        else:
                            if len(line.split('\t')) > 0:
                                hlp = line.split('\t')
                                data.append(hlp)
                else:
                    break

    def get_well(self, wellnumber: str) -> pd.DataFrame:
        wellnums_unique = self.working_df["wellnumber"].unique()
        spec_df = self.working_df[self.working_df["wellnumber"] == wellnumber]

        if spec_df.empty:
            print("ERROR: Wellnumber does not exist.")
            print(f"Accessible wellnumbers: {wellnums_unique}")
        else:
            return spec_df

    def correction_matrix(self, measurement_cy3: str, measurement_cy5: str, wellnumber: str, ex1 = 'default', ex2 = 'default'):

        meas_cy3 = self.get_well(dataframe=self.measurements, measurement=measurement_cy3, wellnumber=wellnumber)
        meas_cy5 = self.get_well(dataframe=self.measurements, measurement=measurement_cy5, wellnumber=wellnumber)

        rows = [[f"ex_{ex1}", list(meas_cy3["value (x)"])[0], list(meas_cy3["value (x)"])[1]], [f"ex_{ex2}", 0.0, list(meas_cy5["value (x)"])[0]]]
        corrmat_cols = pd.unique(self.measurements[measurement_cy3]["emission wavelength (nm)"])
        cols = ["Ex/Em", f"em_{corrmat_cols[0]}", f"em_{corrmat_cols[1]}"]

        self.correct_matrix = pd.DataFrame(rows, columns=cols)

        return self.correct_matrix

    # def calculate_bleedthrough(dataframe: DataFrame, don_acc_type: str) -> float:
    #     """
    #     Function to calculate bleedthrough for the acceptor or the donor.
        
    #     parameters
    #     ----------
    #     dataframe: DataFrame
    #             the correction matrix dataframe 
    #     don_acc_type: str
    #             either "A" for acceptor or "D" for donor

    #     example use: 
    #     bleedthrough_acceptor = calculate_bleedthrough(corrMat_cy5, "A")

    #     returns
    #     -------
    #     the calculated bleedthrough value
    #     """
    #     valid_type = {"A", "D"}
    #     if don_acc_type not in valid_type:
    #         print("Type unknown. Must be one of %r (A: acceptor, D: donor)" % valid_type)
    #     else:
    #         if don_acc_type == "D":
    #             print("You chose donor bleedthrough calculations.")
    #             if dataframe.iloc[0][2] == 0.0 or dataframe.iloc[0][1] == 0.0:
    #                 print("Division with zero encountered. That is unfortunate. \nBleedthrough of donor set to zero.")
    #             else:
    #                 bt_D = dataframe.iloc[0][2] / dataframe.iloc[0][1] # [row][col]
    #                 print(f"Bleedthrough of donor: {round(bt_D, 4)} ({format(round(bt_D*100, 2),'.2f')}%).")
    #                 return bt_D
    #         elif don_acc_type == "A":
    #             print("You chose acceptor bleedthrough calculations.")
    #             if dataframe.iloc[1][1] == 0.0 or dataframe.iloc[1][2] == 0.0:
    #                 print("Division with zero encountered. It is what it is. \nBleedthrough of Acceptor set to zero.")
    #                 return 0.0
    #             else:
    #                 bt_A = dataframe.iloc[1][1] / dataframe.iloc[1][2] # [row][col]
    #                 print(f"Bleedthrough of Acceptor: {round(bt_A, 4)} ({format(round(bt_A*100,2),'.2f')}%).")
    #                 return bt_A
    
    # def calculate_directExcitation(dataframe: DataFrame, don_acc_type: str):
    #     """
    #     Function to calculate direct excitation for the acceptor or the donor.
        
    #     parameters
    #     ----------
    #     dataframe: DataFrame
    #             the correction matrix dataframe 
    #     don_acc_type: str
    #             either "A" for acceptor or "D" for donor

    #     example use: 
    #     directExcitation_acceptor = calculate_directExcitation(corrMat_cy5, "A")

    #     returns
    #     -------
    #     the calculated direct excitation value
    #     """
    #     valid_type = {"A", "D"}
    #     if don_acc_type not in valid_type:
    #         print("Type unknown. Must be one of %r (A: Acceptor, D: Donor)" % valid_type)
    #     else:
    #         if don_acc_type == "D":
    #             print(f"You chose donor direct excitation calculations.")
    #             if dataframe.iloc[1][1] == 0.0 or dataframe.iloc[0][1] == 0.0:
    #                 print("Divide by zero encountered. It is what it is.")
    #                 print(f"Direct excitation of donor set to zero.")
    #                 return 0.0
    #             else:
    #                 dE_D = dataframe.iloc[1][1] / dataframe.iloc[0][1] # row, col
    #                 print(f"Direct Excitation of Donor: {dE_D} ({format(round(dE_D*100,2), '.2f')}%).")
    #                 return dE_D
            
    #         elif don_acc_type == "A":
    #             print(f"You chose acceptor bleedthrough calculations.")
    #             if dataframe.iloc[0][2] == 0.0 or dataframe.iloc[1][2] == 0.0:
    #                 print("Encountered a zero in the calculation. Should it be there?")
    #                 print(f"Direct Excitation of Acceptor set to zero.")
    #                 return 0.0
    #             else:
    #                 dE_A = dataframe.iloc[0][2] / dataframe.iloc[1][2] # row, col
    #                 print(f"Direct Excitation of Acceptor: {dE_A} ({format(round(dE_A*100,2), '.2f')}%).")
    #                 return dE_A

    # def calculate_bt_correction(self, dataframe: DataFrame, measurement: str, wellnumber: str, bt_var, don_acc_type: str) -> float:
    #     """
    #     Function to calculate bleedthrough correction for the acceptor or the donor.
        
    #     parameters
    #     ----------
    #     dataframe: DataFrame
    #             the dataframe/dictionary, that is generated after reading the ID5-file with the "read_ID5_data" function. 
    #     measurement: str
    #             desired measurement (eg. "Measurement5_20 °C")
    #     wellnumber: str
    #             number of the well (eg. "A3")
    #     bt_var: a variable
    #             the beforehand calculated bleedthrough value variable
    #     don_acc_type: str
    #             either "A" for acceptor or "D" for donor

    #     example use: 
    #     bt_corr_D = calculate_bt_correction(data, "Measurement5_20 °C", "A3", bt_a, "D")

    #     returns
    #     -------
    #     the calculated bleedthrough correction value

    #     citation: 
    #     Börner R, Kowerko D, Hadzic MCAS, König SLB, Ritter M, et al. (2018) 
    #     Simulations of camera-based single-molecule fluorescence experiments. PLOS ONE 13(4): e0195277. 
    #     https://doi.org/10.1371/journal.pone.0195277
    #     """
    #     valid_type = {"A", "D"}
    #     if don_acc_type not in valid_type:
    #         print("Type unknown. Must be one of %r (A: Acceptor, D: Donor)" % valid_type)
    #     else:
    #         if don_acc_type == "D":
    #             print("You chose donor direct excitation calculations. \
    #                 The input bleedthrough variable (bt_var) must be of bt acceptor.")
    #             data = self.get_well(dataframe=dataframe, measurement=measurement, wellnumber=wellnumber)
    #             bt_corr_D = data.iloc[0][3] - bt_var * data.iloc[1][3]
    #             print(f"I'^[D.em]_[D.ex] = {bt_corr_D}")
    #             return bt_corr_D
            
    #         elif don_acc_type == "A":
    #             print(f"You chose acceptor bleedthrough calculations. \
    #                 The input bleedthrough variable (bt_var) must be of bt donor.")
    #             data = self.get_well(dataframe=dataframe, measurement=measurement, wellnumber=wellnumber)
    #             bt_corr_A = data.iloc[1][3] - bt_var * data.iloc[0][3]
    #             print(f"I'^[A.em]_[D.ex] = {bt_corr_A}")
    #             return bt_corr_A

    # def calculate_dED_correction(self, dataframe: DataFrame, measurement: str, wellnumber: str, dE_var):
    #     """
    #     Function to calculate direct Excitation correction for the donor.
        
    #     parameters
    #     ----------
    #     dataframe: DataFrame
    #             the dataframe/dictionary, that is generated after reading the ID5-file with the "read_ID5_data" function. 
    #     measurement: str
    #             desired measurement (eg. "Measurement5_20 °C")
    #     wellnumber: str
    #             number of the well (eg. "A3")
    #     dE_var: a variable
    #             the beforehand calculated direct Excitation value variable

    #     example use: 
    #     dE_correction_donor = calculate_dED_correction(data, "Measurement5_20 °C", "A3", dE_A)

    #     returns
    #     -------
    #     the calculated direct Excitation correction value

    #     citation:
    #     Börner R, Kowerko D, Hadzic MCAS, König SLB, Ritter M, et al. (2018) 
    #     Simulations of camera-based single-molecule fluorescence experiments. PLOS ONE 13(4): e0195277. 
    #     https://doi.org/10.1371/journal.pone.0195277
    #     """
    #     data = self.get_well(dataframe=dataframe, measurement=measurement, wellnumber=wellnumber)
    #     dE_corr = data.iloc[0, 3] - dE_var * data.iloc[1, 3]
    #     print(f"I''_[D.ex]^[D.em] = {dE_corr}")
    #     return dE_corr

    # def calculate_dEA_correction(dataframe: DataFrame, bt_corr_A, dE_var):
    #     """
    #     Function to calculate direct Excitation correction for the acceptor.
        
    #     parameters
    #     ----------
    #     dataframe: DataFrame
    #             the correction matrix dataframe of Cy5
    #     bt_corr_A: a variable
    #             the beforehand calculated corrected bleedthrough value variable
    #     dE_var: a variable
    #             the beforehand calculated direct Excitation value variable

    #     example use: 
    #     dE_correction_acceptor = calculate_dEA_correction(corrMat_cy5, bt_corr_A, dE_A)

    #     returns
    #     -------

    #     the calculated direct Excitation correction value

    #     citation:
    #     Börner R, Kowerko D, Hadzic MCAS, König SLB, Ritter M, et al. (2018) 
    #     Simulations of camera-based single-molecule fluorescence experiments. PLOS ONE 13(4): e0195277. 
    #     https://doi.org/10.1371/journal.pone.0195277
    #     """
    #     dE_corr = bt_corr_A - dE_var * dataframe.iloc[1, 2]
    #     print(f"I''_[D.ex]^[A.em] = {dE_corr}")
    #     return dE_corr

    # def calculate_FRET(dE_corr_A, dE_corr_D):
    #     '''
    #     Function to calculate FRET.

    #     parameters
    #     ----------
    #     dE_corr_A: a variable
    #             direct Excitation correction of acceptor
    #     dE_corr_D: a variable
    #             direct Excitation correction of donor
    #     example use: 
    #     fret_value = calculate_FRET(dE_corr_A, dE_corr_D)

    #     returns 
    #     -------
    #     the calculated FRET value

    #     citation:
    #     Börner R, Kowerko D, Hadzic MCAS, König SLB, Ritter M, et al. (2018) 
    #     Simulations of camera-based single-molecule fluorescence experiments. PLOS ONE 13(4): e0195277. 
    #     https://doi.org/10.1371/journal.pone.0195277
    #     '''

    #     FRET = dE_corr_A / (dE_corr_D + dE_corr_A)
    #     print(round(FRET, 3))
    #     return FRET

    # def calculate_bundleFRET(self, dataframe: DataFrame, corrMat_cy5, bt_d, dE_A):
    #     '''
    #     Function to calculate FRET in bundle.

    #     parameters
    #     ----------
    #     dataframe: DataFrame
    #             the dataframe with all temperatures (temperatur_data)
    #     corrMat_cy5: DataFrame
    #             correction matrix of Cy5
    #     bt_d: a variable
    #             the beforehand calculated bleedthrough (donor) value variable
    #     dE_A: a variable
    #             the beforehand calculated direct Excitation (acceptor) value variable

    #     example use: 
    #     fret_bundle = calculate_bundleFRET(temperature_data, bt_d=bt_d, dE_A=dE_A)

    #     returns 
    #     -------
    #     the calculated FRET values in a dataframe
    #     '''

    #     well_list = dataframe["wellnumber"].unique()
    #     listfret = []

    #     for i in well_list:
    #             well_i = self.get_well(dataframe=dataframe, wellnumber=i)
    #             for temp in dataframe["temperature (°C)"].unique():
    #                 temp_i = well_i[well_i["temperature (°C)"] == temp]
    #                 btcorrA = temp_i.iloc[1, 3] - (bt_d * temp_i.iloc[0, 3])
    #                 dEcorrD = temp_i.iloc[0, 3] - (dE_A * temp_i.iloc[1, 3])
    #                 dEcorrA = btcorrA - (dE_A * corrMat_cy5.iloc[1, 2])
    #                 fret_value = dEcorrA / (dEcorrD + dEcorrA)
    #                 liste = [i, temp, dEcorrA, dEcorrD, fret_value]
    #                 listfret.append(liste)
    #     fret_df = pd.DataFrame(listfret, columns=["wellnumber", "temperature (°C)", "dE_corr_A", "dE_corr_D", "FRET"])
    #     return fret_df



class Genesis:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.measurements = {}
        self.read_genesis_data()

    def read_genesis_data(self):

        df_raw = pd.read_csv(self.file_path, header=0, sep="\t", encoding='utf-8')
        df_raw = df_raw.dropna(axis=1)
        df_col_sorted = df_raw.sort_index(axis=1)
        df_with_means_of_all_measurements = pd.DataFrame(
            df_col_sorted['Wavelength(nm)'])  # leeres df erstellen für später

        col_list = [c.split(' ')[0] for c in df_col_sorted.columns if c != 'Wavelength(nm)']

        for index, item in enumerate(col_list):
            if item == 'Wavelength(nm)':
                pass
            else:
                df_measurement_x = df_col_sorted.filter(like=col_list[index], axis=1)
                measurement_df_with_meanval = pd.concat([df_col_sorted['Wavelength(nm)'], df_measurement_x.iloc[:, :],
                                                         df_measurement_x.iloc[:, :].mean(axis=1)], axis=1)
                measurement_df_with_meanval.columns = [f'mean Abs {item}' if x == 0 else x for x in
                                                       measurement_df_with_meanval.columns]
                self.measurements[f"Measurement_{item}"] = measurement_df_with_meanval

                df_with_means_of_all_measurements = pd.concat(
                    [df_with_means_of_all_measurements, measurement_df_with_meanval.iloc[:, -1:]], axis=1)
                index += 1

        self.measurements['Means_all'] = df_with_means_of_all_measurements.iloc[:,
                                         ~df_with_means_of_all_measurements.columns.duplicated()]
        self.measurements['Means_all'] = self.measurements['Means_all'].melt(id_vars=['Wavelength(nm)'])
        self.measurements['Means_all'] = self.measurements['Means_all'].rename(
            columns={'Wavelength(nm)': "wavelength (nm)", 'variable': "measurement", 'value': "mean Abs"})

        print(f"\nTo access the data dictionary (measurements), use the following keys:")
        print(''.join(str(key) + '\n' for key in self.measurements.keys()))

class Nanodrop:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.working_df = self.read_nano_data()

    @staticmethod
    def is_date(string, fuzzy=False):
        """
        Return whether the string can be interpreted as a date.

        :param string: str, string to check for date
        :param fuzzy: bool, ignore unknown tokens in string if True
        """
        try:
            parse(string, fuzzy=fuzzy)
            return True

        except ValueError:
            return False

    def read_nano_data(self) -> pd.DataFrame:
        """
        Function for reading Nanodrop .tsv files. Reads the spectrum data and creates a dataframe from all samples and
        returns it.
        The column names are: "sample", "wavelength (nm)", "Abs", "date_time"
        :return: Dataframe with all absorption spectra of all samples
        """

        sample = ""
        data = []

        with open(self.file_path, "r", encoding='UTF-8') as file:

            lines = file.readlines()

            for line in lines:
                if not line.isspace():
                    line = line.strip('\n')
                    if line.startswith("//"):
                        pass
                    elif self.is_date(line):
                        current_date = line
                    elif line.startswith("Wavelength"):
                        pass
                    elif line[0].isdigit():
                        line_splitted = line.split("\t")
                        if len(line_splitted) < 2:
                            pass
                        else:
                            data.append([sample, float(line_splitted[0]), float(line_splitted[1]), current_date])
                    else:
                        sample = line

            return pd.DataFrame(data, columns=["sample", "wavelength (nm)", "Abs", "date_time"])

    def print_samples(self) -> None:
        """
        Prints all sample names of the data.
        :return: None
        """
        sample_list = self.working_df["sample"].unique()
        print("The following samples are available in your measurement")
        for s in sample_list:
            print(f"{s}")

    def get_sample(self, sample_name) -> pd.DataFrame:
        """
        Filters a sample by its name from the whole data and returns it as a dataframe
        :param sample_name: string --> Name of the sample.
        :return: Dataframe with an absorption spectra of one sample
        """
        if sample_name in self.working_df["sample"].unique():
            sliced_df = self.working_df[self.working_df["sample"] == sample_name]
            return sliced_df
        else:
            print("Sample name not found in dataframe. Please check the sample name")

    def plot_sample(self, sample_name: str, color='#e34a33') -> plt.subplots():
        """
        Generates a plot of an absorption spectrum.
        :param sample_name: string --> Name of the sample.
        :param color: color of the data points (optional)
        :return: matplotlib figure object
        """
        if sample_name in self.working_df["sample"].unique():
            sample_df = self.get_sample(sample_name)
            # set plot size and font size
            plt.rcParams['figure.figsize'] = [11, 11]
            plt.rcParams['xtick.labelsize'] = 9
            plt.rcParams['ytick.labelsize'] = 9

            fig, ax = plt.subplots(figsize=(5, 3))

            # create a scatter plot
            ax.scatter(sample_df['wavelength (nm)'], sample_df['Abs'], marker='x', color=color, linewidth=0.5, s=10)

            # set axis labels
            ax.set_xlabel(r'wavelength $\lambda$ (nm)')
            ax.set_ylabel('Abs')
            legend_info = str(sample_df.iloc[1, 0])
            ax.legend([legend_info], loc='best')
            plt.subplots_adjust(bottom=0.2)

            # show plot
            plt.show()
            return fig
        else:
            print("Sample name not found in dataframe. Please check the sample name")


if __name__ == '__main__':
    test_id5 = ID5("C:/Users/reuss/Documents/GitHub/Visual_FRET/src/id5_data/220718_FRET_problematic.txt", "1 2 3")
    # test_id5 = ID5("C:/Users/reuss/Documents/GitHub/Visual_FRET/src/id5_data/test_dataset_id5_mitAllinklProblems.txt", "1 2 3")
    # test_id5 = ID5("id5_data/id5_test_data_fl.txt")
    m1 = test_id5.measurements['Measurement_1']

    mx = test_id5.correction_matrix(measurements=test_id5.measurements, measurement_cy3='Measurement_1', measurement_cy5='Measurement_3', wellnumber='A2')
    print(m1.working_df)

    # A1 = m1.get_well("A2")
    # print(A1)
    #cm = m1.correction_matrix("Measurement1_Corr. Matrix Cy3", "Measurement1_Corr. Matrix Cy5", "A2", 595, 666)
    #print(type(A1))

    # test_genesis = Genesis("C:/Users/reuss/Documents/GitHub/Visual_FRET/src/id5_data/2023-03-06_F_400-600_JM.csv")

    # test_nano = Nanodrop("nanodrop_data/2023-02-14_concentration_RNA_VS.tsv")
    # whole_data = test_nano.working_df
    # test_nano.print_samples()
    # KL_1_2 = test_nano.get_sample("KL 1.2 1")
    # plot_1 = test_nano.plot_sample("KL 1.2 3", color="lightblue")

    # carry_data = Carry("carry_data/Export Data 2023_03_31_DNA_verdunnt_Schmelzkurve_PL.csv")
    # carry_data.add_column_data("Concentration", [0, 1, 5, 10, 20, 40, 60, 80, 100, 1, 10, 100, 1, 10, 100, 1])
    # print(carry_data.data)
    # m1 = test_nano.measurements['Means_all']
    # A1 = m1.get_well("A12")
    # m1.print_meta_data()
