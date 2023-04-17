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
            # create short elements in list for naming keys in data_dict
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
            # measure["name"] = element[0]
            measure.insert(0, "measure_number", number_of_measurement)
            measure.insert(1, "name", element[0])
            measure.insert(2, "wavelength_nm", element[1])
            measure.insert(3, "temperature_range_celcius", element[2])
            # measure.assign(name=element[0])
            # measure.assign(wavelength=element[1])
            # measure.assign(temperature_range=element[2])
            # measure["wavelength (nm)"] = element[1]
            # measure["temperature_range (°C)"] = element[2]
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

        @staticmethod
        def create_plate_id_list() -> list:
            letter = ["A", "B", "C", "D", "E", "F", "G", "H"]
            plate = []
            for l in letter:
                for i in range(1, 13):
                    plate.append(l + str(i))
            return plate

        @staticmethod
        def get_well(dataframe: pd.DataFrame, wellnumber: str, measurement: str = None) -> pd.DataFrame:
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
                    meas_well = meas_x[(meas_x.loc[:, 'wellnumber'] == wellnumber)]
                    if meas_well.empty:
                        print("Wellnumber does not exist.")
                    else:
                        return meas_well

        def print_meta_data():
            if self.type_read_mode == 'Absorbance':
                print(
                    f"Experiment '{self.section_name}': emission wavelength {self.start_wavelength}nm - {self.end_wavelength}nm in steps of {self.wavelength_step}nm")


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
        self.number_of_wavelength = metadata[15]
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

        spec_df = self.working_df[self.working_df["wellnumber"] == wellnumber]

        if spec_df.empty:
            print("ERROR: Wellnumber does not exist.")
        else:
            return spec_df

    def print_meta_data(self):
        if self.type_read_mode == 'Fluorescence' and self.wavelengths is None:
            print(
                f"Experiment '{self.section_name}': emission wavelength {self.start_wavelength}nm - {self.end_wavelength}nm in steps of {self.wavelength_step}nm, {self.sweep_waves}: {self.sweep_fixed_wavelength}nm")
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
            # df = df.iloc[1:, :]
            df = df.replace("", np.nan)
            df = df.dropna(axis=1, how="any")
            df = df.melt(id_vars=df.columns[:2],
                         value_vars=list(set(self.create_plate_id_list()).intersection(df.columns)))
            # print(df)
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
                self.emission_wavelength = list(
                    map(int, input("Please enter Emission Wavelength (3) without comma: \n")))

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
        self.read_id5_data()

    def read_id5_data(self) -> None:
        number_of_measurements = 0
        meta = []
        data = []
        iterator = 0

        ex_wl_data = None
        contains_wavelength = False
        # input wavelength_fret händisch. BSP 595 660 660
        # em_wl_values = list(map(int, input("Please enter Emission Wavelengths (3) without comma:\n").split()))
        # ex_wl_values = list(map(int, input("Please enter Excitation Wavelengths (3) without comma:\n").split()))

        with open(self.file_path, 'r', encoding='UTF-8') as file:
            # print('Using ID5-function to read ID5-files...')
            # next(file)
            lines = file.readlines()

            for line in lines:
                if iterator <= int(number_of_measurements):
                    if not line.isspace():
                        line = line.strip('\n')
                        if line.startswith("##BLOCKS="):
                            number_of_measurements = line.split(" ")[1]
                        elif line.startswith("Plate"):
                            meta = line.split("\t")
                            meta = [None if x == "" else x for x in meta]
                        elif line.startswith("~End"):
                            iterator += 1
                            read_mode = meta[5]
                            if read_mode == "Absorbance":
                                pass
                                # self.measurements[f"Measurement_{iterator}"] = ID5MeasureAbsorbance(meta, data)
                            elif read_mode == "Fluorescence":
                                self.measurements[f"Measurement_{iterator}"] = ID5MeasureFluorescence(meta, data,
                                                                                                      self.emission_wavelength)

                            elif read_mode == "Luminescence" or "Time Resolved" or "Imaging":
                                print("Data reading routine is not implemented for these types of experiments")
                            else:
                                print(
                                    "Unknown type of experiment. Check your file or implement new data reading routine")
                            data = []
                            meta = []
                        else:
                            if len(line.split('\t')) > 0:
                                hlp = line.split('\t')
                                data.append(hlp)
                else:
                    break
        # print(f"\nTo access the data dictionary (measurements and meta data), use the following keys:")
        # print(''.join(str(key) + '\n' for key in data_dict.keys()))
        # return data_dict


if __name__ == '__main__':
    test_id5 = ID5("id5_data/id5_test_data_fl.txt", "555 535 666")
    m1 = test_id5.measurements["Measurement_5"]
    print(m1)
    A1 = m1.get_well("A12")
    print(A1)
    m1.print_meta_data()

