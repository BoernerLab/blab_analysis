import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from more_itertools import split_at
from dateutil.parser import parse
from scipy import optimize
from scipy.signal import find_peaks
from pathlib import Path
from enum import Enum, unique
from typing import TextIO
import re


NO_STAGES_REGEX = re.compile('Number of stages,[0-9]+')
METADATA_REGEX = re.compile('.*_(\d+[.]\d+){1}nm_(\d+[.]\d+){1}-(\d+[.]\d+){1}C.*')
STAGE = 'Stage'


class InvalidCaryFormatError(Exception):
    def __init__(self):
        self.message: str = "The format of the Cary file is not valid for parsing. " \
                            "Please make sure to read the correct file."

    def __str__(self):
        return f"{self.message}"


class InvalidHyperparameterError(Exception):
    def __init__(self):
        self.message: str = "The format of the Hyperparameters does not match Key and Value." \
                            "If this needs to be implemented, reach out to mweber95."

    def __str__(self):
        return f"{self.message}"


class InvalidHyperparameterHeaderError(Exception):
    def __init__(self):
        self.message: str = "The first value of a hyperparameter section doesn't match " \
                            "the implemented options for the 'Cary'." \
                            "If this needs to be implemented, reach out to mweber95."

    def __str__(self):
        return f"{self.message}"


class EnumToList(Enum):
    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))


@unique
class CaryCases(EnumToList):
    Scan = 'Scan'
    Thermal = 'Thermal'


@unique
class CaryHyperparameters(EnumToList):
    Method = 'METHOD'
    TempColl = 'Temperature Collection'
    Derivative = 'Derivative'


@unique
class CaryMeasurement(EnumToList):
    Name = 'Name'
    CollTime = 'Collection Time'
    CellNo = 'Cell Number'


@unique
class CaryDataframe(EnumToList):
    Temperature = 'Temperature (°C)'
    Wavelength = 'Wavelength (nm)'
    Absorbance = 'Absorbance'
    Measurement = 'Measurement'
    Meta = 'Meta'
    Date = 'Date'
    Cell_Number = 'Cell_Number'


class Cary:
    def __init__(self, file_path: str):
        self.hyperparameters: dict = {}
        self.data: pd.DataFrame = pd.DataFrame()
        self.data_meta: pd.DataFrame = pd.DataFrame()

        self.file_path: Path = Path(file_path)
        self.file_content: list = self._read_data()
        self.raw_data, self.raw_hyperparameters, self.raw_measurements, self.device_measurement = \
            self._split_data_to_chunks()
        self._parse_hyperparameters()
        self._define_cary_case()

    def _read_data(self):
        with open(self.file_path, 'r', encoding='UTF-8') as file:
            file_content = file.readlines()
        return file_content

    def _split_data_to_chunks(self):
        device_measurement = self.file_content[1].strip().split(',')[1]
        raw_data = [[item.rstrip() for item in sublist]
                    for sublist in split_at(self.file_content, lambda i: i == '\n')
                    if sublist]
        raw_hyperparameters = raw_data[1:-1]
        raw_measurements = raw_data[-1]
        return raw_data, raw_hyperparameters, raw_measurements, device_measurement

    def _define_cary_case(self):
        try:
            if self.device_measurement in CaryCases.list():
                df_data, information = self._parse_measurement()
                if self.device_measurement == CaryCases.Scan.value:
                    self._tidy_up_df(df_data, information, CaryDataframe.Wavelength)
                elif self.device_measurement == CaryCases.Thermal.value:
                    self._tidy_up_df(df_data, information, CaryDataframe.Temperature)
                self._extract_info_to_dataframe()
            else:
                raise InvalidCaryFormatError
        except InvalidCaryFormatError as error:
            raise error

    def _parse_measurement(self):
        df_information_raw = [line.split(',')[1:] for line in self.raw_measurements
                              if line.split(',')[0] in CaryMeasurement.list()]
        df_data = [line[1:].split(',') for line in self.raw_measurements if not line.split(',')[0]]
        # df_numeric = [list(map(float, lst)) for lst in df_data[1:]]
        dataframe = pd.DataFrame(df_data[1:], columns=df_data[0])
        dataframe.replace(r'^\s*$', np.nan, regex=True, inplace=True)
        df_information_filtered = [[values for checker, values in zip(dataframe.any().tolist(), sublist) if checker]
                                   for sublist in df_information_raw]
        df_information = [list(filter(None, information)) for information in df_information_filtered]
        dataframe.dropna(axis=1, how="all", inplace=True)
        return dataframe, df_information

    def _tidy_up_df(self, dataframe: pd.DataFrame, information: list, cary_dataframe: Enum):
        self.data = pd.concat([pd.DataFrame(np.concatenate((array, np.array([[i+1] * dataframe.shape[0]]).T), axis=1),
                                            columns=[cary_dataframe.value,
                                                     CaryDataframe.Absorbance.value,
                                                     CaryDataframe.Measurement.value])
                              for i, array in enumerate(list(np.hsplit(np.array(dataframe), dataframe.shape[1]/2)))],
                              ignore_index=True)
        self.data[[CaryDataframe.Temperature.value, CaryDataframe.Absorbance.value, CaryDataframe.Measurement.value]] =\
            self.data[[CaryDataframe.Temperature.value,
                       CaryDataframe.Absorbance.value,
                       CaryDataframe.Measurement.value]].apply(pd.to_numeric)
        for new_column, column_name in zip(information, [CaryDataframe.Meta.value,
                                                         CaryDataframe.Date.value,
                                                         CaryDataframe.Cell_Number.value]):
            new_column = np.repeat(np.array(new_column), (self.data.shape[0])/len(information[0]))
            self.data[column_name] = new_column
        self.data.dropna(inplace=True)

    def _extract_info_to_dataframe(self):
        meta_collection = []
        for measurement_string, measurement_index in zip(self.data['Meta'].tolist(), self.data['Measurement'].tolist()):
            match = re.match(METADATA_REGEX, measurement_string)
            if match:
                procedure = 'Cooling' if float(match.group(2)) >= float(match.group(3)) else 'Heating'
                meta_collection.append([measurement_index,
                                        float(match.group(1)),
                                        float(match.group(2)),
                                        float(match.group(3)),
                                        procedure])
        self.data_meta = pd.DataFrame(meta_collection, columns=['Measurement',
                                                                'Wavelength (nm)',
                                                                'Temperature Start (°C)',
                                                                'Temperature End (°C)',
                                                                'Ramp Type']).drop_duplicates(keep='first',
                                                                                              ignore_index=True)

    def _parse_hyperparameters(self):
        for hyperparameter_block in self.raw_hyperparameters:
            if hyperparameter_block[0] in CaryHyperparameters.list():
                hyperparameter_category = hyperparameter_block.pop(0)
                self.hyperparameters[hyperparameter_category] = {}
                for element in hyperparameter_block:
                    processed_element = element.split(",")
                    try:
                        if len(processed_element) == 2:
                            self.hyperparameters[hyperparameter_category][processed_element[0]] = processed_element[1]
                        elif len(processed_element) == 3:
                            placeholder = []
                            for value in processed_element[1:]:
                                placeholder.append(value)
                            self.hyperparameters[hyperparameter_category][processed_element[0]] = placeholder
                        else:
                            self.hyperparameters[hyperparameter_category].clear()
                            raise InvalidHyperparameterError
                    except InvalidHyperparameterError as error:
                        raise error
            elif NO_STAGES_REGEX.match(hyperparameter_block[0]):
                hyperparameter_category = hyperparameter_block.pop(0).split(",")[0]
                self.hyperparameters[hyperparameter_category] = {}
                for measurement in hyperparameter_block:
                    col_names = measurement.split(",")
                    if col_names[0] == STAGE:  # regex required
                        for column in col_names:
                            self.hyperparameters[hyperparameter_category][column] = []
                    else:
                        col_names = list(self.hyperparameters[hyperparameter_category].keys())
                        for column, value in zip(col_names, measurement.split(',')):
                            self.hyperparameters[hyperparameter_category][column].append(value)
            else:
                raise InvalidHyperparameterHeaderError()


class Carry:
    def __init__(self, file_path: str, sample_names: list = None) -> None:
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
        self.measurements = {}
        self.sample_names = sample_names
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
        :param coloumn_name: Name of the column. Example: "cK (mM)"
        :param values: List of values per measurement (cuvette). Example: ["0","1","10","100"]
        :return: Dataframe with appended Date
        """

        measures = sorted(set(self.data["Measurement"]))
        for i, value in enumerate(values):
            self.data.loc[self.data['Measurement'] == measures[i], coloumn_name] = value

    def parse_melting_curve_data(self):
        data = pd.read_csv(self.file_path, header=1)

        new_df = pd.concat([pd.DataFrame({"Temperature (C)": data[t_col], "Abs": data[abs_col], "Measurement": i + 1})
                            for i, (t_col, abs_col) in enumerate(zip(data.filter(like="Temperature"), data.filter(like="Abs")))],
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
        col_name_1 = ""

        with open(self.file_path, 'r', encoding='UTF-8') as file:
            lines = file.readlines()
            print(lines[0])
            for line in lines:
                if not line.isspace():
                    line = line.strip('\n')
                    if line.startswith("CSV Report") or line.startswith("Worksheet"):
                        pass
                    elif line.startswith('METHOD'):
                        meth_to_samp_lines = True
                        continue
                    elif line.startswith('SAMPLES'):
                        meth_to_samp_lines = False
                        pass

                    elif meth_to_samp_lines:
                        line = line.strip('\n')
                        meta = line.split(',')
                        meta_list.append(meta)

                    elif line.startswith('Name'):
                        names_line = line.split(',')
                        names_line = list(filter(None, names_line))
                        self.sample_names = names_line[1:]
                        name_to_wl_lines = True
                        continue
                    elif line.startswith(',Wavelength'):
                        cols = line.split(',')
                        name_to_wl_lines = False
                        col_name_1 = "Wavelength (nm)"
                        pass
                    elif line.startswith(',Temperature (°C)'):
                        cols = line.split(',')
                        name_to_wl_lines = False
                        col_name_1 = "Temperature (°C)"
                        pass
                    elif name_to_wl_lines:
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
        data_fin = data_fin[[col_name_1, "Abs"]]
        data_fin = data_fin.iloc[:, 3:]
        names_line.insert(1, col_name_1)
        del names_line[0]
        data_fin.columns = names_line
        data_fin.iloc[:] = data_fin.iloc[:].apply(pd.to_numeric)
        data_fin = data_fin.melt(id_vars=[col_name_1],
                                value_vars=self.sample_names, #toDo - nicht fest rein, sonst veränderbar! ['KL 1.1', 'KL 1.2', 'TL 2.1', 'TL 2.2']
                                var_name='sample',
                                value_name='value (RFU)')

        dct = dict((item[0], item[1:]) for item in data_list)
        dctdat = dict((item[0], item[1:]) for item in meta_list)

        self.measurements[f"Metadata_Method_{iterator}"] = dctdat
        self.measurements[f"Metadata_Samples_{iterator}"] = dct
        self.measurements[f"Measurement_{iterator}"] = data_fin

        print(f"\nTo access the data dictionary (measurements and meta data), use the following keys:")
        print(''.join(str(key) + '\n' for key in self.measurements.keys()))
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
        self.correction_matrix()

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
    def __init__(self, metadata: list, data: list, wavelength_pairs: dict) -> None:
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
        self.wavelength_pairs = wavelength_pairs
        self.restructure_data()
        # self.correction_matrix()
        # self.correct_mat = self.correction_matrix(measurement_cy3, measurement_cy5, wellnumber)

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
            print(
                f"Experiment '{self.section_name}': \n"
                f"{self.wavelength_pairs}")

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

            for i, data_line in enumerate(self.data):
                wl_pair = []
                del data_line[0]
                del data_line[len(data_line) - 1]
                if i == 0:
                    data_line.insert(0, "excitation_wl")
                    data_line.insert(1, "emission_wl")
                if i > 0:
                    wl_pair_key_list = list(self.wavelength_pairs.keys())
                    wl_pair = self.wavelength_pairs[wl_pair_key_list[i - 1]]
                    data_line.insert(0, wl_pair[0])
                    data_line.insert(1, wl_pair[1])

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
            self._working_df = df_sort
            df.empty
            df_sort.empty

        else:
            print(f"Current read type = {self.read_type}. If you can read this, implementation not done.")
    @property
    def working_df(self):
        return self._working_df

    @working_df.setter
    def working_df(self, new_df):
        self._working_df = new_df

    def add_new_column(self, value_column_name, dictionary, new_column_name):
        """
        Adds a new column to a pandas DataFrame based on a value in a specified column, using a dictionary to map the value to
        a new value for the new column.

        Parameters:
        dataframe (pandas.DataFrame): The DataFrame to modify.
        value_column_name (str): The name of the column to use as the source for the value to map to a new value.
        dictionary (dict): The dictionary to use to map the value to a new value for the new column.
        new_column_name (str): The name of the new column to add to the DataFrame.

        Returns:
        pandas.DataFrame: The modified DataFrame with the new column added.
        """
        new_column_values = []
        current_df = self.working_df
        for index, row in current_df.iterrows():
            value = row[value_column_name]

            if value in dictionary:
                new_value = dictionary[value]
            else:
                new_value = None

            new_column_values.append(new_value)

        current_df[new_column_name] = new_column_values

        self.working_df = current_df

class ID5:
    def __init__(self, file_path: str, wavelength_pairs: dict = None):
        self.file_path = file_path
        self.measurements = {}
        self.wavelength_pairs = wavelength_pairs
        self.read_id5_data()

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
                                                                                                    self.wavelength_pairs)
                            elif read_mode == "Luminescence" or "Time Resolved" or "Imaging":
                                print("Data reading routine is not implemented for these types of experiments.")
                            else:
                                print(
                                    "Unknown type of experiment. Check your file or implement new data reading routine.")
                            print(f"Measurement_{iterator} = {meta[1]}: {read_type}, {read_mode}")  # print accessible keys of dict
                            data = []
                            meta = []
                        else:
                            if len(line.split('\t')) > 0:
                                hlp = line.split('\t')
                                data.append(hlp)
                else:
                    break

    def calculate_blank(self, measurement_name_list: list, well_numbers: list):
        """
        Collect all blank values from given well_number list. Calculate mean blank value and substract the blank from
        values
        :param measurement_list: list of measurements from which the blank is to be calculated and subtracted.
        :param well_numbers: list of wellnumbers with blank measurements (e.g.: ["A1","B1","C1"])
        :return:
        """
        blanks = []
        for measurement_name in measurement_name_list:
            current_df = self.measurements[measurement_name].working_df
            for index, row in current_df.iterrows():
                value = row["wellnumber"]
                if value in well_numbers:
                    blanks.append(row["RFU"])
        mean_blank = sum(blanks) / len(blanks)
        for measurement_name in measurement_name_list:
            current_df = self.measurements[measurement_name].working_df
            current_df["bg corrected RFU"] = current_df["RFU"] - mean_blank
            self.measurements[measurement_name].working_df = current_df

        return mean_blank

    # def calculate_correction_matrix(self,measurement_cy3: str, measurement_cy5: str, wellnumber: str):
    #     """
    #     :param measurement_cy3:
    #     :param measurement_cy5:
    #     :param wellnumber:
    #     :param ex1:
    #     :param ex2:
    #     :return:
    #     """
    #     # gilt nur für objekte der klasse ID5fluo endpoint

    #     cy3 = self.measurements[measurement_cy3]
    #     cy3df = cy3.working_df
    #     meas_cy3 = cy3.get_well(wellnumber)

    #     cy5 = self.measurements[measurement_cy5]
    #     meas_cy5 = cy5.get_well(wellnumber)

    #     row_names = [self.wavelength_pairs["Dex_Dem"][0], self.wavelength_pairs["Dex_Aem"][0]]
        
    #     rows = [[f"ex_{row_names[0]}", list(meas_cy3["RFU"])[0], list(meas_cy3["RFU"])[1]],
    #             [f"ex_{row_names[1]}", 0.0, list(meas_cy5["RFU"])[0]]]

    #     col_names = pd.unique(cy3df["emission wavelength (nm)"])
    #     cols = ["Ex/Em", f"em_{col_names[0]}", f"em_{col_names[1]}"]

    #     correct_matrix = pd.DataFrame(rows, columns=cols)

    #     return correct_matrix

    # def calculate_bleedthrough(self, correction_matrix: pd.DataFrame, don_acc_type: str) -> float:
    #     """
    #     Function to calculate bleedthrough for the acceptor or the donor.
        
    #     parameters
    #     ----------
    #     correction_matrix: DataFrame
    #             the correction matrix 
    #     don_acc_type: str
    #             either "A" for acceptor or "D" for donor

    #     returns
    #     -------
    #     the calculated bleedthrough value
    #     """
    #     valid_type = {"A", "D"}
    #     if don_acc_type not in valid_type:
    #         print("Type unknown. Must be one of %r (A: acceptor, D: donor)" % valid_type)
    #     else:
    #         if don_acc_type == "D":
    #             print(f"You chose donor bleedthrough calculations.")
    #             if correction_matrix.iloc[0][2] == 0.0 or correction_matrix.iloc[0][1] == 0.0:
    #                 print("Division with zero encountered. That is unfortunate.")
    #                 print(f"Bleedthrough of donor set to zero. \n")
    #             else:
    #                 bt_D = correction_matrix.iloc[0][2] / correction_matrix.iloc[0][1] # row, col
    #                 print(f"Bleedthrough of donor: {round(bt_D, 4)} ({format(round(bt_D*100, 2),'.2f')}%). \n")
    #                 return bt_D
    #         elif don_acc_type == "A":
    #             print(f"You chose acceptor bleedthrough calculations.")
    #             if correction_matrix.iloc[1][1] == 0.0 or correction_matrix.iloc[1][2] == 0.0:
    #                 print("Division with zero encountered. It is what it is.")
    #                 print(f"Bleedthrough of Acceptor set to zero. \n")
    #                 return 0.0
    #             else:
    #                 bt_A = correction_matrix.iloc[1][1] / correction_matrix.iloc[1][2] # row, col
    #                 print(f"Bleedthrough of Acceptor: {round(bt_A, 4)} ({format(round(bt_A*100,2),'.2f')}%). \n")
    #                 return bt_A

    # def calculate_directExcitation(self, correction_matrix: pd.DataFrame, don_acc_type: str) -> float:
    #     """
    #     Function to calculate direct excitation for the acceptor or the donor.
        
    #     parameters
    #     ----------
    #     dataframe: DataFrame
    #             the correction matrix dataframe 
    #     don_acc_type: str
    #             either "A" for acceptor or "D" for donor

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
    #             if correction_matrix.iloc[1][1] == 0.0 or correction_matrix.iloc[0][1] == 0.0:
    #                 print("Divide by zero encountered. It is what it is.")
    #                 print(f"Direct excitation of donor set to zero. \n")
    #                 return 0.0
    #             else:
    #                 dE_D = correction_matrix.iloc[1][1] / correction_matrix.iloc[0][1] # row, col
    #                 print(f"Direct Excitation of Donor: {round(dE_D, 4)} ({format(round(dE_D*100,2), '.2f')}%). \n")
    #                 return dE_D
            
    #         elif don_acc_type == "A":
    #             print(f"You chose acceptor direct excitation calculations.")
    #             if correction_matrix.iloc[0][2] == 0.0 or correction_matrix.iloc[1][2] == 0.0:
    #                 print("Encountered a zero in the calculation. Should it be there?")
    #                 print(f"Direct Excitation of Acceptor set to zero. \n")
    #                 return 0.0
    #             else:
    #                 dE_A = correction_matrix.iloc[0][2] / correction_matrix.iloc[1][2] # row, col
    #                 print(f"Direct Excitation of Acceptor: {round(dE_A, 4)} ({format(round(dE_A*100,2), '.2f')}%). \n")
    #                 return dE_A

    def restructure_working_df(self, measurement_name_list: list):
        '''
        Function to restructure the working df for all measurements. idd = IDem_Dex, iad= IAem_Dex, iaa = IAem_Aex
        '''
        for measurement_name in measurement_name_list:
            current_df = self.measurements[measurement_name].working_df

            idd = current_df[(current_df["excitation wavelength (nm)"] == list(self.wavelength_pairs.items())[0][1][0]) & \
                            (current_df["emission wavelength (nm)"] == list(self.wavelength_pairs.items())[0][1][1])] # list(wavelength_pairs.items())[0] # = ('Dex_Dem', [530, 595])
            idd = idd.sort_values(by=['wellnumber', 'temperature (°C)']) 
            #idd.columns = [f'Dem_Dex cons' if x == 'RFU' else x for x in idd.columns] 
            idd.columns = [f'I^Dem_Dex' if x == 'bg corrected RFU' else x for x in idd.columns] 
            #idd.columns = [f'Dem_Dex concentration (mM)' if x == 'concentration (mM)' else x for x in idd.columns]
            idd = idd.reset_index(drop=True) 
            idd = idd.drop(columns=['excitation wavelength (nm)', 'emission wavelength (nm)', 'RFU']) 


            iad = current_df[(current_df["excitation wavelength (nm)"] == list(self.wavelength_pairs.items())[1][1][0]) & \
                            (current_df["emission wavelength (nm)"] == list(self.wavelength_pairs.items())[1][1][1])]
            iad = iad.sort_values(by=['wellnumber', 'temperature (°C)']) 
            #iad.columns = [f'Dex_Aem RFU' if x == 'RFU' else x for x in iad.columns] 
            iad.columns = [f'I^Aem_Dex' if x == 'bg corrected RFU' else x for x in iad.columns] 
            #iad.columns = [f'Dex_Aem concentration (mM)' if x == 'concentration (mM)' else x for x in iad.columns]
            iad = iad.reset_index(drop=True)
            iad = iad.loc[:, ['temperature (°C)', 'wellnumber', 'I^Aem_Dex']]

            iaa = current_df[(current_df["excitation wavelength (nm)"] == list(self.wavelength_pairs.items())[2][1][0]) & \
                            (current_df["emission wavelength (nm)"] == list(self.wavelength_pairs.items())[2][1][1])]
            iaa = iaa.sort_values(by=['wellnumber', 'temperature (°C)']) 
            #iaa.columns = [f'Aex_Aem RFU' if x == 'RFU' else x for x in iaa.columns] 
            iaa.columns = [f'I^Aem_Aex' if x == 'bg corrected RFU' else x for x in iaa.columns] 
            #iaa.columns = [f'Aex_Aem concentration' if x == 'concentration (mM)' else x for x in iaa.columns]
            iaa = iaa.reset_index(drop=True)
            iaa = iaa.loc[:, ['temperature (°C)', 'wellnumber', 'I^Aem_Aex']]

            new_df = pd.merge(idd, iad, on=['temperature (°C)', 'wellnumber'])
            new_df = pd.merge(new_df, iaa, on=['temperature (°C)', 'wellnumber'])
            new_df = new_df.iloc[:,[0,1,3,2,4,5]]
            
            self.measurements[measurement_name].FRET_df = new_df

        # Fret Berechnen
        # spalten = temp, wn, DD, AD, AA, 2xI', 2xI'', FRET (wenn vorher drangeklatscht: conzis + sample/construct -> test damit: doppelte raus)

    def calculate_bt_correction(self, measurement_name_list: list, bt_variable: int):
        """
        Function to calculate bleedthrough correction.

        :param measurement_name_list: list of measurements from which the bt correction is to be calculated and subtracted.
        :param bt_variable: the bleedthrough value
        :return: calculated bleedthrough values

        """
        if bt_variable == 0:
            print(f"Bleedthrough variable is equal to {bt_variable}.")
            print("Therefore I'^Dem_Dex is equal to I^Dem_Dex. No I'^Dem_Dex column will be added.")
        else: 
            for measurement_name in measurement_name_list:
                current_df = self.measurements[measurement_name].FRET_df
                current_df = current_df.dropna()
                current_df["I'^Aem_Dex"] = current_df["I^Aem_Dex"] - (bt_variable * current_df["I^Dem_Dex"])
                current_df = current_df.iloc[:,[0,1,2,3,4,6,5]]
                self.measurements[measurement_name].FRET_df = current_df

    def calculate_de_correction(self, measurement_name_list: list, de_variable: int):
        """
        Function to calculate direct excitation correction.

        :param measurement_name_list: list of measurements from which the de correction is to be calculated and subtracted.
        :param de_variable: the direct excitation value
        :return: calculated direct excitation values

        """
        for measurement_name in measurement_name_list:
            current_df = self.measurements[measurement_name].FRET_df
            current_df = current_df.dropna()
            # current_df["I''^Dem_Dex"] = current_df["I'^Dem_Dex"] - (de_variable * 0)
            current_df["I''^Aem_Dex"] = current_df["I'^Aem_Dex"] - (de_variable * current_df["I^Aem_Aex"])
            current_df = current_df.iloc[:,[0,1,2,3,4,5,7,6]]
            self.measurements[measurement_name].FRET_df = current_df

        print("I''^Dem_Dex equals I^Dem_Dex as I^Dem_Aex cannot physically and technically be measured.")
        print("No I''^Dem_Dex column will be added to the FRET dataframe. Please use I^Dem_Dex.")
        print("I''^Aem_Dex has been added to the FRET dataframe.")

    def calculate_FRET(self, measurement_name_list: list):
        """
        Function to calculate FRET.

        :param measurement_name_list: list of measurements from which FRET is to be calculated and subtracted.
        :return: calculated FRET values
        """
        for measurement_name in measurement_name_list:
            current_df = self.measurements[measurement_name].FRET_df
            current_df = current_df.dropna()
            current_df["FRET"] = current_df["I''^Aem_Dex"] / (current_df["I^Dem_Dex"] + current_df["I''^Aem_Dex"])
            self.measurements[measurement_name].FRET_df = current_df

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
        date_pattern = re.compile("[0-9]+/[0-9]+/[0-9]{4}.*")
        with open(self.file_path, "r", encoding='UTF-8') as file:

            lines = file.readlines()

            for line in lines:
                if not line.isspace():
                    line = line.strip('\n')
                    if line.startswith("//"):
                        pass
                    elif re.match(date_pattern, line):
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

            return pd.DataFrame(data, columns=["sample", "wavelength (nm)", "absorbance", "date_time"])

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
            ax.scatter(sample_df['wavelength (nm)'], sample_df['absorbance'], marker='x', color=color, linewidth=0.5, s=10)

            # set axis labels
            ax.set_xlabel(r'Wavelength $\lambda$ (nm)')
            ax.set_ylabel('Absorbance (OD)')
            legend_info = str(sample_df.iloc[1, 0])
            ax.legend([legend_info], loc='best')
            plt.subplots_adjust(bottom=0.2)

            # show plot
            plt.show()

            return fig
        else:
            print("Sample name not found in dataframe. Please check the sample name.")

    def plot_all_samples(self, colormap = "jet") -> plt.subplots(): # col_map = "Paired"
        """
        Generates a plot of all absorption spectra.
        :param color: color of the data points (optional)
        :return: matplotlib figure object
        """

        # set plot size and font size
        plt.rcParams['figure.figsize'] = [11, 11]
        plt.rcParams['xtick.labelsize'] = 9
        plt.rcParams['ytick.labelsize'] = 9

        sampl_num = len(self.working_df['sample'].unique())

        fig, ax = plt.subplots(figsize=(10, 8))

        # if colormap is not None:
        #     colormap = colormap
        #     i = 0
        #     for name, group in self.working_df.groupby('sample'):
        #         # create a scatter plot
        #         ax.scatter(group['wavelength (nm)'], group['absorbance'], c = colormap[i], label = name, marker='x', linewidth=0.5, s=10)
        #         i += 1
        # else:
        colormap = plt.get_cmap('jet', sampl_num)
        i = 0
        for name, group in self.working_df.groupby('sample'):
            # create a scatter plot
            ax.plot(group['wavelength (nm)'], group['absorbance'], c = colormap(i), label = name, marker='x', linewidth=0.5)
            i += 1

        # set axis labels
        ax.set_xlabel(r'Wavelength $\lambda$ (nm)')
        ax.set_ylabel('Absorbance (OD)')
        # legend_info = str(self.working_df["sample"].unique())
        plt.legend(loc='best')
        plt.subplots_adjust(bottom=0.2)

        # show plot
        plt.show()

        return fig


if __name__ == '__main__':
    # cary_data = Cary("carry_data/fuer_Mirko/5_07_2023_Gruppe1.csv")
    # data = cary_data.data
    # cary_data_2 = Cary("carry_data/fuer_Mirko/2023_05_22_DNA_Na_PL (1).csv")
    # data2 = cary_data_2.data
    # 5_07_2023_Gruppe1.csv
    # cary_data = Cary("carry_data/fuer_Mirko/5_07_2023_Gruppe1.csv")
    cary_data = Cary("carry_data/fuer_Mirko/2023_08_01_DNA_K_MOPS_pH65_PL.csv")
    wavelength_pairs = {
        "Dex_Dem": [530, 595],
        "Dex_Aem": [530, 670],
        "Aex_Aem": [630, 670]
    }

    #my_id_5_data = ID5("id5_data/2023-03-15_Praktikum_FRET3_VS.txt", wavelength_pairs)
    #print(my_id_5_data.measurements["Measurement_1"].working_df)

    #test_nano = ID5("C:/Users/reuss/Documents/GitHub/Visual_FRET/src/id5_data/2023-05-30_MgCl2_titration_VS.txt")
    #whole_data = test_nano.working_df
    # test_nano.print_samples()
    #sample1 = test_nano.get_sample("Sample 1")
    #plot_1 = test_nano.plot_sample("Sample 1")
    # nano_cols = ["#006d2c", "#a50f15", "#feb24c", "#2b8cbe"]

    # plot_all = test_nano.plot_all_samples()



    #plot_all.savefig("nano_plot.png", dpi=300, bbox_inches = 'tight')
    #print(sample1)
    
    #print(my_id_5_data.measurements["Measurement_1"].working_df.columns)

    #cm = my_id_5_data.calculate_correction_matrix("Measurement_1", "Measurement_3", "A2")
    # cm_cy3 = my_id_5_data.calculate_correction_matrix("Measurement_1", "Measurement_3", "A5")
    # print(cm_cy3)
    # cm_cy5 = my_id_5_data.calculate_correction_matrix("Measurement_1", "Measurement_3", "B5")
    # print(cm_cy5)

    # bt_d = my_id_5_data.calculate_bleedthrough(cm_cy3, "D")
    # bt_a = my_id_5_data.calculate_bleedthrough(cm_cy5, "A")

    # de_d = my_id_5_data.calculate_bleedthrough(cm_cy3, "D")
    # de_a = my_id_5_data.calculate_bleedthrough(cm_cy5, "A")

    # FRET_measurements = []
    # measure_names = my_id_5_data.measurements.keys()
    # for name in measure_names:
    #     measurement = my_id_5_data.measurements[name]
    #     name_1 = measurement.section_name
    #     if "FRET" in name_1:
    #         FRET_measurements.append(name)
    # print(FRET_measurements)

    # fret = my_id_5_data.calculate_FRET(FRET_measurements, cm_cy5, bt_d, de_a)
    # print(my_id_5_data.measurements["Measurement_7"].working_df)

    # print(cm)
    # print(my_id_5_data.calculate_directExcitation(cm, "A"))
    # print(my_id_5_data.calculate_bleedthrough(cm, "D"))
    # print(my_id_5_data.measurements["Measurement_1"].working_df)


    # test_id5 = ID5("C:/Users/reuss/Documents/GitHub/Visual_FRET/src/id5_data/test_dataset_id5_mitAllinklProblems.txt", "1 2 3")
    # test_id5 = ID5("id5_data/id5_test_data_fl.txt")
    # FRET_measurements = []
    # measure_names = my_id_5_data.measurements.keys()
    # for name in measure_names:
    #     measurement = my_id_5_data.measurements[name]
    #     name_1 = measurement.section_name
    #     if "°" in name_1:
    #         FRET_measurements.append(name)

    # #print(FRET_measurements)


    # blank_wells = ["A1", "B1", "C1", "D1", "E1", "F1", "G1", "H1"]
    # mean_blank = my_id_5_data.calculate_blank(FRET_measurements, blank_wells)


    #test_id5.calculate_blank()

    # mx = test_id5.correction_matrix(measurements=test_id5.measurements, measurement_cy3='Measurement_1', measurement_cy5='Measurement_3', wellnumber='A2')
    #print(m1.working_df)

    # A1 = m1.get_well("A2")
    # print(A1)
    # cm = m1.correction_matrix("Measurement1_Corr. Matrix Cy3", "Measurement1_Corr. Matrix Cy5", "A2", 595, 666)
    # print(type(A1))

    # test_genesis = Genesis("C:/Users/reuss/Documents/GitHub/Visual_FRET/src/id5_data/2023-03-06_F_400-600_JM.csv")

    # test_nano = Nanodrop("nanodrop_data/2023-02-14_concentration_RNA_VS.tsv")
    # whole_data = test_nano.working_df
    # test_nano.print_samples()
    # KL_1_2 = test_nano.get_sample("KL 1.2 1")
    # plot_1 = test_nano.plot_sample("KL 1.2 3", color="lightblue")

    ### carry_data = Carry("carry_data/5_07_2023_Gruppe1.csv")
    ### print(carry_data.measurements["Measurement_1"])
    # carry_data.add_column_data("Concentration", [0, 1, 5, 10, 20, 40, 60, 80, 100, 1, 10, 100, 1, 10, 100, 1])
    # print(carry_data.data)
    # m1 = test_nano.measurements['Means_all']
    # A1 = m1.get_well("A12")
    # m1.print_meta_data()
