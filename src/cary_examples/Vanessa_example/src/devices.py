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
METADATA_REGEX = re.compile('(.*)_(\d+[.]\d+){1}nm_(\d+[.]\d+){1}-(\d+[.]\d+){1}C_?R?(\d+)?')
STAGE = 'Stage'


class InvalidCarryFormatError(Exception):
    def __init__(self):
        self.message: str = "The format of the Carry file is not valid for parsing. " \
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
                            "the implemented options for the 'Carry'." \
                            "If this needs to be implemented, reach out to mweber95."

    def __str__(self):
        return f"{self.message}"


class EnumToList(Enum):
    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))


@unique
class CarryCases(EnumToList):
    Scan = 'Scan'
    Thermal = 'Thermal'


@unique
class CarryHyperparameters(EnumToList):
    Method = 'METHOD'
    TempColl = 'Temperature Collection'
    Derivative = 'Derivative'


@unique
class CarryMeasurement(EnumToList):
    Name = 'Name'
    CollTime = 'Collection Time'
    CellNo = 'Cell Number'


@unique
class CarryDataframe(EnumToList):
    Temperature = 'Temperature (°C)'
    Wavelength = 'Wavelength (nm)'
    Absorbance = 'Absorbance'
    Measurement = 'Measurement'
    Meta = 'Meta'
    Date = 'Date'
    Cell_Number = 'Cell_Number'


class Carry:
    def __init__(self, file_path: str):
        self.hyperparameters: dict = {}
        self.data: pd.DataFrame = pd.DataFrame()
        self.data_meta: pd.DataFrame = pd.DataFrame()

        self.file_path: Path = Path(file_path)
        self.file_content: list = self._read_data()
        self.raw_data, self.raw_hyperparameters, self.raw_measurements, self.device_measurement = \
            self._split_data_to_chunks()
        self._parse_hyperparameters()
        self._define_Carry_case()

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

    def _define_Carry_case(self):
        try:
            if self.device_measurement in CarryCases.list():
                df_data, information = self._parse_measurement()
                if self.device_measurement == CarryCases.Scan.value:
                    self._tidy_up_df(df_data, information, CarryDataframe.Wavelength)
                elif self.device_measurement == CarryCases.Thermal.value:
                    self._tidy_up_df(df_data, information, CarryDataframe.Temperature)
                self._extract_info_to_dataframe()
            else:
                raise InvalidCarryFormatError
        except InvalidCarryFormatError as error:
            raise error

    def _parse_measurement(self):
        df_information_raw = [line.split(',')[1:] for line in self.raw_measurements
                              if line.split(',')[0] in CarryMeasurement.list()]
        df_data = [line[1:].split(',') for line in self.raw_measurements if not line.split(',')[0]]
        # df_numeric = [list(map(float, lst)) for lst in df_data[1:]]
        dataframe = pd.DataFrame(df_data[1:], columns=df_data[0])
        dataframe.replace(r'^\s*$', np.nan, regex=True, inplace=True)
        df_information_filtered = [[values for checker, values in zip(dataframe.any().tolist(), sublist) if checker]
                                   for sublist in df_information_raw]
        df_information = [list(filter(None, information)) for information in df_information_filtered]
        dataframe.dropna(axis=1, how="all", inplace=True)
        return dataframe, df_information

    def _tidy_up_df(self, dataframe: pd.DataFrame, information: list, Carry_dataframe: Enum):
        self.data = pd.concat([pd.DataFrame(np.concatenate((array, np.array([[i + 1] * dataframe.shape[0]]).T), axis=1),
                                            columns=[Carry_dataframe.value,
                                                     CarryDataframe.Absorbance.value,
                                                     CarryDataframe.Measurement.value])
                              for i, array in enumerate(list(np.hsplit(np.array(dataframe), dataframe.shape[1] / 2)))],
                              ignore_index=True)
        self.data[[CarryDataframe.Temperature.value, CarryDataframe.Absorbance.value, CarryDataframe.Measurement.value]] =\
            self.data[[CarryDataframe.Temperature.value,
                       CarryDataframe.Absorbance.value,
                       CarryDataframe.Measurement.value]].apply(pd.to_numeric)
        for new_column, column_name in zip(information, [CarryDataframe.Meta.value,
                                                         CarryDataframe.Date.value,
                                                         CarryDataframe.Cell_Number.value]):
            new_column = np.repeat(np.array(new_column), (self.data.shape[0]) / len(information[0]))
            self.data[column_name] = new_column
        self.data.dropna(inplace=True)

    def _extract_info_to_dataframe(self):
        meta_collection = []
        for measurement_string, measurement_index in zip(self.data['Meta'].tolist(), self.data['Measurement'].tolist()):
            match = re.match(METADATA_REGEX, measurement_string)
            if match:
                procedure = 'Cooling' if float(match.group(3)) >= float(match.group(4)) else 'Heating'
                meta_collection.append([measurement_index,
                                        match.group(1),
                                        float(match.group(2)),
                                        float(match.group(3)),
                                        float(match.group(4)),
                                        (int(match.group(5)) + 1) if (match.group(5) is not None) else 1,
                                        procedure])
        self.data_meta = pd.DataFrame(meta_collection, columns=['Measurement',
                                                                'Sample Name',
                                                                'Wavelength (nm)',
                                                                'Temperature Start (°C)',
                                                                'Temperature End (°C)',
                                                                'Run',
                                                                'Ramp Type']).drop_duplicates(keep='first',
                                                                                              ignore_index=True)

    def _parse_hyperparameters(self):
        for hyperparameter_block in self.raw_hyperparameters:
            if hyperparameter_block[0] in CarryHyperparameters.list():
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


#data_raw = Carry(file_path='./2023_08_09_DNA_MOPS_75_K.csv')
