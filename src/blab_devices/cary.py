from enum import Enum, unique
from pathlib import Path
import re

from more_itertools import split_at
import numpy as np
import pandas as pd

from src.blab_devices.exceptions import InvalidCaryFormatError, InvalidHyperparameterError, \
    InvalidHyperparameterHeaderError


STAGE = 'Stage'
NO_STAGES_REGEX = re.compile('Number of stages,[0-9]+')
METADATA_REGEX = re.compile('.*_(\d+[.]\d+){1}nm_(\d+[.]\d+){1}-(\d+[.]\d+){1}C.*')


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
        self.extra_information: dict = {}
        self.data: pd.DataFrame = pd.DataFrame()
        self.data_meta: pd.DataFrame = pd.DataFrame()

        self.file_path: Path = Path(file_path)
        self.file_content: list = self._read_data()
        self.raw_data, self.raw_hyperparameters, self.raw_measurements, self.device_measurement = \
            self._split_data_to_chunks()
        self._parse_hyperparameters()
        self._define_cary_case()
        self._add_extra_information()

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

    def _add_extra_information(self):
        # ordentliche json machen aus vanessa extra infos
        # klassen funktion zum laden der json und in self.extra_information ... Path.stem https://docs.python.org/3/library/pathlib.html
        try:
            self.extra_information = json[self.file_path]
        except CustomError

class CaryAnalysis:
    def __init__(self, cary_object: Cary):
        pass



