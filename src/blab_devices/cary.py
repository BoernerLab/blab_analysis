from enum import Enum, unique
import json
from pathlib import Path
import re

from more_itertools import split_at
import numpy as np
import pandas as pd

from src.blab_devices.exceptions import InvalidCaryFormatError, InvalidHyperparameterError, \
    InvalidHyperparameterHeaderError, NoJsonExtraInformationError


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
    Temperature_K = 'Temperature (K)'
    Wavelength = 'Wavelength (nm)'
    Absorbance = 'Absorbance'
    NormalizedAbsorbance = 'Normalized_Absorbance'
    Measurement = 'Measurement'
    Meta = 'Meta'
    Date = 'Date'
    Cell_Number = 'Cell_Number'
    FirstDerivative = 'dAbs/dT'


class Cary:
    def __init__(self, file_path: str, extra_json: str):
        self.list_hyperparameters: list = list()
        self.list_extra_information: list = list()
        self.list_data: list = list()
        self.list_data_meta: list = list()
        self.file_names: list = list()
        self.extra_json: Path = Path(extra_json)

        if Path(file_path).is_file():
            self.hyperparameters: dict = {}
            self.extra_information: dict = {}
            self.data: pd.DataFrame = pd.DataFrame()
            self.data_meta: pd.DataFrame = pd.DataFrame()

            file_path: Path = Path(file_path)
            self.file_names.append(file_path.stem)
            self._collect_data_for_file(file_path, extra_json)
        else:
            file_paths = [file_paths for file_paths in Path(file_path).iterdir()
                          if file_paths.is_file() and Path(file_paths).suffix != '.json']
            for file_path in file_paths:
                self.hyperparameters: dict = {}
                self.extra_information: dict = {}
                self.data: pd.DataFrame = pd.DataFrame()
                self.data_meta: pd.DataFrame = pd.DataFrame()
                self.file_names.append(file_path.stem)
                self._collect_data_for_file(file_path, extra_json)


    def _collect_data_for_file(self, file_path, extra_json):
        self.file_content: list = self._read_data(file_path)
        self.raw_data, self.raw_hyperparameters, self.raw_measurements, self.device_measurement = \
            self._split_data_to_chunks()
        self._parse_hyperparameters()
        self._define_cary_case()
        self._add_extra_information(file_path, extra_json)
        self.list_hyperparameters.append(self.hyperparameters)
        self.list_data_meta.append(self.data_meta)
        self.list_data.append(self.data)
        self.list_extra_information.append(self.extra_information)

    def _read_data(self, file_path):
        with open(file_path, 'r', encoding='UTF-8') as file:
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

    def _add_extra_information(self, file_path, extra_json):
        with open(f'{extra_json}', 'r') as f:
            extra_information = json.load(f)
        list_of_supported_files = list(extra_information['files'].keys())
        if file_path.stem in list_of_supported_files:
            self.extra_information[file_path.stem] = extra_information['files'][file_path.stem]
        else:
            raise NoJsonExtraInformationError


class CaryAnalysis:
    def __init__(self, cary_object: Cary):
        self.cary_object = cary_object
        for i, (data, data_meta, extra_information, hyperparameter, filename) in enumerate(zip(
                self.cary_object.list_data,
                self.cary_object.list_data_meta,
                self.cary_object.list_extra_information,
                self.cary_object.hyperparameters,
                self.cary_object.file_names)):
            self._add_extra_information_to_data_meta(data, data_meta, extra_information, filename)
            self._normalize_absorbance_for_each_measurement(data)
            self._calculate_and_normalize_first_derivative(data)

    def _add_extra_information_to_data_meta(self, data, data_meta, extra_information, filename):
        for k, v in extra_information[filename].items():
            data_meta[k] = v
        data[CaryDataframe.Temperature_K.value] = data[CaryDataframe.Temperature.value] + 273.15

    def _normalize_absorbance_for_each_measurement(self, data):
        for unique_value in set(data[CaryDataframe.Measurement.value].tolist()):
            mask = data[CaryDataframe.Measurement.value] == unique_value
            min_value = data.loc[mask, CaryDataframe.Absorbance.value].min()
            max_value = data.loc[mask, CaryDataframe.Absorbance.value].max()
            data.loc[mask, CaryDataframe.NormalizedAbsorbance.value] = (
                    (data.loc[mask, CaryDataframe.Absorbance.value] - min_value) / (max_value - min_value))

    def set_normalized_absorbance_for_measurement(self, file_name: str, measurement_id: int,
                                                  min_temp: float, max_temp: float):
        file_id = [i for i, file in enumerate(self.cary_object.file_names) if file_name == file][0]
        mask = self.cary_object.list_data[file_id][CaryDataframe.Measurement.value] == measurement_id
        min_absorbance = self.cary_object.list_data[file_id] \
            [(self.cary_object.list_data[file_id].loc[:, CaryDataframe.Temperature.value] == min_temp) &
             (self.cary_object.list_data[file_id].loc[:, CaryDataframe.Measurement.value] == measurement_id)] \
            [CaryDataframe.Absorbance.value].iloc[0]
        max_absorbance = self.cary_object.list_data[file_id] \
            [(self.cary_object.list_data[file_id].loc[:, CaryDataframe.Temperature.value] == max_temp) &
             (self.cary_object.list_data[file_id].loc[:, CaryDataframe.Measurement.value] == measurement_id)] \
            [CaryDataframe.Absorbance.value].iloc[0]
        self.cary_object.list_data[file_id].loc[mask, CaryDataframe.NormalizedAbsorbance.value] = (
                (self.cary_object.list_data[file_id].loc[mask, CaryDataframe.Absorbance.value] - min_absorbance)
                / (max_absorbance - min_absorbance))

    def _calculate_and_normalize_first_derivative(self, data):
        for unique_value in set(data[CaryDataframe.Measurement.value].tolist()):
            mask = data[CaryDataframe.Measurement.value] == unique_value
            absorbance = data.loc[mask, CaryDataframe.Absorbance.value]
            temperature = data.loc[mask, CaryDataframe.Temperature_K.value]
            data.loc[mask, CaryDataframe.FirstDerivative.value] = np.gradient(absorbance, temperature)
            data.loc[mask, CaryDataframe.FirstDerivative.value] = (
                (data.loc[mask, CaryDataframe.FirstDerivative.value] - data.loc[mask, CaryDataframe.FirstDerivative.value].min())
                / (data.loc[mask, CaryDataframe.FirstDerivative.value].max() - data.loc[mask, CaryDataframe.FirstDerivative.value].min()))

        # done until calculation of first derivative + normalization
        # same for second derivative without normalization
        # then peak finder for first derivative -> extract Tm guess
        # fit gauss in first derivative -> Tm or dH --> dS or dG
        # fit baselines for both transitions
        # global fit of complete curve -> baselines as input
        # calculate errors over all calculations
        # standard deviation and mean over triplets
        # functions to filter after specific data
        # write plot functions for desired analysis



    def _calculate_second_derivative(self):
        pass
