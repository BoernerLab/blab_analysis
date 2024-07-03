from enum import Enum, unique
import json
from pathlib import Path
import re

from lmfit import models, Parameters
from more_itertools import split_at
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter

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
    NormalizedAbsorbance = 'normalized_Abs'
    Measurement = 'Measurement'
    Meta = 'Meta'
    Date = 'Date'
    CellNumber = 'Cell_Number'
    FirstDerivative = 'dAbs/dT'
    FirstDerivativeNormalized = 'normalized_dAbs/dT'
    SecondDerivative = 'd2Abs/dT2'
    SecondDerivativeNormalized = 'normalized_d2Abs/dT2'
    FirstDerivativeSavgolPeaks = 'savgol_peaks'
    ExpectedTransitions = 'Expected Transitions'
    BaseLines = 'Baselines'


@unique
class Bounds(EnumToList):
    Lower = 'lower'
    Upper = 'upper'
    Middle = 'middle'


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
                                                         CaryDataframe.CellNumber.value]):
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
    # Todo: fit gauss in first derivative -> Tm or dH --> dS or dG to extract exact Tm and dH to possibly calculate dG
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
            self._calculate_first_derivative(data)
            self._normalize_first_derivative(data)
            self._calculate_second_derivative(data)
            self._normalize_second_derivative(data)
            self._find_peak_for_measurement(data, data_meta, filename)
            self._fit_baselines(data, data_meta)

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

    def set_normalized_first_derivative_for_measurement(self, file_name: str, measurement_id: int,
                                                        min_temp: float, max_temp: float):
        file_id = [i for i, file in enumerate(self.cary_object.file_names) if file_name == file][0]
        mask = self.cary_object.list_data[file_id][CaryDataframe.Measurement.value] == measurement_id
        min_first_derivative = self.cary_object.list_data[file_id] \
            [(self.cary_object.list_data[file_id].loc[:, CaryDataframe.Temperature.value] == min_temp) &
             (self.cary_object.list_data[file_id].loc[:, CaryDataframe.Measurement.value] == measurement_id)] \
            [CaryDataframe.FirstDerivative.value].iloc[0]
        max_first_derivative = self.cary_object.list_data[file_id] \
            [(self.cary_object.list_data[file_id].loc[:, CaryDataframe.Temperature.value] == max_temp) &
             (self.cary_object.list_data[file_id].loc[:, CaryDataframe.Measurement.value] == measurement_id)] \
            [CaryDataframe.FirstDerivative.value].iloc[0]
        self.cary_object.list_data[file_id].loc[mask, CaryDataframe.FirstDerivativeNormalized.value] = (
                (self.cary_object.list_data[file_id].loc[mask, CaryDataframe.FirstDerivative.value] -
                 min_first_derivative) / (max_first_derivative - min_first_derivative))

    def set_normalized_first_derivative_for_measurement_only_min_temp(self, file_name: str, measurement_id: int,
                                                                      min_temp: float):
        file_id = [i for i, file in enumerate(self.cary_object.file_names) if file_name == file][0]
        mask = self.cary_object.list_data[file_id][CaryDataframe.Measurement.value] == measurement_id
        min_first_derivative = self.cary_object.list_data[file_id] \
            [(self.cary_object.list_data[file_id].loc[:, CaryDataframe.Temperature.value] == min_temp) &
             (self.cary_object.list_data[file_id].loc[:, CaryDataframe.Measurement.value] == measurement_id)] \
            [CaryDataframe.FirstDerivative.value].iloc[0]
        max_first_derivative = self.cary_object.list_data[file_id].loc[mask, CaryDataframe.FirstDerivative.value].max()
        self.cary_object.list_data[file_id].loc[mask, CaryDataframe.FirstDerivativeNormalized.value] = (
                (self.cary_object.list_data[file_id].loc[mask, CaryDataframe.FirstDerivative.value] -
                 min_first_derivative) / (max_first_derivative - min_first_derivative))

    def set_normalized_second_derivative_for_measurement(self, file_name: str, measurement_id: int,
                                                         min_temp: float, max_temp: float):
        file_id = [i for i, file in enumerate(self.cary_object.file_names) if file_name == file][0]
        mask = self.cary_object.list_data[file_id][CaryDataframe.Measurement.value] == measurement_id
        min_second_derivative = self.cary_object.list_data[file_id] \
            [(self.cary_object.list_data[file_id].loc[:, CaryDataframe.Temperature.value] == min_temp) &
             (self.cary_object.list_data[file_id].loc[:, CaryDataframe.Measurement.value] == measurement_id)] \
            [CaryDataframe.Absorbance.value].iloc[0]
        max_second_derivative = self.cary_object.list_data[file_id] \
            [(self.cary_object.list_data[file_id].loc[:, CaryDataframe.Temperature.value] == max_temp) &
             (self.cary_object.list_data[file_id].loc[:, CaryDataframe.Measurement.value] == measurement_id)] \
            [CaryDataframe.Absorbance.value].iloc[0]
        self.cary_object.list_data[file_id].loc[mask, CaryDataframe.SecondDerivativeNormalized.value] = (
                (self.cary_object.list_data[file_id].loc[mask, CaryDataframe.SecondDerivative.value] - min_second_derivative)
                / (max_second_derivative - min_second_derivative))

    def _calculate_first_derivative(self, data):
        for unique_value in set(data[CaryDataframe.Measurement.value].tolist()):
            mask = data[CaryDataframe.Measurement.value] == unique_value
            absorbance = data.loc[mask, CaryDataframe.Absorbance.value]
            temperature = data.loc[mask, CaryDataframe.Temperature_K.value]
            data.loc[mask, CaryDataframe.FirstDerivative.value] = np.gradient(absorbance, temperature)

    def _normalize_first_derivative(self, data):
        for unique_value in set(data[CaryDataframe.Measurement.value].tolist()):
            mask = data[CaryDataframe.Measurement.value] == unique_value
            data.loc[mask, CaryDataframe.FirstDerivativeNormalized.value] = (
                    (data.loc[mask, CaryDataframe.FirstDerivative.value] - data.loc[
                        mask, CaryDataframe.FirstDerivative.value].min())
                    / (data.loc[mask, CaryDataframe.FirstDerivative.value].max() - data.loc[
                mask, CaryDataframe.FirstDerivative.value].min()))

    def _calculate_second_derivative(self, data):
        for unique_value in set(data[CaryDataframe.Measurement.value].tolist()):
            mask = data[CaryDataframe.Measurement.value] == unique_value
            first_derivative = data.loc[mask, CaryDataframe.FirstDerivative.value]
            temperature = data.loc[mask, CaryDataframe.Temperature_K.value]
            data.loc[mask, CaryDataframe.SecondDerivative.value] = np.gradient(first_derivative, temperature)

    def _normalize_second_derivative(self, data):
        for unique_value in set(data[CaryDataframe.Measurement.value].tolist()):
            mask = data[CaryDataframe.Measurement.value] == unique_value
            data.loc[mask, CaryDataframe.SecondDerivativeNormalized.value] = (
                    (data.loc[mask, CaryDataframe.SecondDerivative.value] - data.loc[
                        mask, CaryDataframe.SecondDerivative.value].min())
                    / (data.loc[mask, CaryDataframe.SecondDerivative.value].max() - data.loc[
                mask, CaryDataframe.SecondDerivative.value].min()))

    def _find_peak_for_measurement(self, data, data_meta, filename):
        data_meta[CaryDataframe.FirstDerivativeSavgolPeaks.value] = None
        for unique_value in set(data[CaryDataframe.Measurement.value].tolist()):
            mask = data[CaryDataframe.Measurement.value] == unique_value
            mask_meta = data_meta[CaryDataframe.Measurement.value] == unique_value
            measurement_data = data[mask]
            meta = data_meta[mask_meta]
            measurement_data.reset_index(drop=True, inplace=True)
            savgol = savgol_filter(measurement_data[CaryDataframe.FirstDerivative.value], 12, 3)
            savgol_peaks, _ = find_peaks(savgol, height=0, width=5, distance=10)
            savgol_T = measurement_data[CaryDataframe.Temperature_K.value][savgol_peaks]
            savgol_dAbsdT = savgol[savgol_peaks]
            peaks = np.array([(savgol_x, savgol_y) for savgol_x, savgol_y in zip(savgol_T, savgol_dAbsdT)])
            if int(meta[CaryDataframe.ExpectedTransitions.value][unique_value-1]) != int(len(peaks)):
                print(f"WARNING: Number of found peaks doesn't match the expected transitions\n File: {filename}, Measurement: {unique_value}")
            for index in data_meta.index[data_meta[CaryDataframe.Measurement.value] == unique_value]:
                data_meta.at[index, CaryDataframe.FirstDerivativeSavgolPeaks.value] = peaks

    def _fit_baselines(self, data, data_meta):
        data_meta[CaryDataframe.BaseLines.value] = None
        for unique_value in set(data[CaryDataframe.Measurement.value].tolist()):
            mask = data[CaryDataframe.Measurement.value] == unique_value
            mask_meta = data_meta[CaryDataframe.Measurement.value] == unique_value
            measurement_data = data[mask]
            meta = data_meta[mask_meta]
            measurement_data.reset_index(drop=True, inplace=True)
            peaks = meta[CaryDataframe.FirstDerivativeSavgolPeaks.value].to_list()[0]
            if len(peaks) == 1:
                tm = peaks[:, 0]
                slope_low, intercept_low = self._get_fit_for_baseline(measurement_data, tm, Bounds.Lower.value)
                slope_high, intercept_high = self._get_fit_for_baseline(measurement_data, tm, Bounds.Upper.value)
                for index in data_meta.index[data_meta[CaryDataframe.Measurement.value] == unique_value]:
                    data_meta.at[index, CaryDataframe.BaseLines.value] = np.array([[slope_low, intercept_low], [slope_high, intercept_high]])
            elif len(peaks) == 2:
                tm = sorted(peaks[:, 0])
                slope_low, intercept_low = self._get_fit_for_baseline(measurement_data, tm, Bounds.Lower.value)
                slope_middle, intercept_middle = self._get_fit_for_baseline(measurement_data, tm, Bounds.Middle.value)
                slope_high, intercept_high = self._get_fit_for_baseline(measurement_data, tm, Bounds.Upper.value)
                for index in data_meta.index[data_meta[CaryDataframe.Measurement.value] == unique_value]:
                    data_meta.at[index, CaryDataframe.BaseLines.value] = np.array([[slope_low, intercept_low], [slope_middle, intercept_middle], [slope_high, intercept_high]])
            else:
                for index in data_meta.index[data_meta[CaryDataframe.Measurement.value] == unique_value]:
                    data_meta.at[index, CaryDataframe.BaseLines.value] = []

    def _get_fit_for_baseline(self, measurement_data, tm, baseline):
        if baseline == Bounds.Lower.value:
            if len(tm) == 1:
                tm = measurement_data[measurement_data[CaryDataframe.Temperature_K.value] < float(tm)]
            else:
                tm = measurement_data[(measurement_data[CaryDataframe.Temperature_K.value] < float(tm[0])) &\
                                      (measurement_data[CaryDataframe.FirstDerivative.value] >= 0)]

            tm_index = tm[CaryDataframe.FirstDerivative.value].idxmin()
        elif baseline == Bounds.Middle.value:
            tm = measurement_data[(measurement_data[CaryDataframe.Temperature_K.value] > float(tm[0])) & (
                 measurement_data[CaryDataframe.Temperature_K.value] < float(tm[1]))]
            tm_index = tm[CaryDataframe.FirstDerivative.value].idxmin()
        elif baseline == Bounds.Upper.value:
            if len(tm) == 1:
                tm = measurement_data[measurement_data[CaryDataframe.Temperature_K.value] > float(tm)]
            else:
                tm = measurement_data[(measurement_data[CaryDataframe.Temperature_K.value] > float(tm[1])) &\
                                      (measurement_data[CaryDataframe.FirstDerivative.value] >= 0)]
            tm_index = tm[CaryDataframe.FirstDerivative.value].idxmin()
        else:
            raise ValueError(f"Baseline {baseline} not supported")

        start_index = max(tm_index - 5, 0)
        end_index = min(tm_index + 5, len(measurement_data) - 1)
        area_y = tm.loc[start_index:end_index + 1][CaryDataframe.Absorbance.value].to_list()
        area_x = tm.loc[start_index:end_index + 1][CaryDataframe.Temperature_K.value].to_list()
        mod = models.LinearModel()
        fit_function = mod.fit(area_y, x=area_x)
        return fit_function.values["slope"], fit_function.values["intercept"]

        # fit baselines for both transitions #TODO: BASELINES ANGUCKEN
        # global fit of complete curve -> baselines as input
        # calculate errors over all calculations
        # standard deviation and mean over triplets
        # functions to filter after specific data
        # write plot functions for desired analysis




