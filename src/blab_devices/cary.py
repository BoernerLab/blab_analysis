from enum import Enum, unique
import inspect
import json
from pathlib import Path
import re

import lmfit
from lmfit import models, Parameters
from more_itertools import split_at
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter

from src.blab_devices.exceptions import (InvalidCaryFormatError, InvalidHyperparameterError,
                                         InvalidHyperparameterHeaderError, NoJsonExtraInformationError,
                                         MolecularityRegexError, ExpectedTransitionsError,
                                         FillResultsError)


STAGE = 'Stage'
NO_STAGES_REGEX = re.compile('Number of stages,[0-9]+')
METADATA_REGEX = re.compile('(.*)_(\d+[.]\d+){1}nm_(\d+[.]\d+){1}-(\d+[.]\d+){1}C.*')


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
    RampType = 'Ramp Type'
    Name = 'Name'
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
    BaseLinesError = 'Baselines Std'
    ControlWavelength = 'Control Wavelength'
    SingleMeltingCurve = 'Single Melting Curve Object'
    MultiMeltingCurve = 'Multi Melting Curve Object'
    Molecularity = 'Molecularity'


@unique
class CaryResults(EnumToList):
    Filename = 'Filename'
    SampleName = 'SampleName'
    Measurements = 'Measurements'
    RampType = 'Ramp Type'
    FitMethod = 'FitMethod'
    DeltaS = 'deltaS'
    DeltaSError = 'deltaSerr'
    DeltaH = 'deltaH'
    DeltaHError = 'deltaHerr'
    Tm = 'Tm'
    TmError = 'TmErr'


@unique
class Bounds(EnumToList):
    Lower = 'lower'
    Upper = 'upper'
    Middle = 'middle'


@unique
class FitType(EnumToList):
    Single = 'single'
    Multi = 'multi'


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
                procedure = 'Cooling' if float(match.group(3)) >= float(match.group(4)) else 'Heating'
                meta_collection.append([measurement_index,
                                        str(match.group(1)),
                                        float(match.group(2)),
                                        float(match.group(3)),
                                        float(match.group(4)),
                                        procedure])
        self.data_meta = pd.DataFrame(meta_collection, columns=['Measurement',
                                                                'Name',
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
        self.dmf = DirectMeltFit()
        self.molecularity_one_digit_regex = re.compile(r'^[^0-9]*\d[^0-9]*$')
        self.molecularity_two_digit_regex = re.compile(r'^[^0-9]*\d[^0-9]*\d[^0-9]*$')
        self.results = pd.DataFrame(columns=CaryResults.list())
        for i, (data, data_meta, extra_information, hyperparameter, filename) in enumerate(zip(
                self.cary_object.list_data,
                self.cary_object.list_data_meta,
                self.cary_object.list_extra_information,
                self.cary_object.hyperparameters,
                self.cary_object.file_names)):
            self._add_extra_information_to_data_meta(data, data_meta, extra_information, filename)
            self._add_cell_numbers_to_meta(data, data_meta)
            self._normalize_absorbance_for_each_measurement(data)
            self._calculate_first_derivative(data)
            self._normalize_first_derivative(data)
            self._calculate_second_derivative(data)
            self._normalize_second_derivative(data)
            self._find_peak_for_measurement(data, data_meta, filename)
            self._fit_baselines(data, data_meta, filename, extra_information)
            self._single_curve_fit(data, data_meta, filename, extra_information)
            self._multi_curve_fit(data, data_meta, filename, extra_information)
            self._fill_results(data_meta, extra_information, filename)

    def _add_extra_information_to_data_meta(self, data, data_meta, extra_information, filename):
        for k, v in extra_information[filename].items():
            data_meta[k] = v
        data[CaryDataframe.Temperature_K.value] = data[CaryDataframe.Temperature.value] + 273.15


    def _add_cell_numbers_to_meta(self, data, data_meta):
        df_unique = data.drop_duplicates(subset=CaryDataframe.Measurement.value, keep='first')
        df_result = df_unique[[CaryDataframe.Measurement.value, CaryDataframe.CellNumber.value]]
        data_meta = pd.merge(data_meta, df_result, on=CaryDataframe.Measurement.value)


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
            savgol = savgol_filter(measurement_data[CaryDataframe.FirstDerivative.value], 11, 3)
            savgol_peaks, _ = find_peaks(savgol, height=0, width=5, distance=10)
            savgol_T = measurement_data[CaryDataframe.Temperature_K.value][savgol_peaks]
            savgol_dAbsdT = savgol[savgol_peaks]
            peaks = np.array([(savgol_x, savgol_y) for savgol_x, savgol_y in zip(savgol_T, savgol_dAbsdT)])
            if int(meta[CaryDataframe.ExpectedTransitions.value][unique_value-1]) != int(len(peaks)):
                print(f"WARNING: Number of found peaks doesn't match the expected transitions\n File: {filename}, Measurement: {unique_value}")
            for index in data_meta.index[data_meta[CaryDataframe.Measurement.value] == unique_value]:
                data_meta.at[index, CaryDataframe.FirstDerivativeSavgolPeaks.value] = peaks

    def _fit_baselines(self, data, data_meta, filename, extra_information):
        data_meta[CaryDataframe.BaseLines.value] = None
        data_meta[CaryDataframe.BaseLinesError.value] = None
        for unique_value in set(data[CaryDataframe.Measurement.value].tolist()):
            if float(data_meta.loc[data_meta['Measurement'] == unique_value, CaryDataframe.Wavelength.value]) != float(extra_information[filename][CaryDataframe.ControlWavelength.value]):
                mask = data[CaryDataframe.Measurement.value] == unique_value
                mask_meta = data_meta[CaryDataframe.Measurement.value] == unique_value
                measurement_data = data[mask]
                meta = data_meta[mask_meta]
                measurement_data.reset_index(drop=True, inplace=True)
                peaks = meta[CaryDataframe.FirstDerivativeSavgolPeaks.value].to_list()[0]
                if len(peaks) == 1:
                    tm = peaks[:, 0]
                    slope_low, slope_low_err, intercept_low, intercept_low_err = self._get_fit_for_baseline(measurement_data, tm, Bounds.Lower.value)
                    slope_high, slope_high_err, intercept_high, intercept_high_err = self._get_fit_for_baseline(measurement_data, tm, Bounds.Upper.value)
                    for index in data_meta.index[data_meta[CaryDataframe.Measurement.value] == unique_value]:
                        data_meta.at[index, CaryDataframe.BaseLines.value] = np.array([[slope_low, intercept_low], [slope_high, intercept_high]])
                        data_meta.at[index, CaryDataframe.BaseLinesError.value] = np.array([[slope_low_err, intercept_low_err], [slope_high_err, intercept_high_err]])
                elif len(peaks) == 2:
                    tm = sorted(peaks[:, 0])
                    slope_low, slope_low_err, intercept_low, intercept_low_err = self._get_fit_for_baseline(measurement_data, tm, Bounds.Lower.value)
                    slope_middle, slope_middle_err, intercept_middle, intercept_middle_err = self._get_fit_for_baseline(measurement_data, tm, Bounds.Middle.value)
                    slope_high, slope_high_err, intercept_high, intercept_high_err = self._get_fit_for_baseline(measurement_data, tm, Bounds.Upper.value)
                    for index in data_meta.index[data_meta[CaryDataframe.Measurement.value] == unique_value]:
                        data_meta.at[index, CaryDataframe.BaseLines.value] = np.array([[slope_low, intercept_low], [slope_middle, intercept_middle], [slope_high, intercept_high]])
                        data_meta.at[index, CaryDataframe.BaseLinesError.value] = np.array([[slope_low_err, intercept_low_err], [slope_middle_err, intercept_middle_err], [slope_high_err, intercept_high_err]])

                else:
                    for index in data_meta.index[data_meta[CaryDataframe.Measurement.value] == unique_value]:
                        data_meta.at[index, CaryDataframe.BaseLines.value] = []

    def _get_fit_for_baseline(self, measurement_data, tm, baseline):
        if baseline == Bounds.Lower.value:
            if len(tm) == 1:
                # Non-Melting Curves are vulnerable --> will fail
                tm = measurement_data[(measurement_data[CaryDataframe.Temperature_K.value] < float(tm)) &
                                      (measurement_data[CaryDataframe.FirstDerivative.value] >= 0)]
            else:
                tm = measurement_data[(measurement_data[CaryDataframe.Temperature_K.value] < float(tm[0])) &
                                      (measurement_data[CaryDataframe.FirstDerivative.value] >= 0)]

            tm_index = tm[CaryDataframe.FirstDerivative.value].idxmin()
        elif baseline == Bounds.Middle.value:
            tm = measurement_data[(measurement_data[CaryDataframe.Temperature_K.value] > float(tm[0])) & (
                 measurement_data[CaryDataframe.Temperature_K.value] < float(tm[1]))]
            tm_index = tm[CaryDataframe.FirstDerivative.value].idxmin()
        elif baseline == Bounds.Upper.value:
            if len(tm) == 1:
                tm = measurement_data[(measurement_data[CaryDataframe.Temperature_K.value] > float(tm)) &
                                      (measurement_data[CaryDataframe.FirstDerivative.value] >= 0)]
            else:
                tm = measurement_data[(measurement_data[CaryDataframe.Temperature_K.value] > float(tm[1])) &
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
        return (fit_function.values["slope"], fit_function.params["slope"].stderr,
                fit_function.values["intercept"], fit_function.params["intercept"].stderr)

    def _single_curve_fit(self, data, data_meta, filename, extra_information):
        data_meta[CaryDataframe.SingleMeltingCurve.value] = None
        methods = [attr for attr in dir(self.dmf) if callable(getattr(self.dmf, attr)) and not attr.startswith("__")]
        desired_global_fit = [method for method in methods if f'{CaryDataframe.Molecularity.value.lower()}_{extra_information[filename][CaryDataframe.Molecularity.value]}' in method][0]
        method = getattr(self.dmf, desired_global_fit, None)
        for unique_value in list(set(data[CaryDataframe.Measurement.value].tolist())):
            if float(data_meta.loc[data_meta['Measurement'] == unique_value, CaryDataframe.Wavelength.value]) != float(
                    extra_information[filename][CaryDataframe.ControlWavelength.value]):
                model = lmfit.Model(method)
                mask = data[CaryDataframe.Measurement.value] == unique_value
                mask_meta = data_meta[CaryDataframe.Measurement.value] == unique_value
                measurement_data = data[mask]
                meta = data_meta[mask_meta]
                if len(meta[CaryDataframe.FirstDerivativeSavgolPeaks.value][unique_value-1]) == int(extra_information[filename][CaryDataframe.ExpectedTransitions.value]):
                    if self.molecularity_two_digit_regex.search(desired_global_fit):
                        parse = model.make_params(DH1=dict(value=-250000, max=0),
                                                  DH2=dict(value=-250000, max=0),
                                                  Tm1=dict(value=meta[CaryDataframe.FirstDerivativeSavgolPeaks.value][unique_value-1][0][0],
                                                           min=min(measurement_data[CaryDataframe.Temperature_K.value]),
                                                           max=meta[CaryDataframe.FirstDerivativeSavgolPeaks.value][unique_value-1][1][0]),
                                                  Tm2=dict(value=meta[CaryDataframe.FirstDerivativeSavgolPeaks.value][unique_value-1][1][0],
                                                           min=meta[CaryDataframe.FirstDerivativeSavgolPeaks.value][unique_value-1][0][0],
                                                           max=max(measurement_data[CaryDataframe.Temperature_K.value])),
                                                  m1=dict(value=meta[CaryDataframe.BaseLines.value][unique_value-1][0][0],
                                                          min=meta[CaryDataframe.BaseLines.value][unique_value-1][0][0]-meta[CaryDataframe.BaseLinesError.value][unique_value-1][0][0],
                                                          max=meta[CaryDataframe.BaseLines.value][unique_value-1][0][0]+meta[CaryDataframe.BaseLinesError.value][unique_value-1][0][0]),
                                                  n1=dict(value=meta[CaryDataframe.BaseLines.value][unique_value-1][0][1],
                                                          min=meta[CaryDataframe.BaseLines.value][unique_value-1][0][1]-meta[CaryDataframe.BaseLinesError.value][unique_value-1][0][1],
                                                          max=meta[CaryDataframe.BaseLines.value][unique_value-1][0][1]+meta[CaryDataframe.BaseLinesError.value][unique_value-1][0][1]),
                                                  m2=dict(value=meta[CaryDataframe.BaseLines.value][unique_value-1][1][0],
                                                          min=meta[CaryDataframe.BaseLines.value][unique_value-1][1][0]-meta[CaryDataframe.BaseLinesError.value][unique_value-1][1][0],
                                                          max=meta[CaryDataframe.BaseLines.value][unique_value-1][1][0]+meta[CaryDataframe.BaseLinesError.value][unique_value-1][1][0]),
                                                  n2=dict(value=meta[CaryDataframe.BaseLines.value][unique_value-1][1][1],
                                                          min=meta[CaryDataframe.BaseLines.value][unique_value-1][1][1]-meta[CaryDataframe.BaseLinesError.value][unique_value-1][1][1],
                                                          max=meta[CaryDataframe.BaseLines.value][unique_value-1][1][1]+meta[CaryDataframe.BaseLinesError.value][unique_value-1][1][1]),
                                                  m3=dict(value=meta[CaryDataframe.BaseLines.value][unique_value-1][2][0],
                                                          min=meta[CaryDataframe.BaseLines.value][unique_value-1][2][0]-meta[CaryDataframe.BaseLinesError.value][unique_value-1][2][0],
                                                          max=meta[CaryDataframe.BaseLines.value][unique_value-1][2][0]+meta[CaryDataframe.BaseLinesError.value][unique_value-1][2][0]),
                                                  n3=dict(value=meta[CaryDataframe.BaseLines.value][unique_value-1][2][1],
                                                          min=meta[CaryDataframe.BaseLines.value][unique_value-1][2][1]-meta[CaryDataframe.BaseLinesError.value][unique_value-1][2][1],
                                                          max=meta[CaryDataframe.BaseLines.value][unique_value-1][2][1]+meta[CaryDataframe.BaseLinesError.value][unique_value-1][2][1])
                                                  )

                    #TODO: this cant be tested with our test files yet
                    elif self.molecularity_one_digit_regex.search(desired_global_fit):
                        parse = model.make_params(DH=dict(value=-250000, max=0),
                                                  Tm=dict(value=meta[CaryDataframe.FirstDerivativeSavgolPeaks.value][unique_value-1][0][0],
                                                           min=min(measurement_data[CaryDataframe.Temperature_K.value]),
                                                           max=max(measurement_data[CaryDataframe.Temperature_K.value])),
                                                  m1=dict(value=meta[CaryDataframe.BaseLines.value][unique_value-1][0][0],
                                                          min=meta[CaryDataframe.BaseLines.value][unique_value-1][0][0]-meta[CaryDataframe.BaseLinesError.value][unique_value-1][0][0],
                                                          max=meta[CaryDataframe.BaseLines.value][unique_value-1][0][0]+meta[CaryDataframe.BaseLinesError.value][unique_value-1][0][0]),
                                                  n1=dict(value=meta[CaryDataframe.BaseLines.value][unique_value-1][0][1],
                                                          min=meta[CaryDataframe.BaseLines.value][unique_value-1][0][1]-meta[CaryDataframe.BaseLinesError.value][unique_value-1][0][1],
                                                          max=meta[CaryDataframe.BaseLines.value][unique_value-1][0][1]+meta[CaryDataframe.BaseLinesError.value][unique_value-1][0][1]),
                                                  m2=dict(value=meta[CaryDataframe.BaseLines.value][unique_value-1][1][0],
                                                          min=meta[CaryDataframe.BaseLines.value][unique_value-1][1][0]-meta[CaryDataframe.BaseLinesError.value][unique_value-1][1][0],
                                                          max=meta[CaryDataframe.BaseLines.value][unique_value-1][1][0]+meta[CaryDataframe.BaseLinesError.value][unique_value-1][1][0]),
                                                  n2=dict(value=meta[CaryDataframe.BaseLines.value][unique_value-1][1][1],
                                                          min=meta[CaryDataframe.BaseLines.value][unique_value-1][1][1]-meta[CaryDataframe.BaseLinesError.value][unique_value-1][1][1],
                                                          max=meta[CaryDataframe.BaseLines.value][unique_value-1][1][1]+meta[CaryDataframe.BaseLinesError.value][unique_value-1][1][1])
                                                  )
                    else:
                        raise MolecularityRegexError
                    out = model.fit(data[CaryDataframe.Absorbance.value], params=parse, T=data[CaryDataframe.Temperature_K.value])
                    data_meta.at[unique_value-1, CaryDataframe.SingleMeltingCurve.value] = out
                else:
                    pass

    def _objective_1_transitions(self, params, measurement_data, meta, method):
        residual = []
        for measurement in meta[CaryDataframe.Measurement.value].tolist():
            subset = measurement_data[measurement_data[CaryDataframe.Measurement.value] == measurement]
            x = subset['Temperature (K)'].values
            y = subset['Absorbance'].values

            # Shared parameters
            DH = params['DH']
            Tm = params['Tm']

            # Dataset-specific parameters
            m1 = params[f'm1_measurement_{measurement}']
            n1 = params[f'n1_measurement_{measurement}']
            m2 = params[f'm2_measurement_{measurement}']
            n2 = params[f'n2_measurement_{measurement}']

            # Compute the model values
            model_values = method(x, DH, Tm, m1, n1, m2, n2)

            # Calculate the residuals
            residual.extend(model_values - y)

        return np.array(residual)

    def _objective_2_transitions(self, params, measurement_data, meta, method):
        residual = []
        for measurement in meta[CaryDataframe.Measurement.value].tolist():
            subset = measurement_data[measurement_data[CaryDataframe.Measurement.value] == measurement]
            x = subset['Temperature (K)'].values
            y = subset['Absorbance'].values

            # Shared parameters
            DH1 = params['DH1']
            DH2 = params['DH2']
            Tm1 = params['Tm1']
            Tm2 = params['Tm2']

            # Dataset-specific parameters
            m1 = params[f'm1_measurement_{measurement}']
            n1 = params[f'n1_measurement_{measurement}']
            m2 = params[f'm2_measurement_{measurement}']
            n2 = params[f'n2_measurement_{measurement}']
            m3 = params[f'm3_measurement_{measurement}']
            n3 = params[f'n3_measurement_{measurement}']

            # Compute the model values
            model_values = method(x, DH1, DH2, Tm1, Tm2, m1, n1, m2, n2, m3, n3)

            # Calculate the residuals
            residual.extend(model_values - y)

        return np.array(residual)

    def _multi_curve_fit(self, data, data_meta, filename, extra_information):
        data_meta[CaryDataframe.MultiMeltingCurve.value] = None
        methods = [attr for attr in dir(self.dmf) if callable(getattr(self.dmf, attr)) and not attr.startswith("__")]
        desired_global_fit = [method for method in methods if
                              f'{CaryDataframe.Molecularity.value.lower()}_{extra_information[filename][CaryDataframe.Molecularity.value]}' in method][0]
        method = getattr(self.dmf, desired_global_fit, None)

        data_meta_filtered = data_meta[data_meta[CaryDataframe.Wavelength.value] != extra_information[filename][
            CaryDataframe.ControlWavelength.value]]
        data_meta_grouped = data_meta_filtered.groupby([CaryDataframe.RampType.value, CaryDataframe.Name.value])
        for _, data_meta_filtered_group in data_meta_grouped:
            fit_checker = []
            params = lmfit.Parameters()
            mask = data[CaryDataframe.Measurement.value].isin(data_meta_filtered_group[CaryDataframe.Measurement.value].tolist())
            mask_meta = data_meta[CaryDataframe.Measurement.value].isin(data_meta_filtered_group[CaryDataframe.Measurement.value].tolist())
            measurement_data = data[mask]
            meta = data_meta[mask_meta]
            for measurement_index in data_meta_filtered_group[CaryDataframe.Measurement.value].tolist():
                if len(meta[CaryDataframe.FirstDerivativeSavgolPeaks.value][measurement_index - 1]) == int(extra_information[filename][CaryDataframe.ExpectedTransitions.value]):
                    fit_checker.append(True)
                else:
                    fit_checker.append(False)
            if list(set(fit_checker))[0] and len(list(set(fit_checker))) == 1:
                if self.molecularity_one_digit_regex.search(desired_global_fit):
                    Tm_mean = np.mean([peak[0][0] for peak in meta[CaryDataframe.FirstDerivativeSavgolPeaks.value]])
                    params.add("DH",
                               value=-250000,
                               max=0
                               )
                    params.add("Tm",
                               value=Tm_mean,
                               min=min(measurement_data[CaryDataframe.Temperature_K.value]),
                               max=max(measurement_data[CaryDataframe.Temperature_K.value])
                               )
                    for measurement in meta[CaryDataframe.Measurement.value].tolist():
                        params.add(f'm1_measurement_{measurement}',
                                         value=meta[CaryDataframe.BaseLines.value][measurement - 1][0][0],
                                         min=meta[CaryDataframe.BaseLines.value][measurement - 1][0][0] -
                                             meta[CaryDataframe.BaseLinesError.value][measurement - 1][0][0],
                                         max=meta[CaryDataframe.BaseLines.value][measurement - 1][0][0] +
                                             meta[CaryDataframe.BaseLinesError.value][measurement - 1][0][0]
                                         )
                        params.add(f'n1_measurement_{measurement}',
                                         value=meta[CaryDataframe.BaseLines.value][measurement - 1][0][1],
                                         min=meta[CaryDataframe.BaseLines.value][measurement - 1][0][1] -
                                             meta[CaryDataframe.BaseLinesError.value][measurement - 1][0][1],
                                         max=meta[CaryDataframe.BaseLines.value][measurement - 1][0][1] +
                                             meta[CaryDataframe.BaseLinesError.value][measurement - 1][0][1]
                                         )
                        params.add(f'm2_measurement_{measurement}',
                                         value=meta[CaryDataframe.BaseLines.value][measurement - 1][1][0],
                                         min=meta[CaryDataframe.BaseLines.value][measurement - 1][1][0] -
                                             meta[CaryDataframe.BaseLinesError.value][measurement - 1][1][0],
                                         max=meta[CaryDataframe.BaseLines.value][measurement - 1][1][0] +
                                             meta[CaryDataframe.BaseLinesError.value][measurement - 1][1][0]
                                         )
                        params.add(f'n2_measurement_{measurement}',
                                         value=meta[CaryDataframe.BaseLines.value][measurement - 1][1][1],
                                         min=meta[CaryDataframe.BaseLines.value][measurement - 1][1][1] -
                                             meta[CaryDataframe.BaseLinesError.value][measurement - 1][1][1],
                                         max=meta[CaryDataframe.BaseLines.value][measurement - 1][1][1] +
                                             meta[CaryDataframe.BaseLinesError.value][measurement - 1][1][1]
                                         )
                    minimizer = lmfit.Minimizer(self._objective_1_transitions, params,
                                                fcn_args=(measurement_data, meta, method))
                    out = minimizer.minimize()
                elif self.molecularity_two_digit_regex.search(desired_global_fit):
                    Tm1_mean = np.mean([peak[0][0] for peak in meta[CaryDataframe.FirstDerivativeSavgolPeaks.value]])
                    Tm2_mean = np.mean([peak[1][0] for peak in meta[CaryDataframe.FirstDerivativeSavgolPeaks.value]])
                    params.add("DH1",
                               value=-250000,
                               max=0
                               )
                    params.add("DH2",
                               value=-250000,
                               max=0
                               )
                    params.add("Tm1",
                               value=Tm1_mean,
                               min=min(measurement_data[CaryDataframe.Temperature_K.value]),
                               max=Tm2_mean
                               )
                    params.add("Tm2",
                               value=Tm2_mean,
                               min=Tm1_mean,
                               max=max(measurement_data[CaryDataframe.Temperature_K.value])
                               )
                    for measurement in meta[CaryDataframe.Measurement.value].tolist():
                        params.add(f'm1_measurement_{measurement}',
                                         value=meta[CaryDataframe.BaseLines.value][measurement - 1][0][0],
                                         min=meta[CaryDataframe.BaseLines.value][measurement - 1][0][0] -
                                             meta[CaryDataframe.BaseLinesError.value][measurement - 1][0][0],
                                         max=meta[CaryDataframe.BaseLines.value][measurement - 1][0][0] +
                                             meta[CaryDataframe.BaseLinesError.value][measurement - 1][0][0]
                                         )
                        params.add(f'n1_measurement_{measurement}',
                                         value=meta[CaryDataframe.BaseLines.value][measurement - 1][0][1],
                                         min=meta[CaryDataframe.BaseLines.value][measurement - 1][0][1] -
                                             meta[CaryDataframe.BaseLinesError.value][measurement - 1][0][1],
                                         max=meta[CaryDataframe.BaseLines.value][measurement - 1][0][1] +
                                             meta[CaryDataframe.BaseLinesError.value][measurement - 1][0][1]
                                         )
                        params.add(f'm2_measurement_{measurement}',
                                         value=meta[CaryDataframe.BaseLines.value][measurement - 1][1][0],
                                         min=meta[CaryDataframe.BaseLines.value][measurement - 1][1][0] -
                                             meta[CaryDataframe.BaseLinesError.value][measurement - 1][1][0],
                                         max=meta[CaryDataframe.BaseLines.value][measurement - 1][1][0] +
                                             meta[CaryDataframe.BaseLinesError.value][measurement - 1][1][0]
                                         )
                        params.add(f'n2_measurement_{measurement}',
                                         value=meta[CaryDataframe.BaseLines.value][measurement - 1][1][1],
                                         min=meta[CaryDataframe.BaseLines.value][measurement - 1][1][1] -
                                             meta[CaryDataframe.BaseLinesError.value][measurement - 1][1][1],
                                         max=meta[CaryDataframe.BaseLines.value][measurement - 1][1][1] +
                                             meta[CaryDataframe.BaseLinesError.value][measurement - 1][1][1]
                                         )
                        params.add(f'm3_measurement_{measurement}',
                                         value=meta[CaryDataframe.BaseLines.value][measurement - 1][2][0],
                                         min=meta[CaryDataframe.BaseLines.value][measurement - 1][2][0] -
                                             meta[CaryDataframe.BaseLinesError.value][measurement - 1][2][0],
                                         max=meta[CaryDataframe.BaseLines.value][measurement - 1][2][0] +
                                             meta[CaryDataframe.BaseLinesError.value][measurement - 1][2][0]
                                         )
                        params.add(f'n3_measurement_{measurement}',
                                         value=meta[CaryDataframe.BaseLines.value][measurement - 1][2][1],
                                         min=meta[CaryDataframe.BaseLines.value][measurement - 1][2][1] -
                                             meta[CaryDataframe.BaseLinesError.value][measurement - 1][2][1],
                                         max=meta[CaryDataframe.BaseLines.value][measurement - 1][2][1] +
                                             meta[CaryDataframe.BaseLinesError.value][measurement - 1][2][1]
                                         )
                    minimizer = lmfit.Minimizer(self._objective_2_transitions, params,
                                                fcn_args=(measurement_data, meta, method))
                    out = minimizer.minimize()
                else:
                    raise MolecularityRegexError

                for measurement in meta[CaryDataframe.Measurement.value].tolist():
                    data_meta.at[measurement -1, CaryDataframe.MultiMeltingCurve.value] = out

    def _fill_results(self, data_meta, extra_information, filename):
        data_meta_filtered = data_meta[data_meta[CaryDataframe.Wavelength.value] != extra_information[filename][CaryDataframe.ControlWavelength.value]]
        data_meta_grouped = data_meta_filtered.groupby([CaryDataframe.RampType.value, CaryDataframe.Name.value])
        for group, data_meta_filtered_group in data_meta_grouped:
            for fit_type in [FitType.Single.value, FitType.Multi.value]:
                row: dict = {}
                row[CaryResults.Filename.value] = filename
                row[CaryResults.Measurements.value] = data_meta_filtered_group[CaryDataframe.Measurement.value].to_list()
                row[CaryResults.RampType.value] = group[0]
                row[CaryResults.SampleName.value] = group[1]
                row[CaryResults.FitMethod.value] = fit_type
                if fit_type == FitType.Single.value:
                    self.extract_fit_results(extra_information, filename, data_meta_filtered_group[CaryDataframe.SingleMeltingCurve.value].to_list(), row)
                elif fit_type == FitType.Multi.value:
                    self.extract_fit_results(extra_information, filename, data_meta_filtered_group[CaryDataframe.MultiMeltingCurve.value].to_list(), row)
                else:
                    raise FillResultsError
                self.results.loc[len(self.results)] = row
        # TODO: Errors sind gottlos ... sometimes zero ... check fits individually for curves
        # calculate errors over all calculations
        # standard deviation and mean over triplets
        # functions to filter after specific data
        # write plot functions for desired analysis

    def extract_fit_results(self, extra_information, filename, grouped_melting_curves, row):
        if extra_information[filename][CaryDataframe.ExpectedTransitions.value] == 1:
            grouped_melting_curves = [melting_curve for melting_curve in grouped_melting_curves if
                                      melting_curve != None]
            if grouped_melting_curves:
                Tm_raw = [melting_curve.params['Tm'].value for melting_curve in grouped_melting_curves if melting_curve is not None]
                if len(list(set(Tm_raw))) == 1 and list(set(Tm_raw))[0] == None:
                    Tm_mean = [None]
                else:
                    Tm_mean = [np.mean(Tm_raw)]
                row[CaryResults.Tm.value] = Tm_mean

                TmError_raw = [melting_curve.params['Tm'].stderr for melting_curve in grouped_melting_curves if melting_curve is not None]
                if len(list(set(TmError_raw))) == 1 and list(set(TmError_raw))[0] == None:
                    TmError_mean = [None]
                else:
                    TmError_mean = [np.mean(TmError_raw)]
                row[CaryResults.TmError.value] = TmError_mean

                DeltaH_raw = [melting_curve.params['DH'].value for melting_curve in grouped_melting_curves if melting_curve is not None]
                if len(list(set(DeltaH_raw))) == 1 and list(set(DeltaH_raw))[0] == None:
                    DeltaH_mean = [None]
                else:
                    DeltaH_mean = [np.mean(DeltaH_raw)]
                row[CaryResults.DeltaH.value] = DeltaH_mean

                DeltaHError_raw = [melting_curve.params['DH'].stderr for melting_curve in grouped_melting_curves if melting_curve is not None]
                if len(list(set(DeltaHError_raw))) == 1 and list(set(DeltaHError_raw))[0] == None:
                    DeltaHError_mean = [None]
                else:
                    DeltaHError_mean = [np.mean(DeltaHError_raw)]
                row[CaryResults.DeltaHError.value] = DeltaHError_mean

                row[CaryResults.DeltaS.value] = self._get_DS(DeltaH_mean, Tm_mean)
                row[CaryResults.DeltaSError.value] = self._get_DDS(DeltaH_mean, Tm_mean, DeltaHError_mean, TmError_mean)

        elif extra_information[filename][CaryDataframe.ExpectedTransitions.value] == 2:
            grouped_melting_curves = [melting_curve for melting_curve in grouped_melting_curves if
                                      melting_curve != None]
            if grouped_melting_curves:
                Tm_raw = [[melting_curve.params['Tm1'].value, melting_curve.params['Tm2'].value] for melting_curve in grouped_melting_curves]
                Tm_mean = [sum(values) / len(values) if values else None for values in [list(filter(None, pair)) for pair in zip(*Tm_raw)]]
                row[CaryResults.Tm.value] = Tm_mean

                TmError_raw = [[melting_curve.params['Tm1'].stderr, melting_curve.params['Tm2'].stderr] for melting_curve in grouped_melting_curves]
                TmError_mean = [sum(values) / len(values) if values else None for values in [list(filter(None, pair)) for pair in zip(*TmError_raw)]]
                row[CaryResults.TmError.value] = TmError_mean

                DeltaH_raw = [[melting_curve.params['DH1'].value, melting_curve.params['DH2'].value] for melting_curve in grouped_melting_curves]
                DeltaH_mean = [sum(values) / len(values) if values else None for values in [list(filter(None, pair)) for pair in zip(*DeltaH_raw)]]
                row[CaryResults.DeltaH.value] = DeltaH_mean

                DeltaHError_raw = [[melting_curve.params['DH1'].stderr, melting_curve.params['DH2'].stderr] for melting_curve in grouped_melting_curves]
                DeltaHError_mean = [sum(values) / len(values) if values else None for values in [list(filter(None, pair)) for pair in zip(*DeltaHError_raw)]]
                row[CaryResults.DeltaHError.value] = DeltaHError_mean

                row[CaryResults.DeltaS.value] = self._get_DS(DeltaH_mean, Tm_mean)
                row[CaryResults.DeltaSError.value] = self._get_DDS(DeltaH_mean, Tm_mean, DeltaHError_mean, TmError_mean)

        else:
            raise ExpectedTransitionsError

    def _get_DS(self, DH, Tm):
         return [(v1 / v2 if v2 != 0 else None) if v1 is not None and v2 is not None else None for v1, v2 in zip(DH, Tm)]

    def _get_DDS(self, DH, Tm, DDH, DTm):
        results = []
        for dh, tm, ddh, dtm in zip(DH, Tm, DDH, DTm):
            if dh is None or tm is None or ddh is None or dtm is None:
                results.append(None)
            else:
                results.append(abs(1 / tm * ddh)+abs(dh / tm**2 * dtm))
        return results


class DirectMeltFit:
    def __init__(self):
        self.r = 8.31446261815324

    def molecularity_1(self, T, DH, Tm, m1, n1, m2, n2):
        """
        Analytical melting curve function for intramolecular reactions. Böttcher et. al., ...

        Parameters
        ----------
        T : TYPE
            Temperature in K.
        DH : TYPE
            Molar Enthalpy change in .
        Tm : TYPE
            Melting temeprature in K.
        m1 : TYPE
            Lower baseline inclination in K^-1.
        n1 : TYPE
            Lowerr baseline y-intersect.
        m2 : TYPE
            Upper baseline incliantion in K^-1.
        n2 : TYPE
            Upper baseline y-intersect.

        Returns
        -------
        TYPE
            Decadic Absorption E_d(T) for the given parameters.

        """
        x = DH / self.r * (1 / T - 1 / Tm)
        theta = 1 / (1 + np.exp(x))
        return (m1 * T + n1) * theta + (m2 * T + n2) * (1 - theta)

    def molecularity_2(self, T, DH, Tm, m1, n1, m2, n2):
        """
        Analytical melting curve function for bimolecular reactions. Böttcher et. al., ...

       Parameters
       ----------
       T : TYPE
           Temperature in K.
       DH : TYPE
           Molar Enthalpy change in .
       Tm : TYPE
           Melting temeprature in K.
       m1 : TYPE
           Lower baseline inclination in K^-1.
       n1 : TYPE
           Lowerr baseline y-intersect.
       m2 : TYPE
           Upper baseline incliantion in K^-1.
       n2 : TYPE
           Upper baseline y-intersect.

        Returns
        -------
        TYPE
            Decadic Absorption E_d(T) for the given parameters.

        """
        x = DH / self.r * (1 / T - 1 / Tm)
        theta = 1 - 2 / (1 + np.sqrt(1 + 8 * np.exp(-x)))
        return (m1 * T + n1) * theta + (m2 * T + n2) * (1 - theta)

    def molecularity_1_1(self, T, DH1, DH2, Tm1, Tm2, m1, n1, m2, n2, m3, n3):
        """
        Analytical melting curve function for bimolecular reactions. Böttcher et. al., ...

       Parameters
       ----------
       T : TYPE
           Temperature in K.
       DH : TYPE
           Molar Enthalpy change in .
       Tm : TYPE
           Melting temeprature in K.
       m1 : TYPE
           Lower baseline inclination in K^-1.
       n1 : TYPE
           Lower baseline y-intersect.
       m2 : TYPE
           Mid baseline incliantion in K^-1.
       n2 : TYPE
           Mid baseline y-intersect.
       m3 : TYPE
           Upper baseline incliantion in K^-1.
       n3 : TYPE
           Upper baseline y-intersect.

        Returns
        -------
        TYPE
            Decadic Absorption E_d(T) for the given parameters.

        """
        x1 = DH1 / self.r * (1 / T - 1 / Tm1)
        x2 = DH2 / self.r * (1 / T - 1 / Tm2)
        theta11 = 1 / (1 + np.exp(x1))
        theta12 = 1 / (1 + np.exp(x2))
        return (m1 * T + n1) * theta11 + (m2 * T + n2) * (theta12 - theta11) + (m3 * T + n3) * (1 - theta12)

    def molecularity_2_2(self, T, DH1, DH2, Tm1, Tm2, m1, n1, m2, n2, m3, n3):
        """
        Analytical melting curve function for bimolecular reactions. Böttcher et. al., ...

       Parameters
       ----------
       T : TYPE
           Temperature in K.
       DH : TYPE
           Molar Enthalpy change in .
       Tm : TYPE
           Melting temeprature in K.
       m1 : TYPE
           Lower baseline inclination in K^-1.
       n1 : TYPE
           Lower baseline y-intersect.
       m2 : TYPE
           Mid baseline incliantion in K^-1.
       n2 : TYPE
           Mid baseline y-intersect.
       m3 : TYPE
           Upper baseline incliantion in K^-1.
       n3 : TYPE
           Upper baseline y-intersect.

        Returns
        -------
        TYPE
            Decadic Absorption E_d(T) for the given parameters.

        """
        x1 = DH1 / self.r * (1 / T - 1 / Tm1)
        x2 = DH2 / self.r * (1 / T - 1 / Tm2)
        theta21 = 1 - 2 / (1 + np.sqrt(1 + 8 * np.exp(-x1)))
        theta22 = 1 - 2 / (1 + np.sqrt(1 + 8 * np.exp(-x2)))
        return (m1 * T + n1) * theta21 + (m2 * T + n2) * (theta22 - theta21) + (m3 * T + n3) * (1 - theta22)

    def molecularity_1_2(self, T, DH1, DH2, Tm1, Tm2, m1, n1, m2, n2, m3, n3):
        """
        Analytical melting curve function for bimolecular reactions. Böttcher et. al., ...

       Parameters
       ----------
       T : TYPE
           Temperature in K.
       DH : TYPE
           Molar Enthalpy change in .
       Tm : TYPE
           Melting temeprature in K.
       m1 : TYPE
           Lower baseline inclination in K^-1.
       n1 : TYPE
           Lower baseline y-intersect.
       m2 : TYPE
           Mid baseline incliantion in K^-1.
       n2 : TYPE
           Mid baseline y-intersect.
       m3 : TYPE
           Upper baseline incliantion in K^-1.
       n3 : TYPE
           Upper baseline y-intersect.

        Returns
        -------
        TYPE
            Decadic Absorption E_d(T) for the given parameters.

        """
        x1 = DH1 / self.r * (1 / T - 1 / Tm1)
        x2 = DH2 / self.r * (1 / T - 1 / Tm2)
        theta11 = 1 / (1 + np.exp(x1))
        theta22 = 1 - 2 / (1 + np.sqrt(1 + 8 * np.exp(-x2)))
        return (m1 * T + n1) * theta11 + (m2 * T + n2) * (theta22 - theta11) + (m3 * T + n3) * (1 - theta22)

    def molecularity_2_1(self, T, DH1, DH2, Tm1, Tm2, m1, n1, m2, n2, m3, n3):
        """
        Analytical melting curve function for bimolecular reactions. Böttcher et. al., ...

       Parameters
       ----------
       T : TYPE
           Temperature in K.
       DH : TYPE
           Molar Enthalpy change in .
       Tm : TYPE
           Melting temeprature in K.
       m1 : TYPE
           Lower baseline inclination in K^-1.
       n1 : TYPE
           Lower baseline y-intersect.
       m2 : TYPE
           Mid baseline incliantion in K^-1.
       n2 : TYPE
           Mid baseline y-intersect.
       m3 : TYPE
           Upper baseline incliantion in K^-1.
       n3 : TYPE
           Upper baseline y-intersect.

        Returns
        -------
        TYPE
            Decadic Absorption E_d(T) for the given parameters.

        """
        x1 = DH1 / self.r * (1 / T - 1 / Tm1)
        x2 = DH2 / self.r * (1 / T - 1 / Tm2)
        theta12 = 1 - 2 / (1 + np.sqrt(1 + 8 * np.exp(-x2)))
        theta21 = 1 / (1 + np.exp(x1))
        return (m1 * T + n1) * theta12 + (m2 * T + n2) * (theta21 - theta12) + (m3 * T + n3) * (1 - theta21)
