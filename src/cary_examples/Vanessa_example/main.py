import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import pickle
from pathlib import Path

from src import devices as blab

import re
from scipy import stats, signal, special
from scipy.optimize import curve_fit, OptimizeWarning, root_scalar
import lmfit


class MeltingCurveAnalysis:
    def __init__(self, extra_dict: dict = {}):
        self.data: dict = {}
        self.filenames: list = list(Path('data_raw/').glob('*.csv'))
        self.current_data = None
        self.extra_dict = extra_dict
        self.meta_overview = None

    def fill_data_dict(self):
        for file in self.filenames:
            self.data[file.name] = {}
            self.current_data = blab.Carry(file_path=f'{file}')
            measurement_numbers = self._get_number_measurements()
            for measurement_number in measurement_numbers:
                meta = self._get_filtered_meta_data(measurement_number)
                meta_extra = MeltingCurveAnalysis._add_extra_meta_file(
                    self, meta, file.name)
                self.data[file.name][f'Measurement_{measurement_number}'] = {
                    'data': self._get_filtered_data(measurement_number),
                    'meta': meta_extra
                }
        MeltingCurveAnalysis._create_meta_overview(self)

    def _get_number_measurements(self):
        measurement_numbers = self.current_data.data['Measurement'].unique()
        return measurement_numbers

    def _get_filtered_data(self, measurement_number):
        filtered_data = self.current_data.data[self.current_data.data['Measurement']
                                               == measurement_number]
        return filtered_data

    def _get_filtered_meta_data(self, measurement_number):
        filtered_meta_data = self.current_data.data_meta[
            self.current_data.data_meta['Measurement'] == measurement_number]
        filtered_meta_data = filtered_meta_data.iloc[:, :]
        return filtered_meta_data.to_dict('records')[0]

    @staticmethod
    def _add_extra_meta_file(self, meta, filename):
        extra_information: dict = self.extra_dict.get(filename, {})
        meta.update(extra_information)
        return meta

 #   @staticmethod
    def save(self, filename: str):
        with open(f'{filename}.pickle', 'wb') as handle:
            pickle.dump(self.data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(filename: str):
        with open(f'{filename}.pickle', 'rb') as handle:
            data = pickle.load(handle)
        return data

    def _create_meta_overview(self):
        gi = 1
        df = pd.DataFrame()
        for file_name in self.data.keys():
            for key, value in self.data[file_name].items():
                value['meta'].update({'gi': gi, 'file_name': file_name})
                newrow = pd.DataFrame.from_dict([value['meta']])
                df = pd.concat([df, newrow], ignore_index=True)
                gi += 1
        self.meta_overview = df

    def meta_from_name_Paul(self, file_names: list, add: dict = {}):
        regex_pattern_concentration = r'(\d*\.?\d?\d?).*([nµmu]+M)'
        regex_pattern_ion = r'.*([A-Z]{1}[a-z]*\(I\)|MgCl2|CaCl2).*'
        regex_pattern_buffer = r'(HEPES|MOPS|TRIS-HCl)'
        regex_pattern_pH = r'pH(\d+[.]?\d?)'
        for file_name in file_names:
            for key, value in self.data[file_name].items():
                string = value['meta']['Sample Name']
                matches = re.match(regex_pattern_concentration, string)
                if matches:
                    value['meta']['Concentration'] = float(matches.group(1))
                    value['meta']['Concentration Unit'] = matches.group(2)
                matches = re.match(regex_pattern_ion, string)
                if matches:
                    value['meta']['Ion'] = matches.group(1)
                matches = re.match(regex_pattern_buffer, string)
                if matches:
                    value['meta']['Buffer'] = matches.group(1)
                matches = re.match(regex_pattern_pH, string)
                if matches:
                    value['meta']['pH'] = float(matches.group(1))
                value['meta'].update(add)
        MeltingCurveAnalysis._create_meta_overview(self)

   
    def apply_filters(self, filters: dict) -> pd.DataFrame:
        """
        Filter 'meta-overview' by criteria in dictionary.

        Parameters
        ----------
        filters : dict
            filters = {'column_name': [value1, value2,...]}

        Returns
        -------
        filtered_df : pd.DataFrame
            Filtered DataFrame.

        """
        filtered_df = self.meta_overview.copy()
        for key, values in filters.items():
            filtered_df = filtered_df[filtered_df[key].isin(values)]
        return filtered_df
    
   
    def normalize(x: list, lower: float = None, upper: float = None):
        """
        Normalize list 'x' to 'lower' and 'upper' value. If None is given to min(x) = 0 and max(x) = 1.
        
        Parameters
        ----------
        x : list
            Values to normalize.
        lower : float, optional
            Value eqivilant to '0'. The default is None.
        upper : float, optional
            Value eqivilant to '1'. The default is None.

        Returns
        -------
        array
            Normalized array values calculated by (x - lower) / (upper - lower).

        """
        if lower is None:
            lower = x.min()
        if upper is None:
            upper = x.max()
        return (x - lower) / (upper - lower)
    
    def normalize_data(df: pd.DataFrame,
                       column_to_normalize: str = 'Absorbance',
                       column_bounds: str = 'Temperature (°C)',
                       column_normalized_name: str = None,
                       normalize_to: tuple = None):
        """
        Normalize melting curve dataframe to a given temperature values or to min and max value.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe with curve data.
        column_to_normalize : str, optional
            Name of 'y'-collumn. The default is 'Absorbance'.
        column_bounds : str, optional
            Name of 'x'-collumn. The default is 'Temperature (°C)'.
        column_normalized_name : str, optional
            Name of 'Normalized-collumn. The default is f'Normalized {column_to_normalize}'.
        normalize_to : tuple, optional
            Temperatures, which Abs-values should be '0' and '1' in given form: (T_lower, T_upper). The default is None.

        Returns
        -------
        df : TYPE
            Dataframe with added normalized collum with the name f'Normalized {column_to_normalize}'

        """
        closests = [min(df[column_to_normalize]), max(df[column_to_normalize])]
        if normalize_to is not None:   
            for i in [0, 1]:
                df_closest = df.iloc[(df[column_bounds] - normalize_to[i]).abs().argsort()[:1]]
                closests[i] = df_closest[column_to_normalize].tolist()[0]
        if column_normalized_name is None:
            column_normalized_name = f'Normalized {column_to_normalize}'
        df[column_normalized_name] = MeltingCurveAnalysis.normalize(
            df[column_to_normalize], closests[0], closests[1])
        return df
 
   
            
    
    
    def filter_by_values(df: pd.DataFrame, filter_bounds: dict = {'column_x': 'Temperature (°C)', 'column_y': 'Absorbance', 'x_min': -np.inf, 'y_min':
                             - np.inf, 'x_max': np.inf, 'y_max': np.inf, 'keep_inside': True}):
        """
        Crop/remove parts of a curve with x and y values.

        Parameters
        ----------
        df : DataFrame
            Data of curve.
        filter_bounds : dict, optional
           Dictionarys which contains a rectengular filter area. You have to specify each axis name and each axis min and max border. Everything inside the rectangular border will remain inside the dictionary. If you want to delete the array set 'keep_inside' as False.

        Returns
        -------
        DataFrame
            Cropped curve.

        """
        filter_bounds_updated = {'column_x': 'Temperature (°C)', 'column_y': 'Absorbance', 'x_min': -np.inf, 'y_min':
                                 - np.inf, 'x_max': np.inf, 'y_max': np.inf, 'keep_inside': True}
        filter_bounds_updated.update(filter_bounds) 
        column_x = filter_bounds_updated.get('column_x')
        column_y = filter_bounds_updated.get('column_y')
        df_filtered = df.copy() 
        inside = df_filtered.loc[
            (df_filtered[column_x] >= filter_bounds_updated['x_min']) & (df_filtered[column_x] <= filter_bounds_updated['x_max']) &
            (df_filtered[column_y] >= filter_bounds_updated['y_min']) & (
                df_filtered[column_y] <= filter_bounds_updated['y_max']), :]
        if filter_bounds_updated.get('keep_inside', True) is True:
            return inside
        else:
            return pd.merge(df_filtered, inside, how='outer', indicator=True).query('_merge=="left_only"').drop(
                columns=['_merge'])
        
    def filter_multiple_times(df: pd.DataFrame, filter_bounds_list: list = [], sum_curve: bool = True):
        """
        Filter a curve multiple times.

        Parameters
        ----------
        df : pd.DataFrame
            Curve.
        filter_bounds_list : list, optional
            List of filter bounds -> See 'filter_by_values'. The default is [].
        sum_curve: bool, optional
            True: Each filter is applied to the raw curve and concatinated at the end. False: Each filter is applied to the remaining curve from the filter process before.
        
        Returns
        -------
        filtered_df: pd.DataFrame
            Concated dataframe from each filter area used on the inputet dataframe. 

        """
        if not filter_bounds_list:
            return df
        filtered_df = df.copy()
        temp_list = np.zeros_like(filter_bounds_list)
        if sum_curve is True:
            for index, value in enumerate(filter_bounds_list):
                temp_list[index]= MeltingCurveAnalysis.filter_by_values(filtered_df, value)
            df_out = pd.concat(temp_list)
            df_out.drop_duplicates()
            return df_out
        else:
            for index, value in enumerate(filter_bounds_list): 
                filtered_df= MeltingCurveAnalysis.filter_by_values(filtered_df, value)
            return filtered_df
        
    
    #### Direct UV-Fit
    def equation_molecularity_1(T, DH, Tm, m1, n1, m2, n2):
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
        R = 8.31446261815324  # m^3* Pa * K^−1 * mol^−1
        x = DH / R * (1 / T - 1 / Tm )
        theta = 1 / (1 + np.exp(x)) 
        return (m1 * T + n1) * theta + (m2 * T + n2) * (1 - theta)

  
    def equation_molecularity_2(T, DH, Tm, m1, n1, m2, n2):
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
        R = 8.31446261815324   # m^3* Pa * K^−1 * mol^−1
        x = DH / R * (1 / T - 1 / Tm)
        theta = 1 - 2/(1 + np.sqrt(1 + 8*np.exp(-x))) 
        return (m1 * T + n1) * theta + (m2 * T + n2) * (1 - theta)


    
    def create_model_e1():
        """
        Create a model for reaction with molecularity 1 suitible for the lmfit-toolbox. A Converison form °C to K is automatically performed. If you already performed a conversion, give XXX = True.

        Returns
        -------
        e1_mod : TYPE
            lmfit-Model.
        pars : TYPE
            Dictionary of paramters: DH...molecular melting enthalpy, Tm...melting temperature, .

        """
        e1_mod = lmfit.Model(MeltingCurveAnalysis.equation_molecularity_1, prefix='e1_')
        pars = e1_mod.make_params(DH=dict(value=-150000, max=0),
                                  Tm=dict(value=330, min=200, max=400),
                                  m1=0.0006,
                                  m2=0.0009,
                                  n1=0.1,
                                  n2=0.2)
        return e1_mod, pars

    def create_model_e2():
        """
        Create a model for reaction with molecularity 2 suitible for the lmfit-toolbox. A Converison form °C to K is automatically performed. If you already performed a conversion, give XXX = True.

        Returns
        -------
        e1_mod : TYPE
            lmfit-Model.
        pars : TYPE
            Dictionary of paramters: DH...molecular melting enthalpy, Tm...melting temperature, .

        """
        e2_mod = lmfit.Model(MeltingCurveAnalysis.equation_molecularity_2, prefix='e2_')
        pars = e2_mod.make_params(DH=dict(value=-150000, max=0), Tm=dict(value=320,
                                  min=200, max=400), m1=0.0006, m2=0.0009, n1=0.1, n2=0.2)
        return e2_mod, pars

    def create_model_e2_e1():
        """
        Create a model for reaction with molecularity 1 and molecularity 2 suitible for the lmfit-toolbox. A Converison form °C to K is automatically performed. If you already performed a conversion, give XXX = True.

        Returns
        -------
        e1_mod : TYPE
            lmfit-Model object for intramolecular reaction.
        e1_mod : TYPE
            lmfit-Model object for bimolecular reaction.
        pars : TYPE
            Dictionary of paramters.

        """
        e2_mod = lmfit.Model(MeltingCurveAnalysis.equation_molecularity_2, prefix='e2_')
        e1_mod = lmfit.Model(MeltingCurveAnalysis.equation_molecularity_1, prefix='e1_')
        pars = e2_mod.make_params(DH=dict(value=-150000, max=0), Tm=dict(value=325,
                                  min=200, max=400), m1=0.0006, m2=0.0009, n1=0.1, n2=0.2)
        pars.update(e1_mod.make_params(DH=dict(value=-150000, max=0), Tm=dict(value=345, min=200, max=400),
                                       m1=dict(expr='e2_m2'), m2=0.0005, n1=dict(expr='e2_n2'), n2=0.2))
        print(f'parameter names: {e1_mod.param_names}')
        print(f'parameter names: {e2_mod.param_names}')
        print(f'independent variables: {e1_mod.independent_vars}')
        print(f'independent variables: {e2_mod.independent_vars}')
        return e2_mod + e1_mod, pars
    
    def create_model_e2_e2():
        """
        Create a model for reaction with molecularity 1 and molecularity 2 suitible for the lmfit-toolbox. A Converison form °C to K is automatically performed. If you already performed a conversion, give XXX = True.

        Returns
        -------
        e1_mod : TYPE
            lmfit-Model object for intramolecular reaction.
        e1_mod : TYPE
            lmfit-Model object for bimolecular reaction.
        pars : TYPE
            Dictionary of paramters.

        """
        e2_mod = lmfit.Model(MeltingCurveAnalysis.equation_molecularity_2, prefix='e2_')
        e1_mod = lmfit.Model(MeltingCurveAnalysis.equation_molecularity_2, prefix='e1_')
        pars = e2_mod.make_params(DH=dict(value=-150000, max=0), Tm=dict(value=325,
                                  min=200, max=400), m1=0.0006, m2=0.0009, n1=0.1, n2=0.2)
        pars.update(e1_mod.make_params(DH=dict(value=-150000, max=0), Tm=dict(value=345, min=200, max=400),
                                       m1=dict(expr='e2_m2'), m2=0.0005, n1=dict(expr='e2_n2'), n2=0.2))
        print(f'parameter names: {e1_mod.param_names}')
        print(f'parameter names: {e2_mod.param_names}')
        print(f'independent variables: {e1_mod.independent_vars}')
        print(f'independent variables: {e2_mod.independent_vars}')
        return e2_mod + e1_mod, pars
    
    def create_model_e1_e1():
        """
        Create a model for reaction with molecularity 1 and molecularity 2 suitible for the lmfit-toolbox. A Converison form °C to K is automatically performed. If you already performed a conversion, give XXX = True.

        Returns
        -------
        e1_mod : TYPE
            lmfit-Model object for intramolecular reaction.
        e1_mod : TYPE
            lmfit-Model object for bimolecular reaction.
        pars : TYPE
            Dictionary of paramters.

        """
        e2_mod = lmfit.Model(MeltingCurveAnalysis.equation_molecularity_1, prefix='e2_')
        e1_mod = lmfit.Model(MeltingCurveAnalysis.equation_molecularity_1, prefix='e1_')
        pars = e2_mod.make_params(DH=dict(value=-150000, max=0), Tm=dict(value=325,
                                  min=200, max=400), m1=0.0006, m2=0.0009, n1=0.1, n2=0.2)
        pars.update(e1_mod.make_params(DH=dict(value=-150000, max=0), Tm=dict(value=345, min=200, max=400),
                                       m1=dict(expr='e2_m2'), m2=0.0005, n1=dict(expr='e2_n2'), n2=0.2))
        print(f'parameter names: {e1_mod.param_names}')
        print(f'parameter names: {e2_mod.param_names}')
        print(f'independent variables: {e1_mod.independent_vars}')
        print(f'independent variables: {e2_mod.independent_vars}')
        return e2_mod + e1_mod, pars
    
    def direct_melt_fit(self, filename, measurement_number,
                               model_pars: lmfit.Model,  # = MeltingCurveAnalysis._create_model_e1(),
                               p0: lmfit.Parameters = lmfit.Parameters(),
                               column_x: str = 'Temperature (K)',
                               column_y: str = 'Absorbance',
                               first_derivative_name: str = None,
                               weights: list = [{'first_derivative_border_x_min': -np.inf, 'first_derivative_border_x_max': np.inf, 'first_derivative_border_y_min': -np.inf, 'first_derivative_border_y_max': np.inf, 'data_x_min': -np.inf, 'data_y_min': -
                                                 np.inf, 'data_x_max': np.inf, 'data_y_max': np.inf, 'deri_weight': 1, 'data_weight': 1}],
                               filter_bounds: list = [{}],
                               plot: bool = True):
        df_data_raw = self.data[filename][f'Measurement_{measurement_number}']['data']
        #Add Collumn for T in K.
        df_data_raw['Temperature (K)'] = df_data_raw['Temperature (°C)'] + 273.15 
        
        #Filter dataframe to user input
        df = MeltingCurveAnalysis.filter_multiple_times(df_data_raw,
                                                        filter_bounds)
        
        #Calculate first derivative for weights
        '''
        df_data, first_derivative_name = MeltingCurveAnalysis._calculate_derivative(
            self.data[filename][f'Measurement_{measurement_number}']['data'],
            column_x=column_x,
            column_y=column_y,
            first_derivative_name=first_derivative_name)
        '''
        df = df.sort_values(column_x)
        #df_data = df_data.sort_values(column_x)

        xdata = np.array(df[column_x])
        ydata = np.array(df[column_y])
        #fddata = np.array(df[first_derivative_name])

        #weights
        #weight_array = np.full(len(xdata), 1)
   #     for value in weights:
     #       weight_array = MeltingCurveAnalysis._set_weights(xdata, ydata, fddata, weight_array, weights_dict_new=value)

        #Fit using the lmfit-toolbox.
        mod, pars = model_pars
        pars.update(p0)
        out = mod.fit(ydata, pars, T=xdata)#, weights=weight_array)

     #   y_diff = out.residual
    #    x_diff = np.diff(xdata)
     #   x_diff = np.append(x_diff, 0)
     #   print(len(x_diff))o
      #  dd = np.diff(ydata)/x_diffd

        #dx = np.sqrt(y_diff**2 + (fddata * x_diff)**2)
        #wi = 1/(dx*ydata)**2

        #
     #   pars.update(out.params)
      #  out = mod.fit(ydata, pars, T=xdata, weights=wi)
        
        if plot is True:
            out.plot(xlabel=column_x, ylabel=column_y, yerr=np.zeros(len(xdata)),
                     show_init=False, data_kws={
                              'markersize': 4})
            plt.title(f'{filename}, {measurement_number}')

        return_dict = {'Fit': out}
        self.data[filename][f'Measurement_{measurement_number}']['direct_melt_fit'] = return_dict
        
        return return_dict
    
    def get_DS(DH, Tm):
        return DH / Tm

    def get_DDS(DH, Tm, DDH, DTm):
        dH = abs(1 / Tm * DDH)
        dtm = abs(DH / Tm**2 * DTm)
        return dH + dtm

    def get_direct_melt_result(self, filename, measurement_number):
        fit_out = self.data[filename][f'Measurement_{measurement_number}']['direct_melt_fit']['Fit']
        name = fit_out.model.name
        ps = {}
        
        
        if name == 'Model(equation_molecularity_1, prefix=\'e1_\')':
            Tm = fit_out.params['e1_Tm'].value
            DTm = fit_out.params['e1_Tm'].stderr
            DH = fit_out.params['e1_DH'].value
            DDH = fit_out.params['e1_DH'].stderr

            ps['Tm_C'] = Tm
            ps['DTm'] = DTm
            ps['DH_J/mol'] = DH
            ps['DDH'] = DDH
            ps['m1'] = fit_out.params['e1_m1'].value
            ps['Dm1'] = fit_out.params['e1_m1'].stderr
            ps['n1'] = fit_out.params['e1_n1'].value
            ps['Dn1'] = fit_out.params['e1_n1'].stderr
            ps['m2'] = fit_out.params['e1_m2'].value
            ps['Dm2'] = fit_out.params['e1_m2'].stderr
            ps['n2'] = fit_out.params['e1_n2'].value
            ps['Dn2'] = fit_out.params['e1_n2'].stderr
            
            ps['DS_J/mol/K'] = MeltingCurveAnalysis.get_DS(DH, Tm)
            ps['DDS'] = MeltingCurveAnalysis.get_DDS(DH, Tm, DDH, DTm)
            
                        
            x = np.linspace(290, 370, 100)
            
            fit_out.plot(data_kws={
                              'markersize': 4})
            plt.plot(x, fit_out.params['e1_m1']*x+fit_out.params['e1_n1'], label = '1')
            plt.plot(x, fit_out.params['e1_m2']*x+fit_out.params['e1_n2'], label = '2')
            plt.legend()
            
        if name == 'Model(equation_molecularity_2, prefix=\'e2_\')':
            Tm = fit_out.params['e2_Tm'].value
            DTm = fit_out.params['e2_Tm'].stderr
            DH = fit_out.params['e2_DH'].value
            DDH = fit_out.params['e2_DH'].stderr

            ps['Tm_C'] = Tm
            ps['DTm'] = DTm
            ps['DH_J/mol'] = DH
            ps['DDH'] = DDH
            ps['m1'] = fit_out.params['e2_m1'].value
            ps['Dm1'] = fit_out.params['e2_m1'].stderr
            ps['n1'] = fit_out.params['e2_n1'].value
            ps['Dn1'] = fit_out.params['e2_n1'].stderr
            ps['m2'] = fit_out.params['e2_m2'].value
            ps['Dm2'] = fit_out.params['e2_m2'].stderr
            ps['n2'] = fit_out.params['e2_n2'].value
            ps['Dn2'] = fit_out.params['e2_n2'].stderr
            
            ps['DS_J/mol/K'] = MeltingCurveAnalysis.get_DS(DH, Tm)
            ps['DDS'] = MeltingCurveAnalysis.get_DDS(DH, Tm, DDH, DTm)
            
                        
            x = np.linspace(290, 370, 100)
            
            fit_out.plot(data_kws={
                              'markersize': 4})
            plt.plot(x, fit_out.params['e2_m1']*x+fit_out.params['e2_n1'], label = '1')
            plt.plot(x, fit_out.params['e2_m2']*x+fit_out.params['e2_n2'], label = '2')
            plt.legend()
        
        if name == '(Model(equation_molecularity_2, prefix=\'e2_\') + Model(equation_molecularity_1, prefix=\'e1_\'))':# | '(Model(equation_molecularity_2, prefix=\'e2\') + Model(equation_molecularity_2, prefix=\'e1_\'))':
            #Bimolecular -> lower Temp -> 1
            Tm1 = fit_out.params['e2_Tm'].value
            DTm1 = fit_out.params['e2_Tm'].stderr
            DH1 = fit_out.params['e2_DH'].value
            DDH1 = fit_out.params['e2_DH'].stderr
            DS1 = MeltingCurveAnalysis.get_DS(DH1, Tm1)
          #  DDS1 = MeltingCurveAnalysis.get_DDS(DH1, Tm1, DDH1, DTm1)
            
            #Monomolecular -> higher Temp -> 2
            Tm2 = fit_out.params['e1_Tm'].value
            DTm2 = fit_out.params['e1_Tm'].stderr
            DH2 = fit_out.params['e1_DH'].value
            DDH2 = fit_out.params['e1_DH'].stderr
            DS2 = MeltingCurveAnalysis.get_DS(DH2, Tm2)
          #  DDS2 = MeltingCurveAnalysis.get_DDS(DH2, Tm2, DDH2, DTm2)


            #Paste into dict
            ps['Tm_K_1'] = Tm1
            ps['DTm_1'] = DTm1
            ps['DH_J/mol_1'] = DH1
            ps['DDH_1'] = DDH1
            ps['DS_J/mol/K_1'] = DS1
    #        ps['DDS_1'] = DDS1
            
            ps['Tm_K_2'] = Tm2
            ps['DTm_2'] = DTm2
            ps['DH_J/mol_2'] = DH2
            ps['DDH_2'] = DDH2
            ps['DS_J/mol/K_2'] = DS2
     #       ps['DDS_2'] = DDS2
            
            #Lowest Baseline (erverything unfolded)
            ps['m1'] = fit_out.params['e2_m1'].value
            ps['Dm1'] = fit_out.params['e2_m1'].stderr
            ps['n1'] = fit_out.params['e2_n1'].value
            ps['Dn1'] = fit_out.params['e2_n1'].stderr
            
            #Line between (still kissing)
            ps['m2'] = fit_out.params['e2_m2'].value
            ps['Dm2'] = fit_out.params['e2_m2'].stderr
            ps['n2'] = fit_out.params['e2_n2'].value
            ps['Dn2'] = fit_out.params['e2_n2'].stderr
            
            #Upper Baseline (tertiary contact)
            ps['m3'] = fit_out.params['e1_m2'].value
            ps['Dm3'] = fit_out.params['e1_m2'].stderr
            ps['n3'] = fit_out.params['e1_n2'].value
            ps['Dn3'] = fit_out.params['e1_n2'].stderr
            
            
            x = np.linspace(290, 370, 100)
            
            fit_out.plot(data_kws={
                              'markersize': 4})
            plt.plot(x, fit_out.params['e2_m1']*x+fit_out.params['e2_n1'], label = '1')
            plt.plot(x, fit_out.params['e2_m2']*x+fit_out.params['e2_n2'], label = '2')
            plt.plot(x, fit_out.params['e1_m2']*x+fit_out.params['e1_n2'], label = '3')
            plt.legend()
            
        self.data[filename][f'Measurement_{measurement_number}']['direct_melt_fit']['Result'] = ps
        return ps
    
    #### Baselines
    def linear_baseline_fit(self,
                                    filename,
                                    measurement_number,
                                    deribounds=(0.0005, 0.0005),
                                    column_x: str = 'Temperature (K)',
                                    column_y: str = 'Absorbance',
                                    filter_bounds: list = [{}],
                                    first_derivative_name=None,
                                    second_derivative_name=None,
                                    plot: bool = True):

        df_data_raw = self.data[filename][f'Measurement_{measurement_number}']['data']

        #Add Collumn for T in K.
        df_data_raw['Temperature (K)'] = df_data_raw['Temperature (°C)'] + 273.15 
                

        df_data_raw, first_derivative_name = MeltingCurveAnalysis._calculate_derivative(
            df_data_raw.sort_values(column_x),
            column_x=column_x,
            column_y=column_y,
            first_derivative_name=first_derivative_name)
        

        #Filter dataframe to user input
        df = MeltingCurveAnalysis.filter_multiple_times(df_data_raw,
                                                        filter_bounds)

        # Convert DataFrame to numpy arrays
        xdata_0 = np.array(self.data[filename][f'Measurement_{measurement_number}']['data'].sort_values(column_x).get(column_x))
        ydata_0 = np.array(self.data[filename][f'Measurement_{measurement_number}']['data'].sort_values(column_x).get(column_y))
        #plt.plot(xdata_0, ydata_0)



        maxima = self.data[filename][f'Measurement_{measurement_number}'].get('derivative_fit', None).get('Maxima', None)
        if maxima == None:
            print(filename, measurement_number)
            self.data[filename][f'Measurement_{measurement_number}']['baselines'] = 'Fit failed'

        else:
            # & (df['Temperature (C)'] >= lower_baseline_bounds[0]) & (df['Temperature (C)'] <= lower_baseline_bounds[1])), :]
            lower_baseline = df.loc[((abs(df[first_derivative_name]) < deribounds[0])
                                     & (df[column_x] < maxima[f'{column_x}_max'])), :]
            # & (df['Temperature (C)'] >= upper_baseline_bounds[0]) & (df['Temperature (C)'] <= upper_baseline_bounds[1])), :]
            upper_baseline = df.loc[((abs(df[first_derivative_name]) < deribounds[1])
                                     & (df[column_x] > maxima[f'{column_x}_max'])), :]

            data = {}
            med_data = {'params': None, 'best_fit': None, 'Intersection': None}
            ps = {}

            for index, value in enumerate([lower_baseline, upper_baseline]):
                if index == 0:

                    xdata = lower_baseline[column_x]
                    ydata = lower_baseline[column_y]
                    if len(value) < 2:
                        xdata = df.head(2)[column_x]
                        ydata = df.head(2)[column_y]
                        print(f'Curve Index {measurement_number} using lower tangent')

                else:

                    xdata = upper_baseline[column_x]
                    ydata = upper_baseline[column_y]
                    if len(value) < 2:
                        xdata = df.tail(2)[column_x]
                        ydata = df.tail(2)[column_y]
                        print(f'Curve Index {measurement_number} using upper tangent')

                mod = lmfit.models.LinearModel()
                pars = mod.guess(ydata, x=xdata)
                out = mod.fit(ydata, pars, x=xdata)
                data[f'baseline_{index}'] = out

                if index == 0:
                    ps['m1'] = out.params['slope'].value
                    ps['Dm1'] = out.params['slope'].stderr
                    ps['n1'] = out.params['intercept'].value
                    ps['Dn1'] = out.params['intercept'].stderr
                elif index == 1:
                    ps['m2'] = out.params['slope'].value
                    ps['Dm2'] = out.params['slope'].stderr
                    ps['n2'] = out.params['intercept'].value
                    ps['Dn2'] = out.params['intercept'].stderr

            '''
            pars = lmfit.Parameters()
            pars.add('slope', value=(data['baseline_0'].values.get('slope') + data['baseline_1'].values.get('slope')) / 2)
            pars.add('intercept', value=(data['baseline_0'].values.get(
                'intercept') + data['baseline_1'].values.get('intercept')) / 2)
            med_data['params'] = pars
            median = mod.eval(params=pars, x=xdata_0)
            med_data['best_fit'] = median
            ps['m_m'] = pars['slope'].value
            ps['n_m'] = pars['intercept'].value
            ps['sigma_m_m'] = 1 / 2 * (abs(ps['sigma_m_o']) + abs(ps['sigma_m_u']))
            ps['sigma_n_m'] = 1 / 2 * (abs(ps['sigma_n_o']) + abs(ps['sigma_n_u']))

            # Create a finer grid of x values for interpolation
            fine_x = np.linspace(maxima[f'{column_x}_max'] - 10, maxima[f'{column_x}_max'] + 10, 500)
            xdata_median = fine_x
            xdata_sigmoid = fine_x
            ydata_sigmoid = np.interp(x=fine_x, xp=xdata_0, fp=ydata_0)
         #   print(pars)

            ydata_median = mod.eval(params=pars, x=fine_x)
            pars.add('slope', value=ps['m_m'] + ps['sigma_m_m'])
            pars.add('intercept', value=ps['n_m'] + ps['sigma_n_m'])
          #  print(pars)
            ydata_median_plus = mod.eval(params=pars, x=fine_x)
            pars.add('slope', value=ps['m_m'] - ps['sigma_m_m'])
            pars.add('intercept', value=ps['n_m'] - ps['sigma_n_m'])
        #    print(pars)
            ydata_median_minus = mod.eval(params=pars, x=fine_x)

            t_temp = np.zeros([3, 2])

            idx = 0
            for ydata in [ydata_median, ydata_median_plus, ydata_median_minus]:
                # Initialize variables to store the closest intersection point
                closest_intersection = None
                closest_distance = float('inf')

                # Iterate through points of the linear function
                for x_median, y_median in zip(xdata_median, ydata):
                    # Find the closest point on the sigmoid function
                    index_sigmoid = np.argmin(np.abs(xdata_sigmoid - x_median))
                    x_sigmoid = xdata_sigmoid[index_sigmoid]
                    y_sigmoid = ydata_sigmoid[index_sigmoid]

                    # Calculate the Euclidean distance between the two points
                    distance = np.sqrt((x_median - x_sigmoid)**2 + (y_median - y_sigmoid)**2)

                    # Update the closest intersection point if a closer one is found
                    if distance < closest_distance:
                        closest_intersection = (x_median, y_median)
                        closest_distance = distance
                t_temp[idx, 0] = closest_intersection[0]
                t_temp[idx, 1] = closest_intersection[1]
              #  print(idx)
                idx += 1
               # print(t_temp)

       #     print('\n', t_temp)
            delta_T = max(abs(t_temp[:, 0] - t_temp[0, 0]))
       #     print('\n', delta_T)
            delta_Ed = max(abs(t_temp[:, 1] - t_temp[0, 1]))
      #      print(delta_Ed, '\n')
            ps['Tm_C'] = t_temp[0, 0]
            ps['DTm'] = delta_T
            ps['Ed'] = t_temp[0, 1]
            ps['DEd'] = delta_Ed
            '''
            data['result'] = ps
            '''
        #    print("Closest intersection point with higher x accuracy:", closest_intersection)
         #   med_data['Intersection'] = closest_intersection
            data['median'] = med_data
            #plt.plot
            '''
            if plot is True:
                #data[f'baseline_{0}'].plot(xlabel=column_x, ylabel=column_y, data_kws={
                #   'markersize': 4, 'marker': 'x'})
                #plt.plot(df[column_x], df[column_y])
               # data[f'baseline_{1}'].plot(xlabel=column_x, ylabel=column_y, data_kws={
                #    'markersize': 4, 'marker': 'x'})
                f, ax = plt.subplots(nrows=1, ncols=1)
                ax.plot(df[column_x], df[column_y], 'x', label='data')
                ax.plot(xdata_0, data[f'baseline_{1}'].eval(x=xdata_0), label='upper')
                ax.plot(xdata_0, data[f'baseline_{0}'].eval(x=xdata_0), label='lower')
              #  data['median'].plot(xlabel=column_x, ylabel=column_y, data_kws={
               #     'markersize': 4, 'marker': 'x'})
            #    ax.plot(closest_intersection[0], closest_intersection[1], 'x', label='Tm')
             #   ax.plot(xdata_0, mod.eval(params=pars, x=xdata_0), label='median')
              #  ax.plot(np.arange(50, 65, 0.1), e, label = 'dd')
                ax.legend()
                #ax.plot(xdata_sigmoid, ydata_sigmoid)
                ax.set_title(f'{filename}, {measurement_number}')
            
            self.data[filename][f'Measurement_{measurement_number}']['baselines'] = data
            
    def _get_baseline_fit_out(self, file_name, measurement_number, baseline: str = 'lower'):
        if baseline == 'lower':
            i = 0
        else:
            i = 1
        return self.data.get(file_name).get(f'Measurement_{measurement_number}').get('baselines').get(f'baseline_{i}')
    
    def _baseline_pars_melt_converter(self, filename, measurement_number, molecularity = 2, i=0):
        i = 0
        data = self._get_baseline_fit_out(filename, measurement_number, baseline='lower').values
      #  print(data)
        if i == 0:
            j = 1
        else:
            j = 2
        temp_pars = lmfit.Parameters()
        temp_pars.add(f'e{molecularity}_m{j}', value=data.get('slope'), vary=False)
        temp_pars.add(f'e{molecularity}_n{j}', value=data.get('intercept'), vary=False)
        i = 1
        data = self._get_baseline_fit_out(filename, measurement_number, baseline='upper').values
       # print(data)
        if i == 0:
            j = 1
        else:
            j = 2
        temp_pars.add(f'e{molecularity}_m{j}', value=data.get('slope'), vary=False)
        temp_pars.add(f'e{molecularity}_n{j}', value=data.get('intercept'), vary=False)
        return temp_pars
    
    def baseline_melt_converter_e2_e1(self, filename, measurement_number):
        data = self._get_baseline_fit_out(filename, measurement_number, baseline='lower').values
        temp_pars = lmfit.Parameters()
        temp_pars.add(f'e2_m1', value=data.get('slope'), vary=False)
        temp_pars.add(f'e2_n1', value=data.get('intercept'), vary=False)
        data = self._get_baseline_fit_out(filename, measurement_number, baseline='upper').values
        temp_pars.add(f'e1_m2', value=data.get('slope'), vary=False)
        temp_pars.add(f'e1_n2', value=data.get('intercept'), vary=False)
        return temp_pars
            
    #### Derivative Fit
    def _calculate_derivative(df: pd.DataFrame,
                              column_x: str = 'Temperature (K)',
                              column_y: str = 'Absorbance',
                              first_derivative_name=None) -> pd.DataFrame:
        first_derivative_name = f'd{column_y} / d{column_x}' if first_derivative_name is None else first_derivative_name
        df.loc[:, first_derivative_name] = np.gradient(df[column_y], df[column_x])
        return df, first_derivative_name

    def _maxima(xdata, ydata, height=0, width=5):
        temp = ydata
        peaks, prop = signal.find_peaks(abs(ydata), height=height, width=width)
        x_peaks = xdata[peaks]
        #y_peaks = prop['peak_heights']
        y_peaks = temp[peaks]
        out = pd.DataFrame({'x_max': x_peaks, 'y_max': y_peaks})
        return out

   
    def _create_model_sum_of_1_to_10_peaks(xdata, ydata, num_peaks: int, model=lmfit.models.SkewedVoigtModel):
        if (num_peaks < 1) | (num_peaks > 10):
            print('Please use a peak count between 1 and 10.')
            return None
        else:
            x_min = min(xdata)
            x_max = max(xdata)
            x_diff = x_max - x_min
            x_step = x_diff / (num_peaks + 1)
            y_max = max(ydata)

            peak1 = model(prefix='p1_')
            if num_peaks == 1:
                pars = peak1.guess(ydata, x=xdata)
            else:
                pars = peak1.make_params(amplitude=dict(value=y_max),
                                         center=dict(value=x_min + 1 * x_step, min=x_min, max=x_max),
                                         sigma=dict(value=2),
                                         gamma=dict(value=0, vary=True))
            if num_peaks >= 2:
                peak2 = model(prefix='p2_')
                pars.update(peak2.make_params(amplitude=dict(value=y_max),
                                              center=dict(value=x_min + 2 * x_step, min=pars['p1_center'], max=x_max),
                                              sigma=dict(value=2),
                                              gamma=dict(value=0, vary=True)))
                if num_peaks >= 3:
                    peak3 = model(prefix='p3_')
                    pars.update(peak3.make_params(amplitude=dict(value=y_max),
                                                  center=dict(value=x_min + 3 * x_step, min=pars['p2_center'],
                                                              max=x_max),
                                                  sigma=dict(value=2),
                                                  gamma=dict(value=0, vary=True)))
                    if num_peaks >= 4:
                        peak4 = model(prefix='p4_')
                        pars.update(peak4.make_params(amplitude=dict(value=y_max),
                                                      center=dict(value=x_min + 4 * x_step, min=pars['p3_center'],
                                                                  max=x_max),
                                                      sigma=dict(value=2),
                                                      gamma=dict(value=0, vary=True)))
                        if num_peaks >= 5:
                            peak5 = model(prefix='p5_')
                            pars.update(peak5.make_params(amplitude=dict(value=y_max),
                                                          center=dict(value=x_min + 5 * x_step, min=pars['p4_center'],
                                                                      max=x_max),
                                                          sigma=dict(value=2),
                                                          gamma=dict(value=0, vary=True)))
                            if num_peaks >= 6:
                                peak6 = model(prefix='p6_')
                                pars.update(peak6.make_params(amplitude=dict(value=y_max),
                                                              center=dict(value=x_min + 6 * x_step,
                                                                          min=pars['p5_center'],
                                                                          max=x_max),
                                                              sigma=dict(value=2),
                                                              gamma=dict(value=0, vary=True)))
                                if num_peaks >= 7:
                                    peak7 = model(prefix='p7_')
                                    pars.update(peak7.make_params(amplitude=dict(value=y_max),
                                                                  center=dict(value=x_min + 7 * x_step,
                                                                              min=pars['p6_center'],
                                                                              max=x_max),
                                                                  sigma=dict(value=2),
                                                                  gamma=dict(value=0, vary=True)))
                                    if num_peaks >= 8:
                                        peak8 = model(prefix='p8_')
                                        pars.update(peak8.make_params(amplitude=dict(value=y_max),
                                                                      center=dict(value=x_min + 8 * x_step,
                                                                                  min=pars['p7_center'],
                                                                                  max=x_max),
                                                                      sigma=dict(value=2),
                                                                      gamma=dict(value=0, vary=True)))
                                        if num_peaks >= 9:
                                            peak9 = model(prefix='p9_')
                                            pars.update(peak9.make_params(amplitude=dict(value=y_max),
                                                                          center=dict(value=x_min + 9 * x_step,
                                                                                      min=pars['p8_center'],
                                                                                      max=x_max),
                                                                          sigma=dict(value=2),
                                                                          gamma=dict(value=0, vary=True)))
                                            if num_peaks == 10:
                                                peak10 = model(prefix='p10_')
                                                pars.update(peak10.make_params(amplitude=dict(value=y_max),
                                                                               center=dict(value=x_min + 10 * x_step,
                                                                                           min=pars['p9_center'],
                                                                                           max=x_max),
                                                                               sigma=dict(value=2),
                                                                               gamma=dict(value=0, vary=True)))
                                                return peak1 + peak2 + peak3 + peak4 + peak5 + peak6 + peak7 + peak8 + peak9 + peak10, pars
                                            else:
                                                return peak1 + peak2 + peak3 + peak4 + peak5 + peak6 + peak7 + peak8 + peak9, pars
                                        else:
                                            return peak1 + peak2 + peak3 + peak4 + peak5 + peak6 + peak7 + peak8, pars
                                    else:
                                        return peak1 + peak2 + peak3 + peak4 + peak5 + peak6 + peak7, pars
                                else:
                                    return peak1 + peak2 + peak3 + peak4 + peak5 + peak6, pars
                            else:
                                return peak1 + peak2 + peak3 + peak4 + peak5, pars
                        else:
                            return peak1 + peak2 + peak3 + peak4, pars
                    else:
                        return peak1 + peak2 + peak3, pars
                else:
                    return peak1 + peak2, pars
            else:
                return peak1, pars
    
    def derivative_fit(self,
                                     filename,
                                     measurement_number,
                                     num_peaks: int = 1,
                                     column_x: str = 'Temperature (K)',
                                     column_y: str = 'Absorbance',
                                     model: lmfit.Model = lmfit.models.GaussianModel,
                                     filter_bounds: list = [{'column_x': 'Temperature (°C)', 'column_y': 'Absorbance', 'x_min': -np.inf, 'y_min':
                                                             - np.inf, 'x_max': np.inf, 'y_max': np.inf, 'keep_inside': True}],
                                     weights: list = [{'first_derivative_border_x_min': -np.inf, 'first_derivative_border_x_max': np.inf, 'first_derivative_border_y_min': -np.inf, 'first_derivative_border_y_max': np.inf, 'data_x_min': -np.inf, 'data_y_min': -
                                                       np.inf, 'data_x_max': np.inf, 'data_y_max': np.inf, 'deri_weight': 1, 'data_weight': 1}],

                                     first_derivative_name=None,
                                     p0: lmfit.Parameters = lmfit.Parameters(),
                                     plot: bool = True):
        
        
        df_data_raw = self.data[filename][f'Measurement_{measurement_number}']['data']

        #Add Collumn for T in K.
        df_data_raw['Temperature (K)'] = df_data_raw['Temperature (°C)'] + 273.15 
                

        df_data_raw, first_derivative_name = MeltingCurveAnalysis._calculate_derivative(
            df_data_raw.sort_values(column_x),
            column_x=column_x,
            column_y=column_y,
            first_derivative_name=first_derivative_name)
        

        #Filter dataframe to user input
        df = MeltingCurveAnalysis.filter_multiple_times(df_data_raw,
                                                        filter_bounds)
        
        df = df.sort_values(column_x)
        df_data = df_data_raw.sort_values(column_x)
        # Convert DataFrame to numpy arrays
        xdata = np.array(df[column_x])
        ydata = np.array(df[first_derivative_name])
        fddata = np.array(df[first_derivative_name])

        #weights
    #    weight_array = np.full(len(xdata), 1)
    #    for value in weights:
     #       weight_array = MeltingCurveAnalysis._set_weights(xdata, ydata, fddata, weight_array, weights_dict_new=value)

        # fit
        mod, pars = MeltingCurveAnalysis._create_model_sum_of_1_to_10_peaks(xdata, ydata, num_peaks, model=model)
        pars.update(p0)
        out = mod.fit(ydata, pars, x=xdata)#, weights=weight_array)
        x = np.arange(min(df_data[column_x]), max(df_data[column_x]), 0.01)
        fit = out.eval(x=x)
        fit_dely = out.eval_uncertainty(x=x)
        fit_pl = fit + fit_dely
        fit_mi = fit - fit_dely

        max_peaks = MeltingCurveAnalysis._maxima(x, fit)
    #    print(max_peaks)
        max_peaks_pl = MeltingCurveAnalysis._maxima(x, fit_pl)['x_max']
   #     print(max_peaks_pl)
        max_peaks_mi = MeltingCurveAnalysis._maxima(x, fit_mi)['x_max']
    #    print(abs(max_peaks_pl - max_peaks['x_max']))
        if len(max_peaks_mi) != len(max_peaks_pl):
            dx = np.max([abs(max_peaks_pl[0] - max_peaks['x_max'][0]), abs(max_peaks['x_max'][0] - max_peaks_mi[0])])
        else:
            dx = np.max([abs(max_peaks_pl - max_peaks['x_max']), abs(max_peaks['x_max'] - max_peaks_mi)])
    #    print(dx)

        '''
        norm_grad = MeltingCurveAnalysis.normalize(fit)
        idx = np.argmax(norm_grad)
        difference_array = np.absolute(norm_grad[idx:] - 0.5)
        index = difference_array.argmin()
        T_34 = (x[index + idx])

        norm_grad_pl = MeltingCurveAnalysis.normalize(fit_pl)
        idx_pl = np.argmax(norm_grad_pl)
        difference_array_pl = np.absolute(norm_grad[idx_pl:] - 0.5)
        index_pl = difference_array.argmin()
        T_34_pl = (x[index_pl + idx_pl])

        norm_grad_mi = MeltingCurveAnalysis.normalize(fit_mi)
        idx_mi = np.argmax(norm_grad_mi)
        difference_array_mi = np.absolute(norm_grad[idx_mi:] - 0.5)
        index_mi = difference_array.argmin()
        T_34_mi = (x[index_mi + idx_mi])

        dT34_p = max([abs(T_34_pl - T_34), abs(T_34 - T_34_mi)])
      #  print(dT34_p)

        DH = MeltingCurveAnalysis._get_DH_from_halfwidth(14583, fit[idx] + 273.15, T_34 + 273.15)
        DDH = MeltingCurveAnalysis._get_DDH_Halfwidth(14583, fit[idx] + 273.15, T_34 + 273.15, dx, dT34_p)

        #df_data[f'{first_derivative_name}_fit_normalized_to_max'] = norm_grad
        '''
        ps = {}
        ps['Tm_C'] = max_peaks['x_max'][0]
        ps['dEd/dT'] = max_peaks['y_max'][0]
       # ps['T_34_C'] = T_34

      #  ps['DT_34'] = dT34_p
        ps['DTm'] = dx
      #  ps['DH_hw_J/mol'] = DH
      #  ps['DDH_hw'] = DDH
      #  ps['DS_J/mol/K'] = MeltingCurveAnalysis._get_DS(DH, max_peaks['x_max'][0])
      #  ps['DDS'] = MeltingCurveAnalysis._get_DDS(DH, max_peaks['x_max'][0], DDH, dx)

       # plt.plot
        if plot is True:
            out.plot(xlabel=column_x, ylabel=first_derivative_name, data_kws={
                     'markersize': 4}, yerr=np.zeros(len(xdata)))
            plt.plot(max_peaks['x_max'], max_peaks['y_max'], marker='o', ms=4, label='Tm')
            plt.title(f'{filename}, {measurement_number}')
            plt.legend()

        max_peaks = max_peaks.rename(columns={'x_max': f'{column_x}_max', 'y_max': f'{first_derivative_name}_max'})
        if max_peaks.empty == True:
            return_dict = {'Fit': out, 'Error': 'Fit failed!'}
            print(filename, measurement_number)
        else:
            temp = max_peaks.to_dict('records')[0]
       #     temp.update({'T_34 in °C': T_34, 'DH in J/mol/K': DH})
            return_dict = {'Fit': out, 'Maxima': temp, 'result': ps}
        self.data[filename][f'Measurement_{measurement_number}']['data'] = df_data
        self.data[filename][f'Measurement_{measurement_number}']['derivative_fit'] = return_dict

       # return df_data, return_dict



    def _add_results_to_meta(self):
        for file_name in self.data.keys():
            for key, value in self.data[file_name].items():

                deri_fit_o = value.get('derivative_fit', None)
                if deri_fit_o is not None:
                    deri_fit = deri_fit_o.get('result', None)
                  #  deri_fit.update(deri_fit_o.get('T_34 in °C', None))
                  #  deri_fit.update(deri_fit_o.get('DH in J/mol/K', None))
                    if deri_fit is None:
                        deri_fit = {}
                else:

                    deri_fit = {}
                deri_fit = {'deri_fit_' + str(key): val for key, val in deri_fit.items()}


                deri_max = value.get('derivative_max', None)
                if deri_max is None:
                    deri_max = {}
                deri_max = {'deri_max_' + str(key): val for key, val in deri_max.items()}


                deri_melt = value.get('deri_melt_fit', None)
                if deri_melt is not None:
                    deri_melt = deri_melt.get('result', None)

                else:
                    deri_melt = {}
                deri_melt = {'deri_melt_' + str(key): val for key, val in deri_melt.items()}


                melt = value.get('direct_melt_fit', None) 
                if melt is not None:
                    melt = melt.get('Result', None)
                else:
                    melt = {}
                melt = {'direct_melt_' + str(key): val for key, val in melt.items()}


                median = value.get('baselines', None)
                if median != 'Fit failed':
                    if median is not None:
                        median = median.get('result', None)  # .get('Intersection', None)
                     #   median = {'x': median[0], 'y': median[1]}
                       # print(median)
                    else:
                        median = {}
                else:
                    median = {}
                median = {'median_' + str(key): val for key, val in median.items()}


             #   print(value.get('meta', 'eee'))
                value['meta'].update(deri_fit)
                value['meta'].update(deri_max)
                value['meta'].update(deri_melt)
                value['meta'].update(melt)
                value['meta'].update(median)





















































'''
    class filter_bounds:
        def __init__(self):
            self.column_x: str = 'Temperature (°C)',
            self.column_y: str = 'Absorbance',
            self.x_min: float = -np.inf,
            self.y_min: float = -np.inf,
            self.x_max: float = np.inf,
            self.y_max: float = np.inf,
        
    

    @staticmethod
    def _folded_fraction_formula(y: list, baseline_lower: list, baseline_upper: list):
        return (y - baseline_upper) / (baseline_lower - baseline_upper)

    @staticmethod
    def _k_a__from_folded_fraction_intramolecular(folded_fractions: list):
        return folded_fractions / (1 - folded_fractions)

    @staticmethod
    def linear_equation(x, slope, intercept):
        return slope * x + intercept

    #@staticmethod
    #def _calculate_folded_fraction():

    def _folded_fraction(self, filename, measurement_number, column_x='Temperature (°C)', column_y='Absorbance', baseline_parameters: dict = {'upper': {'slope': 0, 'intercept': 0}, 'lower': {'slope': 0, 'intercept': 0}}, baseline_median=None, filter_bounds: list = [{'column_x': 'Temperature (°C)', 'column_y': 'Absorbance', 'x_min': -np.inf, 'y_min':
                            - np.inf, 'x_max': np.inf, 'y_max': np.inf, 'keep_inside': True}],yb = tuple([-5, 5]), b2 = (-5, 20)
   ):
        df_data = self.data[filename][f'Measurement_{measurement_number}']['data']
        for value in filter_bounds:
            df = MeltingCurveAnalysis.filter_by_values(df_data,
                                                        value)
            
        xdata = df[column_x] + 273.15
        ydata = df[column_y]
        
        if baseline_median != None:
            #print('ee')
            baseline_0_folded = baseline_median['baseline_1'].eval(x=xdata-273.15)
            baseline_100_folded = baseline_median['baseline_0'].eval(x=xdata-273.15)
            med_res = baseline_median['result']
            Tm = med_res['Tm_C']
            Tm_K = Tm + 273.15
           # print(Tm_K)
            Ed = med_res['Ed']
            mu = med_res['m_u']
            nu = med_res['n_u']
            mo = med_res['m_o']
            no = med_res['n_o']
            
            DT = med_res['DTm']
            DEd =  med_res['DEd']
            Dmu = med_res['sigma_m_u']
            Dnu = med_res['sigma_n_u']
            
            Dmo = med_res['sigma_m_o']
            Dno = med_res['sigma_n_o']
            
            
            xb = tuple([1/(Tm_K + b2[0]), 1/(Tm_K + b2[1])])
            print(xb[1], xb[0])
            tg2 = 10**10
            
            ff = MeltingCurveAnalysis._folded_fraction_formula(ydata, baseline_100_folded, baseline_0_folded)
            Dff = MeltingCurveAnalysis._get_Dtheta(T=xdata, Ed =ydata, Tm=Tm, mu=mu, nu=nu, mo=mo, no=no, DEd=DEd, DT = DT, Dmu =Dmu, Dnu=Dnu, Dmo=Dmo, Dno=Dno)
            K_a = MeltingCurveAnalysis._k_a__from_folded_fraction_intramolecular(ff)
            DlnKa = MeltingCurveAnalysis._get_DlnKa(ff, Dff) 
            lnKa = np.log(K_a)
            data = pd.DataFrame({'lnK': lnKa, '1/T': 1/xdata, 'weight': DlnKa})
            data = data.dropna()
            data['g1'] = np.gradient(data['lnK'], data['1/T'])
            data['g2'] = np.gradient(data['g1'], data['1/T'])
            temp_e = data.loc[(data['1/T'] >= xb[1]) & (data['1/T'] <= xb[0]),:]# & (data['g2'] <= tg2) , :]
            temp_f = temp_e.loc[(data['lnK'] >= yb[0]) & (data['lnK'] <= yb[1]),:]# & (data['g2'] <= tg2) , :]
            temp = temp_f
            fig, ax = plt.subplots()
            ax.plot(temp['1/T'], temp['lnK'], label ='eee')
            mod = lmfit.models.LinearModel()
            pars = mod.guess(temp['lnK'], x=temp['1/T'], nan_policy='omit')
            
            out = mod.fit(temp['lnK'], x=temp['1/T'], weights = temp['weight'])
            out.plot()
            
            DH, DDH = MeltingCurveAnalysis._get_DH_DDH_VH(out.params)
            DS, DDS = MeltingCurveAnalysis._get_DS_DDS_VH(out.params)
            
            ps = {}
           # ps['Abs'] = t_temp[0, 1]
           # ps['sigma_Abs'] = delta_Ed
            ps['DH_J/mol'] = DH
            ps['DDH'] = DDH
            ps['DS_J/mol/K'] = DS
            ps['DDS'] = DDS
            
            self.data[filename][f'Measurement_{measurement_number}']['baselines']['vant_hoff'] = out
            self.data[filename][f'Measurement_{measurement_number}']['baselines']['result'].update(ps)
            
        else:
            baseline_0_folded = MeltingCurveAnalysis.linear_equation(xdata, *baseline_parameters.get('upper').values())
            baseline_100_folded = MeltingCurveAnalysis.linear_equation(xdata, *baseline_parameters.get('lower').values())

            ff = MeltingCurveAnalysis._folded_fraction_formula(ydata, baseline_100_folded, baseline_0_folded)
        
            K_a = MeltingCurveAnalysis._k_a__from_folded_fraction_intramolecular(ff)
            lnKa = np.log(K_a)

        self.data[filename][f'Measurement_{measurement_number}']['data']['Folded Fraction'] = ff
        self.data[filename][f'Measurement_{measurement_number}']['data']['K_a'] = K_a
        self.data[filename][f'Measurement_{measurement_number}']['data']['ln_K_a'] =lnKa
        
        f, ax = plt.subplots(nrows=1, ncols=1)
        ax.plot(xdata - 273.15, ff)
       # ax.plot(xdata - 273.15, baseline_0_folded)
      #  ax.plot(xdata - 273.15, baseline_100_folded)
        f, ax = plt.subplots(nrows=1, ncols=1)
        ax.plot(1 / xdata, np.log(K_a))
        
   # def _median_ff_coneverter(self, filename, measurement_number):
    #    data = self.data[filename][f'Measurement_{measurement_number}']['baselines']
     #   return {'upper': {'slope': m2, 'intercept': n2}, 'lower': {'slope': m1, 'intercept': n1}}

    def _melt_dict_ff_converter(self, filename, measurement_number):
        data = self._get_melt_fit_out(filename, measurement_number).values
        m1 = data.get('e1_m1')
        m2 = data.get('e1_m2')
        n1 = data.get('e1_n1')
        n2 = data.get('e1_n2')
        return {'upper': {'slope': m2, 'intercept': n2}, 'lower': {'slope': m1, 'intercept': n1}}
    
    def _folded_fraction(self, filename,
                         measurement_number, column_x='Temperature (°C)', column_y='Absorbance'):
        baselines = self.data[filename][f'Measurement_{measurement_number}']['baselines']
        T = self.data[filename][f'Measurement_{measurement_number}']['data'].get(column_x)
        A = self.data[filename][f'Measurement_{measurement_number}']['data'].get(column_y)
        baseline_1 = baselines['baseline_0'].eval(x=T)  # untere baseline 0 - 100 % folded
        baseline_0 = baselines['baseline_1'].eval(x=T)  # obere baseline 1 - 0 % folded
        ff = (baseline_0 - A) / (baseline_0 - baseline_1)
        ka = ff / (1 - ff)
        log_ka = np.log(ka)
        test = np.where((log_ka > - 3) & (log_ka < 3))
        xdata = 1 / (T[test[0]] + 273.15)
        ydata = log_ka[test[0]]
        mod = lmfit.models.LinearModel()
        pars = mod.guess(ydata, x=xdata)
        out = mod.fit(ydata, pars, x=xdata)
        out.plot()
        plt.plot(1 / (T + 273.15), np.log(ka), 'x')
        R = 8.314472
        Tm = 1 / (-out.values.get('intercept') / out.values.get('slope')) - 273.15
        DH = -out.values.get('slope') * R
        DS = -out.values.get('intercept') * R

        self.data[filename][f'Measurement_{measurement_number}']['folded_fraction'] = {}
        self.data[filename][f'Measurement_{measurement_number}']['folded_fraction']['Tm'] = Tm
        self.data[filename][f'Measurement_{measurement_number}']['folded_fraction']['DH'] = DH
        self.data[filename][f'Measurement_{measurement_number}']['folded_fraction']['DS'] = DS
        self.data[filename][f'Measurement_{measurement_number}']['folded_fraction']['Fit'] = out

        self.data[filename][f'Measurement_{measurement_number}']['data']['Folded Fractions'] = ff
        self.data[filename][f'Measurement_{measurement_number}']['data']['K_a'] = ka
        self.data[filename][f'Measurement_{measurement_number}']['data']['log(K_a)'] = log_ka
        self.data[filename][f'Measurement_{measurement_number}']['data']['1 / T (1 / K)'] = xdata
        f, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 4))
        ax.plot(T, ff)
    

    def _get_max_derivative(self,
                            filename,
                            measurement_number,
                            column_x: str = 'Temperature (°C)',
                            column_y: str = 'Absorbance',
                            filter_bounds: list = [{'column_x': 'Temperature (°C)', 'column_y': 'Absorbance', 'x_min': -np.inf, 'y_min':
                                                    - np.inf, 'x_max': np.inf, 'y_max': np.inf, 'keep_inside': True}],
                            first_derivative_name=None,
                            plot: bool = True):

        df_data, first_derivative_name = MeltingCurveAnalysis._calculate_derivative(
            self.data[filename][f'Measurement_{measurement_number}']['data'],
            column_x=column_x,
            column_y=column_y,
            first_derivative_name=first_derivative_name)

        for value in filter_bounds:
            df = MeltingCurveAnalysis.filter_by_values(df_data,
                                                        value)
        df = df.sort_values(column_x)
        df_data = df_data.sort_values(column_x)
        # Convert DataFrame to numpy arrays
        xdata = np.array(df[column_x])
        ydata = np.array(df[first_derivative_name])
        
        y_max = max(ydata)
        idx = np.argmax(ydata)
        x_max = xdata[np.argmax(ydata)]
        x_max_p = xdata[np.argmax(ydata) + 1]
        x_max_m = xdata[np.argmax(ydata) - 1]
        dx_p = max([abs(x_max - x_max_m), abs(x_max_p - x_max)]) / 2

        norm_grad = MeltingCurveAnalysis.normalize(ydata, lower=0)
      #  print(norm_grad)
        # calculate the difference array
        difference_array = np.absolute(norm_grad[idx:] - 0.5)
      #  print(difference_array)

        # find the index of minimum element from the array
        index = difference_array.argmin()
        T_34 = (xdata[index + idx])
        T_34_p = xdata[index + idx + 1]
        T_34_m = xdata[index + idx - 1]
        dT34_p = max([abs(T_34 - T_34_m), abs(T_34_p - T_34)]) / 2

        DH = MeltingCurveAnalysis._get_DH_from_halfwidth(14583, x_max + 273.15, T_34 + 273.15)
        DDH = MeltingCurveAnalysis._get_DDH_Halfwidth(14583, x_max + 273.15, T_34 + 273.15, dx_p, dT34_p)

      #  df_data[f'{first_derivative_name}_normalized_to_max'] = norm_grad

        ps = {}
        ps['Tm_C'] = x_max
        ps['dEd/dT'] = y_max
        ps['T_34_C'] = T_34
        ps['DT_34'] = dT34_p
        ps['DTm'] = dx_p
       # ps['Abs'] = t_temp[0, 1]
       # ps['sigma_Abs'] = delta_Ed
        ps['DH_hw_J/mol'] = DH
        ps['DDH_hw'] = DDH
        ps['DS_J/mol/K'] = MeltingCurveAnalysis._get_DS(DH, x_max)
        ps['DDS'] = MeltingCurveAnalysis._get_DDS(DH, x_max, DDH, dx_p)

    #    data['results'] = ps

        # plt.plot
        if plot is True:
            f, ax = plt.subplots(nrows=1, ncols=1)
            ax.plot(xdata, ydata)
            ax.plot(xdata, norm_grad)
            ax.plot([T_34, T_34], [0, 0.5])
            ax.plot(x_max, y_max, marker='o', ms=4, label='Tm')
         #   ax.title(f'{filename}, {measurement_number}')
            ax.legend()
       # return_dict = {'x_max': x_max, 'y_max': y_max, 'T_3/4 in °C': T_34, 'DH in J/mol/K': DH}
        self.data[filename][f'Measurement_{measurement_number}']['derivative_max'] = ps

    def _get_DH_from_halfwidth(alpha, T_max, T_34):
        return -alpha / (1 / T_max - 1 / T_34)

   


    def _set_weights(xdata, ydata, fddata, weight_array, weights_dict_new: dict = {}):
        weights_dict: dict = {'first_derivative_border_x_min': -np.inf, 'first_derivative_border_x_max': np.inf, 'first_derivative_border_y_min': -np.inf, 'first_derivative_border_y_max': np.inf, 'data_x_min': -np.inf, 'data_y_min': -
                              np.inf, 'data_x_max': np.inf, 'data_y_max': np.inf, 'deri_weight': 1, 'data_weight': 1}
        weights_dict.update(weights_dict_new)
        deri_idx = np.where((fddata >= weights_dict.get('first_derivative_border_y_min'))
                            & (fddata <= weights_dict.get('first_derivative_border_y_max')) & (xdata >= weights_dict.get('first_derivative_border_x_min')) & (xdata <= weights_dict.get('first_derivative_border_x_max')))
        data_idx = np.where((ydata >= weights_dict.get('data_y_min'))
                            & (ydata <= weights_dict.get('data_y_max')) & (xdata >= weights_dict.get('data_x_min')) & (xdata <= weights_dict.get('data_x_max')))
        np.put(weight_array, deri_idx, weights_dict.get('deri_weight'))
        np.put(weight_array, data_idx, weights_dict.get('data_weight'))
        return weight_array



    def grad_fct_1(T, DH, Tm, m1, n1, m2, n2):
        R = 8.31446261815324   # m^3* Pa * K^−1 * mol^−1
        x = DH / R * (1 / (T + 273.15) - 1 / (Tm + 273.15))
      #  y = DH / R * (1 / (T + 273.15))
        theta = 1 / (1 + np.exp(x))
        theta_deri = DH / R * np.exp(x) / \
            ((T + 273.15)**2 * (np.exp(x) + 1)**2)
        return (m1 - m2) * theta + ((m1 - m2) * (T + 273.15) + n1 - n2) * theta_deri + m2
    
    def r_fct_1(T, DH, Tm, m1, m2, n1, n2):
        R = 8.31446261815324   # m^3* Pa * K^−1 * mol^−1
        x = DH / R * (1 / (T + 273.15) - 1 / (Tm + 273.15))
        return (m1*(T + 273.15)*((DH+R*(T + 273.15))*np.exp(x)+R*(T + 273.15)) + n1*DH*np.exp(x) + m2*(T + 273.15)*np.exp(x)*(R*(T + 273.15)*(np.exp(x)+1)-DH)-n2*DH*np.exp(x))/(R*(T + 273.15)**2*(np.exp(x)+1)**2)

    def _create_deri_model_e1_b():
        e1_mod = lmfit.Model(MeltingCurveAnalysis.r_fct_1, prefix='e1_')
        pars = e1_mod.make_params(DH=dict(value=-150000, max=0),
                                  Tm=dict(value=60, min=1, max=90),
                                  m1=0.01,
                                  m2=0.01,
                                  n1=0.01,
                                  n2=0.01)
        return e1_mod, pars

    def _create_deri_model_e1():
        e1_mod = lmfit.Model(MeltingCurveAnalysis.grad_fct_1, prefix='e1_')
        pars = e1_mod.make_params(DH=dict(value=-1000000, max=0),
                                  Tm=dict(value=60, min=0, max=90),
                                  m1=dict(value = 0.005, min = -.1, max =.1),
                                  m2=dict(value = 0.005, min = -.1, max =.1),
                                  n1=dict(value = 0.005, min = -.1, max =.1),
                                  n2=dict(value = 0.005, min = -.1, max =.1))
        return e1_mod, pars

    def _get_filtered_melt_deri_fit(self,
                                    model_pars: lmfit.Model,  # = MeltingCurveAnalysis._create_model_e1(),
                                    filename,
                                    measurement_number,
                                    column_x: str = 'Temperature (°C)',
                                    column_y: str = 'Absorbance',
                                    first_derivative_name: str = None,
                                    filter_bounds: list = [{'column_x': 'Temperature (°C)', 'column_y': 'Absorbance', 'x_min': -np.inf, 'y_min':
                                                            - np.inf, 'x_max': np.inf, 'y_max': np.inf, 'keep_inside': True}],
                                    p0: lmfit.Parameters = lmfit.Parameters(),
                                    plot: bool = True):
        df_data = self.data[filename][f'Measurement_{measurement_number}']['data']
        for value in filter_bounds:
            df = MeltingCurveAnalysis.filter_by_values(df_data,
                                                        value)

        df_data, first_derivative_name = MeltingCurveAnalysis._calculate_derivative(
            self.data[filename][f'Measurement_{measurement_number}']['data'],
            column_x=column_x,
            column_y=column_y,
            first_derivative_name=first_derivative_name)

        xdata = np.array(df[column_x])
        ydata = np.array(df[column_y])
        fddata = np.array(df[first_derivative_name])

        # fit
        mod, pars = model_pars
        pars.update(p0)
        out = mod.fit(fddata, pars, T=xdata)
        
        Tm = out.params['e1_Tm'].value
        DTm = out.params['e1_Tm'].stderr
        DH = out.params['e1_DH'].value
        DDH = out.params['e1_DH'].stderr
        
        ps = {}
        ps['Tm_C'] = Tm
        ps['DTm'] = DTm
        ps['DH_J/mol'] = DH
        ps['DDH'] = DDH
        ps['m1'] = out.params['e1_m1'].value
        ps['Dm1'] = out.params['e1_m1'].stderr
        ps['n1'] = out.params['e1_n1'].value
        ps['Dn1'] = out.params['e1_n1'].stderr
        ps['m2'] = out.params['e1_m2'].value
        ps['Dm2'] = out.params['e1_m2'].stderr
        ps['n2'] = out.params['e1_n2'].value
        ps['Dn2'] = out.params['e1_n2'].stderr
        
        ps['DS_J/mol/K'] = MeltingCurveAnalysis._get_DS(DH, Tm)
        ps['DDS'] = MeltingCurveAnalysis._get_DDS(DH, Tm, DDH, DTm)

        if plot is True:
            out.plot(xlabel=column_x, ylabel=first_derivative_name, yerr=np.zeros(len(xdata)),
                     show_init=False, data_kws={'markersize': 4, 'marker': 'x'})
            plt.title(f'{filename}, {measurement_number}')
            
        return_dict={'Fit': out, 'result': ps}

        self.data[filename][f'Measurement_{measurement_number}']['deri_melt_fit'] = return_dict
        return return_dict



    

    def _add_results_to_meta(self):
        for file_name in self.data.keys():
            for key, value in self.data[file_name].items():
                
                deri_fit_o = value.get('derivative_fit', None)
                if deri_fit_o is not None:
                    deri_fit = deri_fit_o.get('result', None)
                  #  deri_fit.update(deri_fit_o.get('T_34 in °C', None))
                  #  deri_fit.update(deri_fit_o.get('DH in J/mol/K', None))
                    if deri_fit is None:
                        deri_fit = {}
                else:
                    deri_fit = {}
                deri_fit = {'deri_fit_' + str(key): val for key, val in deri_fit.items()}


                deri_max = value.get('derivative_max', None)
                if deri_max is None:
                    deri_max = {}
                deri_max = {'deri_max_' + str(key): val for key, val in deri_max.items()}


                deri_melt = value.get('deri_melt_fit', None)
                if deri_melt is not None:
                    deri_melt = deri_melt.get('result', None)
                else:
                    deri_melt = {}
                deri_melt = {'deri_melt_' + str(key): val for key, val in deri_melt.items()}


                melt = value.get('melt_fit', None)
                if melt is not None:
                    melt = melt.get('result', None)
                else:
                    melt = {}
                melt = {'direct_melt_' + str(key): val for key, val in melt.items()}


                median = value.get('baselines', None)
                if median != 'Fit failed':
                    if median is not None:
                        median = median.get('result', None)  # .get('Intersection', None)
                     #   median = {'x': median[0], 'y': median[1]}
                       # print(median)
                    else:
                        median = {}
                else:
                    median = {}
                median = {'median_' + str(key): val for key, val in median.items()}
                
                
             #   print(value.get('meta', 'eee'))
                value['meta'].update(deri_fit)
                value['meta'].update(deri_max)
                value['meta'].update(deri_melt)
                value['meta'].update(melt)
                value['meta'].update(median)

    def _get_melt_fit_out(self, file_name, measurement_number):
        return self.data.get(file_name).get(f'Measurement_{measurement_number}').get('melt_fit')



    def _get_derivative_fit_maxima(self, file_name, measurement_number):
        return self.data.get(file_name).get(f'Measurement_{measurement_number}').get('derivative_fit').get('Maxima')

    def _get_all_results(self, file_name, measurement_number):
        meta = self.data.get(file_name).get(f'Measurement_{measurement_number}').get('meta')
        data = self.data.get(file_name).get(f'Measurement_{measurement_number}').get('data')
        melt_out = self.data.get(file_name).get(f'Measurement_{measurement_number}').get('melt_fit', {})
        deri_fit = self.data.get(file_name).get(f'Measurement_{measurement_number}').get('derivative_fit', {})
        deri_max = self.data.get(file_name).get(f'Measurement_{measurement_number}').get('derivative_max', {})
        deri_melt = self.data.get(file_name).get(f'Measurement_{measurement_number}').get('deri_melt_fit', {})
        baselines = self.data.get(file_name).get(f'Measurement_{measurement_number}').get('baselines', {})
        return data, deri_fit, deri_max, melt_out, baselines, meta, deri_melt

    def remove_measurements(self, remove_list: list = []):
        for i in remove_list:
            del self.data[i[0]][f'Measurement_{i[1]}']


    def _get_DDH_Halfwidth(alpha, Tmax, T34, DTmax, DT34, Dalpha=0):
        t34 = abs(alpha * Tmax**2 / (Tmax - T34)**2 * DT34)
        tmax = abs(- alpha * T34**2 / (Tmax - T34)**2 * DTmax)
        dalpha = abs(- 1 / (Tmax - T34)**2 * Dalpha)
        return t34 + tmax + dalpha
    
    def _get_DH_DDH_VH(params):
        R = 8.31446261815324   # m^3* Pa * K^−1 * mol^−1
        DH = -R * params['slope'].value
        DDH= abs(-R * params['slope'].stderr)
        return DH, DDH
    
    def _get_DS_DDS_VH(params):
        R = 8.31446261815324   # m^3* Pa * K^−1 * mol^−1
        DS = R * params['intercept'].value
        DDS= abs(R * params['intercept'].stderr)
        return DS, DDS

    def _get_DDG(T, DDH, DS, DT, DDS):
        ddH = abs(DDH)
        dt = abs(-DS * DT)
        ds = abs(-T * DDS)
        return ddH + dt + ds

    def _get_DlnKa(theta, Dtheta):
        return abs(1 / (theta - theta**2) * Dtheta)

    def _get_DKa(theta, Dtheta):
        return abs((1 - theta)**(-2) * Dtheta)

    def _get_Dtheta(T, Ed, Tm, mu, nu, mo, no, DEd, DT, Dmu, Dnu, Dmo, Dno):
        dEd = abs(DEd / ((mu * T + nu) - (mo * T + no)))
        dT = abs((mu * (no - Ed) + mo * (Ed - nu)) / (T * (mu - mo) + nu + no)**2 * DT)
        dmu = abs(T * ((mo * T + no) - Ed) / (T * (mu - mo) + nu + no)**2 * Dmu)
        dnu = abs(((mo * T + no) - Ed) / (T * (mu - mo) + nu + no)**2 * Dnu)
        dmo = abs(T * ((mu * T + nu) - Ed) / (T * (mu - mo) + nu + no)**2 * Dmo)
        dno = abs((Ed - (mu * T + nu)) / (T * (mu - mo) + nu + no)**2 * Dno)
        return dEd + dT + dmu + dnu + dmo + dno
    
    def _ff_eq1_melt_fit(A, T, m1, m2, n1, n2):
        upper = m2 * T + n2
        lower = m1 * T + n1
        return (A - upper) / (lower - upper)
    
    def _equation_molecularity_3(T, DH, Tm, m1, n1, m2, n2):
        R = 8.31446261815324   # m^3* Pa * K^−1 * mol^−1
        x = DH / R * (1 / (T + 273.15) - 1 / (Tm + 273.15))
        w = np.sqrt(1 + np.exp(x) / 27)
        theta = 1 + np.exp(x / 3) / 8**(1 / 3) * ((w - 1)**(1 / 2) - (w + 1)**(1 / 3))
        return (m1 * (T + 273.15) + n1) * theta + (m2 * (T + 273.15) + n2) * (1 - theta)

    def _equation_molecularity_4(T, DH, Tm, m1, n1, m2, n2):
        R = 8.31446261815324   # m^3* Pa * K^−1 * mol^−1
        x = DH / R * (1 / (T + 273.15) - 1 / (Tm + 273.15))
        r = np.sqrt(1 + 2048 / 27 * np.exp(-x))
        u = np.exp(2 * x / 3) / (4 * 2**(1 / 3)) * (r + 1)**(1 / 2)
        v = np.exp(2 * x / 3) / (4 * 2**(1 / 3)) * (r - 1)**(1 / 3)
        theta = 1 + 1 / 2 * np.sqrt(u - v) - 1 / 2 * np.sqrt(2 * np.sqrt(u**2 + u * v + v**2 - (u - v)))
        return (m1 * (T + 273.15) + n1) * theta + (m2 * (T + 273.15) + n2) * (1 - theta)

    def _equation_molecularity_5(T, DH, Tm, m1, n1, m2, n2):
        R = 8.31446261815324   # m^3* Pa * K^−1 * mol^−1
        x = DH / R * (1 / (T + 273.15) - 1 / (Tm + 273.15))
        theta = 1 - 1 / (1 + np.exp(-x))
        return (m1 * (T + 273.15) + n1) * theta + (m2 * (T + 273.15) + n2) * (1 - theta)

    def _get_linear_baseline_median(self,
                                    filename,
                                    measurement_number,
                                    deribounds=(0.0005, 0.0005),
                                    column_x: str = 'Temperature (°C)',
                                    column_y: str = 'Absorbance',
                                    filter_bounds: list = [{'column_x': 'Temperature (°C)', 'column_y': 'Absorbance', 'x_min': -np.inf, 'y_min':
                                                            - np.inf, 'x_max': np.inf, 'y_max': np.inf, 'keep_inside': True}],
                                    first_derivative_name=None,
                                    second_derivative_name=None,
                                    plot: bool = True):

        df_data, first_derivative_name = MeltingCurveAnalysis._calculate_derivative(
            self.data[filename][f'Measurement_{measurement_number}']['data'].sort_values(column_x),
            column_x=column_x,
            column_y=column_y,
            first_derivative_name=first_derivative_name)

        # Convert DataFrame to numpy arrays
        xdata_0 = np.array(self.data[filename][f'Measurement_{measurement_number}']['data'].sort_values(column_x).get(column_x))
        ydata_0 = np.array(self.data[filename][f'Measurement_{measurement_number}']['data'].sort_values(column_x).get(column_y))
        #plt.plot(xdata_0, ydata_0)

        for value in filter_bounds:
            df = MeltingCurveAnalysis.filter_by_values(df_data,
                                                        value)

        maxima = self.data[filename][f'Measurement_{measurement_number}'].get('derivative_fit', None).get('Maxima', None)
        if maxima == None:
            print(filename, measurement_number)
            self.data[filename][f'Measurement_{measurement_number}']['baselines'] = 'Fit failed'

        else:
            # & (df['Temperature (C)'] >= lower_baseline_bounds[0]) & (df['Temperature (C)'] <= lower_baseline_bounds[1])), :]
            lower_baseline = df.loc[((abs(df[first_derivative_name]) < deribounds[0])
                                     & (df[column_x] < maxima[f'{column_x}_max'])), :]
            # & (df['Temperature (C)'] >= upper_baseline_bounds[0]) & (df['Temperature (C)'] <= upper_baseline_bounds[1])), :]
            upper_baseline = df.loc[((abs(df[first_derivative_name]) < deribounds[1])
                                     & (df[column_x] > maxima[f'{column_x}_max'])), :]

            data = {}
            med_data = {'params': None, 'best_fit': None, 'Intersection': None}
            ps = {}

            for index, value in enumerate([lower_baseline, upper_baseline]):
                if index == 0:

                    xdata = lower_baseline[column_x]
                    ydata = lower_baseline[column_y]
                    if len(value) < 2:
                        xdata = df.head(2)[column_x]
                        ydata = df.head(2)[column_y]
                        print(f'Curve Index {measurement_number} using lower tangent')

                else:

                    xdata = upper_baseline[column_x]
                    ydata = upper_baseline[column_y]
                    if len(value) < 2:
                        xdata = df.tail(2)[column_x]
                        ydata = df.tail(2)[column_y]
                        print(f'Curve Index {measurement_number} using upper tangent')

                mod = lmfit.models.LinearModel()
                pars = mod.guess(ydata, x=xdata)
                out = mod.fit(ydata, pars, x=xdata)
                data[f'baseline_{index}'] = out

                if index == 0:
                    ps['m_u'] = out.params['slope'].value
                    ps['sigma_m_u'] = out.params['slope'].stderr
                    ps['n_u'] = out.params['slope'].value
                    ps['sigma_n_u'] = out.params['slope'].stderr
                elif index == 1:
                    ps['m_o'] = out.params['slope'].value
                    ps['sigma_m_o'] = out.params['slope'].stderr
                    ps['n_o'] = out.params['slope'].value
                    ps['sigma_n_o'] = out.params['slope'].stderr

            pars = lmfit.Parameters()
            pars.add('slope', value=(data['baseline_0'].values.get('slope') + data['baseline_1'].values.get('slope')) / 2)
            pars.add('intercept', value=(data['baseline_0'].values.get(
                'intercept') + data['baseline_1'].values.get('intercept')) / 2)
            med_data['params'] = pars
            median = mod.eval(params=pars, x=xdata_0)
            med_data['best_fit'] = median
            ps['m_m'] = pars['slope'].value
            ps['n_m'] = pars['intercept'].value
            ps['sigma_m_m'] = 1 / 2 * (abs(ps['sigma_m_o']) + abs(ps['sigma_m_u']))
            ps['sigma_n_m'] = 1 / 2 * (abs(ps['sigma_n_o']) + abs(ps['sigma_n_u']))

            # Create a finer grid of x values for interpolation
            fine_x = np.linspace(maxima[f'{column_x}_max'] - 10, maxima[f'{column_x}_max'] + 10, 500)
            xdata_median = fine_x
            xdata_sigmoid = fine_x
            ydata_sigmoid = np.interp(x=fine_x, xp=xdata_0, fp=ydata_0)
         #   print(pars)

            ydata_median = mod.eval(params=pars, x=fine_x)
            pars.add('slope', value=ps['m_m'] + ps['sigma_m_m'])
            pars.add('intercept', value=ps['n_m'] + ps['sigma_n_m'])
          #  print(pars)
            ydata_median_plus = mod.eval(params=pars, x=fine_x)
            pars.add('slope', value=ps['m_m'] - ps['sigma_m_m'])
            pars.add('intercept', value=ps['n_m'] - ps['sigma_n_m'])
        #    print(pars)
            ydata_median_minus = mod.eval(params=pars, x=fine_x)

            t_temp = np.zeros([3, 2])

            idx = 0
            for ydata in [ydata_median, ydata_median_plus, ydata_median_minus]:
                # Initialize variables to store the closest intersection point
                closest_intersection = None
                closest_distance = float('inf')

                # Iterate through points of the linear function
                for x_median, y_median in zip(xdata_median, ydata):
                    # Find the closest point on the sigmoid function
                    index_sigmoid = np.argmin(np.abs(xdata_sigmoid - x_median))
                    x_sigmoid = xdata_sigmoid[index_sigmoid]
                    y_sigmoid = ydata_sigmoid[index_sigmoid]

                    # Calculate the Euclidean distance between the two points
                    distance = np.sqrt((x_median - x_sigmoid)**2 + (y_median - y_sigmoid)**2)

                    # Update the closest intersection point if a closer one is found
                    if distance < closest_distance:
                        closest_intersection = (x_median, y_median)
                        closest_distance = distance
                t_temp[idx, 0] = closest_intersection[0]
                t_temp[idx, 1] = closest_intersection[1]
              #  print(idx)
                idx += 1
               # print(t_temp)

       #     print('\n', t_temp)
            delta_T = max(abs(t_temp[:, 0] - t_temp[0, 0]))
       #     print('\n', delta_T)
            delta_Ed = max(abs(t_temp[:, 1] - t_temp[0, 1]))
      #      print(delta_Ed, '\n')
            ps['Tm_C'] = t_temp[0, 0]
            ps['DTm'] = delta_T
            ps['Ed'] = t_temp[0, 1]
            ps['DEd'] = delta_Ed

            data['result'] = ps

        #    print("Closest intersection point with higher x accuracy:", closest_intersection)
         #   med_data['Intersection'] = closest_intersection
            data['median'] = med_data
            # plt.plot
            if plot is True:
                #data[f'baseline_{0}'].plot(xlabel=column_x, ylabel=column_y, data_kws={
                #   'markersize': 4, 'marker': 'x'})
                #plt.plot(df[column_x], df[column_y])
               # data[f'baseline_{1}'].plot(xlabel=column_x, ylabel=column_y, data_kws={
                #    'markersize': 4, 'marker': 'x'})
                f, ax = plt.subplots(nrows=1, ncols=1)
                ax.plot(df[column_x], df[column_y], 'x', label='data')
                ax.plot(xdata_0, data[f'baseline_{1}'].eval(x=xdata_0), label='upper')
                ax.plot(xdata_0, data[f'baseline_{0}'].eval(x=xdata_0), label='lower')
              #  data['median'].plot(xlabel=column_x, ylabel=column_y, data_kws={
               #     'markersize': 4, 'marker': 'x'})
                ax.plot(closest_intersection[0], closest_intersection[1], 'x', label='Tm')
                ax.plot(xdata_0, mod.eval(params=pars, x=xdata_0), label='median')
              #  ax.plot(np.arange(50, 65, 0.1), e, label = 'dd')
                ax.legend()
                #ax.plot(xdata_sigmoid, ydata_sigmoid)
                ax.set_title(f'{filename}, {measurement_number}')

            self.data[filename][f'Measurement_{measurement_number}']['baselines'] = data

'''
if __name__ == "__main__":

    EXTRA_META = {}
    test = MeltingCurveAnalysis(EXTRA_META)
    test.fill_data_dict()
    test.meta_from_name_Paul(
        ['2023-11-17_RNA_melting_MgCl2_NaCl.csv', '2023-09-19_RNA_melting_KCl.csv'])
 #   test._create_meta_overview()

'''
    i = 8

    # , weights=[{'first_derivative_border_x_min': -np.inf, 'first_derivative_border_x_max': np.inf, 'first_derivative_border_y_min': -np.inf, 'first_derivative_border_y_max': np.inf, 'data_x_min': -np.inf, 'data_y_max':
   # test._get_filtered_derivative_fit(filename='2023_08_09_DNA_MOPS_75_K.csv', measurement_number=i, filter_bounds = [{'column_x': 'Temperature (°C)', 'column_y': 'dAbsorbance / dTemperature (°C)', 'x_min': -np.inf, 'y_min':
   #                         0.002, 'x_max': np.inf, 'y_max': np.inf, 'keep_inside': True}], model = lmfit.models.SkewedVoigtModel)
    test._get_filtered_derivative_fit(filename='2023_08_01_DNA_K_MOPS_pH65_PL.csv',
                                      measurement_number=i, model=lmfit.models.SkewedVoigtModel)

    #  70, 'data_y_min': -np.inf, 'data_y_max': np.inf, 'deri_weight': 1, 'data_weight': 1}])
    test._get_max_derivative(filename='2023_08_09_DNA_MOPS_75_K.csv', measurement_number=i)

    test._get_linear_baseline_median(filename='2023_08_09_DNA_MOPS_75_K.csv', measurement_number=i)

    test._get_filtered_melt_fit(model_pars=MeltingCurveAnalysis._create_model_e1(),
                                filename='2023_08_09_DNA_MOPS_75_K.csv', measurement_number=i)  # , p0=test._baseline_pars_melt_converter(filename='2023_08_09_DNA_MOPS_75_K.csv', measurement_number=i))

   # p0 = test.data['2023_08_09_DNA_MOPS_75_K.csv'][f'Measurement_{i}']['melt_fit'].params
    p0 = lmfit.Parameters()
    p0.add('e1_DH', value=-426801, vary=True)
    p0.add('e1_m1', value=0.000201, vary=True)
    p0.add('e1_m2', value=-0.0000506, vary=True)
    p0.add('e1_n1', value=0.126, vary=True)
    p0.add('e1_n2', value=0.26, vary=True)
    p0.add('e1_Tm', value=59.26, vary=True)
    test._get_filtered_melt_deri_fit(model_pars=MeltingCurveAnalysis._create_deri_model_e1(),
                                     filename='2023_08_09_DNA_MOPS_75_K.csv', measurement_number=i, p0=p0)#test._baseline_pars_melt_converter(filename='2023_08_09_DNA_MOPS_75_K.csv', measurement_number=i))  # , weights=[{'first_derivative_border_x_min': -np.inf, 'first_derivative_border_x_max': np.inf, 'first_derivative_border_y_min': -np.inf, 'first_derivative_border_y_max': np.inf, 'data_x_min': -np.inf, 'data_y_max':
   # test._get_filtered_melt_deri_fit(model_pars=MeltingCurveAnalysis._create_deri_model_e1_b(),
    #                                   filename='2023_08_09_DNA_MOPS_75_K.csv', measurement_number=i, p0 = p0)
    # test._get_filtered_melt_fit(model_pars=MeltingCurveAnalysis._create_model_e1(),
   #                             filename='2023_08_09_DNA_MOPS_75_K.csv', measurement_number=i)
    #   70, 'data_y_min': -np.inf, 'data_y_max': np.inf, 'deri_weight': 1, 'data_weight': 1}])
    test._folded_fraction(filename='2023_08_09_DNA_MOPS_75_K.csv', measurement_number=i,
                          baseline_median =test.data['2023_08_09_DNA_MOPS_75_K.csv'][f'Measurement_{i}']['baselines'])
#  test._add_results_to_meta()
   # p0 = lmfit.Parameters()
   # p0.add('p1_amplitude', value=-1, vary=True)
   # test._get_filtered_derivative_fit(filename='2023_08_09_DNA_MOPS_75_K.csv',
    #                                           measurement_number=1, column_y='Folded Fractions', p0=p0)

    test.meta_from_name_Paul(['2023_08_09_DNA_MOPS_75_K.csv'])
    test._add_results_to_meta()
    test._create_meta_overview()

  #  test.save()
   # test222 = test.apply_filters({'Wavelength (nm)': [260], 'Concentration': [0, 100]})

 #   pickled_data = MeltingCurveAnalysis.load()
  #  print(pickled_data)
    sk = test.data['2023_08_09_DNA_MOPS_75_K.csv'][f'Measurement_{i}']
    '''
