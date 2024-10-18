from src.blab_devices.cary import Cary, CaryAnalysis, CaryDataframe
from matplotlib import pyplot as plt
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")

cary_data = Cary("example_for_cary", "example_for_cary/extra.json")
cary_analysis = CaryAnalysis(cary_data)

cary_analysis.set_new_peak_for_measurement("2023_08_31_DNA_TRIS-HCl_85_K", 3)
cary_analysis.set_baselines("2023_08_31_DNA_TRIS-HCl_85_K", 3, melting_temp=np.array([340.15]))
cary_analysis.redo_single_curve_fit("2023_08_31_DNA_TRIS-HCl_85_K", 1, melting_temp=np.array([335.15]))
cary_analysis.redo_single_curve_fit("2023_08_31_DNA_TRIS-HCl_85_K", 3, melting_temp=np.array([340.15]))
cary_analysis.redo_single_curve_fit("2023_08_31_DNA_TRIS-HCl_85_K", 5, melting_temp=np.array([345.15]))
cary_analysis.redo_multi_curve_fit(filename="2023_08_31_DNA_TRIS-HCl_85_K",
                                   mean_melting_temps=np.array([340.15]),
                                   name="0_mM_K(I)_TRIS-HCl_pH8.5",
                                   ramp="Cooling")
cary_analysis.redo_fill_results(filename="2023_08_31_DNA_TRIS-HCl_85_K")

#cary_analysis.set_normalized_absorbance_for_measurement('file_1', 2, 30, 70)
#cary_analysis.set_normalized_first_derivative_for_measurement_only_min_temp('file_1', 2, 30)
#cary_analysis.set_normalized_second_derivative_for_measurement('file_1', 2, 30, 70)
#cary_analysis.set_normalized_absorbance_for_measurement('file_2', 1, 40, 60)
#cary_analysis.set_normalized_absorbance_for_measurement('file_1', 32, 33, 52)
#cary_analysis.set_normalized_absorbance_for_measurement('file_2', 32, 23, 70)



# data = (cary_analysis.cary_object.list_data[1])[cary_analysis.cary_object.list_data[1]['Measurement'] == 5]
# meta_data = cary_analysis.cary_object.list_data_meta[1]
#
# for i in range(1, 17):
#     data = (cary_analysis.cary_object.list_data[1])[cary_analysis.cary_object.list_data[1]['Measurement'] == i]
#     meta_data = (cary_analysis.cary_object.list_data_meta[1])[cary_analysis.cary_object.list_data_meta[1]['Measurement'] == i]
#
#     fig, ax = plt.subplots(1, 1, figsize=(15, 10))
#
#     ax.scatter(data[CaryDataframe.Temperature_K.value], data[CaryDataframe.Absorbance.value], alpha=0.7)
#     baselines = meta_data[CaryDataframe.BaseLines.value].to_list()[0]
#     x = np.array([280, 380])
#     for baseline in baselines:
#         y = baseline[0]*x + baseline[1]
#         ax.plot(x, y)
#     ax.set_title(f'Temperature vs Absorbance')
#     ax.set_xlabel('Temperature')
#     ax.set_ylabel('Absorbance')
#
#     plt.tight_layout()
#     plt.show()

print(data)
