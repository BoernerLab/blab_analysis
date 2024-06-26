from src.blab_devices.cary import Cary, CaryAnalysis, CaryDataframe
from matplotlib import pyplot as plt


cary_data = Cary("test", "test/extra.json")
cary_analysis = CaryAnalysis(cary_data)
cary_analysis.set_normalized_absorbance_for_measurement('file_1', 2, 30, 70)
cary_analysis.set_normalized_first_derivative_for_measurement_only_min_temp('file_1', 2, 30)
cary_analysis.set_normalized_second_derivative_for_measurement('file_1', 2, 30, 70)
cary_analysis.set_normalized_absorbance_for_measurement('file_2', 1, 40, 60)
cary_analysis.set_normalized_absorbance_for_measurement('file_1', 32, 33, 52)
cary_analysis.set_normalized_absorbance_for_measurement('file_2', 32, 23, 70)

data = (cary_analysis.cary_object.list_data[0])[cary_analysis.cary_object.list_data[0]['Measurement'] == 5]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

abs_columns = [CaryDataframe.Absorbance.value, CaryDataframe.FirstDerivative.value, CaryDataframe.SecondDerivative.value,
               CaryDataframe.NormalizedAbsorbance.value, CaryDataframe.FirstDerivativeNormalized.value,
               CaryDataframe.SecondDerivativeNormalized.value]

for i, ax in enumerate(axes.flat):
    ax.scatter(data[CaryDataframe.Temperature_K.value], data[abs_columns[i]], alpha=0.7)
    ax.set_title(f'Temperature vs {abs_columns[i]}')
    ax.set_xlabel('Temperature')
    ax.set_ylabel(abs_columns[i])

plt.tight_layout()
plt.show()

print(data)
