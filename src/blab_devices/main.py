from src.blab_devices.cary import Cary, CaryAnalysis

cary_data = Cary("test", "test/extra.json")
cary_analysis = CaryAnalysis(cary_data)
cary_analysis.set_normalized_absorbance_for_measurement('file_1', 1, 30, 70)
cary_analysis.set_normalized_absorbance_for_measurement('file_2', 1, 40, 60)
cary_analysis.set_normalized_absorbance_for_measurement('file_1', 32, 33, 52)
cary_analysis.set_normalized_absorbance_for_measurement('file_2', 32, 23, 70)

print(cary_analysis)
