import pandas as pd
d = r"C:\Users\bougu\Real-Time-Embedded-STM32_Acoustic_Diagnostic_System_for_Electromechanical_Equipment_Monitoring\windows\slider"
r = pd.read_csv(d + r"\select_results.csv")
print(r[r.k.isin([30, 58, 65])].pivot_table(index=["variant", "k", "method"], columns="test_id", values="f1").round(2).to_string())