import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
df = pd.read_csv('full_cas_data.csv')
predicted = pd.read_csv('/Users/sadrachpierre/Desktop/analyze_cas/xml_files/COVID-Substance-Property-Content-Set/Properties/full_cas_predicted_experiment.csv')

predicted['cas.rn'] = predicted['CAS#'].str.extract('(\d+)')
predicted['cas.rn'] =predicted['cas.rn'].str[:-3] + '-' + predicted['cas.rn'].str[-3:-1] + '-' + predicted['cas.rn'].str[-1] 

new_df = pd.merge(predicted, df, on='cas.rn')      
