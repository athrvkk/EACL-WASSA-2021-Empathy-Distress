import pandas as pd 
tsv_file='mydata.tsv'
csv_table=pd.read_table(tsv_file,sep='\t')
csv_table.to_csv('my_data.csv',index=False)