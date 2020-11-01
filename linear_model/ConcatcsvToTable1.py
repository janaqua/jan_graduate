import pandas as pd

table1 = pd.DataFrame()
for i in [1,2,5,10,20,50,100,200]:

    table1 = table1.append(pd.read_csv('ToConcatenate\Batch_ASGDandSGD_with_GammaDecay_df%d.csv'%i))

table1.to_csv('Table1.csv', index=False)
