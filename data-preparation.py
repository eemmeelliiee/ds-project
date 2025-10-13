import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
import sys

# Validate input file(s)
def ReadPreviousData(location: str):
    Filepath = Path(location)
    if Filepath.exists():
        data = pd.read_csv(location)
        return data, True 
    else:
        data = pd.DataFrame()

df, df_exists = ReadPreviousData('data/Indian_water_data.csv')

PreviousDataComplete = all([df_exists])

if PreviousDataComplete == False:
    sys.exit("ERROR: Not all required data found.") 
print("All files found")

#Drop irrelevant columns 
irel_cols = 'STN code,Monitoring Location,Year,Conductivity (¬µmho/cm) - Min,Conductivity (¬µmho/cm) - Max,NitrateN (mg/L) - Min,NitrateN (mg/L) - Max,Fecal Coliform (MPN/100ml) - Min,Fecal Coliform (MPN/100ml) - Max,Fecal - Min,Fecal - Max'
irel_cols_arr = irel_cols.split(',')
df.drop(columns=irel_cols_arr, inplace=True)
df.to_csv('data/water_quality_cols.csv') # Contains all columns needed for determining water quality class

#Remove rows with null values
print(df.shape) # 194 rows
print(df.isna().sum())
df.dropna(inplace=True) # 29 rows deleted
print(df.shape) # 165 rows left
df.reset_index(drop=True, inplace=True)

# Find non numerical columns and convert (string) values to numerical equivalents
non_num_cols = df.select_dtypes(exclude=['number']).columns
print(non_num_cols)
for col in non_num_cols: 
    print(df[col].unique()) # Output: String 'BDL' exists in columns 'Dissolved - Min' and 'BOD (mg/L) - Min'
#Replace string value 'BDL' with LOD/2 for relevant columns
df.replace(to_replace={'Dissolved - Min': 'BDL'}, value=0.3/2, inplace=True) # LOD/2 For DO
df.replace(to_replace={'BOD (mg/L) - Min': 'BDL'}, value=1.0/2, inplace=True) # LOD/2 For BOD

# Convert (numerical) object datatype columns to float
obj_cols_to_num = 'Dissolved - Min,BOD (mg/L) - Min,Total Coliform (MPN/100ml) - Min'
obj_cols_to_num_arr = obj_cols_to_num.split(',')
df[obj_cols_to_num_arr] = df[obj_cols_to_num_arr].astype(float)
df.to_csv('data/water_quality_cols_complete_rows.csv')

# Add water quality class column with values based on CPCB water quality qriteria
class_a = ((df['pH - Min'] >= 6.5) & (df['pH - Max'] <= 8.5) & (df['Dissolved - Min'] >= 6) & (df['BOD (mg/L) - Max'] <= 2) & (df['Total Coliform (MPN/100ml) - Max'] <= 50))
class_c = ((df['pH - Min'] >= 6) & (df['pH - Max'] <= 9) & (df['Dissolved - Min'] >= 4) & (df['BOD (mg/L) - Max'] <= 3) & (df['Total Coliform (MPN/100ml) - Max'] <= 5000))
df['water_quality'] = np.select([class_a, class_c],[2, 1], default=0)  # 2 = 'Class A', 1 = 'Class C' 0 = 'Other')

# Drop columns not used for predictive model use case
irel_cols = 'Total Coliform (MPN/100ml) - Min,Total Coliform (MPN/100ml) - Max,BOD (mg/L) - Min,BOD (mg/L) - Max,Dissolved - Max' 
irel_cols_arr = irel_cols.split(',')
df.drop(columns=irel_cols_arr, inplace=True)
df.to_csv('data/water_quality_cols_complete_rows_rel_cols.csv')

# One-hot encode categorical values
df = pd.get_dummies(df, columns=['Type Water Body', 'State Name'], dtype=int)
df.to_csv('data/water_quality_cols_complete_rows_rel_cols_all_num.csv')

# Correlation check
corr = df.corr()
highest_corr = corr.unstack().drop_duplicates().sort_values(ascending=False).head(5) # Get five highest correlation coefficients
print(highest_corr)
# Create heatmap
fig1, ax1 = plt.subplots()
fig1.set_size_inches((27,20))
sns.heatmap(corr,#Use the flightscorrelation matrix
            xticklabels=corr.columns.values, #Use the column names of that matrix
            yticklabels=corr.columns.values,
            annot=True,
            fmt= '.2f', #Annotate the values with 2 decimal places
            cmap=sns.color_palette("coolwarm", as_cmap=True), #Color options
            vmin=-1.0, vmax=1.0, #Ensure the color range is set to the entire possible correlation range
            square=True, ax=ax1) #Set the form of the cells to squares
ax1.set_title("Correlation matrix")
fig1.subplots_adjust(right=0.9) 
fig1.savefig("figures/correlation_matrix.png", format='png', dpi=600)
plt.show

# For lower dimensionality: Remove feature with highest correlation (Type Water Body_WATER TREATMENT PLANT (RAW WATER))
df.drop(columns=['Type Water Body_WATER TREATMENT PLANT (RAW WATER)'], inplace=True)
df.to_csv('data/cleaned_dataset.csv')
