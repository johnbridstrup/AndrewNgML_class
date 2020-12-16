import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

"""
    Multivariable Linear regression script

    Data on cost of insurance from https://github.com/stedy/Machine-Learning-with-R-datasets

    Data:
        Age: Int
        Sex: Category - M/F
        BMI: Float
        Children: Int
        Smoker: Bool
        Region: Category - northeast, northwest, southeast, southwest
        Charges: Float
"""

def formatData(data):
    """Format data with boolean values for categorical data"""
    # Binary data first
    binData = data.replace({'female':1, 'male':0, 'yes':1, 'no':0})
    # Category data
    regionDummies = binData.region.str.get_dummies()
    outData = pd.concat([regionDummies,binData.drop('region', axis=1)],axis=1)
    # Normalize float data (min/max)
    cols_to_norm = ['age','bmi']
    outData[cols_to_norm] = MinMaxScaler().fit_transform(outData[cols_to_norm])
    return outData

# Load data
path = 'data/insurance.csv'
data = pd.read_csv(path)
print(data)
# Format data
formatted_data = formatData(data)
print(formatted_data)
""" Sex and Smoker can be treated as a binary variable [0,1]. 
    Region converted into 4 boolean variables for NE, NW, SE, SW
"""

# 