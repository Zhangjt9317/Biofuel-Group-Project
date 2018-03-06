import pandas as pd

def Database():
    """compile data sets into a data library, the output will be a DataFrame"""
    data_1 = pd.read_excel('data/Flash Point and Cetane Number Predictions for Fuel Compounds.xls', skiprows=3) ###load data
    data = data_1[['Name','Family', 'FP Exp.', 'CN Exp.']]  ###select columns
    result_1 = data.drop(index=0)   ###rearrange the index
    result_1.reset_index(drop=True, inplace=True)
    names = ['Name', 'Family', 'Flash Point', 'Cetane Number']   ###rename columns
    result_1.columns = names
    data_2 = pd.read_excel('data/Flash Point and Cetane Number Predictions for Fuel Compounds.xls', skiprows=4)
    result_2 = data_2.loc[: , '-H': 'aaCa']   ###select specific columns
    smarts = ["[H]", "[CX4H3]", "[CX4H2]", "[CX4H1]", "[CX4H0]", "[CX3H2]", "[CX3H1]", "[CX3H0]", "[CX2H1]","[CX2H0]", "[CX4H2R]", "[CX4H1R]", "[CX4H0R]","[CX3H1R]","[CX3H0R]","[cX3H1](:*):*", "[cX3H0](:*)(:*)*", "[OX2H1]", "[OX2H1][cX3]:[c]", "[OX2H0]", "[OX2H0R]", "[oX2H0](:*):*", "[CX3H0]=[O]", "[CX3H0R]=[O]", "[CX3H1]=[O]", "[CX3H0](=[O])[OX2H1]", "[CX3H0](=[O])[OX2H0]", "[cX3H0](:*)(:*):*"]   ###rename functional groups to SMARTS
    result_2.columns = smarts
    result = pd.concat([result_1, result_2], axis=1)   ###combine two dataframes into one dataframe
    return result