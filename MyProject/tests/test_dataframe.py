import sys
sys.path.append('../..')

import pandas as pd

from MyProject import core


def test_output_database():
    '''Check the shape of the database'''
    data = pd.read_excel('../data/Flash Point and Cetane Number Predictions for Fuel Compounds.xls', skiprows=4)
    if core.Database().size == len(data) * len(core.Database().columns):
        print('Cool! Everything runs well. :)', 'The length of database is', len(core.Database()), 
		',The width of database is', len(core.Database().columns))
    else:
        raise Exception('Oops! Some data is missing :(, please check on the "core function"')
    return
