import core


def test_output_database(x):
    """Check the shape of the database"""
    import pandas as pd
    data = pd.read_excel('data/Flash Point and Cetane Number Predictions for Fuel Compounds.xls', skiprows=4)
    dataframe = x.Database()
    if dataframe.size == len(data) * len(dataframe.columns):
        print('Cool! Everything runs well. :)', 'The size of database is', dataframe.size)
    else:
        raise Exception('Oops! Some data is missing :(, please check on the "core function"')
    return


def test_descriptors(x):
    """Check the descriptors is in our database"""
    if x not in core.Database().columns:
        raise Exception('Sorry! Our database can not find your descriptor. Please try again :)')
    else:
        print('The location of your searched descriptor in our Database is at column',
              core.Database().columns.get_loc(x))
        #Give the location of the descriptor in the database
        print(x)
    return


def test_type_of_biofuel_component(x):
    """Check up the biofuel component you searched is in our database"""
    if x in list(core.Database().xs('Family', axis=1)):
        print("Great!")
    else:
        print('Sorry! Our database can not find your biofuel component. Please try again :)')
    return
