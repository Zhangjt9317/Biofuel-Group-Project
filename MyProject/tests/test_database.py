import sys
sys.path.append('../..')

import MyProject

def test_Database():
    '''Check the shape of the database'''
    result = MyProject.Database()
    assert result.shape == (1159, 32), "The database is not right"  ##test the shape of the database
    return