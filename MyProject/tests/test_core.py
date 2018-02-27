import sys
sys.path.append('../..')

import MyProject


def test_output_database():
    '''Check the shape of the database'''
    result = MyProject.core.Database()
    assert result.shape == (1159, 32), "The database is not right"
    return
