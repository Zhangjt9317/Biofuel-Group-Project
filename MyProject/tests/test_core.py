import sys
sys.path.append('..')

from core import *

def test_Database():
    '''Check the shape of the database'''
    result = Database()
    assert result.shape == (1159, 32), "The database is not right"  ##test the shape of the database
    return

def test_SMILES():
    """Check the SMILES function could generate the correct molecular structure,based on input CID number"""
    result1 = SMILES(175854)  ##check SMILES structures
    result2 = SMILES(13526104)
    result3 = SMILES(887)
    
    assert result1 == 'C(C=CCO)O', 'The chemcial (2E)-but-2-ene-1,4-diol converts wrong!'
    assert result2 == 'CCCCC(CC)(CO)C(C)O', 'The chemcial 2-Butyl-2-ethylbutane-1,3-diol converts wrong!'
    assert result3 == 'CO', 'The chemcial methanol converts wrong'
    return

def test_descriptor_generator():
    """Check descriptor_generator can generate correct numbers of functional groups for 
    specific compound according to CID"""
    result1 = descriptor_generator(887)  ##test methane
    result2 = descriptor_generator(175854)   ##test 2-Butene-1,4-diol, (2E)-
    result3 = descriptor_generator(61038)   ##test 2-Ethyl-2-butyl-1,3-propanediol

    assert result1 == [4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], (
                      "The descriptor_generator does not work properly")
    assert result2 == [8, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], (
                      "The descriptor_generator does not work properly")
    assert result3 == [20, 2, 6, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], (
                      "The descriptor_generator does not work properly")
    return

