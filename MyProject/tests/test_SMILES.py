import sys
sys.path.append('../..')

import MyProject

def test_SMILES():
    """Check the SMILES function could generate the correct molecular structure,based on input CID number"""
    result1 = MyProject.SMILES(175854)  ##check SMILES structures
    result2 = MyProject.SMILES(13526104)
    result3 = MyProject.SMILES(887)
    
    assert result1 == 'C(C=CCO)O', 'The chemcial (2E)-but-2-ene-1,4-diol converts wrong!'
    assert result2 == 'CCCCC(CC)(CO)C(C)O', 'The chemcial 2-Butyl-2-ethylbutane-1,3-diol converts wrong!'
    assert result3 == 'CO', 'The chemcial methanol converts wrong'
    return