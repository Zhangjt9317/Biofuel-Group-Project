import sys
sys.path.append('../..')

from MyProject import SMILES_generator


def test_SMILES():
    """Check the SMILES function could generate the correct molecular structure,based on input CID number"""
    result1 = SMILES_generator.SMILES(175854)
    result2 = SMILES_generator.SMILES(13526104)
    result3 = SMILES_generator.SMILES(887)
    result4 = SMILES_generator.SMILES(138219)
    result5 = SMILES_generator.SMILES(86607438)
    result6 = SMILES_generator.SMILES(21195085)
    
    assert result1 == u'C(C=CCO)O','The chemcials: "(2E)-but-2-ene-1,4-diol" convert wrong!'
    assert result2 == u'CCCCC(CC)(CO)C(C)O', 'The chemcials: "2-Butyl-2-ethylbutane-1,3-diol" convert wrong'
    assert result3 == u'CO', 'The chemcial: "methanol" convert wrong'
    assert result4 == u'C1CC(C1)C2=CC=CC=C2', 'The chemcial: "benzene" convert wrong'
    assert result5 == u'[HH].C1=CC=C(C=C1)O', 'The chemcial: "phenol" convert wrong'
    assert result6 == u'CCC=C.CCC=C', 'The chemcial: "but-1-ene" convert wrong'
    
    if result1 != "O":
        print('The unit test for SMILES work well!')
    else:
        print('The unit test for SMILES runs wrong!')

    return
