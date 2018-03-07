import pubchempy as pcp

def SMILES(CID):
    """
    Description:
    This function produces a SMILES from CID number input 
    """
    #CID number can be gained from URL:https://pubchem.ncbi.nlm.nih.gov/
    c = pcp.Compound.from_cid(CID)
    SMILES = c.canonical_smiles  ###create SMILES structure
    return SMILES