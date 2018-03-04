import openbabel
import pybel
from pybel import Smarts, readstring

import SMILES_generator


def descriptor_generator(CID):
    """Generate the number of each functional group for different components"""
    fg = ["[H]", "[CX4H3]", "[CX4H2]", "[CX4H1]", "[CX4H0]","[CX3H2]", "[CX3H1]", "[CX3H0]", "[CX2H1]", "[CX4H2R]", "[CX4H1R]", "[CX4H0R]", "[cX3H1](:*):*", "[cX3H0](:*)(:*)*", "[OX2H1]", "[OX2H1][cX3]:[c]", "[OX2H0]", "[OX2H0R]", "[oX2H0](:*):*", "[CX3H0]=[O]", "[CX3H0R]=[O]", "[CX3H1]=[O]", "[CX3H0](=[O])[OX2H1]", "[CX3H0](=[O])[OX2H0]", "[cX3H0](:*)(:*):*"]
    counts = []
    result = SMILES_generator(CID)
    for i in range(len(fg)):
        mol = readstring("smi", result)
        smarts = Smarts(fg[i])
        n = smarts.findall(mol)
        counts.append(len(n))
    return counts
