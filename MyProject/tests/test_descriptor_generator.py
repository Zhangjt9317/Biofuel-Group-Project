import sys
sys.path.append('../..')

import MyProject

def test_descriptor_generator():
    """Check descriptor_generator can generate correct numbers of functional groups for 
    specific compound according to CID"""
    result1 = MyProject.descriptor_generator(887)  ##test methane
    result2 = MyProject.descriptor_generator(175854)   ##test 2-Butene-1,4-diol, (2E)-
    result3 = MyProject.descriptor_generator(61038)   ##test 2-Ethyl-2-butyl-1,3-propanediol

    assert result1 == [4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], (
                      "The descriptor_generator does not work properly")
    assert result2 == [8, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], (
                      "The descriptor_generator does not work properly")
    assert result3 == [20, 2, 6, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], (
                      "The descriptor_generator does not work properly")
    return