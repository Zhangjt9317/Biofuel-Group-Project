import descriptor_generator


def test_descriptor_generator():
	"""Check for the descriptor_generator could generate the correct number of functional group for 
different components"""
    result1 = descriptor_generator.descriptor_generator(887)
    result2 = descriptor_generator.descriptor_generator(175854)
    result3 = descriptor_generator.descriptor_generator(13526104)
    result4 = descriptor_generator.descriptor_generator(138219)
    result5 = descriptor_generator.descriptor_generator(21195085)
    result6 = descriptor_generator.descriptor_generator(86607438)

    assert result1 == [4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "The descriptor_generator does not work properly"
    assert result2 == [8, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "The descriptor_generator does not work properly"
    assert result3 == [22, 3, 5, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "The descriptor_generator does not work properly"
    assert result4 == [12, 0, 3, 1, 0, 0, 0, 0, 0, 3, 1, 0, 5, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "The descriptor_generator does not work properly"
    assert result5 == [16, 2, 2, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "The descriptor_generator does not work properly"
    assert result6 == [8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 1, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0], "The descriptor_generator does not work properly"

    return
