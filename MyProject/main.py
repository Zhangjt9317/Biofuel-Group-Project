def main(CID):
    """
    This is the main function of the software.
    """
    #families = ['Alcohols', 'Aromatics', 'Aromatics (Oxygenates)', 'Cyclic (Oxygenates)', 'Esters',
    #          'Naphtenes', 'Olefins', 'Paraffins']
    prop = ['Flash Point', 'Cetane Number']
    result = []  ###place predict values
    fg = descriptor_generator(CID)  ###generate functional groups according to CID
    ### family = classifier(fg)     ###classify the chemical according to functional groups
    for p in prop:
        model = model_selection(family, p, #iteration, #fraction, #test_size)  ###select model for specific property
        value = model.predict(fg)   ###predict property according to functional groups
        result.append(value)
    return result   ###return the result list, [0] is Flash Point, [1] is Cetane Number