def check_input(CID):
    """Check the CID number is correct"""
    if type(CID) != int:    #Check the type of input
        print('Please check the CID number and insert again :)')
    elif CID in range(1000000000):   #Check the range of CID number
        return CID
    else:
        print(('Please check you CID number make sure you input the correct CID number'))
