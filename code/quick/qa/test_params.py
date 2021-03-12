'''

    Quality Assurance: Test parameters 

'''

import sys

def _c(v, desc):
    if not v:
        sys.exit("The consistency check failed when we tested the " + str(desc))

def test_training(params):

    _c(params['learning_rate']>0 && params['learning_rate']<1, "params['learning_rate']")

    _c(params)

