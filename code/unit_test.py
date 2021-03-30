description = '''

 Execute unit test for the DISTANA implementation. With the positional command line argument you can select 
 which unit test should be executed.

'''

import argparse


if __name__ == "__main__":
    '''Starting point of program'''

    parser = argparse.ArgumentParser(description=description, 
            formatter_class=argparse.RawTextHelpFormatter)

    # define the cl arguments corresponding to the unit tests
    parser.add_argument('unit_test', choices=['graph', 'graph2'],  
            help='Which unit test should be executed?')

    # parse command line arguments
    options = parser.parse_args()

    #
    # select the unit test corresponding to the argument
    if options.unit_test == "graph":
        from qa.test_graph_cuda import test_graph
        test_graph()
    if options.unit_test == "graph2":
        from qa.test_graph_cuda2 import test_graph
        test_graph()
