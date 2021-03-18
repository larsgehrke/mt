description = '''

 Execute unit test for the DISTANA implementation.

'''

import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=description, 
            formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('unit_test', choices=['graph', 'graph2'],  
            help='Which unit test should be executed?')

    options = parser.parse_args()

    if options.unit_test == "graph":
        from qa.test_graph_cuda import test_graph
        test_graph()
    if options.unit_test == "graph2":
        from qa.test_graph_cuda2 import test_graph
        test_graph()
