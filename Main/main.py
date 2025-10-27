#!../bin/python3
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ClassDefinition.Utils import Logger, ArgumentParser
required_arguments=["testRequiredArgument1"]
optional_arguments=[]
g_Logger = Logger(__name__)
g_ArgParse = ArgumentParser()
print=g_Logger.print

USAGE="""
main.py testRequiredArgument1=<dummyValue>
"""

def initialize(inputArguments):
    g_ArgParse.setArguments(inputArguments, required_arguments, optional_arguments)

def main(inputArguments):
    initialize(inputArguments)
    print(g_ArgParse.get("testRequiredArgument1"))

    

if __name__ == '__main__':
    try:
        main(sys.argv[1:])
    except Exception as e:
        g_Logger.logger.exception(e)
        exit(1)
