#!../bin/python3
import unittest 
import sys, os, unittest
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ClassDefinition.Utils import Logger, ArgumentParser

class TestUtils(unittest.TestCase):
    def test_argument_parser(self):
        argParse=ArgumentParser()
        with self.assertRaises(Exception):
            argParse.setArguments(
                "KeyValueNotEqualSignSep",
            )
        with self.assertRaises(Exception):
            argParse.setArguments(
                ["key=value"],
                required_arguments=["keyWithDifferentName"]
            )
        argParse.setArguments(
            ["key=value"],
            optional_arguments=["keyWithDifferentName"]
        )
        with self.assertRaises(Exception):
            argPargse.get("key")
        argParse.setArguments(
            ["key1=value1", "key2=value2"],
            required_arguments=["key1"],
            optional_arguments=["key2"]
        )
        self.assertEqual(argParse.get("key1"), "value1")
        self.assertEqual(argParse.get("key2"), "value2")
        print("Passed test_argument_parser")
        

    
    def test_logger(self):
        logger = Logger(__name__)
        print = logger.print
        print("Hello World!!!") # just test that this doesnt raise E 
        print("Passed test_logger")
            

def main():
    ts = TestUtils()
    ts.test_argument_parser()
    ts.test_logger()
    

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"Encountered Exception {e}")
        exit(1)
