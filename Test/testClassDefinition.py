#!../bin/python3
import unittest 
import sys, os, unittest
import torch 
SEMROOT = os.environ['SEMROOT']
sys.path.append(SEMROOT)

from ClassDefinition.Utils import Logger, ArgumentParser
from ClassDefinition.Roberta import Roberta

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
            optional_arguments={"keyWithDifferentName": "defaultValue"}
        )
        with self.assertRaises(Exception):
            argPargse.get("key")
        self.assertEqual(argParse.get("keyWithDifferentName"), "defaultValue")
        argParse.setArguments(
            ["key1=value1", "key2=value2"],
            required_arguments=["key1"],
            optional_arguments={"key2":"defaultValue2"}
        )
        self.assertEqual(argParse.get("key1"), "value1")
        self.assertEqual(argParse.get("key2"), "value2")
        print("Passed test_argument_parser")
        
    
    def test_logger(self):
        logger = Logger(__name__)
        print = logger.print
        print("Hello World!!!") # just test that this doesnt raise E 
        print("Passed test_logger")
            
class TestRoberta(unittest.TestCase):
    def testBasicSentence(self):
        roberta=Roberta()
        roberta.setText("Hello World!")
        self.assertEqual(roberta.getText(), "Hello World!")

        helloWorldEmbedding = roberta.getClsEmbedding()
        self.assertEqual(
            len(helloWorldEmbedding[0]),
            768
        ) # cls embedding is of length 768 
        self.assertEqual(
            torch.equal(roberta.getClsEmbedding()[0],helloWorldEmbedding[0]), True
        ) # just sanity check

        roberta.setText("Goodbye World!")
        self.assertEqual(
            torch.equal(helloWorldEmbedding[0],roberta.getClsEmbedding()[0]), False
        )
        print("Passed TestRoberta")

def main():
    ts = TestUtils()
    ts.test_argument_parser()
    ts.test_logger()

    ts = TestRoberta()
    ts.testBasicSentence()
    

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"Encountered Exception {e}")
        exit(1)
