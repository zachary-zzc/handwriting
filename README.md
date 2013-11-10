handwriting
===========
A handwriting classification using bp network.

Dependencies
-------------------------------------------------
Tested to work under Python 3.2+.

The required dependencies to build the software Numpy >= 1.3.

For runing and testing required matplotlib >= 0.99.1.

Training and Testing DataBase
-------------------------------------------------
MNIST handwritten digit database.

Total 60000 Training Set

Total 10000 Testing Set

Run and Test
-------------------------------------------------
To run the script, type

    python3 handwriteRecognition.py [option] [arg]
    
For help usage print "-h" or "--help"

To run unit test, type 

    python3 bpnetworkTest.py
    python3 handwriteRecognitionTest.py

Current Achievement
-------------------------------------------------
bpNetwork can work properly, One-class classification with 
1000 training sets have more than 95% correct rate over more
then 2000 testing sets.

Problem and Further Work
-------------------------------------------------
1. Multiclass classification

2. Regression term

3. Further functions like gradient check
