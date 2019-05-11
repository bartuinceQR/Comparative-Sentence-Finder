# Comparative-Sentence-Finder
A Python based program that detects whether a given (Turkish) sentence is comparative or not. It makes use of the ITU NLP Pipeline, and requires access to it if one wishes to test sentences outside of the existing test set.

# How to Run
The program can be run by calling sentence_parser.py with a "-mode" argument. The modes are as follows:

-read : Reads an input file that will be sent to the ITU Pipeline. The file can contain multiple sentences separated by newlines. The file can be supplied with the "-readfile" argument. The sentences in the file have to be followed by their class ("comparative" or "non-comparative"), separated by a "|".

-convert : Iterates through the results provided by the Pipeline, and creates a feature list that can be used in the training phase. (Default folder to choose from : "apiresults/train")

-train : Trains an SVM setup depending on the features listed in the previous phase.

-test : Tests sentences with the same SVM setup to predict their class. Keep in mind that these sentences need to be in their Pipeline-processed form in the first place. (Default folder to choose from : "apiresults/test")


"split_train_test2.py" allows you to separate the results (Default folder : "apiresults/") into the train and test folders. a percentage argument has to be specified to determine how many of the results should be taken in total. The file itself can also be edited to change the ratio of comparative or non-comparative results that are chosen.
