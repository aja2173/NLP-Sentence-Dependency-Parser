README_NLP_Dependency_Parser
aja2173
Alek Anichowski

####
Code Instructions:
There are 3 models saved, NNpart1, NNpart2 and NNpart3.

Right now depModel.py will load a model, which is specified in the model.populate(), and decode using that tree. Make sure that the dimensions of the model layers are all the same, as each model will be different.

To train a new model, the code is in nlphw.py, where you can specify dimensions, minibatch, transfer function etc. At the end, you specify a filename where the new model will be saved.

#####
Part1:

Unlabeled attachment score 83.46
Labeled attachment score 80.24

#####
Part2:

Unlabeled attachment score 83.8
Labeled attachment score 80.48

The accuracy increased because we allowed for more complexity in the model-more nodes in the hidden layers made it more flexible and fit the data better.

######
Part3:

Unlabeled attachment score 84.52
Labeled attachment score 81.05

I increased the nodes in the hidden layers to 600, and decreased the minibatch size to 500.

Since part2 implied more complexity improved the model, the increase of nodes in the hidden layer improved the model. 
Also by decreasing the minibatch size we were able to get more updates, essentially making the model learn at a slower pace and more accurately.
