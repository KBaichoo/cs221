# Super Hexagon Bot

Peter Do (peterhdo) and Kevin Baichoo (kbaichoo)

This repo contains all the data, models, and bootstrapping infrastructure to use neural networks to play Super Hexagon. You do need to have Super Hexagon installed, however.

Our models are in the model.py functions (7_layer_model.py, resnet_model.py, model.py, etc.). 

Main.py is used to play the game with the network in a loop.

Our data is stored in frames, where we've split up the training, validation, and test sets. Everything else was used for generating analysis of our results (saliency_map, rotation_experiment, etc.).

There are some helpful functions such as parse_results.py to create a graph of results over time and test_and_cm.py which uses a trained model to generate a confusion matrix. 
