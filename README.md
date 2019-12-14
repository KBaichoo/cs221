# Super Hexagon Bot

Peter Do (peterhdo) and Kevin Baichoo (kbaichoo)

This repo contains all the data, models, and bootstrapping infrastructure to
use neural networks to play Super Hexagon. You do need to have Super Hexagon
installed, however.

Our models are in the model.py functions (7_layer_model.py, resnet_model.py,
model.py, vgg_model.py, etc.).

Main.py is used to play the game with the network in a loop.

Our data is stored in frames, where we've split up the training, validation,
and test sets. Everything else was used for generating analysis of our results
(saliency_map, rotation_experiment, etc.). The raw videos are in video_data.

In scripts you will find handy scripts such as parse_results.py to create a
graph of results over time and test_and_cm.py which uses a trained model to
generate a confusion matrix.

Some NNs parameter files aren't present because the model size is too large
for ordinary github storage.
