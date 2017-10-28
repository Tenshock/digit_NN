# -*- coding: utf-8 -*-

from NN import NeuralNetwork
from backpropagation_v2 import backpropagation

__author__ = "Cédric Prezelin"
__email__ = "cedric.prezelin@outlook.com"


def trainNN():
    images = [[1, 1, 1, 1, 1, 1, 0],
              [0, 1, 1, 0, 0, 0, 0],
              [1, 1, 0, 1, 1, 0, 1],
              [1, 1, 1, 1, 0, 0, 1],
              [0, 1, 1, 0, 0, 1, 1],
              [1, 0, 1, 1, 0, 1, 1],
              [1, 0, 1, 1, 1, 1, 1],
              [1, 1, 1, 0, 0, 0, 1],
              [1, 1, 1, 1, 1, 1, 1],
              [1, 1, 1, 1, 0, 1, 1]]

    y = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    NN = NeuralNetwork()

    for epoch in range(10):
        for image in images:
            NN.calcul(image)
            NN.poids_first_hidden_layer, NN.poids_following_hidden_layers, NN.poids_couche_sortie = backpropagation(image,
                                                                                                                    NN.poids_first_hidden_layer,
                                                                                                                    NN.poids_following_hidden_layers,
                                                                                                                    NN.poids_couche_sortie,
                                                                                                                    NN.resultat,
                                                                                                                    NN.activation_hidden_layers,
                                                                                                                    y,
                                                                                                                    NN.nbr_hidden_layers,
                                                                                                                    NN.nbr_neurones)

    print(NN.resultat)


trainNN()
