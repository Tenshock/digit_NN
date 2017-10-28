# -*- coding: utf-8 -*-

import random, math
import numpy as np

__author__ = "Cédric Prezelin"
__email__ = "cedric.prezelin@outlook.com"

"""
Fonction d'activation de neurone ReLu
value = l'activation des neurones précédents
weight = poids de chaque neurone précédent
Value et weight sont des vecteurs
"""


def activationReLu(values, weights):
    """
    result = 0
        for item in range(0, len(value)):
            result += value[item]*weight[item]
        return max(0, result)
    """
    return 1 / (1 + math.exp(-sum([w*a for w, a in zip(weights, values)])))


def activationFinale(value, weight):
    result = 0
    for item in range(0, len(value)):
        result += value[item]*weight[item]
    return result


"""Déclaration du réseau de neurones"""


nbr_hidden_layers = 6
nbr_entree = 7
nbr_neurones = 12
nbr_sortie = 1

"""
Génère une matrice à 3 dimensions de poids aléatoires compris entre 0 et 1
5 décimales ("%.3f")
"""


def poidsAleatoires3(nbr_couches, nbr_neurones, nbr_entree):
    poids = np.zeros((nbr_couches, nbr_neurones, nbr_entree))
    for i in range(0, nbr_couches):
        for j in range(0, nbr_neurones):
            for k in range(0, nbr_entree):
                poids[i, j, k] = float("%.3f" % random.uniform(-0.01, 0.01))
    return poids

def poidsAleatoires2(nbr_neurones, nbr_entree):
    poids = np.zeros((nbr_neurones, nbr_entree))
    for j in range(0, nbr_neurones):
        for k in range(0, nbr_entree):
            poids[j, k] = float("%.3f" % random.uniform(-0.01, 0.01))
    return poids


"""
Instancie un réseau de neurones
nbr_entree = nombre de valeurs en entrée
nbr_neurones = nombre de neurones dans chaque couche cachée
nbr_sortie = nombre de sortie possible
poids_first_hidden_layer = Poids correspondant à la première couche cachée
poids_following_hidden_layers = Poids correspondant aux couches cachées suivantes. Indexation : [N°Layer, N° neurone, poids Nème neurone précédent]
poids_couche_sortie = Poids correspondant à la couche de sortie
biais_hidden_layers = vecteur regroupant tous les biais Indexation : [0, N° couche]
activation_hidden_layers = Résultat de l'activation des neurones des couches cachées. Indexation : [N° de couche, N° neurone]
resultat = vecteur de résultat du réseau de neurones
"""


class NeuralNetwork:

    def __init__(self):
        self.nbr_entre = nbr_entree
        self.nbr_hidden_layers = nbr_hidden_layers
        self.nbr_neurones = nbr_neurones
        self.nbr_sortie = nbr_sortie

        self.poids_first_hidden_layer = poidsAleatoires2(nbr_neurones, nbr_entree)
        self.poids_following_hidden_layers = poidsAleatoires3(nbr_hidden_layers-1, nbr_neurones, nbr_neurones+1)
        self.poids_couche_sortie = poidsAleatoires2(nbr_sortie, nbr_neurones+1)
        self.biais_hidden_layers = np.zeros((1, nbr_hidden_layers))

        self.activation_hidden_layers = np.zeros((nbr_hidden_layers, nbr_neurones + 1))
        self.resultat = np.zeros((1, nbr_sortie))

    """
    Forward propagation
    """

    def calcul(self, image):
        for neurone in range(0, nbr_neurones + 1):
            for poids in range(0, nbr_entree):
                if neurone == 0:
                    self.activation_hidden_layers[0, neurone] = self.biais_hidden_layers[0, 0]
                else:
                    self.activation_hidden_layers[0, neurone] = activationReLu(image,
                                                                               self.poids_first_hidden_layer[neurone-1, :])

        for layer in range(1, nbr_hidden_layers):
            for neurone in range(0, nbr_neurones+1):
                for poids in range(0, nbr_neurones+1):
                    if neurone == 0:
                        self.activation_hidden_layers[layer, neurone] = self.biais_hidden_layers[0, layer]
                    else:
                        self.activation_hidden_layers[layer, neurone] = activationReLu(self.activation_hidden_layers[layer-1, :],
                                                                                       self.poids_following_hidden_layers[layer-1, neurone-1, :])

        for neurone in range(0, nbr_sortie):
            for poids in range(0, nbr_neurones+1):
                self.resultat[0, neurone] = activationFinale(self.activation_hidden_layers[nbr_hidden_layers - 1, :], self.poids_couche_sortie[neurone, :])



