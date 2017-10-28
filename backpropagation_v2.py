import math
import numpy as np
def cost_function(predicted_values_v, ground_truth_values_v):
    return 1/2 * sum([(a_i - b_i)* (a_i - b_i) for a_i, b_i in zip(predicted_values_v, ground_truth_values_v)])


learning_rate = 0.1


def activation(weights, activations):
    return 1/ (1 + math.exp(-sum([w*a for w,a in zip(weights, activations)])))


def activation_derivative(output):
    # return 0 if output <= 0 else 1
    return output * (1-output)


def calculate_gradient(unit_output, weights_v, gradients_v):
    # print("Gradient output : " + str(unit_output))
    # [print(str(w) + "*" + str(g)) for w,g in zip(weights_v, gradients_v)]
    return round(activation_derivative(unit_output) * (sum([(w*np.array[g]) for w,g in zip(weights_v, gradients_v)])), 3)

def calculate_output_gradient(predicted_value, ground_truth_value):
    return ground_truth_value - predicted_value, 3

def update_weight(old_weight, input, gradient):
    return old_weight * learning_rate * input * gradient


def forward_propagation():
    inputs = [2, -1]
    y = 1

    weights = [[0.5, 1.5],
               [-1, -2],
               [1, 3],
               [-1, -4],
               [1, -3]
               ]

    h_1_n_1 = activation(weights[0], inputs)
    h_1_n_2 = activation(weights[1], inputs)

    h_2_n_1 = activation(weights[2], [h_1_n_1, h_1_n_2])
    h_2_n_2 = activation(weights[3], [h_1_n_1, h_1_n_2])

    output = activation(weights[4], [h_2_n_1, h_2_n_2])

    # print("H1_N1 : " + str(h_1_n_1))
    # print("H1_N2 : " + str(h_1_n_2))
    #
    # print("H2_N1 : " + str(h_2_n_1))
    # print("H2_N2 : " + str(h_2_n_2))
    #
    # print("OUTPUT : " + str(output))

    return inputs, weights, h_1_n_1, h_1_n_2, h_2_n_1, h_2_n_2, output, y


def backpropagation(inputs, weights_input, weights_hidden, weights_output, target, activation, y, nbr_hidden_layer, nbr_neurones):
    g_output = calculate_output_gradient(target, y)

    last_hidden_gradient = []
    for i in range(len(activation[:-1, :])):
        last_hidden_gradient[i] = calculate_gradient(activation[:-1, i], weights_output[:, i], g_output)

    following_hidden_gradient = np.zeros((nbr_hidden_layer-1, nbr_neurones))
    for layer in reversed(range(nbr_hidden_layer-1)):
        for neuron in range(nbr_neurones):
            if layer == nbr_hidden_layer-1:
                following_hidden_gradient[layer, neuron] = calculate_gradient(activation[layer, neuron], weights_hidden[layer+1, :, neuron], last_hidden_gradient)
            else:
                following_hidden_gradient[layer, neuron] = calculate_gradient(activation[layer, neuron], weights_hidden[layer+1, :, neuron], following_hidden_gradient[layer+1, :])

    first_hidden_gradient = []
    for i in range(len(activation[0, :])):
        first_hidden_gradient[i] = calculate_gradient(activation[0, i], weights_input[:, i], following_hidden_gradient[1, :])

    """
    print(g_output)
    print(last_hidden_gradient)
    print(following_hidden_gradient)
    print(first_hidden_gradient)
    """

    weights_input = weights_input + learning_rate * inputs * first_hidden_gradient

    for layer in range(1, nbr_hidden_layer):
        for neurone in range(0, nbr_neurones + 1):
            for poids in range(0, nbr_neurones + 1):
                weights_hidden[layer, neurone, poids] = weights_hidden[layer, neurone, poids] + learning_rate * inputs * following_hidden_gradient[layer, neurone]

    for neurone in range(0, len(weights_output[:, 0])):
        for poids in range(0, nbr_neurones + 1):
            weights_output[neurone, poids] = weights_hidden[neurone, poids] + learning_rate * inputs * \
                                                                                            following_hidden_gradient[
                                                                                                :-1, neurone]

    return weights_input, weights_hidden, weights_output

