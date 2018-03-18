import numpy as np
from pprint import pprint


def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

def sigmoid_prime_rawvalue(z): #We already give calculated value
    return z*(1-z)


class Cost:
    @staticmethod
    def calc(neuron: np.array, expected: np.array) -> np.array:
        raise NotImplementedError

    @staticmethod
    def derative(neuron: np.array, expected: np.array) -> np.array:
        raise NotImplementedError


class QuadraticCost(Cost):
    @staticmethod
    def calc(neuron: np.array, expected: np.array) -> np.array:
        return np.square(neuron - expected)

    @staticmethod
    def derative(neuron: np.array, expected: np.array) -> np.array:
        return 2 * (neuron - expected)


class ECost(Cost):
    @staticmethod
    def calc(neuron: np.array, expected: np.array) -> np.array:
        return np.square(np.e**neuron - np.e**expected)


class Neuron:
    def __init__(self, index: int, is_input: bool = False, parents: list = None, char: str = None):
        if not is_input:
            self.parents = parents
            self.weights = np.random.random(len(parents))
            self.bias = np.random.random()
            self.delta_weights = np.array([0.0] * len(parents))


        self.is_input = is_input
        self.value = 0.0
        self.hidden_value = 0.0
        self.index = index
        self.char = char

    def __mul__(self, other):
        return self.value * other

    def is_son(self, neuron):
        return neuron in self.parents

    def connected_weight(self, neuron) -> float:
        return self.weights[self.parents.index(neuron)]

    @property
    def wxb(self) -> float:
        return np.sum(self.parents * self.weights) #+ self.bias FIXME

    @property
    def parents_input(self) -> float:
        return np.sum(np.array([neuron.value for neuron in self.parents]))

    @property
    def uid(self) -> str:
        to_print = "%.2f" % self.value
        return f"{self.name}-{to_print}"

    def update_value(self) -> None:
        self.hidden_value = float(self.wxb)
        self.value = float(sigmoid(self.wxb))
        #print(self.value)

    @property
    def name(self):
        return f"{self.char}{self.index}"


class NeuronLayer:
    def __init__(self, index: int, neuron_count: int,
                 parent=None,
                 cost: Cost = QuadraticCost):
        char = chr(65+index)
        if index == 0:
            self.neurons = [Neuron(_, True, char=char) for _ in range(neuron_count)]

        else:
            self.neurons = [Neuron(_, parents=parent.neurons, char=char) for _ in range(neuron_count)]

        self.parent = parent
        self.is_input = (index == 0)
        self.index = index
        self._cost = cost

    def cost(self, expected: np.array) -> float:
        return float(np.sum(self._cost.calc(np.array([neuron.value for neuron in self.neurons]), expected)))

    def input(self, input_data: np.array) -> None:
        if not self.is_input:
            raise TypeError("Layer is not an input layer!")
        for neuron, value in zip(self.neurons, input_data):
            neuron.value = value

    def update(self) -> None:
        for neuron in self.neurons:
            neuron.update_value()

    def visualize(self, graph):
        for neuron in self.neurons:
            graph.node(f"{neuron.name} = {round(float(neuron.value), 2)}", shape='circle', style="filled",
                       fillcolor="#"+3*f"{int(round(float(neuron.value*255))):0>2x}")

    def update_weight_bias(self, weight: np.array, bias: np.array) -> None:
        for neuron, new_data, bias_ in zip(self.neurons, weight, bias):
            neuron.weights += np.array(new_data)
            neuron.bias += float(bias_)
            print(f"DEBUG: {neuron.name} Updated!")


class LayerNetwork:
    def __init__(self, size: list, cost: Cost = QuadraticCost):
        self.cost = cost

        self.size = size

        layer = None
        layers = []
        for index, layer_size in enumerate(size):
            layer = NeuronLayer(index, layer_size, parent=layer, cost=cost)
            layers.append(layer)

        self.layers = layers

    def input(self, data: np.array) -> None:
        self.layers[0].input(data)
        for layer in self.layers[1:]:
            #print(f"DEBUG: Layer {layer.index} is updating values...")
            layer.update()

    def out_matrix(self, layer):
        weights = []
        if(not layer == self.layers[-1]):
            forward_layer = self.layers[self.layers.index(layer)+1]
            for neuron in forward_layer.neurons:
                for i in range(len(neuron.weights)): #FIXME needed?
                    weights.append(neuron.weights[i])
        return weights

    #input of given layer
    def input_matrix(self, layer):
        value = []
        previous_index = self.layers.index(layer)-1
        if(previous_index >= 0):
            for neuron in self.layers[previous_index].neurons:
                value.append(neuron.value)
        return value

    def hidden_layer_matrix_prime(self, layer):
        hidden_value = []
        for neuron in layer.neurons:
            hidden_value.append(sigmoid_prime(neuron.hidden_value))
        return hidden_value

    def save(self, fn) -> None:
        import pickle

        with open(fn, "wb") as f:
            pickle.dump(self, f)

    def load(self, fn) -> None:
        import pickle

        with open(fn, "rb") as f:
            pickle.load(f)

    def get_output(self) -> np.array:
        return np.array([neuron.value for neuron in self.layers[-1].neurons])

    def get_cost(self, expected: np.array) -> float:
        return self.layers[-1].cost(expected)

    def get_gradient(self, target: np.array, a: float = 0.000001) -> list:
        delta = {}
        errors = {}
        totalError = 0
        delta_output_sum = 0
        for target, calculated in zip(target,self.get_output()):
            totalError += target-calculated
        delta_output_sum = sigmoid_prime(self.layers[-1].neurons[0].hidden_value) * totalError
        delta_hidden_sum = []


        for layer in self.layers[::-1]:
            for neuron in layer.neurons:
                #OUT LAYER
                if self.layers[-1] == layer:
                    for i in range(len(neuron.weights)):
                        neuron.delta_weights[i] = delta_output_sum / neuron.parents[i].value
            
            #MIDDLE LAYER
            hidden_layer_logic = layer != self.layers[-1] and layer != self.layers[0]
            if(hidden_layer_logic):
                weights = delta_output_sum / self.out_matrix(layer)
                hidden_layer =  self.hidden_layer_matrix_prime(layer)
                delta_hidden_sum = weights * hidden_layer

                delta_weights = []
                for _input in self.input_matrix(layer):
                    for hid_sum in delta_hidden_sum:
                        delta_weights.append(hid_sum/_input)

                for neuron in layer.neurons:
                    for i in range(len(neuron.weights)):
                        neuron.delta_weights[i] = delta_weights.pop(0)



            #INPUT LAYER



                # if self.layers[-1] == layer:
                #     errors[neuron.name] = (target[neuron.index] - neuron.value)
                # else:
                #     error = 0
                #     for son_neuron in self.layers[layer.index+1].neurons:
                #         if son_neuron.is_son(neuron):
                #             error += errors[son_neuron.name] * son_neuron.connected_weight(neuron)
                #     errors[neuron.name] = error
                # #print(f"error for {neuron.name} : {errors[neuron.name]}")
                # delta[neuron.name] = errors[neuron.name] * sigmoid_prime(neuron.value)
                #print(f"delta for {neuron.name} : {delta[neuron.name]}")

                #print(f"Err {neuron.name} : {errors[neuron.name]}")
                #print(f"Delta {neuron.name} : {delta[neuron.name]}")

        return delta



    def visualize(self, fn: str = None, label: str = None) -> None:
        import itertools
        import graphviz

        graph = graphviz.Digraph(format="png")
        graph.attr(rankdir="LR", ranksep="4", label=label)

        for layer in self.layers:
            layer.visualize(graph)

        for old, new in zip(self.layers[:-1], self.layers[1:]):
            for old_, new_ in itertools.product(old.neurons, new.neurons):
                graph.edge(f"{old_.name} = {round(float(old_.value), 2)}",
                           f"{new_.name} = {round(float(new_.value), 2)}",
                           label=str(new_.weights[old.index]))

        if fn is None:
            graph.view()
        else:
            graph.render(fn)

    #Renders the neuron's name with value and weights
    def print(self) -> None:
        from graphviz import Digraph
        f = Digraph('ANN', filename='ann', format="png")
        f.attr(rankdir='LR', size='8.5')
        f.attr('node', shape='circle')
        for layer in self.layers:
            for neuron in layer.neurons:
                f.node(neuron.uid)
                if(not neuron.is_input):
                    for i in range(len(neuron.weights)):
                        we = "%.2f" % neuron.weights[i]
                        f.edge(neuron.parents[i].uid, neuron.uid, label=we)
        f.view()


    def feed(self, data: np.array, expected: np.array = None, iter: int=0) -> np.array:
        self.input(data)
        out = self.get_output()
        if(iter%10 == 0):
            print(f'Iter{iter} Data: {data} \nExpected: {expected}  \nOutput:{out}')

        if expected is not None:
            self.backprop(expected)

        return out

    def backprop(self, expected: np.array):
            #print(f"INFO: Cost: {self.get_cost(expected)}")
            grad = self.get_gradient(expected)
            #for weight, bias_, layer in zip(grad[::-1], bias[::-1], self.layers[1:]):
            #    layer.update_weight_bias(weight, bias_)
            alpha = 0.01
            for layer in self.layers[:0:-1]:
                for neuron in layer.neurons:
                    for i in range(len(neuron.weights)):
                        neuron.weights[i] += neuron.delta_weights[i] * alpha

            # for layer in self.layers[:0:-1]:
            #     for neuron in layer.neurons:
            #         for i in range(len(neuron.weights)):
            #             neuron.weights[i] += alpha * grad[neuron.name] * neuron.parents_input
            #             if(neuron.name == "D70"):
            #                 a=2
                            #print(f"INFO: {grad[neuron.name] * neuron.parents_input} -- {grad[neuron.name]} * {neuron.parents_input}")
                        #print(f"{neuron.name}")

    def get_response(self, data: np.array) -> tuple:
        out = self.feed(data)
        pos = np.argmax(out)
        acc = out[pos]
        return pos, float(acc)


def main():
    #priorStatus()
    #validation()
    #print(sigmoid(1.3))
    simpleEx()

def simpleEx():
    net = LayerNetwork([2, 3, 1])
    net.layers[1].neurons[0].weights[0] = 0.8
    net.layers[1].neurons[0].weights[1] = 0.2
    net.layers[1].neurons[1].weights[0] = 0.4 
    net.layers[1].neurons[1].weights[1] = 0.9
    net.layers[1].neurons[2].weights[0] = 0.3
    net.layers[1].neurons[2].weights[1] = 0.5
    net.layers[2].neurons[0].weights[0] = 0.3
    net.layers[2].neurons[0].weights[1] = 0.5
    net.layers[2].neurons[0].weights[2] = 0.9
    for i in range(1000):
        i_1 = np.random.random()
        if(i_1 > 0.5):
            i_1 = 1
        else:
            i_1 = 0.001
        i_2 = np.random.random()
        if(i_2 > 0.5):
            i_2 = 1
        else:
            i_2 = 0.001
        out = 0
        if(i_1 == 1 and i_2 == 1):
            out = 1
        if(i_1 == 0 and i_2 == 0):
            out = 1
        net.feed(np.array([i_1,i_2]), np.array([out]))


    net.print()





def layerEx():
    inputLayer = NeuronLayer(0, 3, 50)
    layer1 = NeuronLayer(1, 5, inputLayer,10)
    inputLayer.input([1,2,3])
    layer1.update()

def neuronExper():
    inputNeuron = Neuron(1, True, char='a')
    firstNeuron = Neuron(2, parents=[inputNeuron], char='b')

    inputNeuron.value = 5
    firstNeuron.update_value()
    #print(f"{firstNeuron.value}")   

def validation():
    net = LayerNetwork([10, 50 , 100])

    for i in range(10000):
        x = int(np.random.random()*10)
        data = np.array([0.0]*10)
        data[x] = 1 
        y = functionToReplicate(x)
        expected = np.array([0.0]*100)
        expected[y] = 1
        net.feed(data, expected, i)



def functionToReplicate(x: int) -> int:
    return x*x


def priorStatus():
    net = LayerNetwork([784, 16, 16, 10])

    import json
    with open("mnist.json") as f:
        train_data = json.load(f)

    print(train_data[0])

    for i, data in enumerate(train_data[:1000]):
        data['data'] = np.array(data['data']) / 255
        res = np.array([0.0]*10)
        res[int(data['label'])] = 1.0
        net.feed(data['data'], res)
        print(f"Iteration: {i}")

        # net.save("mnist.nn")
        # net.visualize(fn=f"iters/iteration_{i}.gv", label=data['label'])
        # print("done rendering")

    costs = []

    for i, data in enumerate(train_data[:50]):
        data['data'] = np.array(data['data']) / 255
        res = np.array([0.0]*10)
        res[int(data['label'])] = 1.0
        pos, accuracy = net.get_response(data['data'])
        costs.append(net.get_cost(res))
        print(f"Guess: {pos}\nCertainty: {accuracy}\nCorrect: {data['label']}\n-------------------------------")

    print(f"Final cost: {sum(costs)/len(costs)}")

    print("NET STATE")
    for layer in net.layers:
        print(f"LAYER {layer.index}")
        print([n.value for n in layer.neurons])

    return net


if __name__ == "__main__":
    main()
#deltaChange = eTotal regard output * out regard Net * net regard weight
                # err = []
                # weights = []
                # #out - target
                # errorRegardingOutput = neuron.value - target[neuron.index] ##external layer!!!! target
                # #how output change respect total net input
                # outputRegardingNet = sigmoid_prime_rawvalue(output_network[neuron.index])
                # #how the weight is influencing
                # inputRegardingWeight = neuron.parentsInput

                # weight_updates[neuron.index] = (errorRegardingOutput * outputRegardingNet * inputRegardingWeight) * a

                # errors[neuron.index] = (target[neuron.index] - neuron.value) * sigmoid_prime_rawvalue(neuron.value)

                # errors[neuron.index] = ()


                # for prev_neuron in layer.parent.neurons:
                #     if layer == self.layers[-1]:
                #         # d_N = f'(x) * (yN - t)
                #         d_n = sigmoid_prime(prev_neuron.value) * self.cost.derative(neuron.value, output[neuron.index])

                #         errors.append(float(d_n))
                #         weights.append(-a * float(d_n))
                #     else:
                #         # d_n = f'(x) * d_n+1 * w_n
                #         next_layer = errors[len(self.layers) - layer.index - 2]
                #         print(f"{next_layer}")
                #         next_layer_neuron_weight = np.sum([next_neuron[neuron.index]
                #                                            for next_neuron in next_layer])
                #         d_n = (sigmoid_prime(prev_neuron.value) *
                #                next_layer_neuron_weight *
                #                neuron.weights[prev_neuron.index])

                #         errors.append(float(d_n))
                #         weights.append(float(-a * next_layer_neuron_weight * neuron.value))

        #         weight_neurons.append(weights)

        #     errors.append(neurons)
        #     weight_updates.append(weight_neurons)

        # bias_updates = []
        # for layer in self.layers[:0:-1]:
        #     neurons = []
        #     for neuron in layer.neurons:
        #         """
        #         if layer == self.layers[-1]:
        #             # d_N = f'(x) * (yN - t)
        #             d_n = sigmoid_prime(prev_neuron.value) * (neuron.value - output[neuron.index])

        #             neurons.append(-a * float(d_n))
        #         else:
        #             # d_n = f'(x) * d_n+1 * b_n
        #             next_layer = bias_updates[len(self.layers) - layer.index - 2]
        #             inv_a = -1 / a
        #             next_layer_neuron_weight = np.sum(
        #                 inv_a * next_neuron for next_neuron in next_layer)
        #             d_n = (sigmoid_prime(prev_neuron.value) *
        #                    next_layer_neuron_weight *
        #                    neuron.bias)
        #             neurons.append(-a * float(d_n))
        #         """
        #         neurons.append(0.0)

        #     bias_updates.append(neurons)