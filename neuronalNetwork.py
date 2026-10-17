import random
import math
import json


class NeuronalNetwork:
    def __init__(self, input_count, hidden_count, output_count, episode_save_interval, performanceMeasurement, test_mode = False):
        self.input_count = input_count
        self.hidden_count = hidden_count
        self.output_count = output_count

        self.performanceMeasurement = performanceMeasurement

        self.test_mode = test_mode

        self.running = True

        self.measuring_peformance = False

        self.episode_save_interval = episode_save_interval

        # creates the weights for the model
        # here the first value is how many lists are in the list and the second how many weights in this list

        self.weights_input_hidden = self.make_matrix(self.hidden_count, self.input_count)

        # A matching Bias for all the neurons so for hidden count here
        self.input_bias = [0.0] * self.hidden_count

        self.weights_output_hidden = self.make_matrix(self.output_count, self.hidden_count)

        self.output_bias = [0.0] * self.output_count

        self.step = 0

        self.episode = 0

        self.decision_freq = 3

        self.episode_cache = []
        self.episode_actions = []
        self.episode_rewards = []
        self.episode_distances = {}

        self.randcord_x = 0
        self.randcord_y = 0

        self.steering = 0

    def get_random_list(self, count):
        return [random.uniform(-0.5, 0.5) for i in range(count)]

    def make_matrix(self, row, colum):
        return [self.get_random_list(colum) for i in range(row)]

    def linear_layer(self, weights, inputs, biases):
        output = []
        # iterate through all the neurons (weights)
        for neuron_index in range(len(weights)):
            neurone_sum = biases[neuron_index]
            for weight_index in range(len(weights[neuron_index])):
                # multiplying the weights with the input and add the bias
                neurone_sum += (weights[neuron_index][weight_index] * inputs[weight_index])
            output.append(neurone_sum)

        return output

    def relu(self, neurons):
        return [n if n > 0 else 0 for n in neurons]

    # this goes backwards trough the rewards and adds up the returns
    # the higher the gamma the more important will the last steps be
    # rewards is a list of the previous rewards for the Ai
    def calculate_returns(self, rewards, gamma):
        calc_return_rewards = [0.00] * len(rewards)
        current_return = 0
        for r in range(len(rewards) - 1, -1, -1):
            calc_return_rewards[r] = (rewards[r] + gamma * current_return)
            current_return = calc_return_rewards[r]
        return calc_return_rewards


    # function for spreading equal to get probability
    def softmax(self, soft_list):
        # we can subtract the biggest value because if we dont the values get to big
        max_val = max(soft_list)
       # print("maximun is ", max_val)
        exp_vals = []
        for li in soft_list:
            # we can shift the logit because only the difference matters between the values not the absolute value
            shifted_logit = li - max_val
            exp_vals.append(math.exp(shifted_logit))  # e ^ shifted_logit -> difference between the values bigger (natural logaritmus)
        sum_vals = sum(exp_vals)
        return [e / sum_vals for e in exp_vals]

    def choose_option(self, probabilities):
        return random.choices(list(range(len(probabilities))), weights=probabilities)[0]

    def make_zero_matrix(self, rows, columns):
        matrix = []
        for r in range(rows):
            matrix.append([])
            for c in range(columns):
                matrix[r].append(0.0)
        return matrix

    def backpropagation(self):
        returns = self.calculate_returns(self.episode_rewards, 0.99)
        #print("Overall run calculated")
        #print(returns)

        # la = len(self.episode_actions)
        # lc = len(self.episode_cache)
        # lr = len(self.episode_rewards)

        # print(la,lc,lr)
        episode_output_delta = []
        for i in range(len(self.episode_actions)):
            steering_action = self.episode_actions[i][0]
            acceleration_action = self.episode_actions[i][1]
            episode_return = returns[i]
            steering_probs = self.episode_cache[i]['probabilities'][0]
            acceleration_probs = self.episode_cache[i]['probabilities'][1]
            # print("acceratiuon probs")
            # print(len(acceleration_probs))

            acceleration_deltas = []

            steering_deltas = []

            for k, p in enumerate(steering_probs):
                # p - k == action esantially means if the prediction was low and it came out true and the return was postive it will return a high negativ number (what increases the weights)
                # if p is high and k came out true than the increase will be less because the weight is already really likely and must not be increased drasticly
                steering_deltas.append(episode_return * (p - (k == steering_action)))


            for k, p in enumerate(acceleration_probs):
                # p - k == action esantially means if the prediction was low and it came out true and the return was postive it will return a high negativ number (what increases the weights)
                # if p is high and k came out true than the increase will be less because the weight is already really likely and must not be increased drasticly
                acceleration_deltas.append(episode_return * (p - (k == acceleration_action)))
                # print("appending in acceration props run")
                # print(acceleration_deltas)

            deltas = steering_deltas + acceleration_deltas 
            # print("len of steerin delta")
            # print(len(steering_deltas))
            # print(len(acceleration_deltas))        
            episode_output_delta.append(deltas)
        #print("episode")
        #print(episode_output_delta)
        #print("end episode")

        grad_weights_hidden_output = self.make_zero_matrix(self.output_count, self.hidden_count)

        grad_weights_hidden_output = self.output_gradient(grad_weights_hidden_output, episode_output_delta)


        episode_output_bias_gradient = [0.0] * self.output_count
        for i in range(len(episode_output_delta)):
            for k, o in enumerate(episode_output_delta[i]):
                episode_output_bias_gradient[k] += o

        all_hidden_delta = []
        for i in range(len(episode_output_delta)):
            hidden_errors = self.calc_hidden_errors(episode_output_delta[i])
            hidden_z = self.episode_cache[i]['hidden_z']
            all_hidden_delta.append(self.calc_hidden_delta(hidden_z, hidden_errors))
        

        grad_weights_input_hidden = self.make_zero_matrix(
            self.hidden_count,
            self.input_count
        )

        for i in range(len(all_hidden_delta)):
            state = self.episode_cache[i]['state']
            hidden_deltas = all_hidden_delta[i]

            for j in range(self.hidden_count):
                for m in range(self.input_count):
                    grad_weights_input_hidden[j][m] += (
                        hidden_deltas[j] * state[m]
                    )
        
        episode_input_bias_gradient = [0.0] * self.hidden_count

        for i in range(len(all_hidden_delta)):
            for j in range(self.hidden_count):
                episode_input_bias_gradient[j] += all_hidden_delta[i][j]

        # adjusting weights now

        learning_rate = 0.001

        for k in range(self.output_count):
            self.output_bias[k] -= (
                learning_rate *
                episode_output_bias_gradient[k]
            )

        for k in range(self.output_count):
            for j in range(self.hidden_count):
                self.weights_output_hidden[k][j] -= (
                    learning_rate *
                    grad_weights_hidden_output[k][j]
                )

        for k in range(self.hidden_count):
            self.input_bias[k] -= (
                learning_rate *
                episode_input_bias_gradient[k]
            )

        for k in range(self.hidden_count):
            for j in range(self.input_count):
                self.weights_input_hidden[k][j] -= (
                    learning_rate *
                    grad_weights_input_hidden[k][j]
                )

    def stop_episode(self):
        self.backpropagation()
        self.episode_cache.clear()
        self.episode_actions.clear()
        self.episode_rewards.clear()
        self.step = 0
        self.episode_distances.clear()
        self.steering = 0
        self.episode += 1
        if not self.measuring_peformance:
            print(f"Episode {self.episode}")
        
        if self.episode % self.episode_save_interval == 0:
            if not self.test_mode:
                self.save_model(f"models/model-{self.episode}.json")
                if not self.measuring_peformance:
                    print("model saved")


        self.randcord_x = random.uniform(600, 1100) # 800
        self.randcord_y = random.uniform(50, 640) # 400


        

        # self.randcord_x = random.uniform(600, 900) # 800
        # self.randcord_y = random.uniform(300, 500) # 400


    def save_model(self, filename):
        data = {
            "weights_input_hidden": self.weights_input_hidden,
            "input_bias": self.input_bias,
            "weights_output_hidden": self.weights_output_hidden,
            "output_bias": self.output_bias,
        }

        with open(filename, "w", encoding="utf-8") as file:
            json.dump(data, file)

    def load_model(self, filename):
        with open(filename, "r", encoding="utf-8") as file:
            data = json.load(file)

        self.weights_input_hidden = data[
            "weights_input_hidden"
        ]
        self.input_bias = data["input_bias"]

        self.weights_output_hidden = data[
            "weights_output_hidden"
        ]
        self.output_bias = data["output_bias"]

    # derivative = ableitung
    def relu_derivative(self, value):
        if value > 0:
            return 1.0
        else:
            return 0.0

    def calc_hidden_delta(self, hidden_z, hidden_errors):
        hidden_delta = [0.0] * len(hidden_errors)
        for j in range(len(hidden_errors)):
            hidden_delta[j] = hidden_errors[j] * self.relu_derivative(hidden_z[j])
        return hidden_delta
           

    def calc_hidden_errors(self, output_deltas):
        hidden_errors = [0.0] * self.hidden_count

        for j in range(self.hidden_count):
            hidden_error_j = 0.0
            for k in range(self.output_count):
                hidden_error_j += (
                    output_deltas[k] *
                    self.weights_output_hidden[k][j]
                )
            hidden_errors[j] = hidden_error_j

        return hidden_errors


    def output_gradient(self, grad_weights_hidden_output, episode_output_delta):
        for i in range(len(self.episode_actions)):

            hidden_activations = self.episode_cache[i]['hidden_activations']
            for k in range(len(grad_weights_hidden_output)):
                # print(k)
                output_delta = episode_output_delta[i][k]

                for j in range(len(grad_weights_hidden_output[k])):
                    grad_weights_hidden_output[k][j] += hidden_activations[j] * output_delta

        return grad_weights_hidden_output 

# neurons = neuronal.linear_layer(neuronal.weights_input_hidden, [2, -1, 4], neuronal.input_bias)
# activated_neurons = neuronal.relu(neurons)
# second_layer = neuronal.linear_layer(neuronal.weights_output_hidden, activated_neurons, neuronal.output_bias)
# softmax = neuronal.softmax(second_layer)
# print(softmax)

# rewards = [0.1, 0.2, 1]
# gamma = 0.9

# return_rewards = neuronal.calculate_returns(rewards, gamma)
# print(return_rewards)


# hidden_errors = [0.2, -0.96]
# hidden_z = [1.5, -0.4]

#print(neuronal.calc_hidden_delta(hidden_z, hidden_errors))