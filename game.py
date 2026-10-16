import math
import os
import pygame
from car import Car
import random
from neuronalNetwork import NeuronalNetwork
import time
from pathlib import Path
import threading
from cliInterface import CliInterface

from performanceMeasurement import PerformanceMeasurement
# pygame setup
pygame.init()

display_width = 1280
display_height = 720

car_start_x = 50
car_start_y = 400 # 100

fpsCount = 60
episode_save_interval = 40000

max_allowed_steps = 60

measuring_peformance = False

screen = pygame.display.set_mode((display_width, display_height))
clock = pygame.time.Clock()
render_car = False

nn_steering_angles = [
    -20,
    -15,
    -10,
    -5,
    -4,
    -3,
    -2,
    -1,
    -0.5,
    -0.25,
    0,
    0.25,
    0.5,
    1,
    2,
    3,
    4,
    5,
    10,
    15,
    20,
]

nn_acceleration_forces = [
    1,
    0,
    -1
]

SPEED = 8 

debug = False

logging = False

measurement = PerformanceMeasurement()
neuronal = NeuronalNetwork(4, 8, 24, episode_save_interval, measurement)

episodes = input("Input the episode number of the model to load or press Enter: ")

def call_nn_test(input_vars):

    neurons = neuronal.linear_layer(neuronal.weights_input_hidden, input_vars, neuronal.input_bias)
    activated_neurons = neuronal.relu(neurons)
    second_layer = neuronal.linear_layer(neuronal.weights_output_hidden, activated_neurons, neuronal.output_bias)

    steering_softmax = neuronal.softmax(second_layer)
    option = steering_softmax.index(max(steering_softmax))

    steering = nn_steering_angles[option]

    print("For input: ", end="")
    [print(str(f) + ", ", end="") for f in input_vars]
    print("Softmax: ")
    print(steering_softmax)
    print(f"choose {steering} out of: ")
    [print(str(s)) for s in nn_steering_angles]
    


def runDebugMode():
    NnInput = input("Input your desired coords (distance, cos angle, sin angle): ")
    try:
        inputVarsNn =  [int(n.strip()) for n in NnInput.strip().split(",")]
        inputVarsNn[1] = math.cos(math.radians(inputVarsNn[1]))
        inputVarsNn[2] = math.sin(math.radians(inputVarsNn[2]))
        call_nn_test(inputVarsNn)

    except:  # noqa: E722
        print("wrong input please input 3 values seperated by a comma please only input numbers")


if episodes:
    load = Path(__file__).resolve().parent / "models" / f"model-{episodes}.json"
    if load.exists():
        neuronal.load_model(load)
        print("Saved model will be loaded")
        neuronal.episode = int(episodes) + 1
        measurement.starting_episode = neuronal.episode
    else:
        print("New model with new weights")

while True:
    fpsCountInput = input("Do you want to run it in trainings-mode (train), test mode (test), (debug), or accuracy measuring mode (amm): ")
    if fpsCountInput.lower() == "train":
        fpsCount = 600000
        break
    elif fpsCountInput.lower() == "test":
        logging = True
        render_car = True        
        fpsCount = 60
        break
    elif fpsCountInput.lower() == "debug":
        debug = True
        neuronal.running = False
        runDebugMode()
    elif fpsCountInput.lower() == "amm":
        fpsCount = 600000
        measuring_peformance = True
        neuronal.measuring_peformance = True
        break
    else:
        print("wrong input")
    

car = Car(pygame, screen, car_start_x, car_start_y, SPEED)

def get_goal_cords():
    threshold_x = 450
    threshold_y = 110

    randcord_x = (display_width - threshold_x) * random.random() + threshold_x - 80
    randcord_y = (display_height - threshold_y) * random.random() + threshold_y * 0.4
    return randcord_x, randcord_y

def clear_angle_error(angle):
    # now we have to normalize it so 350 and 10 are getting the same
    # for that were adding 180 and taking the modulus of 360 so if sth was
    # over 180 at the start it will then be over 360 and with that the modulus takes away the 360
    # and then we're subtracting 180 because we added it bevor
    # with that we're getting only values between -180 and 180
    # BUT IN PYTHON ANGLE IS ALWAYS RADIANT SO using pi
    return ((angle + math.pi) % (math.pi * 2)) - math.pi

def calcSterringPenalty(sterrings):
    if len(sterrings) > 5:
        return 0.22 * abs(sterrings[-1] - sterrings[-2]) / 40 - sterrings[-1] * 0.004
    return 0


def call_nn(goal_x, goal_y, dt):

    previous_distance = math.hypot(
        goal_x - car.x,
        goal_y - car.y
    )

    previous_distance = previous_distance / display_width

    best_distance_reward = 0
    if previous_distance < car.best_distance:
        best_distance_reward = car.best_distance - previous_distance
        car.best_distance = previous_distance
        
    step_penalty = 0.001

    if neuronal.step != 0:
        sterring_penalty = calcSterringPenalty(neuronal.episode_actions[0])

        neuronal.episode_rewards.append(neuronal.episode_distances[neuronal.step - 1]  - previous_distance - step_penalty - sterring_penalty + (best_distance_reward * 5))



    neuronal.episode_distances[neuronal.step] = previous_distance
    

    if neuronal.step >= max_allowed_steps:
        neuronal.episode_rewards[-1] -= 0.2
        return 
    angle_to_goal = math.atan2(goal_y - car.y, goal_x - car.x)

    angle_error = angle_to_goal - math.radians(car.initialAngle)

    angle_error = clear_angle_error(angle_error)

    # using sin and cos that the NN understands the numbers better
    # e.g 190 deg would be -170 and 170 remain 170 that's pretty far away with just looking at
    # a numeric scalar so its easier to use sin and cos so get those angles close together in numbers
    input_vars = [previous_distance, math.cos(angle_error), math.sin(angle_error), car.acceleration, car.steering]
    # if logging:
        # print("distance: " + str(input_vars[0]))
        # print("cos angle: " + str(input_vars[1]))
        # print("sin angle: " + str(input_vars[2]))


    # initialing NN
    # first layer
    neurons = neuronal.linear_layer(neuronal.weights_input_hidden, input_vars, neuronal.input_bias)
    activated_neurons = neuronal.relu(neurons)
    second_layer = neuronal.linear_layer(neuronal.weights_output_hidden, activated_neurons, neuronal.output_bias)

    steering_softmax = neuronal.softmax(second_layer[:len(nn_steering_angles)])
    acceleration_softmax = neuronal.softmax(second_layer[len(nn_steering_angles):len(nn_steering_angles) + len(nn_acceleration_forces)])
    if fpsCountInput.lower() == "train" or fpsCountInput.lower() == "amm":
        option = neuronal.choose_option(steering_softmax)
        acceleration = neuronal.choose_option(acceleration_softmax)
    elif fpsCountInput.lower() == "test":
        render_car = True  # noqa: F841
        option = steering_softmax.index(max(steering_softmax))
        acceleration = acceleration_softmax.index(max(acceleration_softmax))
        # if logging:
            # print(nn_steering_angles[option])

    neuronal.steering = nn_steering_angles[option]

    car_move_valid = car.move(neuronal.steering, dt, nn_acceleration_forces[acceleration])
        
    state = input_vars

    cache = {
        "state" : state,
        "hidden_z" : neurons,
        "hidden_activations" : activated_neurons,
        "output_layer" : second_layer,
        "probabilities" : [steering_softmax, acceleration_softmax]
    }

    neuronal.episode_cache.append(cache)

    neuronal.episode_actions.append([option, acceleration])

    neuronal.step += 1

    return car_move_valid 


# randcord_x, randcord_y = get_goal_cords()
neuronal.randcord_x = random.uniform(700, 900) # 800
neuronal.randcord_y = random.uniform(100, 500) # 400

car.best_distance = math.hypot(
    neuronal.randcord_x - car.x,
    neuronal.randcord_y - car.y
)



def main(stop_event):
    frame_count = -1

    while neuronal.running:
        if stop_event.is_set():
            neuronal.running = False
            break 
        clock.tick(fpsCount)
        dt = 1 / 60
        frame_count += 1
        # poll for events
        # pygame.QUIT event means the user clicked X to close your window
        if render_car:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    neuronal.running = False


        if render_car:
            screen.fill("grey")
            pygame.draw.arc(screen, 'green', (neuronal.randcord_x, neuronal.randcord_y, 40, 40), 0, 2 * math.pi, 10)

        #print(randcord_x, randcord_y)


        if abs(car.x - neuronal.randcord_x) < 30 and abs(car.y - neuronal.randcord_y) < 30 and (car.acceleration <= 0.05 and car.acceleration >= -0.05):

            neuronal.episode_rewards.append(1)
            measurement.addTurn(neuronal.episode)
            assert (
                len(neuronal.episode_cache)
                == len(neuronal.episode_actions)
                == len(neuronal.episode_rewards)
            )
            if logging:
                time.sleep(2)
            neuronal.stop_episode()
            car.reset_car()
            frame_count = -1
            continue

        if frame_count % 7 == 0:

            move_result = call_nn(neuronal.randcord_x, neuronal.randcord_y, dt)
            if move_result is None:
                assert (
                    len(neuronal.episode_cache)
                    == len(neuronal.episode_actions)
                    == len(neuronal.episode_rewards)
                )

                neuronal.stop_episode()
                car.reset_car()
                frame_count = -1

            elif move_result is False:

                neuronal.episode_rewards.append(-1)

                assert (
                    len(neuronal.episode_cache)
                    == len(neuronal.episode_actions)
                    == len(neuronal.episode_rewards)
                )

                neuronal.stop_episode()
                car.reset_car()
                frame_count = -1

        else:
            if not car.move(neuronal.steering, dt):
                neuronal.episode_rewards.append(-1)

                assert (
                    len(neuronal.episode_cache)
                    == len(neuronal.episode_actions)
                    == len(neuronal.episode_rewards)
                )

                neuronal.stop_episode()
                car.reset_car()
                frame_count = -1

        if render_car:
            car.draw()
            pygame.display.flip()

stop_event = threading.Event()

main_thread = threading.Thread(target=main, args=(stop_event,))

main_thread.start()

if measuring_peformance:
    cli = CliInterface(measurement, neuronal)
    cli.display()

stop_event.set()


#    keys = pygame.key.get_pressed()
#
#    if keys[pygame.K_w]:
#        car.move(1, 0, dt)
#    elif keys[pygame.K_s]:
#        car.move(-1,  0, dt)
#    if keys[pygame.K_d]:
#        car.move(0.5, 45, dt)
#    elif keys[pygame.K_a]:
#        car.move(0.5, -45, dt)


    # flip() the display to put your work on screen


pygame.quit()