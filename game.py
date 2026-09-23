import math
import os
import pygame
from car import Car
import random
from neuronalNetwork import NeuronalNetwork
# pygame setup
pygame.init()

display_width = 1280
display_height = 720

car_start_x = 50
car_start_y = 400 # 100

fpsCount = 60
episode_save_interval = 1000

screen = pygame.display.set_mode((display_width, display_height))
clock = pygame.time.Clock()
running = True
max_steering_nn = 20

nn_steering_angles = [
    -max_steering_nn,
    0,
    max_steering_nn
]

SPEED = 5

neuronal = NeuronalNetwork(3, 8, 3, episode_save_interval)

episodes = input("Input the episode number of the model to load or press Enter: ")
if episodes:
    load = f"models\model-{episodes}.json"
    if os.path.exists(load):
        neuronal.load_model(load)
        print("Saved model will be loaded")
        neuronal.episode = int(episodes) + 1
    else:
        print("New model with new weights")

while True:
    fpsCountInput = input("Do you want to run it in trainings-mode (train) or test mode (test): ")
    if fpsCountInput.lower() == "train":
        fpsCount = 600000
        break
    elif fpsCountInput.lower() == "test":
        fpsCount = 60
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


def call_nn(goal_x, goal_y):

    previous_distance = math.hypot(
        goal_x - car.x,
        goal_y - car.y
    )

    previous_distance = previous_distance / display_width

    step_penalty = 0.001

    if neuronal.step != 0:
        neuronal.episode_rewards.append(neuronal.episode_distances[neuronal.step - 1]  - previous_distance - step_penalty)

    neuronal.episode_distances[neuronal.step] = previous_distance
    

    if neuronal.step >= 40:
        neuronal.episode_rewards[-1] -= 0.2
        return 
    angle_to_goal = math.atan2(goal_y - car.y, goal_x - car.x)

    angle_error = angle_to_goal - math.radians(car.initialAngle)

    angle_error = clear_angle_error(angle_error)

    # using sin and cos that the NN understands the numbers better
    # e.g 190 deg would be -170 and 170 remain 170 that's pretty far away with just looking at
    # a numeric scalar so its easier to use sin and cos so get those angles close together in numbers
    input_vars = [previous_distance, math.cos(angle_error), math.sin(angle_error)]


    # initialing NN
    # first layer
    neurons = neuronal.linear_layer(neuronal.weights_input_hidden, input_vars, neuronal.input_bias)
    activated_neurons = neuronal.relu(neurons)
    second_layer = neuronal.linear_layer(neuronal.weights_output_hidden, activated_neurons, neuronal.output_bias)

    softmax = neuronal.softmax(second_layer)
    if fpsCountInput.lower() == "train":
        option = neuronal.choose_option(softmax)
    elif fpsCountInput.lower() == "test":
        option = softmax.index(max(softmax))

    neuronal.steering = nn_steering_angles[option]

    car_move_valid = car.move(1, neuronal.steering, dt)
        
    state = input_vars

    cache = {
        "state" : state,
        "hidden_z" : neurons,
        "hidden_activations" : activated_neurons,
        "output_layer" : second_layer,
        "probabilities" : softmax
    }

    neuronal.episode_cache.append(cache)

    neuronal.episode_actions.append(option)

    neuronal.step += 1

    return car_move_valid 


# randcord_x, randcord_y = get_goal_cords()
neuronal.randcord_x = random.uniform(700, 900) # 800
neuronal.randcord_y = random.uniform(100, 500) # 400
frame_count = -1
while running:
    dt = clock.tick(fpsCount) / 1000 # limits FPS to 60
    frame_count += 1
    # poll for events
    # pygame.QUIT event means the user clicked X to close your window
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False


    # fill the screen with a color to wipe away anything from last frame
    screen.fill("grey")

    # RENDER YOUR GAME HERE
    #print(randcord_x, randcord_y)
    pygame.draw.arc(screen, 'green', (neuronal.randcord_x, neuronal.randcord_y, 40, 40), 0, 2 * math.pi, 10)


    if abs(car.x - neuronal.randcord_x) < 30 and abs(car.y - neuronal.randcord_y) < 30:

        neuronal.episode_rewards.append(1)
        assert (
            len(neuronal.episode_cache)
            == len(neuronal.episode_actions)
            == len(neuronal.episode_rewards)
        )
        neuronal.stop_episode()
        car.reset_car()
        frame_count = -1
        continue

    if frame_count % 4 == 0:

        move_result = call_nn(neuronal.randcord_x, neuronal.randcord_y)
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
        if not car.move(1,neuronal.steering, 60 / 1000):
            neuronal.episode_rewards.append(-1)

            assert (
                len(neuronal.episode_cache)
                == len(neuronal.episode_actions)
                == len(neuronal.episode_rewards)
            )

            neuronal.stop_episode()
            car.reset_car()
            frame_count = -1
    car.draw()



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

    pygame.display.flip()


pygame.quit()