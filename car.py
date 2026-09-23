import math

import pygame
import random

class Car:
    def __init__(self, pygame, screen, x, y, SPEED, width=100, height=50, window_width=1280, window_height=720):
        self.screen = screen
        self.pygame = pygame
        self.width = width
        self.height = height
        self.speed = SPEED
        self.wh = window_height
        self.ww = window_width
        self.start_x = x
        self.start_y = y
        self.steering = 0
        self.initialAngle = 0
        self.x = x
        self.y = y

    def draw(self):
        car_image = pygame.Surface((100, 50), pygame.SRCALPHA)

        self.pygame.draw.rect(car_image, "black", (0, 0, car_image.get_width(), car_image.get_height()))
        self.pygame.draw.rect(car_image, "red", (80, 0, 20, 20))
        self.pygame.draw.rect(car_image, "blue", (80, 30, 20, 20))

        rotated = pygame.transform.rotate(car_image, -self.initialAngle)

        rect = rotated.get_rect(center=(self.x,self.y))
        self.screen.blit(rotated, rect)

    # for test purposes is acceleration always either -1 for backwards or 1 for forwards
    def move(self, acceleration, steering, dt):
        wheelbase = 70
        velocity = self.speed * 60 * acceleration
        initial_radiant = math.radians(self.initialAngle)
        steering_angle = math.radians(steering)

        initial_radiant += (velocity / wheelbase) * math.tan(steering_angle) * dt

        new_x = (velocity * math.cos(initial_radiant) * dt) + self.x
        new_y = (velocity * math.sin(initial_radiant) * dt) + self.y

        self.initialAngle = math.degrees(initial_radiant)


        if new_x > self.ww - self.width or new_y > self.wh - self.height or new_x < 0 or new_y < 0:
            return False
        self.x = new_x
        self.y = new_y
        return True

    def change_steering(self, steering):
        self.initialAngle += steering

    def reset_car(self):
        self.start_x = random.uniform(50, 400)
        self.start_y = random.uniform(50, 630)
        self.x = self.start_x
        self.y = self.start_y
        self.initialAngle = 0

    def getRect(self):
        return self.x, self.y, self.width, self.height