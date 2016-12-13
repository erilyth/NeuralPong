#!/usr/bin/python

import pygame
import random
import sys
from pygame.locals import *
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD

screen_width = 400
screen_height = 300

paddle_length = 60
paddle_length2 = 300
paddle_width = 10
ball_radius = 5
paddle_speed = 4
ball_speed = 2
max_speed = 5

player1_pos = [0, screen_height/2-paddle_length/2] # Changes from 0, 0 to 0, screen_height-paddle_length-1
ball_pos = [screen_width/2, screen_height/2]
ball_velocity = [0, 0]
final_img = np.zeros((screen_height, screen_width))
old_img = np.zeros((screen_height, screen_width))
diff_img = np.zeros((screen_height, screen_width))

# Define a neural network that will be used to predict the output

total_inputs = []
total_class_outputs = []
max_size = 100000

model = Sequential()
model.add(Dense(output_dim=300, input_dim=screen_width*screen_height))
model.add(Activation("tanh"))
model.add(Dense(output_dim=3))
model.add(Activation("softmax"))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

model.load_weights("model.keras")

def retrieve_image(background):
    global final_img
    global old_img
    global diff_img
    old_img = final_img
    final_img = np.zeros((screen_height, screen_width))
    for i in range(-ball_radius, ball_radius+1):
        for j in range(-ball_radius, ball_radius+1):
            if i * i + j * j <= ball_radius * ball_radius:
                if ball_pos[0] + i < screen_width and ball_pos[0] + i >= 0 and ball_pos[1] + j < screen_height and ball_pos[1] + j >= 0:
                    final_img[ball_pos[1]+j][ball_pos[0]+i] = 1
    for i in range(paddle_width):
        for j in range(paddle_length):
            final_img[int(player1_pos[1])+j][int(player1_pos[0])+i] = 1
    diff_img = final_img - old_img
    # Gives us an RGB image
    # final_img = pygame.surfarray.array3d(pygame.display.get_surface())
    # print(final_img)
    """
    # An alternative approach using pixelArrays but its quite slow
    surface_pixels = pygame.PixelArray(background)
    for i in range(0, screen_height):
        for j in range(0, screen_width):
            cur = surface_pixels[j][i]
            pixel = background.unmap_rgb(cur)
            pixel = pixel[0:3]
            final_img.append(pixel)
    """

def initialize_ball():
    global ball_velocity
    ball_velocity = [random.uniform(0.7, 1.1)*ball_speed, random.uniform(0.7, 1.1)*ball_speed]
    if random.uniform(0,1) > 0.5:
        ball_velocity[0] *= -1
    if random.uniform(0,1) > 0.5:
        ball_velocity[1] *= -1
    ball_velocity[0] = int(round(ball_velocity[0]))
    ball_velocity[1] = int(round(ball_velocity[1]))

def check_collision():
    global ball_pos
    global ball_velocity
    global player1_pos
    rect1 = Rect(player1_pos[0], player1_pos[1], paddle_width, paddle_length)
    rect3 = Rect(ball_pos[0]-ball_radius, ball_pos[1]-ball_radius, 2*ball_radius, 2*ball_radius)
    if rect3.colliderect(rect1):
        return True
    return False

def train_network(status):
    global total_inputs
    global total_class_outputs
    if status == True:
        # Neural net won
        for ix in range(len(total_inputs)):
            cur_inp = np.atleast_2d(total_inputs[ix])
            act_out = np.atleast_2d(total_class_outputs[ix])
            model.fit(cur_inp, act_out, nb_epoch=1, batch_size=1)
    else:
        # Neural net lost
        for ix in range(len(total_inputs)):
            cur_inp = np.atleast_2d(total_inputs[ix])
            cur_out = [0, 0, 0]
            for iy in range(3):
                if total_class_outputs[ix][iy] == 1:
                    cur_out[iy] = 0
                else:
                    cur_out[iy] = 0.5
            act_out = np.atleast_2d(cur_out)
            model.fit(cur_inp, act_out, nb_epoch=1, batch_size=1)
    print("Done updating weights!")
    # Save the model
    model.save_weights("model.keras")

def reset_game():
    global ball_pos
    global player1_pos
    global total_class_outputs
    global total_inputs
    ball_pos = [screen_width / 2, screen_height / 2]
    player1_pos = [0, screen_height / 2 - paddle_length / 2]  # Changes from 0, 0 to 0, screen_height-paddle_length-1
    initialize_ball()
    total_inputs = []
    total_class_outputs = []

def draw_ball(background):
    global ball_pos
    global ball_velocity
    global player1_pos
    ball_pos[0] += ball_velocity[0]
    ball_pos[1] += ball_velocity[1]
    if ball_pos[0] < ball_radius + paddle_width/2:
        train_network(False)
        reset_game()
    if ball_pos[1] > screen_height - ball_radius or ball_pos[1] < ball_radius:
        if ball_pos[1] < ball_radius:
            ball_pos[1] = ball_radius
        else:
            ball_pos[1] = screen_height - ball_radius
        ball_velocity[1] *= -1
    if ball_pos[0] > screen_width - ball_radius:
        ball_pos[0] = screen_width - ball_radius;
        ball_velocity[0] *= -1
    if check_collision():
        print('Collided!')
        gap = abs(ball_pos[1] - (player1_pos[1]+paddle_length/2))*1.0 / (paddle_length / 2)
        ball_velocity[0] = int(round(ball_velocity[0] * (-1) * (max(0.8, gap + 0.7))))
        ball_velocity[1] = int(round(ball_velocity[1] * (max(0.9, gap + 0.4))))
        ball_pos[0] += ball_velocity[0]
        train_network(True)
        global total_inputs
        global total_class_outputs
        total_inputs = []
        total_class_outputs = []
    ball_velocity[0] = max(-max_speed, min(max_speed, ball_velocity[0]))
    ball_velocity[1] = max(-max_speed, min(max_speed, ball_velocity[1]))
    ball_pos[0] = round(ball_pos[0])
    ball_pos[1] = round(ball_pos[1])
    pygame.draw.circle(background, (255, 255, 255), ball_pos, ball_radius)
    return background

def draw_paddles(background):
    global ball_pos
    global ball_velocity
    global player1_pos
    pygame.draw.rect(background, (255, 255, 255), (player1_pos[0], player1_pos[1], paddle_width, paddle_length))
    return background

def main():
    global ball_pos
    global ball_velocity
    global player1_pos
    clock = pygame.time.Clock()
    pygame.init()
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption('Pong!')

    background = pygame.Surface(screen.get_size())
    background = background.convert()
    background.fill((0, 0, 0))
    background_new = draw_paddles(background)
    background_new = draw_ball(background_new)
    screen.blit(background_new, (0, 0))
    pygame.display.flip()

    initialize_ball()
    retrieve_image(background)
    global cur_frame
    global total_inputs
    global total_class_outputs

    while 1:
        clock.tick(60)
        background.fill((0, 0, 0))
        background_new = draw_paddles(background)
        background_new = draw_ball(background_new)
        screen.blit(background_new, (0, 0))
        pygame.display.flip()
        if len(total_inputs)<max_size:
            global diff_img
            retrieve_image(background)
            neural_input = diff_img.flatten()
            neural_input = np.atleast_2d(neural_input)
            total_inputs.append(neural_input)
            output_prob = model.predict(neural_input,1)[0]
            best = np.amax(output_prob)
            if output_prob[0] == best:
                total_class_outputs.append(np.asarray([1,0,0]))
                player1_pos[1] -= paddle_speed
                if player1_pos[1] < 0:
                    player1_pos[1] = 0
            elif output_prob[2] == best:
                total_class_outputs.append(np.asarray([0,0,1]))
                player1_pos[1] += paddle_speed
                if player1_pos[1] >= screen_height-paddle_length:
                    player1_pos[1] = screen_height-paddle_length-1
            else:
                total_class_outputs.append(np.asarray([0,1,0]))
        for event in pygame.event.get():
            if event.type == QUIT:
                return
        keys = pygame.key.get_pressed()
        """
        if keys[K_w]:
            player1_pos[1] -= paddle_speed
            if player1_pos[1] < 0:
                player1_pos[1] = 0
        if keys[K_s]:
            player1_pos[1] += paddle_speed
            if player1_pos[1] > screen_height-paddle_length:
                player1_pos[1] = screen_height-paddle_length
        if random.uniform(0,1) > 0.5:
            # Automated bot motion
            if ball_pos[1] < player2_pos[1]+paddle_length2/2-paddle_speed:
                player2_pos[1] -= paddle_speed
                if player2_pos[1] < 0:
                    player2_pos[1] = 0
            elif ball_pos[1] > player2_pos[1]+paddle_length2/2+paddle_speed:
                player2_pos[1] += paddle_speed
                if player2_pos[1] >= screen_height-paddle_length2:
                    player2_pos[1] = screen_height-paddle_length2-1
        """

if __name__ == '__main__':
    main()