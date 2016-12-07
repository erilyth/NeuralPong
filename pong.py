#!/usr/bin/python

import pygame
import random
import sys
from pygame.locals import *
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation

screen_width = 400
screen_height = 300

paddle_length = 60
paddle_width = 10
ball_radius = 5
paddle_speed = 4
ball_speed = 2
max_speed = 5

player1_pos = [0, 0] # Changes from 0, 0 to 0, screen_height-paddle_length-1
player2_pos = [screen_width-paddle_width, 0] # Changes from 0 to screen_width-paddle_width, 0 to screen_width-paddle_width, screen_height-paddle_length-1
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

model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])

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
            final_img[player1_pos[1]+j][player1_pos[0]+i] = 1
            final_img[player2_pos[1]+j][player2_pos[0]+i] = 1
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
    global player2_pos
    rect1 = Rect(player1_pos[0], player1_pos[1], paddle_width, paddle_length)
    rect2 = Rect(player2_pos[0], player2_pos[1], paddle_width, paddle_length)
    rect3 = Rect(ball_pos[0]-ball_radius, ball_pos[1]-ball_radius, 2*ball_radius, 2*ball_radius)
    if rect3.colliderect(rect1) or rect3.colliderect(rect2):
        return True
    return False

def draw_ball(background):
    global ball_pos
    global ball_velocity
    global player1_pos
    global player2_pos
    ball_pos[0] += ball_velocity[0]
    ball_pos[1] += ball_velocity[1]
    if ball_pos[0] > screen_width - ball_radius - paddle_width/2 or ball_pos[0] < ball_radius + paddle_width/2:
        global total_inputs
        global total_class_outputs
        if ball_pos[0] > screen_width - ball_radius - paddle_width/2:
            # Neural net won. Perform updates
            for ix in range(len(total_inputs)):
                cur_inp = np.atleast_2d(total_inputs[ix])
                act_out = np.atleast_2d(total_class_outputs[ix])
                model.fit(cur_inp, act_out, nb_epoch=1, batch_size=1)
        else:
            # Neural net lost. Perform updates
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
        total_inputs = []
        total_class_outputs = []
        model.save_weights("model.keras")
        ball_pos = [screen_width/2, screen_height/2]
        initialize_ball()
    if ball_pos[1] > screen_height - ball_radius or ball_pos[1] < ball_radius:
        if ball_pos[1] < ball_radius:
            ball_pos[1] = ball_radius
        else:
            ball_pos[1] = screen_height - ball_radius
        ball_velocity[1] *= -1
    if check_collision():
        if ball_velocity[0] < 0:
            gap = abs(ball_pos[1] - (player1_pos[1]+paddle_length/2))*1.0 / (paddle_length / 2)
        else:
            gap = abs(ball_pos[1] - (player2_pos[1]+paddle_length/2))*1.0 / (paddle_length / 2)
        ball_pos[0] -= ball_velocity[0]
        ball_velocity[0] = int(round(ball_velocity[0]*(-1)*(max(0.8, gap+0.7))))
        ball_velocity[1] = int(round(ball_velocity[1]*(max(0.9, gap+0.4))))
        print('Collided!')
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
    global player2_pos
    pygame.draw.rect(background, (255, 255, 255), (player1_pos[0], player1_pos[1], paddle_width, paddle_length))
    pygame.draw.rect(background, (255, 255, 255), (player2_pos[0], player2_pos[1], paddle_width, paddle_length))
    return background

def main():
    global ball_pos
    global ball_velocity
    global player1_pos
    global player2_pos
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
        """
        if keys[K_DOWN]:
            player2_pos[1] += paddle_speed
            if player2_pos[1] >= screen_height-paddle_length:
                player2_pos[1] = screen_height-paddle_length-1
        if keys[K_UP]:
            player2_pos[1] -= paddle_speed
            if player2_pos[1] < 0:
                player2_pos[1] = 0
        if random.uniform(0,1) > 0.5:
            # True motion
            if ball_pos[1] < player2_pos[1]+paddle_length/2-paddle_speed:
                player2_pos[1] -= paddle_speed
                if player2_pos[1] < 0:
                    player2_pos[1] = 0
            elif ball_pos[1] > player2_pos[1]+paddle_length/2+paddle_speed:
                player2_pos[1] += paddle_speed
                if player2_pos[1] >= screen_height-paddle_length:
                    player2_pos[1] = screen_height-paddle_length-1

if __name__ == '__main__':
    main()