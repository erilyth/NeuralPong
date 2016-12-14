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
ball_speed = 3
max_speed = 5
gamma = 0.95
epsilon = 0.3 # Random probability which drops as learning progresses

player1_pos = [0, screen_height/2-paddle_length/2] # Changes from 0, 0 to 0, screen_height-paddle_length-1
ball_pos = [screen_width/2, screen_height/2]
ball_velocity = [0, 0]
final_img = np.zeros((80, 80))
old_img = np.zeros((80, 80))
diff_img = np.zeros((80, 80))

reward_cur = 0.1

# Define a neural network that will be used to predict the output

replay_memory = []
max_size = 5000

model = Sequential()
model.add(Dense(output_dim=200, input_dim=80*80))
model.add(Activation("tanh"))
model.add(Dense(output_dim=2))
model.add(Activation("softmax"))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

model.load_weights("model.keras")

def retrieve_image(background):
    global final_img
    global old_img
    global diff_img
    old_img = final_img
    """
    for i in range(-ball_radius, ball_radius+1):
        for j in range(-ball_radius, ball_radius+1):
            if i * i + j * j <= ball_radius * ball_radius:
                if ball_pos[0] + i < screen_width and ball_pos[0] + i >= 0 and ball_pos[1] + j < screen_height and ball_pos[1] + j >= 0:
                    final_img[ball_pos[1]+j][ball_pos[0]+i] = 1
    for i in range(paddle_width):
        for j in range(paddle_length):
            final_img[int(player1_pos[1])+j][int(player1_pos[0])+i] = 1
    """
    # Gives us an RGB image
    final_img = pygame.surfarray.array3d(pygame.transform.scale(pygame.display.get_surface(), (80, 80)))
    avgs = [[(r * 0.33 + g * 0.33 + b * 0.33) for (r, g, b) in col] for col in final_img]
    final_img = np.array([[avg for avg in col] for col in avgs])
    # This is to visualize the array and make sure it works
    #background.blit(pygame.surfarray.make_surface(diff_img), (200,200))
    diff_img = final_img - old_img
    """
    for i in range(80):
        for j in range(80):
            sys.stdout.write(str(diff_img[i][j]) + " ")
        print("")
    """
    #print(diff_img)
    return background

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

def train_network():
    global replay_memory
    for ix in range(len(replay_memory)-1): #Ignore the terminal state
        state_t = replay_memory[ix][0]
        state_t1 = replay_memory[ix][2]
        action_t = replay_memory[ix][1]
        reward_t = replay_memory[ix][3]
        target_t = model.predict(state_t)[0]
        Q_pred_t1 = model.predict(state_t1)[0]
        if action_t[0] == 1:
            target_t[0] = reward_t + gamma * np.amax(Q_pred_t1)
        else:
            target_t[1] = reward_t + gamma * np.amax(Q_pred_t1)
        model.fit(state_t, np.atleast_2d(target_t), nb_epoch=1, batch_size=1)
    print("Done updating weights!")
    # Save the model
    global epsilon
    epsilon = epsilon * 0.97
    model.save_weights("model.keras")

def reset_game():
    global ball_pos
    global player1_pos
    ball_pos = [screen_width / 2, screen_height / 2]
    player1_pos = [0, screen_height / 2 - paddle_length / 2]  # Changes from 0, 0 to 0, screen_height-paddle_length-1
    initialize_ball()

def draw_ball(background):
    global ball_pos
    global ball_velocity
    global player1_pos
    global reward_cur
    global replay_memory
    ball_pos[0] += ball_velocity[0]
    ball_pos[1] += ball_velocity[1]
    reward_cur = 0.1
    if ball_pos[0] < ball_radius + paddle_width/2:
        reward_cur = -1
        train_network()
        replay_memory = []
        reset_game()
    if ball_pos[1] > screen_height - ball_radius or ball_pos[1] < ball_radius:
        if ball_pos[1] < ball_radius:
            ball_pos[1] = ball_radius
        else:
            ball_pos[1] = screen_height - ball_radius
        ball_velocity[1] *= -1
    if ball_pos[0] > screen_width - ball_radius:
        ball_pos[0] = screen_width - ball_radius
        ball_velocity[0] *= -1
    if check_collision():
        print('Collided!')
        reward_cur = 1
        gap = abs(ball_pos[1] - (player1_pos[1]+paddle_length/2))*1.0 / (paddle_length / 2)
        ball_velocity[0] = int(round(ball_velocity[0] * (-1) * (max(1, gap + 0.3))))
        ball_velocity[1] = int(round(ball_velocity[1] * (max(0.9, gap + 0.4))))
        ball_pos[0] += ball_velocity[0]
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
    global replay_memory
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
    background = retrieve_image(background)
    global cur_frame
    global epsilon

    while 1:
        clock.tick(60)
        background.fill((0, 0, 0))
        # Reward is calculated in these functions
        background_new = draw_paddles(background)
        background_new = draw_ball(background_new)
        pygame.display.flip()
        if len(replay_memory)<max_size:
            global diff_img
            background_new = retrieve_image(background_new)
            current_state = diff_img.flatten()
            current_state = np.atleast_2d(current_state)
            actions_possible = model.predict(current_state,1)[0]
            action_chosen = np.asarray([0, 0])
            best = np.amax(actions_possible)
            if random.uniform(0, 1) < epsilon:
                action_idx = random.randrange(2)
                actions_possible[action_idx] = best
                actions_possible[1-action_idx] = 0
            if actions_possible[0] == best:
                action_chosen = np.asarray([1, 0])
                player1_pos[1] -= paddle_speed
                if player1_pos[1] < 0:
                    player1_pos[1] = 0
            elif actions_possible[1] == best:
                action_chosen = np.asarray([0, 1])
                player1_pos[1] += paddle_speed
                if player1_pos[1] >= screen_height-paddle_length:
                    player1_pos[1] = screen_height-paddle_length-1
            if len(replay_memory) != 0:
                replay_memory[len(replay_memory)-1][2] = current_state
                replay_memory[len(replay_memory)-1][3] = reward_cur
            reward_dummy = 0
            next_state_dummy = current_state
            # The next state and reward are just placeholders for now and will be updated in the next iteration
            replay_memory.append([current_state, action_chosen, next_state_dummy, reward_dummy])
        else:
            train_network()
            replay_memory = []
        screen.blit(background_new, (0, 0))
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