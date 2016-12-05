#!/usr/bin/python

import pygame
import random
from pygame.locals import *

from keras.models import Sequential

screen_width = 800
screen_height = 600

paddle_length = 120
paddle_width = 20
ball_radius = 10
paddle_speed = 8
ball_speed = 4
max_speed = 9

player1_pos = [0, 0] # Changes from 0, 0 to 0, screen_height-paddle_length
player2_pos = [screen_width-paddle_width, 0] # Changes from 0 to screen_width-paddle_width, 0 to screen_width-paddle_width, screen_height-paddle_length
ball_pos = [screen_width/2, screen_height/2]
ball_velocity = [0, 0]

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
		print 'Failed!'
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
		print 'Collided!'
	ball_velocity[0] = max(-max_speed, min(max_speed, ball_velocity[0]))
	ball_velocity[1] = max(-max_speed, min(max_speed, ball_velocity[1]))
	print ball_velocity[0]
	pygame.draw.circle(background, (255, 255, 255), ball_pos, 10)
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

	screen.blit(background, (0, 0))
	pygame.display.flip()

	initialize_ball()

	while 1:
		clock.tick(60)
		for event in pygame.event.get():
			if event.type == QUIT:
				return
		keys = pygame.key.get_pressed()
		if keys[K_w]:
			player1_pos[1] -= paddle_speed
			if player1_pos[1] < 0:
				player1_pos[1] = 0
		if keys[K_s]:
			player1_pos[1] += paddle_speed
			if player1_pos[1] > screen_height-paddle_length:
				player1_pos[1] = screen_height-paddle_length
		if keys[K_DOWN]:
			player2_pos[1] += paddle_speed
			if player2_pos[1] > screen_height-paddle_length:
				player2_pos[1] = screen_height-paddle_length
		if keys[K_UP]:
			player2_pos[1] -= paddle_speed
			if player2_pos[1] < 0:
				player2_pos[1] = 0
		background.fill((0, 0, 0))
		background_new = draw_paddles(background)
		background_new = draw_ball(background_new)
		screen.blit(background_new, (0, 0))
		pygame.display.flip()

if __name__ == '__main__':
	main()