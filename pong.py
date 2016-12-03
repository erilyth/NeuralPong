#!/usr/bin/python

import pygame
from pygame.locals import *

screen_width = 800
screen_height = 600

paddle_length = 120
paddle_width = 20

player1_pos = [0, 0] # Changes from 0 to screen_height-paddle_length
player2_pos = [screen_width-paddle_width, 0] # Changes from 0 to screen_height-paddle_length
ball_pos = [screen_width/2, screen_height/2]

def draw_ball(background):
	pygame.draw.circle(background, (255, 255, 255), ball_pos, 10)
	return background

def draw_paddles(background):
	pygame.draw.rect(background, (255, 255, 255), (player1_pos[0], player1_pos[1], paddle_width, paddle_length))
	pygame.draw.rect(background, (255, 255, 255), (player2_pos[0], player2_pos[1], paddle_width, paddle_length))
	return background

def main():
	pygame.init()
	screen = pygame.display.set_mode((screen_width, screen_height))
	pygame.display.set_caption('Pong!')

	background = pygame.Surface(screen.get_size())
	background = background.convert()
	background.fill((0, 0, 0))

	screen.blit(background, (0, 0))
	pygame.display.flip()

	while 1:
		for event in pygame.event.get():
			if event.type == QUIT:
				return
		player1_pos[1] += 1
		background.fill((0, 0, 0))
		background_new = draw_paddles(background)
		background_new = draw_ball(background_new)
		screen.blit(background_new, (0, 0))
		pygame.display.flip()

if __name__ == '__main__':
	main()