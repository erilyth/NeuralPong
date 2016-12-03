#!/usr/bin/python

import pygame
from pygame.locals import *

screen_width = 800
screen_height = 600

paddle_length = 100
paddle_width = 20
player1_pos = 0 # Changes from 0 to screen_height-paddle_length
player2_pos = 0 # Changes from 0 to screen_height-paddle_length

def draw_paddles(background):
	bar1 = pygame.Surface((paddle_width, paddle_length))
	bar2 = pygame.Surface((paddle_width, paddle_length))
	bar1.fill((255, 255, 255))
	bar2.fill((255, 255, 255))
	background.blit(bar1, (0, player1_pos))
	background.blit(bar2, (screen_width-paddle_width, player2_pos))
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
		background_new = draw_paddles(background)
		screen.blit(background_new, (0, 0))
		pygame.display.flip()

if __name__ == '__main__':
	main()