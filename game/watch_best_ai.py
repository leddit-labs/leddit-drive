import numpy as np
import pygame

from ai.agent import Agent
from ai.config import SAVE_FILE

from environment.world import World
from game.ui.ui import UI


def run():
    pygame.init()

    screen = pygame.display.set_mode((1200, 750))

    clock = pygame.time.Clock()

    genome = np.load(SAVE_FILE)

    agent = Agent(genome)

    world = World()

    ui = UI(screen)

    running = True

    while running:
        screen.fill((63, 124, 65))      # green background
        world.track.draw(screen)        # draw track

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        state = world.get_state()

        action = agent.act(state)

        _, _, done = world.step(action)

        if done:
            world.reset()

        

        pygame.draw.circle(screen, (255, 0, 0), (int(world.car.x), int(world.car.y)), world.car.radius) # draw car

        # world.track.debug_draw_sensors(screen,world.car,)

        #ui draw
        ui.draw(world, clock)

        # REQUIRED pygame LOGIC
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    run()
