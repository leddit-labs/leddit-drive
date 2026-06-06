import pygame
from environment.world import World
from game.ui.ui import UI


def run():
    pygame.init()
    screen = pygame.display.set_mode((1200, 750))
    clock = pygame.time.Clock()

    world = World()
    ui = UI(screen)
    running = True

    while running:
        keys = pygame.key.get_pressed() # get input

        screen.fill((63, 124, 65))      # green background
        world.track.draw(screen)        # draw track

        # iterates over events (is required to be here since the pygame event stack would overflow)
        for event in pygame.event.get():
            if event.type == pygame.QUIT or keys[pygame.K_ESCAPE]:
                running = False

        #reset car controls - later AI will control what human controls below
        steer = 0
        throttle = 0

        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            # print("pressed left")
            steer = -1
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            # print("pressed right")
            steer = 1
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            # print("pressed up")
            throttle = 1

        # move car and game tics
        try:
            state, reward, done, checkpoint_hit = world.step((steer, throttle))

            #print(state)
            #print(reward)

        except Exception as e:
            print("CRASH INSIDE STEP:", e)
            running = False

        if done:
            world.reset() # reset the game

        # DRAW
        pygame.draw.circle(screen, (255, 0, 0), (int(world.car.x), int(world.car.y)), world.car.radius) # draw car

        # DEBUG STUFF
        #print(world.debug_get_sensors())                  # print sensor values to console
        #world.track.debug_draw_sensors(screen, world.car)   # draw sensor
        #world.track.debug_draw_checkpoints(screen)

        #to visualy see what walls are skipped in collision detection
        #done, checked, skipped, hit = world.track.debug_is_collision(world.car)
        #world.track.debug_draw_wall_filter(screen, checked, skipped, hit)

        #UI
        ui.draw(world, clock)

        # REQUIRED pygame LOGIC
        pygame.display.flip()
        clock.tick(60)  # hard coded to 60fps

    pygame.quit()


if __name__ == "__main__":
    run()
