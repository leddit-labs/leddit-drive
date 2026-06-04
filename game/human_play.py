import pygame
from environment.world import World


def run():
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    clock = pygame.time.Clock()

    world = World()

    running = True

    while running:
        #iterates over events
        for event in pygame.event.get():
            #print(event)
            if event.type == pygame.QUIT:
                running = False
        

        screen.fill((63, 124, 65))  #green background
        world.track.draw(screen)    #white track

        keys = pygame.key.get_pressed()

        steer = 0
        throttle = 0
        
        if keys[pygame.K_LEFT]:
            #print("pressed left")
            steer = -1
        if keys[pygame.K_RIGHT]:
            #print("pressed right")
            steer = 1
        if keys[pygame.K_UP]:
            #print("pressed up")
            throttle = 1
        



        #move car and game tics
        try:
            state, reward, done = world.step((steer, throttle))
            #print(world.car.x, world.car.y, world.car.speed)
            #print(done)

        except Exception as e:
            print("CRASH INSIDE STEP:", e)
            running = False

        if done:
            world.reset()
        
        #DRAW
        pygame.draw.circle(screen, (255, 0, 0), (int(world.car.x), int(world.car.y)), 5)

        #DEBUG STUFF
        print(world.debug_get_sensors())                    # print sensor values to console
        world.track.debug_draw_sensors(screen, world.car)   # draw sensor

        pygame.display.flip()
        clock.tick(60) #hard coded to 60fps

    pygame.quit()


if __name__ == "__main__":
    run()
