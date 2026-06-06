import pygame
from environment.world import World


def run():
    pygame.init()
    screen = pygame.display.set_mode((1200, 750))
    clock = pygame.time.Clock()

    world = World()

    running = True

    # for making new tracks
    debug_points = []

    # checkpoint tool state
    checkpoint_start = None
    checkpoint_index = 1

    while running:
        # iterates over events
        for event in pygame.event.get():
            # print(event)
            if event.type == pygame.QUIT:
                running = False

            # LEFT click = add point
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                x, y = event.pos

                debug_points.append((x, y))

                # prints directly usable format
                print(f"({x}, {y}),")

            # RIGHT click = undo last point
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
                if len(debug_points) > 0:
                    removed = debug_points.pop()
                    print(f"REMOVED: {removed}")

            # MIDDLE click = checkpoint tool
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 2:
                x, y = event.pos

                # first click → start checkpoint
                if checkpoint_start is None:
                    checkpoint_start = (x, y)
                    # print(f"Checkpoint start: {checkpoint_start}")

                # second click → finish checkpoint
                else:
                    a = checkpoint_start
                    b = (x, y)

                    print(f"(( {a[0]}, {a[1]} ), ( {b[0]}, {b[1]} )),")

                    checkpoint_index += 1
                    checkpoint_start = None

        screen.fill((63, 124, 65))  # green background
        world.track.draw(screen)  # draw the white track

        keys = pygame.key.get_pressed()

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
            state, reward, done = world.step((steer, throttle))
            # print(world.car.x, world.car.y, world.car.speed)
            # print(done)

        except Exception as e:
            print("CRASH INSIDE STEP:", e)
            running = False

        if done:
            world.reset()

        # DRAW
        pygame.draw.circle(
            screen, (255, 0, 0), (int(world.car.x), int(world.car.y)), world.car.radius
        )

        # DEBUG STUFF
        # print(world.debug_get_sensors())                    # print sensor values to console
        # world.track.debug_draw_sensors(screen, world.car)  # draw sensor

        # world.track.debug_draw_checkpoints(screen)

        # for making new tracks - this is just for dev QOL visuals
        # draw placed points
        for x, y in debug_points:
            pygame.draw.circle(screen, (255, 255, 255), (x, y), 4)

        # draw connected lines
        if len(debug_points) > 1:
            pygame.draw.lines(screen, (255, 255, 255), False, debug_points, 2)

        # grey preview line for easier track making
        if len(debug_points) > 0:
            mx, my = pygame.mouse.get_pos()
            lx, ly = debug_points[-1]

            pygame.draw.line(screen, (160, 160, 160), (lx, ly), (mx, my), 2)

        # preview checkpoint line
        if checkpoint_start is not None:
            mx, my = pygame.mouse.get_pos()
            pygame.draw.line(screen, (0, 0, 255), checkpoint_start, (mx, my), 2)

        # draw car
        pygame.draw.circle(
            screen, (255, 0, 0), (int(world.car.x), int(world.car.y)), world.car.radius
        )

        # REQUIRED pygame LOGIC
        pygame.display.flip()
        clock.tick(60)  # hard coded to 60fps

    pygame.quit()


if __name__ == "__main__":
    run()
