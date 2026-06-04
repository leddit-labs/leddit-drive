import pygame
from environment.world import World


def run():
    pygame.init()
    screen = pygame.display.set_mode((1200, 750))
    clock = pygame.time.Clock()

    world = World()

    running = True

    # for making new tracks
    debug_lines = []

    while running:
        # iterates over events
        for event in pygame.event.get():
            # print(event)
            if event.type == pygame.QUIT:
                running = False

            # LEFT click = add line
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                x, y = event.pos

                if len(debug_lines) > 0:
                    last = debug_lines[-1][1]
                    debug_lines.append((last, (x, y)))

                    print(f"(({last[0]}, {last[1]}), ({x}, {y})),")
                else:
                    debug_lines.append(((x, y), (x, y)))

            # RIGHT click = undo last line
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
                if len(debug_lines) > 0:
                    removed = debug_lines.pop()
                    print(f"^DELETE LINE ABOVE^: {removed}")

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
        pygame.draw.circle(screen, (255, 0, 0), (int(world.car.x), int(world.car.y)), world.car.radius)

        # DEBUG STUFF
        # print(world.debug_get_sensors())                    # print sensor values to console
        #world.track.debug_draw_sensors(screen, world.car)  # draw sensor

        #world.track.debug_draw_checkpoints(screen)

        # for making new tracks
        for (x1, y1), (x2, y2) in debug_lines:
            pygame.draw.line(screen, (255, 255, 255), (x1, y1), (x2, y2), 2)

        # grey preview line (last click -> mouse)
        if len(debug_lines) > 0:
            mx, my = pygame.mouse.get_pos()
            _, (lx, ly) = debug_lines[-1]

            pygame.draw.line(screen, (160, 160, 160), (lx, ly), (mx, my), 2)

        # REQUIRED pygame LOGIC
        pygame.display.flip()
        clock.tick(60)  # hard coded to 60fps

    pygame.quit()


if __name__ == "__main__":
    run()
