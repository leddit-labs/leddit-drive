import math

import pygame


class Track:
    def __init__(self, built_track):
        self.segments = built_track["segments"]
        self.checkpoints = built_track["checkpoints"]

        self.walls = self.segments
        self.outer = built_track["outer"]
        self.inner = built_track["inner"]

    #a raycast of sorts. car has 3 sensors. 
    def get_sensors(self, car):
        sensor_angles = [
            car.angle - 0.5,  # left sensor
            car.angle,  # front sensor
            car.angle + 0.5,  # right sensor
        ]

        sensor_distances = []

        max_distance = 200  # how far car can see in pixels

        for angle in sensor_angles:
            distance = max_distance

            # march forward along the ray
            for i in range(max_distance):
                # point along the sensor ray
                test_x = car.x + math.cos(angle) * i
                test_y = car.y + math.sin(angle) * i

                # check if this point touches any wall
                hit = False

                for (x1, y1), (x2, y2) in self.walls:
                    if self._point_line_distance(test_x, test_y, x1, y1, x2, y2) < 3:
                        distance = i
                        hit = True
                        break
                if hit:
                    break
            sensor_distances.append(distance)
        return sensor_distances

    def is_collision(self, car):
        for (x1, y1), (x2, y2) in self.walls:
            if self._circle_line_collision(car.x, car.y, car.radius, x1, y1, x2, y2):
                return True
        return False

    def get_reward(self, car):
        # not implemented, will be used for getting the current score
        return car.speed

    def draw(self, screen):
        if not len(self.outer) == 0:
            pygame.draw.polygon(screen, (90, 90, 90), self.outer)   # draw the entire track grey
        
        if not len(self.inner) == 0:
            pygame.draw.polygon(screen, (63, 124, 65), self.inner)  # mask the inside grass hole and make that green

        for (a, b) in self.segments:
            pygame.draw.line(screen, (200, 200, 200), a, b, 3)


    # checks collision with radius. car has a radius hitbox
    def _circle_line_collision(self, cx, cy, r, x1, y1, x2, y2):
        closest_x, closest_y = self._closest_point_on_segment(cx, cy, x1, y1, x2, y2)
        return math.dist((cx, cy), (closest_x, closest_y)) < r

    def _point_line_distance(self, px, py, x1, y1, x2, y2):
        closest_x, closest_y = self._closest_point_on_segment(px, py, x1, y1, x2, y2)

        return math.dist((px, py), (closest_x, closest_y))

    # finds the closes point on a line (a helper function used for car sensor and collision checks)
    def _closest_point_on_segment(self, px, py, x1, y1, x2, y2):
        dx = x2 - x1
        dy = y2 - y1

        if dx == 0 and dy == 0:
            return x1, y1

        line_len_sq = dx * dx + dy * dy

        t = ((px - x1) * dx + (py - y1) * dy) / line_len_sq
        t = max(0, min(1, t))

        closest_x = x1 + t * dx
        closest_y = y1 + t * dy

        return closest_x, closest_y

    # -------DEBUG--------
    def debug_draw_sensors(self, screen, car):
        sensor_angles = [car.angle - 0.5, car.angle, car.angle + 0.5]

        sensor_distances = self.get_sensors(car)

        for angle, dist in zip(sensor_angles, sensor_distances):
            end_x = car.x + math.cos(angle) * dist
            end_y = car.y + math.sin(angle) * dist

            pygame.draw.line(
                screen,
                (255, 255, 0),  # yellow
                (car.x, car.y),
                (end_x, end_y),
                2,  # thickness of line
            )

    def debug_draw_checkpoints(self, screen):
        for i, (x, y) in enumerate(self.checkpoints):
            pygame.draw.circle(screen, (0, 0, 255), (int(x), int(y)), 5)

            # index label for points
            font = pygame.font.SysFont(None, 18)
            img = font.render(str(i), True, (255, 255, 255)) # white
            screen.blit(img, (x + 5, y + 5)) # blit basically replaces img with a offset 