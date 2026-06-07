import math

import pygame

from ai.config import SENSOR_RANGE_PIXELS

WALL_THICKNESS_PIXELS = 5  # wall width in pixels. used for better raycasting

MAX_CHECK_COLLISION_DISTANCE_PIXELS = 200


class Track:
    def __init__(self, built_track):
        self.segments = built_track["segments"]
        self.checkpoints = built_track["checkpoints"]

        self.walls = self.segments
        self.outer = built_track["outer"]
        self.inner = built_track["inner"]

        self.amount_of_checkpoints = len(self.checkpoints)

    # a raycast of sorts. car has 3 sensors.
    def get_sensors(self, car, should_normalize=True):
        sensor_angles = [
            car.angle - 0.5,
            car.angle,
            car.angle + 0.5,
        ]

        max_distance = SENSOR_RANGE_PIXELS
        step_size = 6

        sensor_distances = []

        cos = math.cos
        sin = math.sin

        local_car_x = car.x
        local_car_y = car.y

        sensor_check_range = (
            SENSOR_RANGE_PIXELS + 100
        )  # add 100 pixels as a safe margin - still check walls a bit further away than sensor range
        MAX_SENSOR_CHECK_DIST_SQ = sensor_check_range * sensor_check_range

        # pre-filter walls once
        nearby_walls = []

        #discard walls far away
        for (x1, y1), (x2, y2) in self.walls:
            mx = (x1 + x2) * 0.5
            my = (y1 + y2) * 0.5

            dx = mx - local_car_x
            dy = my - local_car_y

            if dx * dx + dy * dy <= MAX_SENSOR_CHECK_DIST_SQ:
                nearby_walls.append(((x1, y1), (x2, y2)))

        # raycast using only nearby walls
        for angle in sensor_angles:
            ca = cos(angle)
            sa = sin(angle)

            distance = max_distance

            for i in range(0, max_distance, step_size):
                test_x = local_car_x + ca * i
                test_y = local_car_y + sa * i

                # check if this point is close enough to any wall segment - WALL_THICKNESS is used to define how wide wall is
                # If true, the sensor has hit a wall, break to get out.
                for (x1, y1), (x2, y2) in nearby_walls:
                    if (
                        self._point_line_distance(test_x, test_y, x1, y1, x2, y2)
                        < WALL_THICKNESS_PIXELS
                    ):
                        distance = i
                        break
                else:
                    continue  # No wall hit at this step, continue marching forward

                break  # Wall was hit -> stop checking further steps on this ray

            if should_normalize:
                sensor_distances.append(distance / max_distance)
            else:
                sensor_distances.append(distance)

        print((f"checked sensor walls: {len(nearby_walls)}, total walls: {len(self.walls)}"))
        return sensor_distances

    def is_collision(self, car):
        # save to local variable. removes the lookups needed. small performace boost
        local_walls = self.walls
        local_car_x = car.x
        local_car_y = car.y
        local_radius = car.radius

        MAX_CHECK_DIST_SQ = (
            MAX_CHECK_COLLISION_DISTANCE_PIXELS * MAX_CHECK_COLLISION_DISTANCE_PIXELS
        )

        for (x1, y1), (x2, y2) in local_walls:
            # a way to skip walls far away. we don't want to check walls for collision if they are far away
            mx = (x1 + x2) * 0.5
            my = (y1 + y2) * 0.5

            dx = mx - local_car_x
            dy = my - local_car_y

            if dx * dx + dy * dy > MAX_CHECK_DIST_SQ:
                continue  # skip this wall

            if self._circle_line_collision(
                local_car_x, local_car_y, local_radius, x1, y1, x2, y2
            ):
                return True
        return False

    def draw(self, screen):
        if not len(self.outer) == 0:
            pygame.draw.polygon(
                screen, (90, 90, 90), self.outer
            )  # draw the entire track grey

        if not len(self.inner) == 0:
            pygame.draw.polygon(
                screen, (63, 124, 65), self.inner
            )  # mask the inside grass hole and make that green

        for a, b in self.segments:
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

    def checkpoint_crossed(self, car):
        for i, (a, b) in enumerate(self.checkpoints):
            if self._circle_line_collision(
                car.x, car.y, car.radius, a[0], a[1], b[0], b[1]
            ):
                return i

    # -------DEBUG--------
    def debug_draw_sensors(self, screen, car):
        sensor_angles = [car.angle - 0.5, car.angle, car.angle + 0.5]

        sensor_distances = self.get_sensors(car, False)

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
        for i, (a, b) in enumerate(self.checkpoints):
            pygame.draw.line(screen, (0, 0, 255), a, b, 2)

            mx = (a[0] + b[0]) // 2
            my = (a[1] + b[1]) // 2

            font = pygame.font.SysFont(None, 18)
            img = font.render(
                str(i + 1), True, (255, 255, 255)
            )  # plus 1 since index start on 0
            screen.blit(img, (mx + 5, my + 5))  # blip nudges the img some pixels

    # this is a clone of is_collision. but is used to visually see what walls are skipped in collision detection
    # is is called in human_play.py
    def debug_is_collision(self, car):
        local_walls = self.walls
        local_car_x = car.x
        local_car_y = car.y
        local_radius = car.radius

        MAX_CHECK_DIST_SQ = (
            MAX_CHECK_COLLISION_DISTANCE_PIXELS * MAX_CHECK_COLLISION_DISTANCE_PIXELS
        )

        checked = []
        skipped = []
        hit = []

        for (x1, y1), (x2, y2) in local_walls:
            mx = (x1 + x2) * 0.5
            my = (y1 + y2) * 0.5

            dx = mx - local_car_x
            dy = my - local_car_y

            if dx * dx + dy * dy > MAX_CHECK_DIST_SQ:
                skipped.append(((x1, y1), (x2, y2)))
                continue

            checked.append(((x1, y1), (x2, y2)))

            if self._circle_line_collision(
                local_car_x, local_car_y, local_radius, x1, y1, x2, y2
            ):
                hit.append(((x1, y1), (x2, y2)))
                return True, checked, skipped, hit

        return False, checked, skipped, hit

    def debug_draw_wall_filter(self, screen, checked, skipped, hit):
        # checked walls = green
        for a, b in checked:
            pygame.draw.line(screen, (0, 255, 0), a, b, 2)

        # skipped walls = red
        for a, b in skipped:
            pygame.draw.line(screen, (255, 0, 0), a, b, 2)

        # hit wall = yellow
        for a, b in hit:
            pygame.draw.line(screen, (255, 255, 0), a, b, 4)
