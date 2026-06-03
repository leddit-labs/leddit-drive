import math

import pygame


class Track:
    def __init__(self):
        self.walls = [
            # outer boundaries
            ((100, 100), (700, 100)),  # outer upper
            ((700, 100), (700, 500)),  # outer right
            ((700, 500), (100, 500)),  # outer buttom
            ((100, 500), (100, 100)),  # outer left
            # inner boundaries
            ((250, 250), (500, 250)),  # inner upper
            ((500, 350), (250, 350)),  # inner lower
            ((500, 250), (500, 350)),  # inner right
            ((250, 250), (250, 350)),  # inner left
        ]

    def get_sensors(self, car):
        # simple fake sensors. it should check around the car, so algorithm can decide what to do
        return [50, 50, 50, 50, 50]

    def is_collision(self, car):
        for (x1, y1), (x2, y2) in self.walls:
            if self._circle_line_collision(car.x, car.y, car.radius, x1, y1, x2, y2):
                return True
        return False




    # this function checks if the car colided with a wall segment
    #this refference below was used to implement function
    #https://www.geeksforgeeks.org/dsa/minimum-distance-from-a-point-to-the-line-segment-using-vectors/
    def _circle_line_collision(self, cx, cy, r, x1, y1, x2, y2):
        # cx and cy is the car position
        # r is the car radius - defined in car.py
        # x1, y1, x2, y2 is a wall segment

        # direction vector of the wall segment
        dx = x2 - x1
        dy = y2 - y1

        # If the line has zero length, we can't compute collision
        if dx == 0 and dy == 0:
            return False

        # find how far along the line the closest point is

        # The dot product part finds projection onto the line.
        t = ((cx - x1) * dx + (cy - y1) * dy) / (dx * dx + dy * dy)

        # Clamp t so the closest point stays ON the segment
        t = max(0, min(1, t))

        # convert t back into actual closest point coordinates
        closest_x = x1 + t * dx
        closest_y = y1 + t * dy

        # measure distance from car center to closest point
        # If this distance is smaller than the car radius,
        # the circle is touching or overlapping the wall.
        dist = math.sqrt((cx - closest_x) ** 2 + (cy - closest_y) ** 2)

        return dist < r

    def get_reward(self, car):
        #not implemented, maybe used by AI idk yet
        return car.speed

    def draw(self, screen):
        for (x1, y1), (x2, y2) in self.walls:
            pygame.draw.line(screen, (200, 200, 200), (x1, y1), (x2, y2), 5)
