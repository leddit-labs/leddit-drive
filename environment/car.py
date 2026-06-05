import numpy as np

class Car:
    def __init__(self):
        self.x = 150    #car initial x spawn position
        self.y = 150    #car initial x spawn position
        self.angle = -30
        self.speed = 0
        self.radius = 5 #radius of car collision uses this

    def update(self, action):
        steer, throttle = action

        self.angle += steer * 0.05
        self.speed += throttle * 0.2
        self.speed *= 0.98

        self.x += np.cos(self.angle) * self.speed
        self.y += np.sin(self.angle) * self.speed

    def get_state(self, track):
        sensors = track.get_sensors(self)
        return sensors + [self.speed]