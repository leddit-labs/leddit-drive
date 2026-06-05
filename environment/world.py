from environment.car import Car
from environment.track import Track
from environment.util.track_builder import TRACK_DEFINITION, TrackBuilder

class World:
    def __init__(self):
        builder = TrackBuilder(TRACK_DEFINITION)
        built_track = builder.build()
        
        self.track = Track(built_track)
        self.car = Car()

    def reset(self):
        self.car = Car()
        return self.get_state()

    def get_state(self):
        return self.car.get_state(self.track)

    def step(self, action):
        self.car.update(action) # moves car

        reward = self.track.get_reward(self.car)
        done = self.track.is_collision(self.car)

        return self.get_state(), reward, done
    
    #-------DEBUG--------
    def debug_get_sensors(self):
        return self.track.get_sensors(self.car)