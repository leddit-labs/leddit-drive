from environment.car import Car
from environment.track import Track
from environment.util.track_builder import TRACK_DEFINITION, TrackBuilder

class World:
    def __init__(self):
        builder = TrackBuilder(TRACK_DEFINITION)
        built_track = builder.build()
        
        self.track = Track(built_track)
        self.car = Car()

        self.current_checkpoint = 0
        self.score = 100

    def reset(self):
        self.car = Car()
        self.current_checkpoint = 0
        self.score = 100
        return self.get_state()

    def get_state(self):
        return self.car.get_state(self.track)

    def step(self, action):
        self.car.update(action)

        reward = 0
        
        # should prevent car standing still as 
        reward -= 0.01
        
        #motivates car to go fast?
        reward += self.car.speed * 0.01

        #checkpoint logic - a bonus is added to reward if checkpoint is hit
        hit = self.track.checkpoint_crossed(self.car)
        if hit is not None and hit == self.current_checkpoint:
            self.current_checkpoint += 1

            bonus = 10 + self.current_checkpoint * 2
            self.score += bonus
            reward += bonus # this is the reward for the AI. robots love this <3

            #print("checkpoint:", self.current_checkpoint)

        done = self.track.is_collision(self.car) # if done --> car crashes.
        

        return self.get_state(), reward, done

    def _hit_checkpoint_add_to_score(self):
        self.score += 10 + self.current_checkpoint * 2 #flat 10points + checkpoint index * 2. this should motivate agent to go futher
    

    #-------DEBUG--------
    def debug_get_sensors(self):
        return self.track.get_sensors(self.car)