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
        self.score = 0

    def reset(self):
        self.car = Car()
        self.current_checkpoint = 0
        self.score = 0
        return self.get_state()

    def get_state(self):
        return self.car.get_state(self.track)


    def step(self, action):
        self.car.update(action)

        reward = 0

        checkpoint_hit = False

        # discourage standing still
        reward -= 0.01

        # reward speed
        reward += self.car.speed * 0.01

        # checkpoint reward
        hit = self.track.checkpoint_crossed(self.car)

        if hit is not None and hit == self.current_checkpoint:
            checkpoint_hit = True

            self.current_checkpoint += 1

            bonus = 10 + self.current_checkpoint * 2

            self.score += bonus
            reward += bonus

            # lap completed
            if self.current_checkpoint == self.track.amount_of_checkpoints:
                self.current_checkpoint = 0

        done = self.track.is_collision(self.car)

        return (
            self.get_state(),
            reward,
            done,
            checkpoint_hit,
        )

    def _hit_checkpoint_add_to_score(self):
        self.score += (
            10 + self.current_checkpoint * 2
        )  # flat 10points + checkpoint index * 2. this should motivate agent to go futher

    # -------DEBUG--------
    def debug_get_sensors(self):
        return self.track.get_sensors(self.car, False)
