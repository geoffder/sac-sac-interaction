# TODO: a simple class that can handle the generation / acces of BP releasers
# that handles multiple velocities. Currently, bp_releasers is part of model config
# but velocities, which I want to control input, comes later. So instead of having
# it be set in stone, I need to revamp to allow generation of the velocity
# approriate rates and quanta generators from them in a way that maintains
# compatibility with simply providing a [time] (as is default right now, since a toy
# non-train input may be desired)
class Releaser:
    """WIP: idea is that `of_velocity` is a closure that will generate a poisson
    process function that takes an rng provider and an onset time. This is memoized
    so that it isn't recomputed over multiple trials etc."""

    def __init__(self, of_velocity):
        self.of_velocity = of_velocity  # vel -> train generator
        self.memo = {}

    def train(self, vel, rng, t):
        if vel not in self.memo:
            self.memo[vel] = self.of_velocity(vel)
        return self.memo[vel](rng, t)


mini_releaser = Releaser(lambda _: (lambda _, t: [t]))
