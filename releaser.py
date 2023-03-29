class Releaser:
    """`of_velocity` is a closure that will generate a poisson process function
    that takes an rng provider and an onset time. This is memoized so that it
    isn't recomputed over multiple trials etc."""

    def __init__(self, of_velocity):
        self.of_velocity = of_velocity  # vel -> train generator
        self.memo = {}

    def train(self, vel, rng, t):
        if vel not in self.memo:
            self.memo[vel] = self.of_velocity(vel)
        return self.memo[vel](rng, t)


mini_releaser = Releaser(lambda _: (lambda _, t: [t]))
