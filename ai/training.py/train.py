from environment.world import World
from ai.agent import Agent
from ai.ga import GA

def fitness(agent):
    world = World()
    total = 0

    for _ in range(300):
        state = world.get_state()
        action = agent.forward(state)

        _, reward, done = world.step(action)

        total += reward
        if done:
            break

    return total


def train():
    ga = GA()
    agents = [Agent() for _ in range(ga.size)]

    for gen in range(50):
        fitnesses = [fitness(a) for a in agents]

        print("Gen", gen, "best:", max(fitnesses))

        agents = ga.evolve(agents, fitnesses)


if __name__ == "__main__":
    train()