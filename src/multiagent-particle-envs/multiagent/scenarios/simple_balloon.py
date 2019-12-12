import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        num_good_agents = 2
        num_adversaries = 2
        num_balloons = 5
        num_landmarks = 3
        vocabulary_size = 2

        world.dim_c = vocabulary_size

        num_agents = num_adversaries + num_good_agents
        # add agents
        world.agents = [Agent() for _ in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.index = i
            agent.collide = True
            agent.adversary = True if i < num_adversaries else False
            agent.size = 0.1 if agent.adversary else 0.1
            agent.accel = 3.0 if agent.adversary else 4.0
            agent.max_speed = 1.0 if agent.adversary else 1.3

            # Added attributes for balloons and observation range
            agent.num_balloons = num_balloons
            agent.init_num_balloons = num_balloons
            agent.obs_range = 3.0 if agent.adversary else 3.0
            agent.atk_range = 0.5 if agent.adversary else 0.5

            agent.team = 0 if agent.adversary else 1

            agent.state.c = np.zeros((num_agents, world.dim_c), dtype=int)

        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = True
            landmark.movable = False
            landmark.size = 0.1
            landmark.boundary = False

        # make initial conditions
        self.obs = {}
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.85, 0.35]) if not agent.adversary else np.array([0.85, 0.35, 0.35])
            # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])

        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)

        # reset agent attributes
        for agent in world.agents:
            agent.num_balloons = agent.init_num_balloons

        for i, landmark in enumerate(world.landmarks):
            if not landmark.boundary:
                landmark.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)
                landmark.state.p_vel = np.zeros(world.dim_p)


    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]

    def reward(self, agent, world):
        # Agents are rewarded based on the total number of balloons left on all teammates and opponents
        rew = 0

        if agent.adversary:
            opponents = [a for a in world.agents if not a.adversary]
        else:
            opponents = [a for a in world.agents if a.adversary]

        # less rewards if more balloons in opponents
        num_balloon_opponent = sum([o.num_balloons for o in opponents])
        rew -= num_balloon_opponent

        # more rewards if more balloons in self
        rew += agent.num_balloons

        return rew


    def distance(self, a1, a2):
        return np.sqrt(np.sum(np.square(a1.state.p_pos - a2.state.p_pos)))

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            if not entity.boundary and self.distance(agent, entity) <= agent.obs_range:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # communication of all other agents
        other_comm = []
        other_pos = []
        other_vel = []
        for other in world.agents:
            if other is agent or self.distance(agent, other) > agent.obs_range:
                continue

            # print(other.index, other.state.c)
            other_comm.append(other.state.c[agent.index])
            # other_comm.append(list(other.state.c.flatten()[agent.index]))
            other_pos.append(other.state.p_pos - agent.state.p_pos)
            if not other.adversary:
                other_vel.append(other.state.p_vel)

        # dictionary format

        self.obs["own velocity"] = agent.state.p_vel
        self.obs["own position"] = agent.state.p_pos
        self.obs["own message"] = agent.state.c.flatten()
        self.obs["obstacle positions"] = entity_pos
        self.obs["other positions"] = other_pos
        self.obs["other velocities"] = other_vel
        self.obs["other messages"] = other_comm
        print("other comm", other_comm)
        return np.asarray([agent.state.p_vel] + [agent.state.p_pos] + [agent.state.c.flatten()] + entity_pos + other_pos + other_vel + other_comm)
        # return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + [agent.state.c.flatten()] + entity_pos + other_pos + other_vel + other_comm)

    def info(self, agent, world):
        # # get positions of all entities in this agent's reference frame
        # entity_pos = []
        # for entity in world.landmarks:
        #     if not entity.boundary and self.distance(agent, entity) <= agent.obs_range:
        #         entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # # communication of all other agents
        # other_comm = []
        # other_pos = []
        # other_vel = []
        # for other in world.agents:
        #     if other is agent or self.distance(agent, other) <= agent.obs_range:
        #         continue
        #     other_comm.append(list(other.state.c[agent.index]))
        #     other_pos.append(other.state.p_pos - agent.state.p_pos)
        #     if not other.adversary:
        #         other_vel.append(other.state.p_vel)
        #
        # # dictionary format
        #
        # self.obs["own velocity"] = agent.state.p_vel
        # self.obs["own position"] = agent.state.p_pos
        # self.obs["own message"] = agent.state.c.flatten()
        # self.obs["obstacle positions"] = entity_pos
        # self.obs["other positions"] = other_pos
        # self.obs["other velocities"] = other_vel
        # self.obs["other messages"] = other_comm

        return self.obs
        # TODO: !!!!!!!!!!!
