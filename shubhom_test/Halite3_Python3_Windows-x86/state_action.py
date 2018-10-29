'''
A Halite game update has a definition of state depending on each players' resources and their distribution
on the grid. Each ship and agent has a selection of possible actions all dependent on state.
'''


class State():
    def __init__(self, ships, halite, dropoffs, map_size, map=None):
        '''
        :param ships: List of ships in agent's possession (list)
        :param halite: Amount of halite we currently have (int)
        :param dropoffs: List of dropoffs with coordinates (list)
        :param map_size: Size of maps (int tuple, invariant over episodes)
        :param map: Optional map parameter. TODO: Do we need to know global map locations of opponents?
        '''
        self.ships = ships
        self.halite = halite
        self.dropoffs = dropoffs
        self.map_size = map_size
        self.map = map


        #TODO: VECTOR ENCODE THIS GARBAGE
    def update_state(self,ships,halite,dropoffs,terminal,map=None):
        '''
        Define state transitions.
        :param ships: Updated list of ships in agent's posession (list)
        :param halite: Updated amount of halite (int)
        :param dropoffs: Updated listm of agent's dropoffs with coordinates (list)
        :param terminal: Game over (boolean)
        :param map: Optional map parameter
        '''
        self.ships = ships
        self.halite = halite
        self.dropoffs = dropoffs
        self.terminal = terminal

        #TODO: VECTOR ENCODING UPDATE
    def action_set(self,state):
        '''
        Spawn, dropoff, move NSEW, stay still. One-hot encode all actions.
        '''
        #TODO: Constraint actions by Halite available
        action_set = []
        if
