import numpy as np
import logging

import hlt
from hlt import constants
from hlt.positionals import Direction, Position

#from multiagent.core import World, Agent, Landmark
#from multiagent.scenario import BaseScenario

#class HaliteScenario(BaseScenario):
#    pass

class HaliteGame(object):
    def __init__(self, game)