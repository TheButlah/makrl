# essential imports
import hlt
from hlt import constants
from hlt.positionals import Direction
import random
import logging

# nonessential imports
import numpy as np
import tensorflow as tf

# initialize graph
sess = tf.Session()
new_saver = tf.train.import_meta_graph('./model/model.ckpt.meta')
new_saver.restore(sess, tf.train.latest_checkpoint('./model/'))

# shape of [None, 11, 11, 9]
"""
    0 : halite amount
    1 : friendly ship
    2 : friendly dropoff
    3 - 8 same for enemies
"""
input_ph = tf.get_default_graph().get_tensor_by_name('ship_input_ph:0')
# shape of [None, 5]
"""
    0 : u
    1 : l
    2 : r
    3 : d
    4 : stay still
"""
actions = tf.get_default_graph().get_tensor_by_name('ship_output/BiasAdd:0')

# initialize game
game = hlt.Game()

# initialize game state with 3 enemies
def initialize_game():
    """
    returns : a shuffled list of length 3 that determines which player is which
    """
    game.ready("MyRLBot")
    logging.info("Successfully created bot! My Player ID is {}.".format(game.my_id))
    num_enemies = 3
    vals = range(num_enemies)
    vals = random.shuffle(vals)
    return vals

# method to extract information
def generate_state(game_map, me):
    """
    game_map : a game map of the world. Has metadata to be extracted. Cell values can be extracted using
               ._cells
    me : player object referring to me
    """
    map_shape = (game_map.height, game_map.width)

    # Build a ndarray to work on
    state = np.zeros((*map_shape, 9), dtype=np.int8)

    # Update halite counts
    for r, row in enumerate(game_map._cells):
        # bit length is used for fast log_2 
        state[r, :, 0] = map(lambda x: min(4096, x.halite).bit_length(), row)

    i = 0
    # Update ship locations
    for player in game.players.values():
        ships = player._ships.values()
        drops = player._dropoffs.values()
        
        if player is me:
            # Update Ships
            for ship in ships:
                state[ship.position.y, ship.position.x, 1] = 1
            # Update dropoffs
            for drop in drops:    
                state[drop.position.y, drop.position.x, 2] = 1
        else:
            # Update Ships
            for ship in ships:
                state[ship.position.y, ship.position.x, 2 * i + 1] = 1
            # Update dropoffs
            for drop in drops:
                state[drop.position.y, drop.position.x, 2 * i  + 2] = 1
            i += 1
    return state
    
# method to extract 