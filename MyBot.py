#!/usr/bin/env python3
# Python 3.6

# Import the Halite SDK, which will let you interact with the game.
import hlt

# This library contains constant values.
from hlt import constants

# This library contains direction metadata to better interface with the game.
from hlt.positionals import Direction

# This library allows you to generate random numbers.
import random

# Logging allows you to save messages for yourself. This is required because the regular STDOUT
#   (print statements) are reserved for the engine-bot communication.
import logging

# import various dependencies
import numpy as np

""" <<<Game Begin>>> """

# This game object contains the initial game state.
game = hlt.Game()
# At this point "game" variable is populated with initial map data.
# This is a good place to do computationally expensive start-up pre-processing.
# As soon as you call "ready" function below, the 2 second per turn timer will start.
game.ready("MyPythonBot")

# Now that your bot is initialized, save a message to yourself in the log file with some important information.
#   Here, you log here your id, which you can always fetch from the game object by using my_id.
logging.info("Successfully created bot! My Player ID is {}.".format(game.my_id))

""" <<<Game Loop>>> """
# SET UP MAPPINGS FOR FEATURE DICTS #
#####################################
num_enemies = len(game.players)-1
# list of features along with lists of corresponding aliases
feature_list = [
    ["halite", "h"],
    ["friendly_ships", "fs"],
    ["friendly_drops", "fd"],
    ["enemy_ships", "es"],
    ["enemy_drops", "ed"],
]
num_features = len(feature_list)

f_idx = feature_idx = {}

for idx, names in enumerate(feature_list):
    for name in names:
        feature_idx[name] = idx

# generate state
def generate_state(game_map, me, old_state=None):
    map_shape = (game_map.height, game_map.width)

    # Build a ndarray to work on
    if old_state is None:
        state = np.zeros((*map_shape, num_features), dtype=np.int8)
    else:
        state = old_state

    # Update halite counts
    for r, row in enumerate(game_map._cells):
        # bit length is used for fast log_2 
        state[r, :, feature_idx['h']] = map(lambda x: min(4096, x.halite).bit_length(), row)

    # Update ship locations
    for player in game.players.values():
        ships = player._ships.values()
        drops = player._dropoffs.values()
        
        # Helper code to deal with friendly vs enempy
        if player is me:
            s = 'fs'
            d = 'fd'
        else:
            s = 'es'
            d = 'ed'

        # Update Ships
        for ship in ships:
            state[ship.position.y, ship.position.x, feature_idx[s]] = 1
        # Update dropoffs
        for drop in drops:    
            state[drop.position.y, drop.position.x, feature_idx[d]] = 1

    return state

while True:
    # This loop handles each turn of the game. The game object changes every turn, and you refresh that state by
    #   running update_frame().
    game.update_frame()
    # You extract player metadata and the updated map metadata here for convenience.
    me = game.me
    game_map = game.game_map

    # A command queue holds all the commands you will run this turn. You build this list up and submit it at the
    #   end of the turn.
    command_queue = []

    for ship in me.get_ships():
        # For each of your ships, move randomly if the ship is on a low halite location or the ship is full.
        #   Else, collect halite.
        if game_map[ship.position].halite_amount < constants.MAX_HALITE / 10 or ship.is_full:
            command_queue.append(
                ship.move(
                    random.choice([ Direction.North, Direction.South, Direction.East, Direction.West ])))
        else:
            command_queue.append(ship.stay_still())

    # If the game is in the first 200 turns and you have enough halite, spawn a ship.
    # Don't spawn a ship if you currently have a ship at port, though - the ships will collide.
    if game.turn_number <= 200 and me.halite_amount >= constants.SHIP_COST and not game_map[me.shipyard].is_occupied:
        command_queue.append(me.shipyard.spawn())

    # Send your moves back to the game environment, ending this turn.
    game.end_turn(command_queue)

