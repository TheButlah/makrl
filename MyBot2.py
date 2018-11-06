#!/usr/bin/env python3
# Python 3.6

# Import the Halite SDK, which will let you interact with the game.
import hlt

# This library contains constant values.
from hlt import constants

# This library contains direction metadata to better interface with the game.
from hlt.positionals import Direction, Position

# This library allows you to generate random numbers.
import random

# Logging allows you to save messages for yourself. This is required because the regular STDOUT
#   (print statements) are reserved for the engine-bot communication.
import logging

import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns

import sys
if sys.version_info[0] < 3:
    raise Exception("Must be using Python 3")

""" <<<Game Begin>>> """

# This game object contains the initial game state.
game = hlt.Game()
# At this point "game" variable is populated with initial map data.
# This is a good place to do computationally expensive start-up pre-processing.

# TODO: Install Tensorflow, everything else

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


logging.info(f_idx)

# TODO: Initialize model

# As soon as you call "ready" function below, the 2 second per turn timer will start.
game.ready("MyPythonBot")

# Now that your bot is initialized, save a message to yourself in the log file with some important information.
#   Here, you log here your id, which you can always fetch from the game object by using my_id.
logging.info("Successfully created bot! My Player ID is {}.".format(game.my_id))

# logging.info(np.asarray(map(lambda x: x.halite_amount, game.game_map[0])))

""" <<<Game Loop>>> """

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


def plot_state(state):
    for feat in range(state.shape[2]):
       sns.heatmap(state[:, :, feat])
       plt.show()



    # grad_x = 0;grad_y= 0;
    # for s in my_ships
    #     for o in all_objects
    #         d = (s.x - o.x)**2 + (s.y-o.y)**2
    #         f = o.w * math.exp(-o.g * d) 
    #         grad_x += o.w * o.g * f * (s.x - o.x)
    #         grad_y += o.w * o.g * f * (s.y - o.y)

    # Wx*Exp(-Gx d(x,s)^2) )
def plot_vector_field(frames,game_map,me):
    logging.info("Frames: " + str(frames.shape))
    h,f,e = frames[:,:,0], frames[:,:,1], frames[:,:,2]
    grad_x = np.zeros(e.shape)
    grad_y = np.zeros(e.shape)
    #ship = me.get_ships()[0]
    g = [5,10,21]
    w = [-1,1,0.25]

    x = np.arange(0,e.shape[0])
    y = np.arange(0,e.shape[1])
    x, y = np.meshgrid(x,y)

    for ship in me.get_ships():
        d = (x - ship.position.x)**2 + (y - ship.position.y)**2
        for i in range(0,3):
            f = w[i]*np.exp(g[i] - d)
            logging.info(d.shape)
            logging.info(f.shape)
            logging.info(x.shape)
            logging.info(y.shape)
            grad_x += w[i] * g[i] * f * (ship.position.x - x)
            grad_y += w[i] * g[i] * f * (ship.position.y - y)

    grad_x = grad_x / np.max(abs(grad_x))
    grad_y = grad_y / np.max(abs(grad_y))

    plt.quiver(x,y,grad_x,grad_y)
    plt.show()






state = None


while True:
    # This loop handles each turn of the game. The game object changes every turn, and you refresh that state by
    #   running update_frame().
    game.update_frame()
    

    # You extract player metadata and the updated map metadata here for convenience.
    me = game.me
    game_map = game.game_map

    start_time = time.time()
    #logging.info(time.time() - start_time)

    if len(me.get_ships()) != 0:
        pass
        #logging.info(me.get_ships()[0].position)

    # A command queue holds all the commands you will run this turn. You build this list up and submit it at the
    #   end of the turn.
    command_queue = []

    state = generate_state(game_map,me,state)
    plot_state(state)
    # plot_vector_field(frames,game_map,me)


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

