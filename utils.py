import numpy as np
import vizdoom as vzd
import random


"""
ret: (x,y,z)
"""
def get_player_pos(game):
    return np.array((game.get_game_variable(vzd.POSITION_X),
            game.get_game_variable(vzd.POSITION_Y),
            game.get_game_variable(vzd.POSITION_Z)), np.float)

"""
ret: 1D float
"""
def get_player_dist_from_start(game, start_point):
    player = get_player_pos(game)
    return np.linalg.norm(player - start_point)

def print_state_sectors(game):
    state = game.get_state()
    for s in state.sectors:
        print("Sector lines:", [(l.x1, l.y1, l.x2, l.y2, l.is_blocking) for l in s.lines])

def print_state_objects(game):
    state = game.get_state()
    for o in state.objects:
        print("Object id:", o.id, "object name:", o.name)
        print("Object position: x:", o.position_x, ", y:", o.position_y, ", z:", o.position_z)

"""
convert action to mixed domain (the one to be passed to game)
"""
def convert_action_to_mixed(action_cont):
    action_mixed = np.where(action_cont > 0.0, True, False).tolist()
    action_mixed[14] = action_cont[14]*10.0
    return action_mixed

"""
convert action to continuous domain
"""
def convert_action_to_continuous(action_mixed):
    action_cont = np.where(action_mixed, 0.9, -0.9)
    action_cont[14] = action_mixed[14] / 10.0
    return action_cont

"""
return: list length of 15: 14 booleans and 1 float
"""
def get_random_action(turn_delta_sigma=3.3, weapon_switch_prob=0.05):
    random_action = random.choices([True, False], k=7)
    random_action += [False, False, False, False, False, False, False]

    if random.random() < weapon_switch_prob:
        random_action[random.randint(7,13)] = True


    # random_action = [False, False, False, False, False, False, False, False, False, False, False, False, False, False]
    #random_action = [False, False, False] + random.choices([True, False], k=4) + [False, False, False, False, False, False, False]


    # prevent simultaneous left/right or forward/back presses
    if random_action[3] and random_action[4]:
        random_action[random.randint(3, 4)] = False
    if random_action[5] and random_action[6]:
        random_action[random.randint(5, 6)] = False

    random_action.append(random.gauss(0.0, turn_delta_sigma))
    random_action[14] = np.clip(random_action[14], -9.0, 9.0)

    return random_action

"""
get null (mixed) action
"""
def get_null_action():
    action = np.zeros((15,), dtype=bool).tolist()
    action[14] = 0.0
    return action

"""
Apply a mutation to an action
"""
def mutate_action(action, max_flipped_buttons=4, turn_delta_sigma=1.0, turn_damping=0.9,
    weapon_switch_prob=0.03,):
    # flip buttons
    if max_flipped_buttons > 0:
        flipped_buttons = random.randint(1,max_flipped_buttons)
        for i in range(flipped_buttons):
            button_id = random.randint(0,13) # id of button to flip
            action[button_id] = not action[button_id]
    
    # reset weapon switching
    for i in range(7,14):
        action[i] = False
    
    if random.random() < weapon_switch_prob:
        action[random.randint(7,13)] = True
    
    # apply deviation to the turning delta
    action[14] *= turn_damping
    action[14] += random.gauss(0.0, turn_delta_sigma)
    action[14] = np.clip(action[14], -9.0, 9.0)

    return action