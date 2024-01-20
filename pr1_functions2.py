from utils import *
from example import example_use_of_gym_env

MF = 0  # Move Forward
TL = 1  # Turn Left
TR = 2  # Turn Right
PK = 3  # Pickup Key
UD = 4  # Unlock Door
actions = [MF, TL, TR, PK, UD]
orientations = [[0,-1] , [-1,0], [0,1], [1,0]] # up, left, down, right 
CC = 0 # O:open, C:close
CO = 1
OC = 2
OO = 3
door_states_B = [CC, CO, OC, OO]
key_positions_B = [[1,1], [2,3], [1,6]]
wall_locs_B = [[4,0], [4,1], [4,3], [4,4], [4,6], [4,7]]
door_locs_B = [[4,2], [4,5]]
grid_size_B = [8, 8]

class stateA: # for partA
    def __init__(self, pos, ori, type, iskey, isdoor):
        self.pos = pos
        self.ori = ori
        self.type = type
        self.iskey = iskey
        self.isdoor = isdoor
        self.list_idx = None # index in state sublist
        self.stage_cost = [] # 4x1
        self.terminal_cost = float('inf')
        self.value_func = [] # Tx1
        self.control_policy = [] # Tx1
        self.optimal_act = None
        self.child = {}
        self.parent = {}
        self.child['states'] = []
        self.child['actions'] = []
        self.parent['states'] = []
        self.parent['actions'] = []

class stateB: # for partB
    def __init__(self, pos, ori, type, key_pos, iskey, door_status):
        self.pos = pos
        self.ori = ori
        self.type = type
        self.iskey = iskey
        self.door_status = door_status
        self.key_pos = key_pos
        self.list_idx = None # index in state sublist
        self.stage_cost = [] # 4x1
        self.terminal_cost = float('inf')
        self.value_func = [] # Tx1
        self.control_policy = [] # Tx1
        self.optimal_act = None
        self.child = {}
        self.parent = {}
        self.child['states'] = []
        self.child['actions'] = []
        self.parent['states'] = []
        self.parent['actions'] = []

########## OUTSIDE CLASS FUNCTIONS ###########
def isvalid(size, x, y):
    if x < 0 or y < 0:
        return False
    if x >= size[1] or y >= size[0]:
        return False
    return True

def isgoal(state_x, goal_pos):
    if (state_x.pos == goal_pos).all():
        return True
    return False

################################ PART A ######################################
def init_statesA(env, info):
    grid_size = [info['height'], info['width']]
    state_grid = [[[] for j in range(grid_size[1])] for i in range(grid_size[0])]
    goal = np.array(info['goal_pos'])
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            for k1 in range(4): # ori
                for k2 in range(2): # key
                    for k3 in range(2): # door
                        cur_st = stateA(pos=np.array([j,i]), ori=np.array(orientations[k1]), type=None, iskey=k2, isdoor=k3)
                        if env.grid.get(j, i) is not None:
                            cur_st.type = env.grid.get(j, i).type
                        list_index = k3 + 2*k2 + 4*k1
                        cur_st.list_idx = k3 + 2*k2 + 4*k1
                        state_grid[i][j].append(cur_st)
    
    num_states = grid_size[0] * grid_size[1] * 4 * 2 * 2
    return state_grid, num_states

def motion_model_A(state_grid, state_x, act_u, env, info):
    grid_size = [info['height'], info['width']]
    cur_pos = state_x.pos
    cur_ori = state_x.ori
    ori_id = orientations.index(list(cur_ori))
    front_pos = cur_pos + cur_ori
    front_pos_invalid = 0 
    if isvalid(grid_size, front_pos[0], front_pos[1]):
        front_pos_invalid = 0 
        front_type = env.grid.get(front_pos[0], front_pos[1])
    else:
        front_pos_invalid = 1
    iskey = state_x.iskey
    isdoor = state_x.isdoor

    if act_u == MF:
        if front_pos_invalid == 1:
            return None
        if front_type is not None and front_type.type == 'door' and isdoor == 0:
            return None
        return state_grid[front_pos[1]][front_pos[0]][isdoor + 2*iskey + 4*ori_id]
    if act_u == TL:
        next_ori_id = (ori_id + 1) % 4
        return state_grid[cur_pos[1]][cur_pos[0]][isdoor + 2*iskey + 4*next_ori_id]
    if act_u == TR:
        next_ori_id = (ori_id - 1) % 4
        return state_grid[cur_pos[1]][cur_pos[0]][isdoor + 2*iskey + 4*next_ori_id]
    if act_u == PK:
        if iskey == 1 or front_pos_invalid == 1:
            return None
        if front_type is not None and front_type.type == 'key':
            return state_grid[cur_pos[1]][cur_pos[0]][isdoor + 2*(iskey+1) + 4*ori_id]
        return None
    if act_u == UD:
        if isdoor == 1 or front_pos_invalid == 1: #door already open
            return None
        if front_type is not None and front_type.type == 'door':
            return state_grid[cur_pos[1]][cur_pos[0]][(isdoor+1) + 2*iskey + 4*ori_id]
        return None

def compute_allcosts_A(state_x, env, info, goal, state_grid):
    if state_x.type == 'wall':
        state_x.value_func.append(float('inf'))
        for _ in actions:
            state_x.stage_cost.append(float('inf'))
        return
    pos = state_x.pos
    if (pos == goal).all():
        state_x.terminal_cost = 1
        state_x.value_func.append(1)
    else:
        state_x.value_func.append(float('inf'))
    ori = state_x.ori
    front_pos = pos + ori
    front_type = env.grid.get(front_pos[0], front_pos[1])
    for act in actions:
        next_state = motion_model_A(state_grid, state_x, act, env, info)
        if next_state is not None:
            state_x.child['states'].append(next_state)
            state_x.child['actions'].append(act)
            next_state.parent['states'].append(state_x)
            next_state.parent['actions'].append(act)

        if act == MF:
            if front_type is not None:
                if front_type.type == 'wall': # front position is wall
                    state_x.stage_cost.append(float('inf'))
                if front_type.type == 'door' and state_x.isdoor == 0:
                    state_x.stage_cost.append(float('inf')) # front position is locked door
                else: # in case if it is other not-None types 
                    state_x.stage_cost.append(1)    
            else:
                state_x.stage_cost.append(1)
        if act == TL or act == TR:
            state_x.stage_cost.append(1)
        if act == PK:
            if front_type is not None and front_type.type == 'key' and state_x.iskey == 0 and state_x.isdoor == 0:
                state_x.stage_cost.append(1)
            else:
                state_x.stage_cost.append(float('inf'))
        if act == UD:
            if front_type is not None and front_type.type == 'door' and state_x.isdoor == 0 and state_x.iskey == 1:
                state_x.stage_cost.append(1)
            else:
                state_x.stage_cost.append(float('inf'))
    return

def find_costs_children_parents_A(state_grid, env, info):
    grid_size = [info['height'], info['width']]
    goal = np.array(info['goal_pos'])
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            for k1 in range(4): # ori
                for k2 in range(2): # key
                    for k3 in range(2): # door
                        cur_state = state_grid[i][j][k3 + 2*k2 + 4*k1]
                        compute_allcosts_A(cur_state, env, info, goal, state_grid)

################################ PART B ######################################
def get_Index_B(k3, k2, k1, k0):
    return int(k3 + len(door_states_B)*k2 + len(door_states_B)*2*k1 + len(door_states_B)*2*len(key_positions_B)*k0)

def is_list_present_in_list(list1, list2): 
    isthere = 0
    for lst in list2:
        if list1==lst:
            isthere = 1
            break
    return isthere

def init_statesB():
    state_grid = [[[] for j in range(grid_size_B[1])] for i in range(grid_size_B[0])]
    for i in range(grid_size_B[0]):
        for j in range(grid_size_B[1]):
            if i == 5 and j == 3:
                stap = 9
            for k0 in range(4): # ori
                for k1 in range(len(key_positions_B)): # key positions
                    for k2 in range(2): # key   
                        for k3 in range(len(door_states_B)): # doors
                            cur_st = stateB(pos=np.array([j,i]), ori=np.array(orientations[k0]), type=None, key_pos = key_positions_B[k1], iskey=k2, door_status=k3)
                            if is_list_present_in_list([j, i], wall_locs_B):  # RECHECK THIS
                                cur_st.type = 'wall'
                            if is_list_present_in_list([j, i], door_locs_B):
                                cur_st.type = 'door'
                            #list_index = k3 + len(door_states_B)*k2 + len(door_states_B)*2*k1 + len(door_states_B)*2*len(key_positions_B)*k0
                            # list_index = get_Index_B(k3, k2, k1, k0)
                            # print(list_index)
                            cur_st.list_idx = get_Index_B(k3, k2, k1, k0)
                            state_grid[i][j].append(cur_st)
            cd=5
    num_states = grid_size_B[0] * grid_size_B[1] * 4 * len(key_positions_B) * 2 * len(door_states_B)
    return state_grid, num_states

def motion_model_B(state_grid, state_x, act_u):
    cur_pos = state_x.pos
    cur_ori = state_x.ori
    ori_id = orientations.index(list(cur_ori))
    iskey = state_x.iskey
    door_status = state_x.door_status
    key_pos = state_x.key_pos
    key_pos_id = key_positions_B.index(key_pos)
    
    front_pos = cur_pos + cur_ori 
    isWall = 0
    for wall in wall_locs_B:
        if (front_pos==wall).all():
            isWall = 1
            break
    isDoor = 0
    for Door in door_locs_B:
        if (front_pos==Door).all():
            isDoor = 1
            break

    front_type = 'None'
    if isWall or (not isvalid(grid_size_B, front_pos[0], front_pos[1])): # wall or boundary
        front_type = 'wall'
    if isDoor:
        front_type = 'door'

    if act_u == MF:
        if front_type == 'wall': # wall
            return None
        if front_type == 'door': # locked door
            if (front_pos == door_locs_B[0]).all():
                if door_status == 0 or door_status == 1:
                    return None
            if (front_pos == door_locs_B[1]).all():
                if door_status == 0 or door_status == 2:
                    return None
        return state_grid[front_pos[1]][front_pos[0]][get_Index_B(door_status, iskey, key_pos_id, ori_id)]
    if act_u == TL:
        next_ori_id = (ori_id + 1) % 4
        return state_grid[cur_pos[1]][cur_pos[0]][get_Index_B(door_status, iskey, key_pos_id, next_ori_id)]
    if act_u == TR:
        next_ori_id = (ori_id - 1) % 4
        return state_grid[cur_pos[1]][cur_pos[0]][get_Index_B(door_status, iskey, key_pos_id, next_ori_id)]
    if act_u == PK:
        if (front_pos == key_pos).all() and iskey == 0:
            return state_grid[cur_pos[1]][cur_pos[0]][get_Index_B(door_status, iskey+1, key_pos_id, ori_id)]
        return None
    if act_u == UD:
        if front_type == 'door': # locked door and has key
            if (front_pos == door_locs_B[0]).all():
                if (door_status == 0 or door_status == 1) and iskey == 1:
                    return state_grid[cur_pos[1]][cur_pos[0]][get_Index_B(door_status+2, iskey, key_pos_id, ori_id)]
            if (front_pos == door_locs_B[1]).all():
                if (door_status == 0 or door_status == 2) and iskey == 1:
                    return state_grid[cur_pos[1]][cur_pos[0]][get_Index_B(door_status+1, iskey, key_pos_id, ori_id)]
        return None

def compute_allcosts_B(state_x, state_grid, goal_pos):
    if state_x.type == 'wall':
        state_x.value_func.append(float('inf'))
        for _ in actions:
            state_x.stage_cost.append(float('inf'))
        return
    pos = state_x.pos
    if (pos == goal_pos).all():
        state_x.terminal_cost = 1
        state_x.value_func.append(1)
    else:
        state_x.value_func.append(float('inf'))
    ori = state_x.ori
    front_pos = pos + ori
    isWall = 0
    for wall in wall_locs_B:
        if (front_pos==wall).all():
            isWall = 1
            break
    isDoor = 0
    for Door in door_locs_B:
        if (front_pos==Door).all():
            isDoor = 1
            break

    front_type = 'None'
    if isWall or (not isvalid(grid_size_B, front_pos[0], front_pos[1])): # wall or boundary
        front_type = 'wall'
    if isDoor:
        front_type = 'door'
    door_status = state_x.door_status
    iskey = state_x.iskey

    for act in actions:
        next_state = motion_model_B(state_grid, state_x, act)
        if next_state is not None:
            state_x.child['states'].append(next_state)
            state_x.child['actions'].append(act)
            next_state.parent['states'].append(state_x)
            next_state.parent['actions'].append(act)
        
        if act == MF:
            if front_type == 'wall':
                state_x.stage_cost.append(float('inf'))
            elif front_type == 'door': # locked door
                if (front_pos == door_locs_B[0]).all():
                    if door_status == 0 or door_status == 1:
                        state_x.stage_cost.append(float('inf'))
                    else:
                        state_x.stage_cost.append(1)
                if (front_pos == door_locs_B[1]).all():
                    if door_status == 0 or door_status == 2:
                        state_x.stage_cost.append(float('inf'))
                    else:
                        state_x.stage_cost.append(1)
            else:
                state_x.stage_cost.append(1)
        if act == TL or act == TR:
            state_x.stage_cost.append(1)
        if act == PK:   # RECHECK THE LOGIC
            if door_status == 3: # both doors open
                state_x.stage_cost.append(float('inf'))
            else:
                state_x.stage_cost.append(1)
        if act == UD:
            if front_type == 'door': # locked door and has key
                if (front_pos == door_locs_B[0]).all() and iskey == 1:
                    if door_status == 0 or door_status == 1:
                        state_x.stage_cost.append(1)
                    else: 
                        state_x.stage_cost.append(float('inf'))
                elif (front_pos == door_locs_B[1]).all() and iskey == 1:
                    if door_status == 0 or door_status == 2:
                        state_x.stage_cost.append(1)
                    else: 
                        state_x.stage_cost.append(float('inf'))
                else:
                    state_x.stage_cost.append(float('inf'))
            else:
                state_x.stage_cost.append(float('inf'))
    return

def find_costs_children_parents_B(state_grid, goal_pos):
    for i in range(grid_size_B[0]):
        for j in range(grid_size_B[1]):
            for k0 in range(4): # ori
                for k1 in range(len(key_positions_B)): # key positions
                    for k2 in range(2): # key
                        for k3 in range(len(door_states_B)): # doors
                            cur_state = state_grid[i][j][get_Index_B(k3, k2, k1, k0)]
                            compute_allcosts_B(cur_state, state_grid, goal_pos)


