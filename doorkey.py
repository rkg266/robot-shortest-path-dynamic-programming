from utils import *
from example import example_use_of_gym_env
from pr1_functions2 import *

MF = 0  # Move Forward
TL = 1  # Turn Left
TR = 2  # Turn Right
PK = 3  # Pickup Key
UD = 4  # Unlock Door
actions = [MF, TL, TR, PK, UD]
actions_str = ['MF', 'TL', 'TR', 'PK', 'UD']
orientations = [[0,-1] , [-1,0], [0,1], [1,0]] # up, left, down, right 
goal_locs_B = [[5,1], [6,3], [5,6]]
key_positions_B = [[1,1], [2,3], [1,6]]
grid_size_B = [8, 8]

def doorkey_problem(env, info, state_grid, num_states):
    grid_size = [info['height'], info['width']]
    goal_pos = info['goal_pos']
    goal_states = state_grid[goal_pos[1]][goal_pos[0]] 

    # Backward DPA
    T = num_states - 1
    cur_states = goal_states
    for t in reversed(range(T)):
        next_states = []
        for t_state in cur_states:
            parents = t_state.parent['states']
            parent_acts = t_state.parent['actions']
            for p_t in range(len(parents)):
                if isgoal(parents[p_t], goal_pos):
                    continue
                val_f = parents[p_t].stage_cost[parent_acts[p_t]] + t_state.value_func[0]
                if val_f < parents[p_t].value_func[0]:
                    parents[p_t].value_func[0] = val_f
                    parents[p_t].optimal_act = parent_acts[p_t]
                    next_states.append(parents[p_t])
        cur_states = next_states
    
    # Traverse for optimal path
    agent_pos = env.agent_pos
    agent_dir = env.dir_vec  # or env.agent_dir
    agent_ori_id = orientations.index(list(agent_dir))
    door = env.grid.get(info["door_pos"][0], info["door_pos"][1])
    is_locked = door.is_locked
    if is_locked:
        is_door = 0
    else:
        is_door = 1
    is_key = int(env.carrying is not None)

    start_state = state_grid[agent_pos[1]][agent_pos[0]][is_door + 2*is_key + 4*agent_ori_id]
    optim_act_seq = []
    optim_act_seq_str = []
    present_state = start_state
    while not isgoal(present_state, goal_pos):
        optim_act_seq.append(actions[present_state.optimal_act])
        optim_act_seq_str.append(actions_str[present_state.optimal_act])
        present_state = motion_model_A(state_grid, present_state, present_state.optimal_act, env, info)
    return optim_act_seq

def compute_door_state(info):
    isdoor = info['door_open'] 
    if isdoor[0] is False and isdoor[1] is False:
        return 0
    if isdoor[0] is False and isdoor[1] is True:
        return 1
    if isdoor[0] is True and isdoor[1] is False:
        return 2
    if isdoor[0] is True and isdoor[1] is True:
        return 3

def DPA_B(state_grid, num_states, goal_pos):
    goal_states = state_grid[goal_pos[1]][goal_pos[0]] 
    # Backward DPA
    T = num_states - 1
    cur_states = goal_states
    for t in reversed(range(T)):
        next_states = []
        for t_state in cur_states:
            parents = t_state.parent['states']
            parent_acts = t_state.parent['actions']
            for p_t in range(len(parents)):
                if isgoal(parents[p_t], goal_pos):
                    continue
                val_f = parents[p_t].stage_cost[parent_acts[p_t]] + t_state.value_func[0]
                if val_f < parents[p_t].value_func[0]:
                    parents[p_t].value_func[0] = val_f
                    parents[p_t].optimal_act = parent_acts[p_t]
                    next_states.append(parents[p_t])
        cur_states = next_states
    return
 
def compute_optimal_policy_B(state_env_grid_B, num_states_B, goal_pos):
    find_costs_children_parents_B(state_env_grid_B, goal_pos)
    DPA_B(state_env_grid_B, num_states_B, goal_pos)
    return

def get_optimal_policy_B(state_env_grids_goalwise, env, info):
    agent_pos = env.agent_pos
    agent_dir = env.dir_vec  # or env.agent_dir
    agent_ori_id = orientations.index(list(agent_dir))
    key_pos = info['key_pos']
    key_pos_id = key_positions_B.index(list(key_pos))
    door_status = compute_door_state(info)
    is_key = int(env.carrying is not None)
    goal_pos = info['goal_pos']
    goal_pos_id = goal_locs_B.index(list(goal_pos))
    front_pos = agent_pos + agent_dir
    front_type = 'None'
    if is_list_present_in_list(list(front_pos), wall_locs_B) or (not isvalid(grid_size_B, front_pos[0], front_pos[1])): # wall or boundary
        front_type = 'wall'
    if is_list_present_in_list(list(front_pos), door_locs_B):
        front_type = 'door'

    cur_state_grid = state_env_grids_goalwise[goal_pos_id]
    start_state = cur_state_grid[agent_pos[1]][agent_pos[0]][get_Index_B(door_status, is_key, key_pos_id, agent_ori_id)]
    optim_act_seq = []
    optim_act_seq_str = []
    present_state = start_state
    while not isgoal(present_state, goal_pos):
        optim_act_seq.append(actions[present_state.optimal_act])
        optim_act_seq_str.append(actions_str[present_state.optimal_act])
        present_state = motion_model_B(cur_state_grid, present_state, present_state.optimal_act)
        #print(actions_str[present_state.optimal_act])
    return optim_act_seq

def partA(env_path):
    env, info = load_env(env_path)  # load an environment
    state_env_grid, num_states = init_statesA(env, info)
    find_costs_children_parents_A(state_env_grid, env, info)
    seq = doorkey_problem(env, info, state_env_grid, num_states)  # find the optimal action sequence
    # print sequence
    seq_str_list = []
    for s in seq:
        seq_str_list.append(actions_str[s])
    seq_str = ', '.join(seq_str_list)
    print(seq_str)

    a1 = env_path.find('doorkey')
    a2 = env_path.find('.env')
    out_gif_name = env_path[a1 : a2] + '.gif'
    out_gif_path = os.path.join('/home/renukrishna/ece276b/ECE276B_PR1/envs/known_opt_gifs/', out_gif_name)
    draw_gif_from_seq(seq, load_env(env_path)[0], out_gif_path)  # draw a GIF & save


def partB(env_folder):
    state_env_grids_goalwise = {}
    for g_p in range(len(goal_locs_B)): # goal wise optimal policies
        goal_pos = goal_locs_B[g_p]
        state_env_grid_B, num_states_B = init_statesB()
        compute_optimal_policy_B(state_env_grid_B, num_states_B, goal_pos)
        state_env_grids_goalwise[g_p] = state_env_grid_B

    # Run for random env
    env, info, env_path = load_random_env(env_folder)  # load an environment
    seq = get_optimal_policy_B(state_env_grids_goalwise, env, info)

    # Run for each env
    for id in range(36):
        env, info, env_path = load_random_env_sequential(env_folder, id)  # load an environment
        seq = get_optimal_policy_B(state_env_grids_goalwise, env, info)
        # print sequence
        seq_str_list = []
        for s in seq:
            seq_str_list.append(actions_str[s])
        seq_str = ', '.join(seq_str_list)
        print(seq_str)

        a1 = env_path.find('DoorKey')
        a2 = env_path.find('.env')
        out_gif_name = env_path[a1 : a2] + '.gif'
        out_gif_path = os.path.join('/home/renukrishna/ece276b/ECE276B_PR1/envs/random_opt_gifs/', out_gif_name)
        draw_gif_from_seq(seq, load_env(env_path)[0], out_gif_path)  # draw a GIF & save
    


if __name__ == "__main__":
    #example_use_of_gym_env()
    # Part A
    datapath = r"/home/renukrishna/ece276b/ECE276B_PR1/envs/known_envs"
    files_list_all = sorted(os.listdir(datapath))
    env_list = [x for x in files_list_all if x.endswith('.env')]
    for env_fil in env_list:
        partA(os.path.join(datapath, env_fil))
    print('Part A done')

    # Part B
    datapath = r"/home/renukrishna/ece276b/ECE276B_PR1/envs/random_envs"
    partB(datapath)
    print('Part B done')
