# Created on Thu Nov 02 13:15:01 2017
# Author: Chaitanya Reddy

#%%
import os
import h5py
import numpy as np
import vreppy as vp
import impclass as ic
import matplotlib.pyplot as plt

#%%
#
#        0      =>      No Action
#
# Increase Action    Decrease Action    Robot Joint
#
#        1                  2          robotS_joint1
#        3                  4          robotS_joint2
#        5                  6          robotS_joint3
#        7                  8          robotS_joint4
#        9                 10          robotS_gripAction
#
#       11                 12          robotE_joint1
#       13                 14          robotE_joint2
#       15                 16          robotE_joint3
#       17                 18          robotE_joint4
#       19                 20          robotE_gripAction

#%%
path_to_robotS = os.getcwd() + '/models/StartSideRobot.ttm'
path_to_robotE = os.getcwd() + '/models/EndSideRobot.ttm'
path_to_object = os.getcwd() + '/models/Object.ttm'
path_to_scene = os.getcwd() + '/scenes/emptyScene.ttt'
path_to_goal = os.getcwd() + '/models/Goal.ttm'
               
#%%
dt = 0.05
PID = [1, 8, 0.0017]
maxGripForce = 2.5

#%%
def setup_env():
    
    clientID = vp.connectToRemoteAPIServer(port_num = 19997, is_sync = True)
    
    vp.loadVREPScene(clientID, path_to_scene)
    
    robotS_base_handle = vp.loadModelIntoScene(clientID, path_to_robotS, 
                                               [0, 0.3, None], [90, 90, 90])                                           
    robotE_base_handle = vp.loadModelIntoScene(clientID, path_to_robotE, 
                                               [0, -0.3, None], [90, -90, 90])
                                        
    robotS_grip_handle = vp.getObjectHandle(clientID, 'StartSideRobot_link5')
    object_base_handle = vp.loadModelIntoScene(clientID, path_to_object,
                                               [0, -0.05, 0.055], [90, 0, 0], 
                                               frame = robotS_grip_handle)
    
    robotS = ic.PhantomRobot('Start', clientID, robotS_base_handle, PID, maxGripForce)
    robotE = ic.PhantomRobot('End', clientID, robotE_base_handle, PID, maxGripForce)
    
    robotS.freeze_joints(['gripRotate'])
    robotE.freeze_joints(['gripRotate'])
    
    env = ic.HandoverEnv(clientID, robotS, robotE, object_base_handle, debug = False)
    
    vp.loadModelIntoScene(clientID, path_to_goal, env.goal, [0, 0, 0])
    
    return clientID, robotS, robotE, env

def perform_episode(e):
    
    clientID, robotS, robotE, env = setup_env()

    vp.setSimulationTimeStep(clientID, dt)
    vp.startSimulation(clientID)
    
    vp.syncSpinOnce(clientID)
    vp.performBlockingOp(clientID)
    
    state = env._get_state()
    state = env._convert_dict_to_tuple()
    state = np.reshape(state, [1, state_size])
    
    net_epi_score = 0
    all_epi_scores = []
    
    for step in range(MAX_STEPS):
        action1, action2 = agent.choose_action2(state)
        
        next_state, reward, done = env.step(action1)
        
        net_epi_score += reward 
        all_epi_scores.append((net_epi_score, reward))
        
        next_state = np.reshape(next_state, [1, state_size])
        agent.remember_experience(state, action1, reward, next_state, done)
        state = next_state
        if done and step != 0:
            print (agent.model.predict(state)[0])
            print (action1)
            break
        
        next_state, reward, done = env.step(action2)
        
        net_epi_score += reward 
        all_epi_scores.append((net_epi_score, reward))
        
        next_state = np.reshape(next_state, [1, state_size])
        agent.remember_experience(state, action2, reward, next_state, done)
        state = next_state
        if done and step != 0:
            print (agent.model.predict(state)[0])
            print (action2)
            break
        
    vp.stopSimulation(clientID)
    vp.closeConnection(clientID)
    return net_epi_score, all_epi_scores, step

#%%
EPISODES = 300
MAX_STEPS = 150
BATCH_SIZE = 128

state_size = 13
action_size = 21

all_scores = []
all_time = []
done  = False

agent = ic.DQNAgent(state_size, action_size)

for e in range(EPISODES):
    epi_score, epi_score_list, end_step = perform_episode(e)
    
    print("episode: {}/{}, time_step: {}, score: {}, exploration: {}"
        .format(e+1, EPISODES, end_step, round(epi_score, 4), round(agent.epsilon, 4)))
    
    all_scores.append(epi_score)
    all_time.append(end_step)
    
    if len(agent.memory) > BATCH_SIZE:
        print ('----------------------------------------------')
        mean_target_err = agent.replay_experience(BATCH_SIZE)
        print ('----------------------------------------------')

    if e % 100 == 0:
        plt.plot(all_scores)
        plt.show()
        plt.show(all_time)
        plt.show()
        #agent.save_model('imp_model.h5')

agent.save_model('after_500.h5')

plt.plot(all_scores)
plt.show()
plt.show(all_time)
plt.show()

#%%
