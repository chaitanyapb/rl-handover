# Created on Wed Nov 15 18:04:18 2017
# Author: Chaitanya Pb

#%%
import random
import numpy as np
import vreppy as vp
from collections import deque

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

#%%
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=3000)
        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.learning_rate = 0.0005
        self.dropout_rate = 0.15
        self.model = self._build_model()
    
    def _build_model(self):
        model = Sequential()
        model.add(Dense(32, input_dim=self.state_size, activation='relu'))
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model
    
    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        action_values = self.model.predict(state)
        return np.argmax(action_values[0])

    def choose_action2(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size), random.randrange(self.action_size)
        action_values = self.model.predict(state)
        action1 = np.argmax(action_values[0])
        
        action_values[0][action1] = np.min(action_values[0])
        action2 = np.argmax(action_values[0])
        
        return action1, action2
    
    def remember_experience(self, s, a, r, ns, d):
        self.memory.append((s, a, r, ns, d))
    
    def replay_experience(self, batch_size):
        
        target_err = []
        
        minibatch = random.sample(self.memory, batch_size - 10)
        for i in range(10):
            minibatch.append(self.memory[-(i+1)])
        
        for s, a, r, ns, d in minibatch:
            if d:
                q_target = r
            else:
                q_target = r + self.gamma*np.amax(self.model.predict(ns)[0])
            state_target = self.model.predict(s)
            
            target_err.append(abs((q_target - state_target[0][a])/q_target))
            if len(target_err) % 40 == 0: print ('Q = {}, Pred = {}'.format(round(q_target, 4), round(state_target[0][a], 4)))
                
            state_target[0][a] = q_target
            self.model.fit(s, state_target, nb_epoch=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        print ('Mean Prediction Error = {}'.format(np.mean(target_err)))
        return np.mean(target_err)
    
    def load_model(self, filename):
        self.model = keras.models.load_model(filename)
    
    def save_model(self, filename):
        self.model.save(filename)

#%%
class PhantomRobot:
    def __init__(self, side, clientID, base_handle, PID, maxGripForce):
        self.name = side + 'SideRobot'
        self.clientID = clientID
        self.base = base_handle
        self.gripClosed = True
        
        self.joint_dict = {}
        self.all_joint_handles = []
        self.all_joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'gripperCenter_joint', 'gripperClose_joint']
        self.all_joint_keys = ['joint1', 'joint2', 'joint3', 'joint4', 'gripRotate', 'gripAction']
        
        for key, jname in zip(self.all_joint_keys, self.all_joint_names):
            jhandle = vp.getObjectHandle(self.clientID, self.name + '_' + jname)            
            vp.setJointPID(self.clientID, jhandle, PID[0], PID[1], PID[2])
            self.all_joint_handles.append(jhandle)
            self.joint_dict[key] = jhandle
        self._update_joints()
        
        vp.setJointForce(self.clientID, self.joint_dict['gripAction'], maxGripForce)
        #print ('SUCCESS: PhantomRobot instance created')
        
    def _update_joints(self):
        self.active_joint_keys = [key for key in self.all_joint_keys if self.joint_dict[key] is not None]
        self.active_joint_handles = [self.joint_dict[key] for key in self.active_joint_keys]
        self.num_active_joints = len(self.active_joint_keys)
    
    def freeze_joints(self, joint_keys):
        for key in joint_keys:
            self.joint_dict[key] = None
        self._update_joints()

#%%
class HandoverEnv:
    def __init__(self, clientID, robotS, robotE, obj_handle, debug = False):
        self.clientID = clientID
        self.robotS = robotS
        self.robotE = robotE
        self.object_handle = obj_handle
        self.egrip1_handle = vp.getObjectHandle(self.clientID, 'EndSideRobot_fingerLeft_visible')
        self.egrip2_handle = vp.getObjectHandle(self.clientID, 'EndSideRobot_fingerLeft_visible0')
        
        self.dtheta = vp.d2r(4)   
        self.goal = [0, -0.6, 0]
        self.goal_radius = 0.05
        self.goal_height = 0.12
        self.precision = 4
        self.maxGripOpen = 0.0368
        self.cylinderSize = 0.025
        self.debugMode = debug
        self.e_factor = 2000
        
        self.total_active_joints = self.robotS.num_active_joints + self.robotE.num_active_joints
        self.state_size = self.total_active_joints + 3
        self.action_size = 2*self.total_active_joints + 1
        self.all_state_keys = ['rS_' + key for key in self.robotS.active_joint_keys] + \
                              ['rE_' + key for key in self.robotE.active_joint_keys] + \
                              ['obj_x', 'obj_y', 'obj_z']
        
        self.state_dict = self._get_state()
        self.state_tuple_deg = self._convert_dict_to_tuple()
        self.reward = 0
        self.done = False
        self.transfer_done = False
        
        obj_pos = [self.state_dict['obj_x'], self.state_dict['obj_y'], self.state_dict['obj_z']]
        egrip1_pos = vp.getAbsolutePosition(self.clientID, self.egrip1_handle)
        egrip2_pos = vp.getAbsolutePosition(self.clientID, self.egrip2_handle)
        egrip_pos = [(e1+e2)/2.0 for e1, e2 in zip(egrip1_pos, egrip2_pos)]
        self.curr_dist = self._euclidean(obj_pos, egrip_pos)
        #print ('SUCCESS: Handover Env created')
        #print ('State Size = {}, Action Size = {}'.format(self.state_size, self.action_size))
    
    def _euclidean(self, list1, list2):
        diff = [i - j for i, j in zip(list1, list2)]
        return np.linalg.norm(diff)
    
    def _reached_goal(self):
        obj_xy = [self.state_dict['obj_x'], self.state_dict['obj_y']]
        obj_z = [self.state_dict['obj_z']]
        is_goal_xy = self._euclidean(obj_xy, self.goal[0:2]) <= self.goal_radius
        is_goal_z = self._euclidean(obj_z, [self.goal[2]]) <= self.goal_height
        if is_goal_xy and is_goal_z:
            return True
        else:
            return False
                
    def _object_in_grip(self):
        obj_in_grip_count = 0
        if self.robotS.gripClosed and self.state_dict['rS_gripAction'] >= 0.9*self.cylinderSize:
            obj_in_grip_count += 1
        if self.robotE.gripClosed and self.state_dict['rE_gripAction'] >= 0.9*self.cylinderSize:
            obj_in_grip_count += 1
        return obj_in_grip_count
    
    def _collision(self):
        return False
        
    def _take_action(self, action):
        tag = int(np.floor((action-1) / 2))
        increase = True if action % 2 == 1 else False
        
        if tag == -1:
            prefix = joint_key = 0
            set_val = get_val = 0
            joint_handle = None
        elif tag >= 0 and tag < self.robotS.num_active_joints:
            prefix = 'rS_'
            joint_key = self.robotS.active_joint_keys[tag]
            joint_handle = self.robotS.joint_dict[joint_key]
        elif tag >= 0 and tag < self.total_active_joints:
            prefix = 'rE_'
            tag = tag - self.robotS.num_active_joints
            joint_key = self.robotE.active_joint_keys[tag]
            joint_handle = self.robotE.joint_dict[joint_key]
        
        if prefix + joint_key != 0:
            curr_val = self.state_dict[prefix + joint_key]
            if joint_key == 'gripAction':
                set_val = self.maxGripOpen if increase else 0.0
                if prefix == 'rS_':
                    self.robotS.gripClosed = True if set_val == 0 else False
                elif prefix == 'rE_':
                    self.robotE.gripClosed = True if set_val == 0 else False
            else:
                set_val = curr_val+self.dtheta if increase else curr_val-self.dtheta
            vp.setJointPosition(self.clientID, joint_handle, set_val)
        
        vp.syncSpinOnce(self.clientID)
        vp.performBlockingOp(self.clientID)
        
        if self.debugMode:        
            get_val = vp.getJointPosition(self.clientID, joint_handle) if joint_handle != None else 0
            gripS = vp.getJointForce(self.clientID, self.robotS.joint_dict['gripAction'])
            gripE = vp.getJointForce(self.clientID, self.robotE.joint_dict['gripAction'])
            print ('----------------------------------------------')
            print ('Action taken = {}'.format(action))
            print ('Tag = {}, Increase = {}, State Key = {}'.format(tag, increase, prefix+joint_key))
            print ('SetVal = {}, GetVal = {}'.format(round(vp.r2d(set_val), self.precision), round(vp.r2d(get_val), self.precision)))
            print ('GripS Force = {}, GripE Force = {}'.format(round(gripS, self.precision), round(gripE, self.precision)))
            print ('Robots holding object = {}'.format(self._object_in_grip()))
            print ('----------------------------------------------')
                
    def _get_state(self):
        state = {}
        
        for key in self.robotS.active_joint_keys:
            state['rS_' + key] = round(vp.getJointPosition(self.clientID, self.robotS.joint_dict[key]), self.precision)
        for key in self.robotE.active_joint_keys:
            state['rE_' + key] = round(vp.getJointPosition(self.clientID, self.robotE.joint_dict[key]), self.precision)
        
        obj_pos = vp.getAbsolutePosition(self.clientID, self.object_handle)
        for i in range(len(obj_pos)):
            obj_pos[i] = round(obj_pos[i], self.precision)
        state['obj_x'], state['obj_y'], state['obj_z'] = obj_pos
        
        return state
    
    def _convert_dict_to_tuple(self):
        state_tuple = ()
        for key in self.all_state_keys:
            if key[0] == 'r':
                state_tuple = state_tuple + (round(vp.r2d(self.state_dict[key]), self.precision),)
            elif key[0] == 'o':
                state_tuple = state_tuple + (self.state_dict[key],)
            else:
                raise Exception('FAIL: Unrecognized State Key')
        return state_tuple
    
    def _get_reward(self):
        reward_time_step = -0.01
        reward_dropped = -500 if self._object_in_grip() == 0 else 0
        reward_collision = -500 if self._collision() else 0
        reward_goal = 10000 if self._reached_goal() else 0
        
        obj_pos = [self.state_dict['obj_x'], self.state_dict['obj_y'], self.state_dict['obj_z']]
        
        if self._object_in_grip() == 2 and not self.transfer_done:
            reward_transfer = 2500
            self.transfer_done = True
            self.curr_dist = self._euclidean(obj_pos, self.goal)
        else:
            reward_transfer = 0
        
        if self.transfer_done:
            next_dist = self._euclidean(obj_pos, self.goal)
        else:
            egrip1_pos = vp.getAbsolutePosition(self.clientID, self.egrip1_handle)
            egrip2_pos = vp.getAbsolutePosition(self.clientID, self.egrip2_handle)
            egrip_pos = [(e1+e2)/2.0 for e1, e2 in zip(egrip1_pos, egrip2_pos)]
            next_dist = self._euclidean(obj_pos, egrip_pos)
        
        reward_euclidean = self.e_factor*(self.curr_dist - next_dist)
        self.curr_dist = next_dist
        
        if reward_dropped == -500: print ('Oops! Object dropped! -500')
        if reward_transfer == 2500: print ('Yay! Transfer occured! +2500')
        if reward_goal == 10000: print ('Awesome! Goal reached! +10000')
        
        net_reward = reward_time_step + reward_dropped + reward_collision + \
                     reward_goal + reward_transfer + reward_euclidean
        return net_reward
    
    def _is_done(self):
        if self._reached_goal():
            return True
        elif self._object_in_grip() == 0:
            return True
        #elif self.transfer_done:
            #return True
        elif self._collision():
            return True
        else:
            return False
    
    def step(self, action):
        self._take_action(action)
        self.state_dict = self._get_state()
        self.state_tuple_deg = self._convert_dict_to_tuple()
        self.reward = self._get_reward()
        self.done = self._is_done()
        return self.state_tuple_deg, self.reward, self.done
        