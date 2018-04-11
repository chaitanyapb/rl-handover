# Created on Mon Nov 06 21:16:22 2017
# Author: Chaitanya Pb

#%% Package Imports
import os
import vreppy as vp

#%% Define important paths
path_to_vrep = 'C:/Program Files/V-REP3/V-REP_PRO_EDU/vrep.exe'
path_to_pxp = os.getcwd() + '/models/PhantomXPincher.ttm'
path_to_scene = os.getcwd() + '/scenes/emptyScene.ttt'

joint_names = ['PhantomXPincher_joint1', 'PhantomXPincher_joint2', 
               'PhantomXPincher_joint3', 'PhantomXPincher_joint4', 
               'PhantomXPincher_gripperCenter_joint', 
               'PhantomXPincher_gripperClose_joint']
               
#%% Set parameters
dt = 0.001
stop_sim_time = 0.3

sim_time = 0
set_joint_val = 0
dval = 0.003

#%% Start
vp.setPathToVREP(path_to_vrep)
vp.startVREP(port_num = 19997, wait = 15) # Wait time in seconds

clientID = vp.connectToRemoteAPIServer(port_num = 19997, is_sync = True)

vp.loadVREPScene(clientID, path_to_scene)

pxp_base_handle = vp.loadModelIntoScene(clientID, path_to_pxp, 
                                        [0, -0.5, None], [90, 0, 90])

joint_handles = vp.getAllJointHandles(clientID, joint_names)

vp.setSimulationTimeStep(clientID, dt)
vp.startSimulation(clientID)

#%% Simulate
while sim_time < stop_sim_time:
    
    print 't =', sim_time
    
    joint_handle = joint_handles[1]
    
    get_joint_val = vp.getJointPosition(clientID, joint_handle)
    vp.setJointPosition(clientID, joint_handle, set_joint_val)
    
    sim_time += dt
    set_joint_val += dval
    print get_joint_val
    
    vp.syncSpinOnce(clientID)

#%% Close
vp.stopSimulation(clientID)
vp.closeConnection(clientID)
