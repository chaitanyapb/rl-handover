# Created on Mon Nov 06 18:06:33 2017
# Author: Chaitanya Pb

#%% Package Imports
import vrep
from numpy import pi

#%% Important variables
blocking = vrep.simx_opmode_blocking
oneshot = vrep.simx_opmode_oneshot
return_ok = vrep.simx_return_ok
novalue = vrep.simx_return_novalue_flag

#%% Set the path to V-REP here or through setPathToVREP()
path_to_vrep_exe = 'C:/Program Files/V-REP3/V-REP_PRO_EDU/vrep.exe'

#%%
def d2r(deg):
    
    rad = deg*(pi/180)
    return rad

def r2d(rad):
    
    deg = rad*(180/pi)
    return deg

#%%
def setPathToVREP(path):
    
    global path_to_vrep_exe
    path_to_vrep_exe = path

#%% Launch V-REP
def startVREP(port_num, headless = False, wait = 0):

    import subprocess as sp
    global path_to_vrep_exe

    args = [path_to_vrep_exe, 
            '-gREMOTEAPISERVERSERVICE_' 
            + str(port_num) 
            + '_FALSE_TRUE']
    if headless:
        args.append('-h')
    
    sp.Popen(args)
    
    if wait:
        import time
        print ('Waiting for V-REP ...')
        time.sleep(wait)

#%%
def connectToRemoteAPIServer(port_num = 19997, is_sync = True):
    
    vrep.simxFinish(-1)
    clientID = vrep.simxStart('127.0.0.1', 
                              port_num,
                              waitUntilConnected = True, 
                              doNotReconnectOnceDisconnected = True, 
                              timeOutInMs = 1000, 
                              commThreadCycleInMs = 5)
    if clientID == -1:
        raise Exception('FAIL: Connection to remote API server failed')        
    #else:
        #print ('SUCCESS: Connected to remote API server')
    
    if is_sync: 
        vrep.simxSynchronous(clientID, True)
        
    return clientID

#%%
def setSimulationTimeStep(clientID, dt = 0.01):
    
    vrep.simxSetFloatingParameter(clientID, 
                                  vrep.sim_floatparam_simulation_time_step, 
                                  dt, 
                                  oneshot)

#%%
def startSimulation(clientID):
    
    vrep.simxStartSimulation(clientID, blocking)
    #print ('SUCCESS: Simulation started')

#%%
def syncSpinOnce(clientID):
    
    vrep.simxSynchronousTrigger(clientID)

#%%
def stopSimulation(clientID):

    vrep.simxStopSimulation(clientID, blocking)
    #print ('SUCCESS: Simulation stopped')

#%%
def closeConnection(clientID):

    vrep.simxGetPingTime(clientID)
    vrep.simxFinish(clientID)
    #print ('SUCCESS: Connection closed')

#%%
def performBlockingOp(clientID):
    
    vrep.simxGetPingTime(clientID)

#%%
def loadVREPScene(clientID, path_to_scene):
    
    ret = vrep.simxLoadScene(clientID, 
                             path_to_scene, 
                             False, 
                             blocking)
    if ret != return_ok:
        raise Exception('FAIL: Scene failed to load')
    #else:
        #print ('SUCCESS: Scene loaded')

#%%
def loadVREPModel(clientID, path_to_model):
    
    ret, base_handle = vrep.simxLoadModel(clientID, 
                                          path_to_model, 
                                          False, 
                                          blocking)
    if ret != return_ok:
        raise Exception('FAIL: Model failed to load')
    #else:
        #print ('SUCCESS: Model loaded')
    
    return base_handle

#%%
def placeModelInScene(clientID, base_handle, position, orientation, 
                      frame = 'absolute'):
    
    if frame == 'absolute':
        frame = -1
    
    ret = vrep.simxSetObjectPosition(clientID, 
                                     base_handle, 
                                     frame, 
                                     position, 
                                     oneshot)
    
    if ret != return_ok and ret != novalue:
        raise Exception('FAIL: Unable to position model')
                                     
    ret = vrep.simxSetObjectOrientation(clientID,
                                        base_handle,
                                        frame,
                                        orientation,
                                        oneshot)
                                     
    if ret != return_ok and ret != novalue:
        raise Exception('FAIL: Unable to orient model')
    
    #print ('SUCCESS: Model placed in scene')

#%%
def loadModelIntoScene(clientID, path_to_model, position, orientation,
                       frame = 'absolute'):
    
    if frame == 'absolute':
        frame = -1
    
    px, py, pz = position
    ox, oy, oz = orientation
    
    base_handle = loadVREPModel(clientID, path_to_model)
    
    if pz == None:
        xyz = getAbsolutePosition(clientID, base_handle)        
        pz = xyz[2]
    
    position = (px, py, pz)
    orientation = (d2r(ox), d2r(oy), d2r(oz))
    
    placeModelInScene(clientID, base_handle, position, orientation, frame)
    
    return base_handle
    

#%%
def getObjectHandle(clientID, name):
    
    ret, handle = vrep.simxGetObjectHandle(clientID, 
                                           name, 
                                           blocking)
    if ret != return_ok:
        raise Exception('FAIL: Object ' + name + ' does not exist')
    
    return handle

#%%    
def getObjectParameter(clientID, handle, parameter_code, is_float = True):
    
    if is_float:
        ret, value = vrep.simxGetObjectFloatParameter(clientID, 
                                                      handle, 
                                                      parameter_code, 
                                                      blocking)
    else:
        ret, value = vrep.simxGetObjectIntParameter(clientID, 
                                                    handle, 
                                                    parameter_code, 
                                                    blocking)
    if ret != return_ok:
        raise Exception('FAIL: getObjectParameter failed')
    
    return value

#%%
def setObjectParameter(clientID, handle, parameter_code, set_val, is_float = True):
    
    if is_float:
        vrep.simxSetObjectFloatParameter(clientID, 
                                         handle, 
                                         parameter_code,
                                         set_val,
                                         oneshot)
    else:
        vrep.simxSetObjectIntParameter(clientID, 
                                       handle, 
                                       parameter_code,
                                       set_val,
                                       oneshot)

#%%
def getAbsolutePosition(clientID, handle):
    
    ret, xyz = vrep.simxGetObjectPosition(clientID, 
                                          handle, 
                                          -1, 
                                          blocking)
    if ret != return_ok:
        raise Exception('FAIL: getAbsolutePosition failed')
    
    return xyz

#%%
def getJointPosition(clientID, joint_handle):
    
    if joint_handle == None:
        get_joint_val = None
        ret = return_ok
    else:
        ret, get_joint_val = vrep.simxGetJointPosition(clientID, 
                                                       joint_handle, 
                                                       blocking)
    if ret != return_ok: 
        raise Exception('FAIL: GetJointPosition failed')
    
    return get_joint_val

#%%
def getJointForce(clientID, joint_handle):
    
    ret, get_joint_val = vrep.simxGetJointForce(clientID, 
                                                joint_handle, 
                                                blocking)
    if ret != return_ok: 
        raise Exception('FAIL: GetJointForce failed')
    
    return get_joint_val

#%%
def setJointPosition(clientID, joint_handle, set_joint_val):
    
    vrep.simxSetJointTargetPosition(clientID, 
                                    joint_handle, 
                                    set_joint_val, 
                                    oneshot)

#%%
def setJointVelocity(clientID, joint_handle, set_joint_val):
    
    vrep.simxSetJointTargetVelocity(clientID, 
                                    joint_handle, 
                                    set_joint_val, 
                                    oneshot)

#%%
def setJointForce(clientID, joint_handle, set_joint_val):
    
    vrep.simxSetJointForce(clientID, 
                           joint_handle, 
                           set_joint_val, 
                           oneshot)

#%%
def getAllJointHandles(clientID, joint_names):
    
    joint_handles = []
    for name in joint_names:
        ret, handle = vrep.simxGetObjectHandle(clientID, 
                                               name, 
                                               blocking)
        if ret != return_ok:
            raise Exception('FAIL: Joint ' + name + ' does not exist')
        else:
            joint_handles.append(handle)
    
    return joint_handles

#%%
def getAllJointPositions(clientID, joint_handles):
    
    joint_poses = []
    for handle in joint_handles:
        if handle != None:
            joint_poses.append(getJointPosition(clientID, handle))
        else:
            joint_poses.append(None)
            
    return joint_poses

#%%
def setJointPID(clientID, handle, P, I, D):
    
    setObjectParameter(clientID, handle, 2002, P, True)
    setObjectParameter(clientID, handle, 2003, I, True)
    setObjectParameter(clientID, handle, 2004, D, True)

#%%