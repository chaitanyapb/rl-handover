# Robot-robot Handover via Reinforcement Learning

This project was done as part of RBE595 Deep Reinforcement Learning (Fall 2017) at Worcester Polytechnic Institute (WPI). The main contributors to this work are Chaitanya Perugu, Akshay Kumar and Gaurav Vikhe, with guidance from Dr. Carlos Morato (Principal Scientist, Microsoft).

The idea behind the project was to teach collaborative behavior in robots without explicit instruction, i.e. through reinforcement learning. For this purpose, we define a simple object transport task for two arm manipulators to learn to perform. The catch here is that the initial location of the object can be reached by only one of the manipulators while the goal location of the object is within the workspace of the other manipulator only. As a result, the two manipulators are forced to collaborate with each other for successful object transport.

We use a DQN-based RL agent to act as the "brain" to the two "hands" and train the agent through custom-modified Experience Replay to learn to complete the task. Since the only way to do so is to work together, the agent learns a collaborative behavior policy for the two arm manipulators. We employ V-REP as our main simulation platform and use PhantomX Pincher robots as our manipulators.

The code was built on the Windows OS, but, with small changes to the addresses, should work on Linux OS too.
