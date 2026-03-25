# -*- coding: utf-8 -*-
"""
Multi-agent leader follower simulation

(c) N. Tabti
"""

import numpy as np
import matplotlib.pyplot as plt
import Robot
import Simulation



# fleet definition
nbOfRobots = 6
fleet = Robot.Fleet(nbOfRobots, dynamics='singleIntegrator2D')#, initState=initState)    


# random initial positions
np.random.seed(100)
for i in range(0, nbOfRobots):
    fleet.robot[i].state = 10*np.random.rand(2, 1)-5  # random init btw -5, +5


# simulation parameters
Te = 0.01
simulation = Simulation.FleetSimulation(fleet, t0=0.0, tf=20.0, dt=Te)


# reference definition (with respect to elader)
rRef = None #  **** A COMPLETER EN TP ****


# gains for leader (L) and follower (F) robots
kL = 1.0  #  **** A MODIFIER EN TP ****
kF = 1.0  #  **** A MODIFIER EN TP ****



# main loop of simulation
for t in simulation.t:


    # computation for each robot of the fleet
    for i in range(0, fleet.nbOfRobots):

		# control input of robot i
        fleet.robot[i].ctrl = None #  **** A COMPLETER EN TP **** #
		
        
    # store simulation data
    simulation.addDataFromFleet(fleet)
    # integrat motion over sampling period
    fleet.integrateMotion(Te)


# plots
simulation.plot(figNo=2)
simulation.plotFleet(figNo=2, mod=100, links=True)

#Bomboclat