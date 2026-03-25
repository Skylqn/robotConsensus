# -*- coding: utf-8 -*-
"""
Multi-agent consensus simulation

(c) S. Bertrand
"""


import numpy as np
import Robot
import Graph
import Simulation
import matplotlib.pyplot as plt

# Tes listes de points
coordonnées = [
    np.array([[4.0], [4.0]]) 
    # np.array([[-4.0], [4.0]]), 
    # np.array([[-4.0], [-4.0]]),
    # np.array([[4.0], [-4.0]])
]

obstacles = [
    np.array([[1.9], [2.0]])
    # np.array([[0], [4.0]]),
    # np.array([[-4], [0]]),
    # np.array([[0], [-4]]),
    # np.array([[4], [0]])
]

index_cible = 0
rayon_validation = 1 
# fleet definition
nbOfRobots = 6  
fleet = Robot.Fleet(nbOfRobots, dynamics='singleIntegrator2D')#, initState=initState)    


# random initial positions
np.random.seed(100)
for i in range(0, nbOfRobots):
    fleet.robot[i].state = 10*np.random.rand(2, 1)-5  # random init btw -5, +5

# communication graph
communicationGraph = Graph.Graph(nbOfRobots)
# adjacency matrix
communicationGraph.adjacencyMatrix = np.ones((nbOfRobots,nbOfRobots)) # tout le monde est connecté avec tout le monde 
# communicationGraph.adjacencyMatrix = np.array([[1, 1, 0, 0, 0, 1],
#                                               [1, 1, 1, 0, 0, 0],
#                                               [0, 1, 1, 1, 0, 0],
#                                               [0, 0, 1, 1, 1, 0],
#                                               [0, 0, 0, 1, 1, 1],# 
#                                               [1, 0, 0, 0, 1, 1]]) # connecté avec 2 voisins



# plot communication graph
communicationGraph.plot(figNo=1)


# simulation parameters
Te = 0.01
simulation = Simulation.FleetSimulation(fleet, t0=0.0, tf=20.0, dt=Te)

# control gain for consensus
kp = 1 #  **** A MODIFIER EN TP ****
kt = 0.2
k_rep = 2
D_s = 1
D_obs = 0.9



# main loop of simulation
for t in simulation.t:

    cible_actuelle = coordonnées[index_cible]
    if np.linalg.norm(fleet.robot[0].state - cible_actuelle) < rayon_validation:
        if index_cible < len(coordonnées) - 1:
            index_cible += 1
            cible_actuelle = coordonnées[index_cible]

    for i in range(0, fleet.nbOfRobots):
        u_i = np.zeros((2,1))
        pos_i = fleet.robot[i].state

        voisins = communicationGraph.getNeighbors(i)
        for j in voisins:
            pos_j = fleet.robot[j].state
            vecteur_ij = pos_j - pos_i
            distance_inter_robots = np.linalg.norm(vecteur_ij)
            
            if distance_inter_robots > 0:
                u_i += kp * vecteur_ij 
                if distance_inter_robots < D_s:
                    force_repulsion = k_rep * (D_s - distance_inter_robots)
                    u_i += force_repulsion * (-vecteur_ij / distance_inter_robots)

        for obs in obstacles:
            pos_obs = obs.reshape(2, 1)
            vecteur_io = pos_obs - pos_i 
            distance_obs = np.linalg.norm(vecteur_io)
            
            if distance_obs < D_obs and distance_obs > 0:
                force_repulsion_obs = k_rep * (D_obs - distance_obs)
                u_i += force_repulsion_obs * (-vecteur_io / distance_obs)

        vecteur_cible = cible_actuelle - pos_i
        u_i += kt * vecteur_cible
        
        fleet.robot[i].ctrl = u_i
        
    simulation.addDataFromFleet(fleet)
    fleet.integrateMotion(Te)


# plot
simulation.plot(figNo=2)

simulation.plotFleet(figNo=5, mod=20)

plt.show()

fig, ax = plt.subplots(figsize=(8, 8))

mod_affichage = 10

# Boucle de lecture de la vidéo
for tt in range(0, len(simulation.t), mod_affichage):
    ax.clear() # On efface l'image précédente
    
    # Paramètres de la fenêtre
    ax.set_aspect('equal')
    ax.set_xlim(-6, 6) # N'hésite pas à agrandir si les robots sortent de l'écran
    ax.set_ylim(-6, 6)
    ax.grid(True)
    ax.set_title(f"Temps : {simulation.t[tt]:.2f} s")

    # 1. On dessine TOUTES les cibles (croix rouges)
    for c in coordonnées:
        ax.plot(c[0,0], c[1,0], 'rx', markersize=12)
            
    # 2. On dessine TOUS les obstacles (carrés noirs)
    for obs in obstacles:
        ax.plot(obs[0,0], obs[1,0], 'ks', markersize=12)

    # 3. On dessine les robots et leurs liens
    for i in range(fleet.nbOfRobots):
        # Récupération de la position du robot i à l'instant tt
        xi = simulation.robotSimulation[i].state[0, tt]
        yi = simulation.robotSimulation[i].state[1, tt]
        
        # Dessin du robot
        ax.plot(xi, yi, marker='o', markersize=10)
        
        # Dessin des liens de communication (lignes grises pointillées)
        voisins = communicationGraph.getNeighbors(i)
        for j in voisins:
            xj = simulation.robotSimulation[j].state[0, tt]
            yj = simulation.robotSimulation[j].state[1, tt]
            ax.plot([xi, xj], [yi, yj], color='gray', linestyle='--', alpha=0.4)

    # Petite pause pour créer l'illusion du mouvement
    plt.pause(0.01) 

# Garde la dernière image ouverte à la fin de la vidéo
plt.show()
