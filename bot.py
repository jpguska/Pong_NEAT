import neat
import pickle
import os
import numpy as np
import random
from envpong import PongLogic

class BotPlayer:
    def __init__(self, env):
        self.env = env
        self.obs = None
        
        # Caminhos para os arquivos
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, 'config-feedforward.txt')
        genome_path = os.path.join(local_dir, 'melhor_ia_neat.pkl')
        
        # Recria a rede neural
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config_path)
                             
        with open(genome_path, "rb") as f:
            genome = pickle.load(f)
            
        self.net = neat.nn.FeedForwardNetwork.create(genome, config)
    
    def act(self):
        # Se não houver observação ainda, fica parado
        if self.obs is None:
            return PongLogic.PaddleMove.STILL
            
        # A rede processa os inputs
        output = self.net.activate(self.obs)
        
        # Pega o índice do maior valor (0, 1 ou 2) e subtrai 1
        # Resultados: -1 (DOWN), 0 (STILL) ou 1 (UP)
        action_idx = np.argmax(output)
        action = action_idx - 1
        
        return action
    
    def observe(self, obs):
        self.obs = obs

# Inimigo
class BotRight:
    def __init__(self, env):
        self.env = env
        self.obs = None
    
    def act(self):
        # Joga aleatoriamente
        action = random.choice([PongLogic.PaddleMove.DOWN, PongLogic.PaddleMove.STILL, PongLogic.PaddleMove.UP])  
        return action
    
    def observe(self, obs):
        self.obs = obs

class BotLeft:
    pass