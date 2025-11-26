import neat
import os
import pickle
import numpy as np
from envpong import PongEnv

# Um oponente simples que segue a bola (Bot "Perfeito" para treino)
class BotProfessor:
    def __init__(self):
        self.obs = None
    
    def observe(self, obs):
        self.obs = obs
        
    def act(self):
        if self.obs is None: return 0
        
        # O estado do jogo (obs) tem 14 valores.
        # Índice 5: Posição Y da Raquete 2 (Direita/Oponente)
        # Índice 9: Posição Y da Bola
        p2_y = self.obs[5]
        ball_y = self.obs[9]
        
        # Lógica simples: Se a bola está acima, sobe. Se está abaixo, desce.
        if p2_y < ball_y:
            return 1 # UP
        elif p2_y > ball_y:
            return -1 # DOWN
        return 0 # STILL

def eval_genomes(genomes, config):
    # Cria o ambiente sem GUI (rápido)
    env = PongEnv(debugPrint=False)
    opponent = BotProfessor() 

    for genome_id, genome in genomes:
        genome.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        
        obs, info = env.reset()
        opponent.observe(obs)
        
        done = False
        steps = 0
        
        # Limita a 5000 passos para não ficar infinito
        while not done and steps < 5000:
            # 1. IA decide
            output = net.activate(obs)
            action_idx = np.argmax(output)
            action_p1 = action_idx - 1 # Converte para -1, 0, 1
            
            # 2. Professor decide
            action_p2 = opponent.act()
            
            # 3. Executa
            obs, reward, done, truncated, info = env.step(action_p1, action_p2)
            opponent.observe(obs)
            
            # --- NOVA LÓGICA DE PONTUAÇÃO (FITNESS) ---
            # O reward original acumula a diferença de placar a cada frame (causa bugs).
            # Vamos recompensar comportamento ativo:
            
            # Se a IA (P1) estiver ganhando, ótimo.
            diff_score = env.game.states[-1].player1Score - env.game.states[-1].player2Score
            
            # Se o Professor fizer ponto, penaliza muito a IA
            if env.game.states[-1].player2Score > 0:
                genome.fitness -= 20 # Punição severa por perder
                done = True # Encerra o treino desse genoma se ele tomou gol
            
            # Se a IA fizer ponto no professor, RECOMPENSA GIGANTE
            if env.game.states[-1].player1Score > 0:
                genome.fitness += 100
                env.game.states[-1].player1Score = 0 # Reseta placar virtual p/ continuar treinando
            
            # Recompensa por sobreviver (rebatidas)
            # Cada passo vivo vale um pouco, incentivando a rebater a bola
            genome.fitness += 0.1
            
            steps += 1

def run(config_file):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    p.add_reporter(neat.StatisticsReporter())

    # Roda por até 50 gerações
    winner = p.run(eval_genomes, 50)

    with open("melhor_ia_neat.pkl", "wb") as f:
        pickle.dump(winner, f)
    print('\nNovo cérebro treinado salvo!')

if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    run(config_path)