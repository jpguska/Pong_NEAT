import arcade
import pickle
import neat
import os
import numpy as np
import threading
import time
from envpong import PongGUIEnv, PongLogic
from bot import BotRight  # Usaremos o BotRight como oponente para teste

# Classe Wrapper para jogar com a rede NEAT salva
class BotNEAT:
    def __init__(self, genome, config):
        self.genome = genome
        self.config = config
        # Recria a rede neural exata que foi treinada
        self.net = neat.nn.FeedForwardNetwork.create(genome, config)
        self.obs = None

    def observe(self, obs):
        self.obs = obs

    def act(self):
        if self.obs is None:
            return 0
            
        # A rede recebe os inputs e cospe 3 valores de ativação
        output = self.net.activate(self.obs)
        
        # Lógica: O maior valor vence.
        # Índices: 0, 1, 2.
        # Mapeamento para Ação (-1, 0, 1): Índice - 1
        action = np.argmax(output) - 1
        return action

def runLoop(env, bot_p1, bot_p2):
    """Loop lógico do jogo (roda em thread separada da GUI)"""
    time.sleep(1) # Espera 1 seg para janela abrir
    
    while True:
        # Pede ação para os bots
        actionp1 = bot_p1.act()
        actionp2 = bot_p2.act()
         
        # Atualiza o ambiente
        obs, reward, done, truncated, info = env.step(actionp1, actionp2)
        
        # Entrega nova observação aos bots
        bot_p1.observe(obs)
        bot_p2.observe(obs)
        
        # Sincronia visual: espera o tempo de um frame (dt)
        # Se quiser ver super rápido, diminua ou remova este sleep
        time.sleep(env.game.dt)

def main():
    local_dir = os.path.dirname(__file__)
    
    # 1. Carregar Configurações
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    genome_path = os.path.join(local_dir, 'melhor_ia_neat.pkl')

    if not os.path.exists(genome_path):
        print("ERRO: Arquivo 'melhor_ia_neat.pkl' não encontrado.")
        print("Execute 'python3 treino_neat.py' primeiro.")
        return

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    # 2. Carregar o Cérebro (Genoma)
    with open(genome_path, "rb") as f:
        genome = pickle.load(f)

    # 3. Inicializar Jogo
    env = PongGUIEnv() # Janela do Arcade
    
    # Player 1 é a nossa IA, Player 2 é o Bot Aleatório (Sparring)
    player_ia = BotNEAT(genome, config)
    opponent = BotRight(env)

    # Observação inicial
    obs, _ = env.reset()
    player_ia.observe(obs)
    opponent.observe(obs)
    
    # Inicia a thread de lógica (Game Loop)
    t = threading.Thread(target=runLoop, args=(env, player_ia, opponent))
    t.daemon = True # Fecha a thread se a janela fechar
    t.start()
    
    # Inicia a thread de interface gráfica (Arcade)
    arcade.run()

if __name__ == "__main__":
    main()