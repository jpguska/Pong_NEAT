# Solução IA Pong - NEAT (PPCI 2025)

Solução desenvolvida para o desafio de Pong da 2ª Competição de IA PPCI.

A solução utiliza o algoritmo **NEAT (NeuroEvolution of Augmenting Topologies)**.

## Abordagem Técnica

Foi optado o uso de Algoritmos Genéticos em vez de Aprendizado por Reforço, devido à eficiência do algoritmo NEAT para ambientes como o Pong, que possui baixa dimensionalidade e resposta contínua.

### Arquitetura da Rede Neural
O genoma da IA foi configurado com as seguintes características:

* **Entradas (14 inputs):** Posições X/Y e Velocidades X/Y das raquetes e da bola, normalizadas entre 0 e 1 (fornecido pelo arquivo `envpong.py`).
* **Saídas (3 outputs):** Neurônios correspondentes às ações:
    * Descer (-1)
    * Ficar Parado (0)
    * Subir (1)
* **Ativação:** Função `tanh` (Tangente Hiperbólica).
* **Evolução:** A topologia da rede é dinâmica, começando simples e se torna mais complexa apenas se necessário.

### Processo de Treinamento
O treinamento foi realizado simulando gerações de partidas aceleradas e sem renderização gráfica.
* **População:** Possui 50 indivíduos por geração.
* **Função de Fitness:** É baseada na pontuação acumulada (Pontuação_Jogador 1 - Pontuação_Jogador 2) e na sobrevivência.

## 📂 Estrutura dos Arquivos

* `bot.py`: **[Arquivo Principal]** Contém a classe `BotPlayer` que carrega a IA treinada e implementa a interface `act/observe` para a competição.
* `melhor_ia_neat.pkl`: O arquivo binário contendo o genoma da melhor IA treinada.
* `config-feedforward.txt`: Arquivo de configuração obrigatório do NEAT, que define parâmetros genéticos e estruturais.
* `treino_neat.py`: Script para treinar a IA e que gera o arquivo .pkl.
* `rodar_campeao.py`: Script auxiliar para visualizar a IA jogando graficamente.
* `envpong.py`: Ambiente de simulação do jogo.

## 🚀 Como Executar

### Pré-requisitos
A solução foi desenvolvida em Python 3. As dependências necessárias estão listadas abaixo:

```bash
pip install neat-python numpy arcade gym shimmy
