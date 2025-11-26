# Daniel Cavalcanti Jeronymo
# Pong game using Arcade library with Neural Network training
# Atualizado para ter compatibilidade com o Arcade 3.0

import math
import numpy as np
import itertools
from enum import IntEnum
import copy
import arcade
import gym
from gym.spaces import Tuple, Box, Discrete

class Rect:
    def __init__(self, center, shape):
        self.center = center
        self.shape = shape
        self.box = self.calculateBox()
        self.vertices = self.calculateVertices()
        self.box = np.array(self.box)
        self.vertices = np.array(self.vertices)

    def calculateBox(self):
        offsets = []
        for x, width in zip(self.center, self.shape):
            offsets.append((x - width/2, x + width/2))
        return offsets

    def calculateVertices(self):
        return [v for v in itertools.product(*self.box)]

    def intersect(self, other):
        for v in other.vertices:
            if ((self.box[:,0] < v) & (v < self.box[:,1])).all():
                return True
        return False

class PongLogic:
    class PaddleMove(IntEnum):
        UP = 1
        STILL = 0
        DOWN = -1

    class GameState():
        def __init__(self, player1action, player2action, paddle1Position, paddle2Position, paddle1Velocity, paddle2Velocity, ballPosition, ballVelocity, player1Score, player2Score, time, totalTime):
            self.paddle1Position = paddle1Position
            self.paddle2Position = paddle2Position
            self.paddle1Velocity = paddle1Velocity
            self.paddle2Velocity = paddle2Velocity
            self.ballPosition = ballPosition
            self.ballVelocity = ballVelocity
            self.player1action = player1action
            self.player2action = player2action
            self.player1Score = player1Score
            self.player2Score = player2Score
            self.time = time
            self.totalTime = totalTime

    def randomBallVelocity(self, mag):
        maxAngle = math.radians(60)
        r = np.random.uniform(-1, 1, 1)[0]*maxAngle
        leftRight = np.random.choice([-1,1])
        r1 = mag*leftRight*math.cos(r)
        r2 = mag*math.sin(r)
        v = np.array([r1, r2], dtype=np.float64)
        return v

    def bounceBallTop(self, state):
        state.ballVelocity[1] = -state.ballVelocity[1]
        state.ballPosition += state.ballVelocity*self.dt

    def bounceBallPaddle(self, id, state):
        maxAngle = math.pi/3
        BOUNCE_FACTOR = 1.1
        ballSpeed = np.linalg.norm(state.ballVelocity)

        if id == 1:
            paddleposY = state.paddle1Position[1]
        else:
            paddleposY = state.paddle2Position[1]

        relativeIntersect = (state.ballPosition[1] - paddleposY) / (self.paddleShape[1]/2 + self.ballShape[1]/2)
        bounceAngle = relativeIntersect * maxAngle

        state.ballVelocity[0] = ballSpeed*BOUNCE_FACTOR*math.cos(bounceAngle)
        state.ballVelocity[1] = ballSpeed*BOUNCE_FACTOR*math.sin(bounceAngle)

        if id == 2:
            state.ballVelocity[0] *= -1

        state.ballVelocity = np.minimum(state.ballVelocity, self.ballVelocityMag*100)
        state.ballVelocity = np.maximum(state.ballVelocity, -self.ballVelocityMag*100)
        state.ballPosition += state.ballVelocity*self.dt

    def __init__(self, dt, windowShape, paddleShape, paddleOffset, paddleVelocity, ballShape, ballPosition, ballVelocityMag, debugPrint=True):
        self.windowWidth, self.windowHeight = windowShape
        self.boundTop = 0
        self.boundBottom = self.windowHeight
        self.boundLeft = 0
        self.boundRight = self.windowWidth

        self.paddleShape = paddleShape
        self.paddleVelocity = np.array([0, paddleVelocity])
        self.ballShape = ballShape
        self.ballVelocityMag = ballVelocityMag

        paddle1Position = np.array([self.windowWidth*paddleOffset, self.windowHeight//2])
        paddle2Position = np.array([self.windowWidth*(1-paddleOffset), self.windowHeight//2])
        paddle1Velocity = np.zeros(2)
        paddle2Velocity = np.zeros(2)
        ballPosition = np.array(ballPosition)
        ballVelocity = self.randomBallVelocity(ballVelocityMag)
        self.s0 = self.GameState(PongLogic.PaddleMove.STILL, PongLogic.PaddleMove.STILL, paddle1Position, paddle2Position, paddle1Velocity, paddle2Velocity, ballPosition, ballVelocity, 0, 0, 0.0, 0.0)

        self.states = [self.s0]
        self.dt = dt
        self.debugPrint = debugPrint

    def reset(self, winnerId):
        state = copy.deepcopy(self.s0)
        state.player1Score = self.states[-1].player1Score + (winnerId == 1)
        state.player2Score = self.states[-1].player2Score + (winnerId == 2)
        state.totalTime = self.states[-1].totalTime
        state.ballVelocity = self.randomBallVelocity(self.ballVelocityMag)
        self.states += [state]
        if self.debugPrint:
            print(f"---SCORE--- P1: {self.states[-1].player1Score} | P2: {self.states[-1].player2Score}")

    def update(self, player1action, player2action):
        state = copy.deepcopy(self.states[-1])
        state.time += self.dt
        state.totalTime += self.dt
        state.player1action, state.player2action = player1action, player2action

        state.paddle1Velocity = self.paddleVelocity*player1action
        state.paddle2Velocity = self.paddleVelocity*player2action
        state.paddle1Position += state.paddle1Velocity*self.dt
        state.paddle2Position += state.paddle2Velocity*self.dt

        paddle1Rect = Rect(state.paddle1Position, self.paddleShape)
        paddle2Rect = Rect(state.paddle2Position, self.paddleShape)
        ballRect = Rect(state.ballPosition, self.ballShape)

        paddleOffset = paddle1Rect.box[1][0] - self.boundTop
        if paddleOffset < 0: state.paddle1Position[1] -= paddleOffset
        paddleOffset = paddle2Rect.box[1][0] - self.boundTop
        if paddleOffset < 0: state.paddle2Position[1] -= paddleOffset
        paddleOffset = self.boundBottom - paddle1Rect.box[1][1]
        if paddleOffset < 0: state.paddle1Position[1] += paddleOffset
        paddleOffset = self.boundBottom - paddle2Rect.box[1][1]
        if paddleOffset < 0: state.paddle2Position[1] += paddleOffset

        state.ballPosition += state.ballVelocity*self.dt

        if state.ballPosition[1] <= self.boundTop or state.ballPosition[1] >= self.boundBottom:
            self.bounceBallTop(state)

        if paddle1Rect.intersect(ballRect) or ballRect.intersect(paddle1Rect):
            self.bounceBallPaddle(1, state)
        if paddle2Rect.intersect(ballRect) or ballRect.intersect(paddle2Rect):
            self.bounceBallPaddle(2, state)

        self.states += [state]

        if state.ballPosition[0] < self.boundLeft:
            self.reset(2)
        if state.ballPosition[0] > self.boundRight:
            self.reset(1)

class PongEnv(gym.Env):
    def __init__(self, width=400, height=400, FPS=30.0, debugPrint=False):
        super().__init__()
        self.createGame(width, height, FPS, debugPrint)
        self.action_space = Discrete(3, start=-1)
        self.observation_space = Tuple((
            Box(0, 1), Box(0, 1), Box(0, 1), Box(0, 1),
            Box(0, 1), Box(0, 1), Box(0, 1), Box(0, 1),
            Box(0, 1), Box(0, 1), Box(0, 1), Box(0, 1),
            Discrete(3, start=-1), Discrete(3, start=-1)
        ))

    def step(self, actionp1, actionp2):
        reward = 0
        done = False
        truncated = False
        info = {}
        self.game.update(actionp1, actionp2)
        reward = self.game.states[-1].player1Score - self.game.states[-1].player2Score
        obs = self.getInputs(self.game.states[-1])
        self.steps +=1
        return obs, reward, done, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.createGame(self.game.windowWidth, self.game.windowHeight, 1/self.game.dt, self.game.debugPrint)
        obs = self.getInputs(self.game.states[-1])
        info = {}
        return obs, info

    def render(self):
        pass
    
    def createGame(self, width, height, simFPS, debugPrint):
        self.game = PongLogic(1/simFPS, windowShape=(width, height), paddleShape=(10,30), paddleOffset=0.15, paddleVelocity=200, ballShape=(5,5), ballPosition=(width/2,height/2), ballVelocityMag=100, debugPrint=debugPrint)
        self.steps = 0

    def getInputs(self, state):
        inputs = []
        inputs += [state.paddle1Position[0]/self.game.windowWidth]
        inputs += [state.paddle1Position[1]/self.game.windowHeight]
        inputs += [state.paddle1Velocity[0]/(self.game.ballVelocityMag*100)]
        inputs += [state.paddle1Velocity[1]/(self.game.ballVelocityMag*100)]
        inputs += [state.paddle2Position[0]/self.game.windowWidth]
        inputs += [state.paddle2Position[1]/self.game.windowHeight]
        inputs += [state.paddle2Velocity[0]/(self.game.ballVelocityMag*100)]
        inputs += [state.paddle2Velocity[1]/(self.game.ballVelocityMag*100)]
        inputs += [state.ballPosition[0]/self.game.windowWidth]
        inputs += [state.ballPosition[1]/self.game.windowHeight]
        inputs += [state.ballVelocity[0]/(self.game.ballVelocityMag*100)]
        inputs += [state.ballVelocity[1]/(self.game.ballVelocityMag*100)]
        inputs += [state.player1action]
        inputs += [state.player2action]
        return inputs
    
class PongGUIEnv(arcade.Window, PongEnv):
    def __init__(self, width=400, height=400, FPS=30.0):
        # Configurações da janela arcade
        super().__init__(width, height, 'CYBERPONG', update_rate=1/FPS, draw_rate=1/FPS)
        PongEnv.__init__(self)
        
        self.background_color = arcade.color.ARSENIC
        self.player1action = PongLogic.PaddleMove.STILL
        self.player2action = PongLogic.PaddleMove.STILL

    def on_draw(self):
        self.clear()
        
        # Pega as posições
        p1_pos = self.game.states[-1].paddle1Position
        p2_pos = self.game.states[-1].paddle2Position
        ball_pos = self.game.states[-1].ballPosition
        
        w_pad, h_pad = self.game.paddleShape
        w_ball, h_ball = self.game.ballShape

    
        # Desenha Raquete 1
        rect1 = arcade.XYWH(p1_pos[0], p1_pos[1], w_pad, h_pad)
        arcade.draw_rect_filled(rect1, arcade.color.WHITE_SMOKE)
        
        # Desenha Raquete 2
        rect2 = arcade.XYWH(p2_pos[0], p2_pos[1], w_pad, h_pad)
        arcade.draw_rect_filled(rect2, arcade.color.WHITE_SMOKE)
        
        # Desenha Bola
        rect_ball = arcade.XYWH(ball_pos[0], ball_pos[1], w_ball, h_ball)
        arcade.draw_rect_filled(rect_ball, arcade.color.WHITE_SMOKE)

    def update(self, dt):
        pass 

    def on_key_press(self, key, key_modifiers):
        if key == arcade.key.W:
            self.player1action = PongLogic.PaddleMove.UP
        elif key == arcade.key.S:
            self.player1action = PongLogic.PaddleMove.DOWN
        if key == arcade.key.UP:
            self.player2action = PongLogic.PaddleMove.UP
        elif key == arcade.key.DOWN:
            self.player2action = PongLogic.PaddleMove.DOWN

    def on_key_release(self, key, key_modifiers):
        if key == arcade.key.W or key == arcade.key.S:
            self.player1action = PongLogic.PaddleMove.STILL
        if key == arcade.key.UP or key == arcade.key.DOWN:
            self.player2action = PongLogic.PaddleMove.STILL