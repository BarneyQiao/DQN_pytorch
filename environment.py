# -*- coding: utf-8 -*-
"""
@Time          : 2020/11/7 12:26
@Author        : BarneyQ
@File          : environment.py
@Software      : PyCharm
@Description   : Environment
@Modification  :
    @Author    :
    @Time      :
    @Detail    :

"""
import gym
import numpy as np

class MedicalEnv(gym.Env):
    #----- image show -----
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }
    #-----Initialize action space and state space -----
    def __init__(self):
        self.W = 64
        self.L = 64
        self.channel = 3
        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = gym.spaces.Box(0,255, shape=(self.W,self.L,self.channel))
        # print(self.observation_space)
        self.state = [self.W/2,self.L/2,self.W/2,self.L/2]
    #----- communications between agent and env -----
    def step(self, action):
        # action(5):
        #   delta_x,
        #   delta_y,
        #   flatter_x,
        #   flatter_y
        #   scale
        x_tp = self.state[0]
        y_tp = self.state[1]
        x_br = self.state[2]
        y_br = self.state[3]

        # operate x,y
        delta_x = action[0]
        delta_y = action[1]
        flatter_x = action[2]
        flatter_y = action[3]
        scale = action[4]
        # move
        x_tp = x_tp + delta_x
        y_tp = y_tp + delta_y
        x_br = x_br + delta_x
        y_br = y_br + delta_y
        # flatter
        if flatter_x>0:
            x_br = x_br + flatter_x
        elif flatter_x<0:
            x_tp = x_tp + flatter_x

        if flatter_y>0:
            y_tp = y_tp + flatter_y
        elif flatter_y<0:
            y_br = y_br + flatter_y
        # scale
        if scale != 0:
            x_tp = x_tp + scale
            y_tp = y_tp + scale
            x_br = x_br + scale
            y_br = y_br + scale

        # cannot move out of range
        if x_tp <= 0:
            x_tp = 0
        if x_tp >= self.W:
            x_tp = self.W

        if y_tp <= 0:
            y_tp = 0
        if y_tp >= self.L:
            y_tp = self.L

        if x_br <= 0:
            x_br = 0
        if x_br >= self.W:
            x_br = self.W

        if y_br <= 0:
            y_br = 0
        if y_br >= self.L:
            y_br = self.L

        # reward
        if x_br>x_tp | y_br>y_tp :
            reward = -1 # no box punishment
            done = True
        else:
            reward = 0
            boxes2 = [226, 132, 455, 323]
            iou_c = self.iou(self.state,boxes2)
            self.state = [x_tp, y_tp, x_br, y_br]
            iou_n = self.iou(self.state,boxes2)

            if iou_n > iou_c:
                if iou_n >= 0.6:
                    reward = 1
                    done = True
                else:
                    done = False
                    reward = 1
            elif iou_n < iou_c:
                reward = -1
                done = True
            else:
                reward = 0
                done = False




        return self.state, reward, done, {}
    #----- reset env -----
    def reset(self):
        self.state = [self.W / 2, self.L / 2, self.W / 2, self.L / 2]
        return self.state
    #----- image render-----
    def render(self, mode='human'):
        return None
    #----- close -----
    def close(self):
        return None

    #----- compute IOU -----
    def iou(self,boxes1,boxes2):

        mx = min(boxes1[0], boxes2[0])
        Mx = max(boxes1[2], boxes2[2])
        my = min(boxes1[1], boxes2[1])
        My = max(boxes1[3], boxes2[3])
        w1 = boxes1[2] - boxes1[0]
        h1 = boxes1[3] - boxes1[1]
        w2 = boxes2[2] - boxes2[0]
        h2 = boxes2[3] - boxes2[1]

        uw = Mx - mx
        uh = My - my
        cw = w1 + w2 - uw
        ch = h1 + h2 - uh
        mask = ((cw <= 0) + (ch <= 0) > 0)
        area1 = w1 * h1
        area2 = w2 * h2
        carea = cw * ch
        carea[mask] = 0
        uarea = area1 + area2 - carea
        return carea / uarea