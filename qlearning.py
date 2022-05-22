import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import glob
import argparse
import os, sys
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils import data
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, f1_score
import pickle
import sys, os, logging, glob
from logging import info
logging.basicConfig(level=logging.INFO, format='%(pathname)s:%(lineno)d: [%(asctime)ss%(msecs)03d]:%(message)s', datefmt='%Hh%Mm%S')

from sklearn.tree import DecisionTreeClassifier
import random

from ptflops import get_model_complexity_info
from thop import profile
from torchsummary import summary
from torchscan import summary as scan
from torchvision.models import resnet50
from utillc import *

t  = lambda x : torch.tensor(x).to('cuda')

class Env :
    W, H = 100, 100
    actions = [ (1,0), (0, -1), (-1, 0), (0, 1) ]
    state = (20, 20)

    def reset(self) :
        self.state = (20, 20)
        self.state = np.random.randint((self.W, self.H))
        self.state = t(self.state)
        self.traj = [ self.state ]                
    
    def __init__(self) :
        self.center = t([self.W/2, self.H/2])
        KK = 10
        self.actions = t(self.actions) / KK
        self.reset()

        W, H = self.W, self.H

        self.obstacles = []
        self.obstacles += [ (x, 0) for x in range(W)]
        self.obstacles += [ (x, H) for x in range(W)]
        self.obstacles += [ (0, y) for y in range(H)]
        self.obstacles += [ (W, y) for y in range(H)]
        self.obstacles += [ (x, H/3) for x in range(W//3, 2*W//3)]
        self.obstacles += [ (W/3, y) for y in range(H//3, 2*H//3)]
        self.obstacles += [ (W*2/3, y) for y in range(H//3, 2*H//3)]

        self.epsi = torch.tensor(0.01).to('cuda')        
        self.draw()


        
    def draw(self) :
        EKO()
        x = np.linspace(0, self.W, self.W)
        y = np.linspace(0, self.H, self.H)
        xv, yv = np.meshgrid(x, y, indexing='ij')
        EKOX(xv.shape)
        xy = np.stack((xv, yv), axis=2)
        EKOX(xy.shape)
        im = self.reward(t(xy))[0].cpu().numpy()
        EKOX(im.shape)
        EKOX(TYPE(im))
        
        EKO()        
        xs = [ p[0] for p in self.obstacles]
        ys = [ p[1] for p in self.obstacles]
        xts = [ p.cpu().numpy()[0] for p in self.traj]
        yts = [ p.cpu().numpy()[1] for p in self.traj]
        ss = self.state.cpu().numpy()
        def fff() :
            plt.imshow(im)
            plt.scatter(ys, xs, s=5, color='red')
            plt.scatter(yts, xts, s=5, color='blue')
            plt.scatter(yts[0], xts[0], s=5, color='orange')
            plt.scatter(self.W/2, self.H/2, s=5, color='green')
        def ff(fig) :
            fff()
        EKOF(ff)
        #fff();  plt.show()
        

    def reward(self, state) :
        #EKOX(state.shape)
        d = (state - self.center).norm(2, dim=2) / self.W # 0.5 = max, 0 = min
        #EKOX(d.shape)
        #EKOX(TYPE(d.cpu().numpy()))
        rewrd  = 1. / torch.max(d, self.epsi) / 100 # very close => reward = 100


        # loin des obstacles => big reward
        o = [ (state - t(e)).norm(2, dim=2) / self.W for e in self.obstacles]
        o = torch.stack(o, dim=2)
        o1 = torch.min(o, dim=2)[0]
        #EKOX(o.shape)
        #rewrd = o1

        #EKOX(torch.min(o))
        done = (d < 0.01).any() or (torch.min(o) < 0.01).any()
        
        return rewrd, done
    
    def next(self, action_idx) :
        #EKOX(action_idx)
        action = self.actions[action_idx]
        new_state  = self.state + action
        #EKOX(new_state.shape)
        self.state = new_state
        self.traj.append(self.state)

        reward, done = self.reward(new_state.unsqueeze(0).unsqueeze(0))
        #EKOX(reward)
        return reward, new_state, done
        

class QLearning :

    def __init__(self, env) :
        self.env = env
        EKOX(env.actions.shape)
        EKOX(env.state.shape)
        depth = 3
        D = 7
        din = env.actions.shape[1] + env.state.shape[0]
        EKOX(din)
        kk = [ nn.Linear(D, D), nn.ReLU() ] * depth
        layers = ([  nn.Linear(din, D), nn.ReLU() ] +  kk + [ nn.Linear(D, 1), nn.Sigmoid()])
        self.model = nn.Sequential(*layers)
        self.model = self.model.cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.SmoothL1Loss()
        self.alpha =  0.1
        self.gamma =  0.99
        self.store = {}
        
    def F(self, state, action) :
        input = torch.cat((state, action)).to('cuda').float()
        EKOX(input.shape)
        return self.model(input)

    def G(self, state) :
        actionsn = env.actions.shape[0]
        states = state.repeat(actionsn, 1)
        input = torch.cat(( env.actions, states), axis=1).float()
        return self.model(input)
        
    def train(self) :
        for i in range(50) : self.episode()


        
    def episode(self) :
        self.env.reset()
        for i in tqdm.tqdm(range(2000)) :
            qs = self.G(self.env.state)
            best = torch.argmax(qs)
            reward, next_state, done = env.next(best)
            self.store[(self.env.state, best, next_state)] = reward            
            qs1 = self.G(next_state)
            best1 = torch.argmax(qs1)
            Y =  ( reward + self.gamma *  qs1[best1])[0]
            #EKOX(Y.shape)
            #EKOX(qs[best].shape)
            loss = self.criterion(qs[best], Y)
            loss.backward()
            self.optimizer.step()
            if done : break
        self.env.draw()
        

if __name__ == "__main__":

    
    parser = argparse.ArgumentParser(description='qlearning')
    parser.add_argument('--date', default="")
    args = parser.parse_args()
    env = Env()
    qlearning = QLearning(env)
    qlearning.train()
        
        
        

        

        
        
