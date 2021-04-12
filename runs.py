#!/usr/bin/python3

from which_pyqt import PYQT_VER
if PYQT_VER == 'PYQT5':
    from PyQt5.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT4':
    from PyQt4.QtCore import QLineF, QPointF
else:
    raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))


import time
import numpy as np
from TSPClasses import *
from TSPSolver import *
from Proj5GUI import *
import heapq
import itertools
np.seterr(all="raise")
import pickle


class Run:

    def __init__( self ):
        super(Run,self).__init__()
        self._MAX_SEED = 1000

        self._scenario = None
        self.solver = TSPSolver([])
        self.genParams = {'size':None,'seed':None,'diff':None}
        self.ALGORITHMS = [('Default                            ', 'defaultRandomTour'), ('Greedy', 'greedy'), ('Branch and Bound', 'branchAndBound'), ('Fancy', 'fancy')]


    def newPoints(self):
        # TODO - ERROR CHECKING!!!!

        seed = int(self.curSeed)
        random.seed(seed)
        self.data_range = {'x': [-1.5, 1.5], 'y': [-1.0, 1.0]}
        ptlist = []
        RANGE = self.data_range
        xr = self.data_range['x']
        yr = self.data_range['y']
        npoints = int(self.size)
        while len(ptlist) < npoints:
            x = random.uniform(0.0, 1.0)
            y = random.uniform(0.0, 1.0)
            if True:
                xval = xr[0] + (xr[1] - xr[0]) * x
                yval = yr[0] + (yr[1] - yr[0]) * y
                ptlist.append(QPointF(xval, yval))
        return ptlist

    def generateNetwork(self, npoints, seed, time, algorithm_idx):
        self.size = npoints
        self.curSeed = seed
        self.timeLimit = time
        self.alg_index = algorithm_idx
        points = self.newPoints()  # uses current rand seed
        diff = 'Hard (Deterministic)'
        rand_seed = int(self.curSeed)
        self._scenario = Scenario(city_locations=points, difficulty=diff,
                                  rand_seed=rand_seed)

        self.genParams = {'size': self.size,
                          'seed': self.curSeed, 'diff': diff}


    def randSeedClicked(self):
        new_seed = random.randint(0, self._MAX_SEED-1)
        self.curSeed.setText( '{}'.format(new_seed) )
        self.view.repaint()

    def solveClicked(self):								# need to reset display??? and say "processing..." at bottom???
        self.solver.setupWithScenario(self._scenario)

        max_time = float( self.timeLimit )
        # TODO - start on a separate thread

        #self.view.repaint()
        #app.processEvents()
        solve_func = 'self.solver.'+self.ALGORITHMS[self.alg_index][1]
        results = eval(solve_func)(time_allowance=max_time)
        if results:
            return results
        else:
            print('GOT NULL SOLUTION BACK!!')		#probably shouldn't ever use this...

#		app.processEvents()

if __name__ == '__main__':
    # Alg_idx: 0 -> Random, 1 -> Greedy, 2 -> Branch and Bound, 3 -> Fancy
    size_list = [15,30,60,100,150,200, 205, 210,220,250,300]
    seed_list = [1,2,3,4,5]

    size_list = [15,30]
    seed_list = [1,2]

    t = Run()
    fancy = []
    bb = []
    greedy = []
    rndom = []

    rndom_cost = 0
    greedy_cost = 0
    bb_cost = 0
    fancy_cost = 0

    rndom_time = 0
    greedy_time = 0
    bb_time = 0
    fancy_time = 0

    rndom_time_avg = []
    rndom_cost_avg = []

    greedy_time_avg = []
    greedy_cost_avg = []

    bb_time_avg = []
    bb_cost_avg = []

    fancy_time_avg = []
    fancy_cost_avg = []

    for i in size_list:
        for j in seed_list:
            t.generateNetwork(i, j, 300, 0)
            rndom.append(t.solveClicked())

            t.generateNetwork(i, j, 300, 1)
            greedy.append(t.solveClicked())

            t.generateNetwork(i, j, 300, 2)
            bb.append(t.solveClicked())

            t.generateNetwork(i, j, 300, 3)
            fancy.append(t.solveClicked())

        rndom_cost = 0
        greedy_cost = 0
        bb_cost = 0
        fancy_cost = 0

        rndom_time = 0
        greedy_time = 0
        bb_time = 0
        fancy_time = 0

        r_count = 0
        gr_count = 0
        bb_count = 0
        ge_count = 0
        for i in range((len(fancy) - len(seed_list)), len(fancy)):
            if not np.isinf(rndom[i]['cost']):
                rndom_cost += rndom[i]['cost']
                rndom_time += rndom[i]['time']
                r_count += 1

            if not np.isinf(greedy[i]['cost']):
                greedy_cost += greedy[i]['cost']
                greedy_time += greedy[i]['time']
                gr_count += 1

            if not np.isinf(bb[i]['cost']):
                bb_cost += bb[i]['cost']
                bb_time += bb[i]['time']
                bb_count += 1

            if not np.isinf(fancy[i]['cost']):
                fancy_cost += fancy[i]['cost']
                fancy_time += fancy[i]['time']
                ge_count += 1

        rndom_cost = rndom_cost / r_count
        greedy_cost = greedy_cost / gr_count
        bb_cost = bb_cost / bb_count
        fancy_cost = fancy_cost / ge_count

        rndom_time = rndom_time / r_count
        greedy_time = greedy_time / gr_count
        bb_time = bb_time / bb_count
        fancy_time = fancy_time / ge_count

        rndom_time_avg.append(rndom_time)
        rndom_cost_avg.append(rndom_cost)

        greedy_time_avg.append(greedy_time)
        greedy_cost_avg.append(greedy_cost)

        bb_time_avg.append(bb_time)
        bb_cost_avg.append(bb_cost)

        fancy_time_avg.append(fancy_time)
        fancy_cost_avg.append(fancy_cost)

    with open('results.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([rndom, rndom_cost_avg, rndom_time_avg, greedy, greedy_cost_avg, greedy_time_avg, bb, bb_cost_avg, bb_time_avg, fancy, fancy_cost_avg, fancy_time_avg], f)

    print("\n\n\n")
    for i in range(len(fancy_cost_avg)):
        print('Size: {}'.format(size_list[i]))
        print('Random Time: {}, Random Cost: {}, Greedy Time: {}, Greedy Cost: {}, Greedy Percent: {}, BB Time: {}, BB Cost: {}, BB Percent: {}, Fancy Time: {}, Fancy Cost: {}, Fancy Percent: {}'.format(rndom_time_avg[i], rndom_cost_avg[i], greedy_time_avg[i], greedy_cost_avg[i], greedy_cost_avg[i]/rndom_cost_avg[i], bb_time_avg[i], bb_cost_avg[i], bb_cost_avg[i]/greedy_cost_avg[i], fancy_time_avg[i], fancy_cost_avg[i], fancy_cost_avg[i]/greedy_cost_avg[i]))

    with open('results.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
        rndom, rndom_cost_avg, rndom_time_avg, greedy, greedy_cost_avg, greedy_time_avg, bb, bb_cost_avg, bb_time_avg, fancy, fancy_cost_avg, fancy_time_avg = pickle.load(f)

    junk = 1