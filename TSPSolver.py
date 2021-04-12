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
import heapq
import itertools
np.seterr(all="raise")



class TSPSolver:
	def __init__( self, gui_view ):
		self._scenario = None
		self.dist_array = np.empty([])
		self.path = np.empty([])

	def setupWithScenario( self, scenario ):
		self._scenario = scenario


	''' <summary>
		This is the entry point for the default solver
		which just finds a valid random tour.  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of solution, 
		time spent to find solution, number of permutations tried during search, the 
		solution found, and three null values for fields not used for this 
		algorithm</returns> 
	'''
	
	def defaultRandomTour( self, time_allowance=60.0 ):
		results = {}
		cities = self._scenario.getCities()
		ncities = len(cities)
		foundTour = False
		count = 0
		bssf = None
		start_time = time.time()
		while not foundTour and time.time()-start_time < time_allowance:
			# create a random permutation
			perm = np.random.permutation( ncities )
			route = []
			# Now build the route using the random permutation
			for i in range( ncities ):
				route.append( cities[ perm[i] ] )
			bssf = TSPSolution(route)
			count += 1
			if bssf.cost < np.inf:
				# Found a valid route
				foundTour = True
		end_time = time.time()
		results['cost'] = bssf.cost if foundTour else math.inf
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results


	''' <summary>
		This is the entry point for the greedy solver, which you must implement for 
		the group project (but it is probably a good idea to just do it for the branch-and
		bound project as a way to get your feet wet).  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number of solutions found, the best
		solution found, and three null values for fields not used for this 
		algorithm</returns> 
	'''



	# Helper function for building the initial distance matrix. It loops
	# throught the matrix and looks up the cost from city to city.
	def distanceCalc(self):
		edges = self._scenario.getEdges()
		cities = self._scenario.getCities()
		self.dist_array = np.ones_like(edges) * np.inf
		for i in range(self.dist_array.shape[0]):
			for j in range(self.dist_array.shape[1]):
				if edges[i, j]:
					self.dist_array[i,j] = cities[i].costTo(cities[j])

	# Greedy run algorithm. This algorithm runs a single greedy run from
	# start city. This is later called in a loop.
	def greedy_run(self, start):
		# Initialize distance matrix
		self.distanceCalc()
		# Initialize cost
		cost = 0
		# Create a copy of the dist_array to edit.
		tmp_dist = self.dist_array.copy()
		edges = self._scenario.getEdges()
		cities = self._scenario.getCities()
		# Starts at random city

		# Greedily grabs the shortest edge
		row = tmp_dist[start, :]
		next = np.argmin(row)
		current = start
		# Appends to path
		self.path = [cities[start]]
		self.path += [cities[next]]
		idx = [start]
		idx += [next]

		while not np.isinf(np.min(row)):
			# Infities out the taken edge so that you can't go back
			cost += np.min(row)
			tmp_dist[:, current] += 1
			tmp_dist[:, current] *= np.inf

			current = next
			row = tmp_dist[current, :]
			next = np.argmin(row)
			self.path += [cities[next]]
			idx += [next]

		# Check that path can loop back to first node
		if edges[idx[-2], start] and len(self.path) > \
				self.dist_array.shape[0]:
			self.path[-1] = cities[start]
			cost += self.dist_array[idx[-2], start]
		else:
			return [], np.inf

		return self.path, cost


	def greedy( self,time_allowance=60.0 ):
		time1 = time.time()
		bssf_time = np.inf
		bssf = []
		bssf_cost = np.inf
		count = 0
		cities = self._scenario.getCities()
		results = {}
		# Run the greedy run for the time allowance
		while(time.time() - time1 < time_allowance):

			for i in range(len(cities) - 1):
				path, cost = self.greedy_run(start=i)
				if cost < bssf_cost:
					# Save the best solution information.
					bssf_cost = cost
					bssf_time = time.time() - time1
					bssf = path

				if cost < np.inf:
					count += 1
			break
		# Return out the required parameters.
		bssf_sol = TSPSolution(bssf[:-1])
		results['cost'] = bssf_sol.cost
		results['time'] = bssf_time
		results['count'] = count
		results['soln'] = bssf_sol
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results
		# return {'cost':bssf_cost, 'time':bssf_time, 'count':count, 'soln':TSPSolution(bssf), 'max':0, 'total':0, 'prunned':0}




	
	
	
	''' <summary>
		This is the entry point for the branch-and-bound algorithm that you will implement
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number solutions found during search (does
		not include the initial BSSF), the best solution found, and three more ints: 
		max queue size, total number of states created, and number of pruned states.</returns> 
	'''

	# Helper function for finding lower bound of a state by computing
	# the reduce matrix.
	def lower_bound(self, tmp_dist_array, path):
		# Gets a copy of the matrix to reduce
		reduce_matrix = tmp_dist_array.copy()
		cost = 0
		# Steps through the rows and reduces the rows while keeping
		# a running sum of the cost.
		for i in range(self.dist_array.shape[0]):
			my_min = np.min(reduce_matrix[i,:])
			if i not in path[:-1]:
				cost += my_min
				if not np.isinf(my_min):
					reduce_matrix[i,:] -= my_min
				else:
					break

		# Steps through all the columns and reduces the columns while
		# keeping a running sum of the cost.
		for j in range(self.dist_array.shape[1]):
			my_min = np.min(reduce_matrix[:,j])
			if j not in path[1:]:
				cost += my_min
				if not np.isinf(my_min):
					reduce_matrix[:,j] -= my_min
				else:
					break
		return cost, reduce_matrix

	def branchAndBound( self, time_allowance=60.0 ):
		# Initialize variables and start timer.
		bssf_cost = np.inf
		time1 = time.time()
		new_path = []
		cost_so_far = 0
		child_count = 0
		max_len = 0
		cities = self._scenario.getCities()
		list_cities = []
		count = 0
		temp_q = []
		pruned = 0
		# Bools to control whether the round-robin is running or not
		# If pop_from_q is true, the algorithm will pop from main
		# priority queue, and in essence prioritize breadth.
		# If round_robin is true, the algorithm will pop from a
		# sub priority queue and prioritize depth. The first iteration sets
		# both bools to true, as the first run must pop from the main
		# priority queue, but the subsequent loops will pop from the
		# sub priority queue
		pop_from_q = True
		round_robin = True

		bssf_cost = np.inf
		# Use greedy to get intial BSSF.
		bssf_path = []
		for i in range(len(cities)):
			path, cost = self.greedy_run(start=np.random.randint(0,len(cities)-1))
			if cost < bssf_cost:
				bssf_time = time.time() - time1
				bssf_cost = cost
				bssf_path = path.copy()


		bssf_path = TSPSolution(bssf_path[:-1])
		bssf_cost = bssf_path.cost
		start = np.random.randint(0, self.dist_array.shape[0] - 1)
		tmp_dist_array = self.dist_array.copy()
		greedy_cost = bssf_cost
		print(bssf_cost)
		new_path = [start]
		q = []
		sub_q = []
		loop_itr = 0

		temp_cost, reduce_matrix = self.lower_bound(self.dist_array,
													[])

		# Build main and sub priority queue
		heapq.heappush(sub_q, (temp_cost, (new_path, reduce_matrix)))
		heapq.heappush(q, (temp_cost, (new_path, reduce_matrix)))

		# Start main loop.
		while len(q) > 0 and (time.time() - time1) < time_allowance:
			loop_itr += 1
			# Get max queue size
			temp_qlen = len(q)
			if (count > 1 and bssf_cost < (1 - len(cities) * 5/(loop_itr)) * greedy_cost):
				# Turn off round-robin once a good enough solution has been found.
				# print("round robin = false")
				round_robin = False
			if max_len < temp_qlen:
				max_len = temp_qlen
			if pop_from_q:
				branch_cost, new_path = heapq.heappop(q)
				new_path, dist_array = new_path
				pop_from_q = False

			if branch_cost > bssf_cost:
				pruned += 1
				continue
			# Create a solution if a solution has been found
			if len(new_path) == dist_array.shape[0]:
				# Create Solution
				list_cities = [cities[idx] for idx in new_path]
				sol = TSPSolution(list_cities)
				count += 1
				sub_q = q.copy()
				# Set flag so next iteration pops from q
				pop_from_q = True
				if sol.cost < bssf_cost:
					# If solution is better update bssf.
					print(start, branch_cost)
					bssf_cost = sol.cost
					bssf_path = sol
					bssf_time = time.time() - time1
					temp_q = q.copy()
					q = []
					for t in temp_q:
						if t[0] < bssf_cost:
							heapq.heappush(q, t)
						else:
							pruned += 1
				continue

			tmp_dist_array = dist_array.copy()
			sub_q = []
			# Get each edge.
			for i in range(tmp_dist_array.shape[0]):
				# Make sure edge isn't already in path.

				if i not in new_path:
					path_cost = dist_array[new_path[-1],i]
					tmp_dist_array[new_path[-1], :] += 1
					tmp_dist_array[new_path[-1], :] *= np.inf
					tmp_dist_array[:,i] += 1
					tmp_dist_array[:,i] *= np.inf
					tmp_dist_array[i,new_path[-1]] = np.inf
					temp_cost, reduce_matrix = self.lower_bound(tmp_dist_array, new_path + [i])

					temp_cost += branch_cost + path_cost
					child_count += 1
					if temp_cost < bssf_cost:
						# heapq.heappush(q, (temp_cost, (new_path + [i], reduce_matrix)))
						# Push results onto the sub_q. This forces depth
						# Prioritization
						heapq.heappush(sub_q, (temp_cost, (new_path + [i], reduce_matrix)))
					else:
						# Count pruned states not added to queue
						pruned += 1

					tmp_dist_array[:, i] = dist_array[:, i].copy()
					tmp_dist_array[i,new_path[-1]] = dist_array[i,new_path[-1]].copy()
			if len(sub_q) > 0 and round_robin:
				# Pop off for round robin
				branch_cost, new_path = heapq.heappop(sub_q)
				new_path, dist_array = new_path
				# Merge sub_q into q, i.e. the leftovers
				q = list(heapq.merge(q, sub_q))
			elif not round_robin:
				# Switch to popping from q
				pop_from_q = True
				q = list(heapq.merge(q, sub_q))
			else:
				# Current branch ended in inf, and needs to restart.
				pop_from_q = True

		# print(loop_itr)
		# print(len(q))
		# print(1 - len(cities) * 10/loop_itr)
		print("Running Time {}".format(time.time() - time1))
		print(bssf_cost / greedy_cost)
		results = {}
		results['cost'] = bssf_path.cost
		results['time'] = bssf_time
		results['count'] = count
		results['soln'] = bssf_path
		results['max'] = max_len
		results['total'] = child_count
		results['pruned'] = pruned
		return results



	''' <summary>
		This is the entry point for the algorithm you'll write for your group project.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number of solutions found during search, the 
		best solution found.  You may use the other three field however you like.
		algorithm</returns> 
	'''

	def breed(self, parent1, parent2):
		child = []
		city1 = np.random.randint(0,len(parent1))
		city2 = np.random.randint(0,len(parent1))

		start = min(city1,  city2)
		end = max(city1, city2)

		for i in range(start, end):
			child.append(parent1[i])

		for city in parent2:
			if city not in child:
				child.append(city)

		if np.random.random() < 0.10:
			child = self.mutate(child)

		return child



	def mutate(self, solution_in):
		solution = solution_in.copy()
		city1 = np.random.randint(0, len(solution) - 1)
		city2 = np.random.randint(0, len(solution) - 1)

		temp = solution[city1]
		solution[city1] = solution[city2]
		solution[city2] = temp

		return solution


	def make_Return(self, bssf_cost, greedy_cost, bssf_city_list, bssf_time, count, population_length):
		results = {}
		bssf_path = TSPSolution(bssf_city_list)
		results['cost'] = bssf_path.cost
		results['time'] = bssf_time
		results['count'] = count
		results['soln'] = bssf_path
		results['max'] = population_length
		results['total'] = 0
		results['pruned'] = 0
		return results
		
	def fancy( self,time_allowance=60.0 ):
		# Initialize variables and start timer.
		bssf_cost = np.inf
		time1 = time.time()
		new_path = []
		cost_so_far = 0
		child_count = 0
		max_len = 0
		cities = self._scenario.getCities()
		list_cities = []
		bssf_cost = np.inf
		population = []
		population_cost = []

		# Use greedy to get intial BSSF.
		for i in range(len(cities)):
			path, cost = self.greedy_run(start=i)
			if len(path)-1 == len(cities):
				greedy_sol = TSPSolution(path[:-1])
			else:
				continue
			if greedy_sol.cost < np.inf:
				population += [path]
				population_cost += [cost]
			if greedy_sol.cost < bssf_cost:
				bssf_path = path.copy()
				bssf_cost = greedy_sol.cost
				bssf_time = time.time() - time1
				greedy_time = bssf_time

		print("Greedy Time: {}".format(greedy_time))
		bssf_city_list = bssf_path[:-1]
		bssf_path = TSPSolution(bssf_path[:-1])
		bssf_cost = bssf_path.cost
		greedy_cost = bssf_cost
		print("Greedy: {}".format(bssf_cost))

		# bssf_path = []
		# for city in cities:
		# 	bssf_path += [city]
		# bssf_path += [cities[0]]
		# bssf_city_list = bssf_path[:-1]
		# bssf_path = TSPSolution(bssf_path[:-1])
		# bssf_cost = bssf_path.cost
		# greedy_cost = bssf_cost
		# print("Junk: {}".format(bssf_cost))


		population_count = 0

		count = 0



		while time.time() - time1 < time_allowance:

			for i in range(len(cities) - 1):
				population_cost, _, population = (list(t) for t in
								zip(*sorted(zip(population_cost, np.arange(len(population)), population))))
				for j in range(len(population)):
					# child = self.haveChildren(population[i])
					child = self.breed(population[i], population[j])

					# child = self.breed(population[i], population[np.random.randint(0, len(population)-1)])
					child_sol = TSPSolution(child)
					if child_sol.cost < np.inf:
						population += [child]
						population_cost += [child_sol.cost]
						population_count += 1
						count += 1

					if child_sol.cost < bssf_cost:
						bssf_city_list = child
						bssf_cost = child_sol.cost
						print("Genetic: {}".format(bssf_cost))
						bssf_time = time.time() - time1


					if time.time() - time1 > time_allowance:
						print(bssf_cost / greedy_cost)
						population_length = len(population)
						results = self.make_Return(bssf_cost, greedy_cost,
												   bssf_city_list,
												   bssf_time, count,
												   population_length)
						print("Total Time: {}".format(
							time.time() - time1))
						print("BSSF Time: {}".format(bssf_time))
						return results




			zipped_cities = zip(population_cost, np.arange(len(population)) ,population)
			sorted_pairs = sorted(zipped_cities)

			# selection = np.random.normal(0, len(population) / 3, size=(1, len(cities)))
			# selection = np.abs(selection)
			# selection[selection > (len(sorted_pairs)-1)] = len(sorted_pairs) - 1
			# selection = np.array(selection, int)
			# selection = selection.squeeze()
			selection = np.random.random_integers(len(cities), len(population)-1, len(cities))
			population_length = len(population)
			population = []
			population_cost = []

			for j in range(len(cities)):
				population += [sorted_pairs[j][2]]
				population_cost += [sorted_pairs[j][0]]

			for j in selection:
				population += [sorted_pairs[j][2]]
				population_cost += [sorted_pairs[j][0]]


		results = {}
		population_length = len(population)

		bssf_path = TSPSolution(bssf_city_list)
		results['cost'] = bssf_path.cost
		results['time'] = bssf_time
		results['count'] = count
		results['soln'] = bssf_path
		results['max'] = population_length
		results['total'] = 0
		results['pruned'] = 0
		return results


