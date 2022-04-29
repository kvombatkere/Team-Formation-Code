## Implementation for Team Formation problem
## Balancing Task Coverage vs. Maximum Expert Load
## Karan Vombatkere, Dec 2021

#Import and logging config
import numpy as np
from heapq import heappop, heappush, heapify
import time, random
import matplotlib.pyplot as plt
import logging
logging.basicConfig(format='%(asctime)s |%(levelname)s: %(message)s', level=logging.INFO)


class TeamFormationProblem:
    '''
    Class to implement base framework for Team Formation problem
    The goal is to optimally Balance Task Coverage vs. Maximum Expert Load
    Implement lazy greedy evaluation using a max heap to create Team Assignment

    Parameters
    ----------------------------------------------------------------------
        m_tasks : list of m tasks; each task is a list of skills (stored as self.tasks)
        n_experts : list of n experts; each expert is a list of skills (stored as self.experts)
    
    Internal variables
    ----------------------------------------------------------------------
    self.taskAssignment  : Created (n x m) matrix to store task assignment
                            n rows represent the n experts, m columns are for m tasks
    self.currentCoverageList    : list of coverages of each task (length m)
    self.currentExpertUnionSkills   : list of sets of union of skills of experts assigned to tasks (length m)
    self.maxHeap    : max heap for ordering expert-task edges (size n x m)
    self.maxWorkloadThreshold   : thresholds for max expert workload to check upto
    '''

    def __init__(self, m_tasks, n_experts, max_workload_threshold=200):
        '''
        Initialize problem instance with m tasks, n experts
        ARGS:
            m_tasks : list of m tasks; each task is a list of skills
            n_experts : list of n experts; each expert is a list of skills
            max_workload_threshold: maximum workload threshold to attempt to solve for, default=2 for testing
        '''
        self.tasks = m_tasks
        self.m = len(self.tasks)

        self.experts = n_experts
        self.n = len(self.experts)
        self.maxWorkloadThreshold = max_workload_threshold

        #self.taskAssignment = np.zeros((self.n, self.m))
        logging.info('------------Team Formation Problem initialized with {} tasks and {} experts---------'.format(self.m,self.n))

    
    
    def getExpertsAssignedToTask(self, taskAssignment, task_index):
        '''
        Given a taskAssignment, and task index get the experts currently assigned to this task
        ARGS:
            taskAssignment  : current task assigment
            task_index  : index of task in original list m_tasks
        RETURN:
            assigned_experts_indices    : list of indices of experts assigned to task with index=task_index
        '''
        assigned_experts_arr = taskAssignment[:,task_index]
        assigned_experts_indices = []
        
        #Retrieve indices of experts assigned to task
        for expert_index, assigned in enumerate(assigned_experts_arr):
            if assigned == 1:
                assigned_experts_indices.append(expert_index)

        return assigned_experts_indices


    
    def displayTaskAssignment(self, taskAssignment):
        '''
        Given a taskAssignment, print the assignment out for each expert
        ARGS:
            taskAssignment  : current task assigment
        '''
        for j in range(self.m):
            #Get experts assigned to T_j
            T_j_assigned_experts_indices = self.getExpertsAssignedToTask(taskAssignment, j)
            
            print("Experts Assigned to Task {}: {}".format(j, T_j_assigned_experts_indices))


    def maximumExpertLoad(self, taskAssignment):
        '''
        Given a taskAssignment, compute the maximum expert load
        ARGS:
            taskAssignment  : current task assigment
        RETURN: 
            max_load    : Maximum expert workload
        '''
        max_load = 0

        for i in range(self.n):
            expert_i_tasks = taskAssignment[i,:]
            expert_i_load = np.sum(expert_i_tasks)
            
            if expert_i_load > max_load:
                max_load = expert_i_load
                
                if max_load == self.m:
                    return max_load

        return max_load



    def updateCurrentCoverageList(self, bestExpertTaskEdge, delta_coverage):
        '''
        Given a bestExpertTaskEdge and delta_coverage, update the current coverage list
        Updates self.currentCoverageList
        ARGS:
            bestExpertTaskEdge  : best expert task edge
            delta_coverage  : increase in coverage
        '''
        self.currentCoverageList[bestExpertTaskEdge['task_index']] += delta_coverage
        return

    

    def updateExpertUnionSkillsList(self, bestExpertTaskEdge):
        '''
        Given a bestExpertTaskEdge, update the expert union skills list
        Updates self.currentExpertUnionSkills
        ARGS:
            bestExpertTaskEdge  : best expert task edge
        '''
        self.currentExpertUnionSkills[bestExpertTaskEdge['task_index']] = self.currentExpertUnionSkills[bestExpertTaskEdge['task_index']].union(self.experts[bestExpertTaskEdge['expert_index']])
        return



    def getBestExpertTaskEdge(self, taskAssignment, experts_copies):
        '''
        Greedily compute the best expert-task edge (assigment) that maximizes coverage in that iteration
        ARGS:
            taskAssignment : current task assigment
            experts_copies  : current list of number of copies available of experts
        
        RETURN:
            max_edge_coverage  : maximum value of change in task coverage by adding edge (i,j)
            bestExpertTaskEdge  : indices of expert and task as a dictionary
        '''
        bestExpertTaskEdge = {'expert_index':None, 'task_index':None}
        max_edge_coverage = 0
        
        for i, E_i in enumerate(self.experts):
            for j, T_j in enumerate(self.tasks):

                #Check if this expert is not yet assigned to T_j (A[i,j] = 0) AND copies are left
                if (taskAssignment[i,j] == 0) and (experts_copies[i] != 0):
                    #Retrieve union of skills of all experts assigned to T_j and add expert E_i
                    expert_skills = self.currentExpertUnionSkills[j].union(set(E_i))
                    task_skills = set(T_j)    #Get task skills as a set
                    
                    #Compute task coverage with expert added
                    T_j_coverage = len(expert_skills.intersection(task_skills))/len(task_skills)
                    delta_coverage = T_j_coverage - self.currentCoverageList[j]
                    #logging.debug("deltaCoverage={:.1f}, T_j_coverage={}".format(delta_coverage, T_j_coverage))

                    if delta_coverage > max_edge_coverage:
                        max_edge_coverage = delta_coverage
                        bestExpertTaskEdge['expert_index'] = i
                        bestExpertTaskEdge['task_index'] = j
                        
                        #Return immediately if all skills in a task are covered
                        if max_edge_coverage == 1:
                            return max_edge_coverage, bestExpertTaskEdge

        return max_edge_coverage, bestExpertTaskEdge
    

    def thresholdPlotter(self, threshold_arr, f_obj_arr, obj_labels):
        '''
        Function to plot the objective and threshold
        ARGS:
            f_obj_arr   : Nested list of objective arrays (correspond to F and baselines)
            obj_labels    : string labels for plotting
            threshold_arr  : Threshold array
        
        RETURN:
            None (plots the array)        
        '''
        plt.figure(figsize=(9,6))
        for i, f_obj in enumerate(f_obj_arr):
            plt.plot(threshold_arr, f_obj, label=obj_labels[i])
        
        title_text = 'Threshold, T_i vs. Objectives, F'
        plt.title(title_text, fontsize=11)
        plt.xlabel('Threshold, T_i')
        plt.ylabel('Objective, F')
        plt.legend(loc='upper right')
        plt.show()
        
        return None


    def greedyTaskAssignment(self, expert_copies_list):
        '''
        Greedily compute a Task Assignment given a set of tasks and experts
        ARGS:
            expert_copies_list  : list of number of copies of each expert
        
        RETURN:
            taskAssignment  : greedy task assigment, an n x m matrix
        '''
        startTime = time.perf_counter()

        #Create empty task assigment matrix
        taskAssignment_i = np.zeros((self.n, self.m), dtype=np.int8)

        #Initialize current coverage list
        self.currentCoverageList = [0 for i in range(self.m)]

        #Initialize expert union skills list as an empty list of sets
        self.currentExpertUnionSkills = [set() for j in range(self.m)]
        
        #Get first best expert-task assignment and delta coverage value
        deltaCoverage, best_ExpertTaskEdge = self.getBestExpertTaskEdge(taskAssignment_i, expert_copies_list)

        #Assign edges until there is no more coverage possible or no expert copies left
        while (deltaCoverage > 0) and (sum(expert_copies_list) != 0):

            #Add edge to assignment
            taskAssignment_i[best_ExpertTaskEdge['expert_index'], best_ExpertTaskEdge['task_index']] = 1

            ##Perform updates 
            #Decrement expert copy, Update current coverage list and expert union skills list
            expert_copies_list[best_ExpertTaskEdge['expert_index']] -= 1
            self.updateCurrentCoverageList(best_ExpertTaskEdge, deltaCoverage)
            self.updateExpertUnionSkillsList(best_ExpertTaskEdge)
            logging.debug("Current Coverage List = {}".format(self.currentCoverageList))

            #Get next best assignment and coverage value
            deltaCoverage, best_ExpertTaskEdge = self.getBestExpertTaskEdge(taskAssignment_i, expert_copies_list)

            logging.debug("deltaCoverage={:.1f}, best_ExpertTaskEdge={}".format(deltaCoverage, best_ExpertTaskEdge))
        

        runTime = time.perf_counter() - startTime
        logging.debug("Greedy Task Assignment computation time = {:.1f} seconds".format(runTime))

        return taskAssignment_i

    

    def baseline_Random(self, expert_copies_list):
        '''
        Baseline algorithm to compute task assignment. Uses random task assignment for each expert
        ARGS:
            expert_copies_list : list of number of copies available of experts
        
        RETURN:
            taskAssignment  : task assigment, using random assignment
        '''
        startTime = time.perf_counter()

        #Create empty task assigment matrix
        taskAssignment_i = np.zeros((self.n, self.m), dtype=np.int8)

        #Initialize current coverage list
        self.currentCoverageList = [0 for i in range(self.m)]

        #Initialize expert union skills list as an empty list of sets
        self.currentExpertUnionSkills = [set() for j in range(self.m)]

        #Assign each expert to random tasks
        for i, E_i in enumerate(self.experts):
            
            expert_copies_left = expert_copies_list[i]
            counter = 0

            while expert_copies_left > 0 and counter < self.m:

                j = np.random.randint(0, self.m)
                counter += 1
                
                #Check if this expert is not yet assigned to T_j (A[i,j] = 0) AND task is not already fully covered
                if (taskAssignment_i[i,j] == 0) and self.currentCoverageList[j] != 1:
                    #Get Task
                    T_j = self.tasks[j]
                    
                    #Retrieve union of skills of all experts assigned to T_j and add expert E_i
                    expert_skills = self.currentExpertUnionSkills[j].union(set(E_i))
                    task_skills = set(T_j)    #Get task skills as a set
                    
                    #Compute task coverage with expert added
                    T_j_coverage = len(expert_skills.intersection(task_skills))/len(task_skills)
                    delta_coverage = T_j_coverage - self.currentCoverageList[j]

                    #Decrement copy
                    expert_copies_left -= 1

                    #Add edge to task assignment
                    expertTaskEdge = {'expert_index':i, 'task_index':j}

                    taskAssignment_i[expertTaskEdge['expert_index'], expertTaskEdge['task_index']] = 1

                    #Update current coverage list and expert union skills list
                    self.updateCurrentCoverageList(expertTaskEdge, delta_coverage)
                    self.updateExpertUnionSkillsList(expertTaskEdge)
                    logging.debug("Current Coverage List = {}".format(self.currentCoverageList))


        runTime = time.perf_counter() - startTime
        logging.debug("Baseline Random Task Assignment computation time = {:.1f} seconds".format(runTime))

        return taskAssignment_i


    def baseline_NoUpdateGreedy(self, expert_copies_list):
        '''
        Baseline algorithm to compute task assignment. Uses max heap without any updates
        ARGS:
            expert_copies_list : list of number of copies available of experts
        
        RETURN:
            taskAssignment  : task assigment
        '''
        startTime = time.perf_counter()
        #First initialize maxheap to store edge coverages in self.maxHeap
        self.initializeMaxHeap()
        deltaCoverage = 1

        #Create empty task assigment matrix
        taskAssignment_i = np.zeros((self.n, self.m), dtype=np.int8)

        #Initialize current coverage list and expert union skills list as an empty list of sets
        self.currentCoverageList = [0 for i in range(self.m)]
        self.currentExpertUnionSkills = [set() for j in range(self.m)]
        
        #Assign edges from heap until there is no more coverage possible or no expert copies left
        while len(self.maxHeap) > 0 and (sum(expert_copies_list) != 0):
            #Pop top edge from maxHeap
            top_edge = heappop(self.maxHeap)
            top_ExpertTaskEdge = {'expert_index': top_edge[1], 'task_index': top_edge[2]}

            if (expert_copies_list[top_ExpertTaskEdge['expert_index']] != 0):
                
                #Retrieve union of skills of all experts assigned to T_j and add expert E_i
                expert_skills = self.currentExpertUnionSkills[top_ExpertTaskEdge['task_index']].union(set(self.experts[top_ExpertTaskEdge['expert_index']]))
                task_skills = set(self.tasks[top_ExpertTaskEdge['task_index']])   #Get task skills as a set
                
                #Compute delta_coverage of current edge
                edge_coverage = len(expert_skills.intersection(task_skills))/len(task_skills)
                deltaCoverage = edge_coverage - self.currentCoverageList[top_ExpertTaskEdge['task_index']]

                #Add edge to assignment
                taskAssignment_i[top_ExpertTaskEdge['expert_index'], top_ExpertTaskEdge['task_index']] = 1

                #Decrement expert copy, Update current coverage list and expert union skills list
                expert_copies_list[top_ExpertTaskEdge['expert_index']] -= 1
                self.updateCurrentCoverageList(top_ExpertTaskEdge, deltaCoverage)
                self.updateExpertUnionSkillsList(top_ExpertTaskEdge)
                logging.debug("Current Coverage List = {}".format(self.currentCoverageList))


        runTime = time.perf_counter() - startTime
        logging.debug("Baseline No Update Greedy Task Assignment computation time = {:.1f} seconds".format(runTime))

        return taskAssignment_i


    def getBestExpertForTaskGreedy(self, taskAssignment, j):
        '''
        Greedily compute the best expert-task edge (assigment) that maximizes coverage in that iteration
        ARGS:
            taskAssignment : current task assigment
            j : task index
        
        RETURN:
            max_edge_coverage  : maximum value of change in task coverage by adding edge (i,j)
            bestExpertTaskEdge  : indices of expert and task as a dictionary
        '''
        bestExpertTaskEdge = {'expert_index':None, 'task_index':None}
        max_edge_coverage = 0

        T_j = self.tasks[j]

        #traverse experts in random order to assign uniformly
        for i in np.random.permutation(self.n):
            #Check if this expert is not yet assigned to T_j (A[i,j] = 0)
            if (taskAssignment[i,j] == 0):
                E_i = self.experts[i]
                #Retrieve union of skills of all experts assigned to T_j and add expert E_i
                expert_skills = self.currentExpertUnionSkills[j].union(set(E_i))
                task_skills = set(T_j)    #Get task skills as a set
                
                #Compute task coverage with expert added
                T_j_coverage = len(expert_skills.intersection(task_skills))/len(task_skills)
                delta_coverage = T_j_coverage - self.currentCoverageList[j]
                #logging.debug("deltaCoverage={:.1f}, T_j_coverage={}".format(delta_coverage, T_j_coverage))

                if delta_coverage > max_edge_coverage:
                    max_edge_coverage = delta_coverage
                    bestExpertTaskEdge['expert_index'] = i
                    bestExpertTaskEdge['task_index'] = j
                    
        return max_edge_coverage, bestExpertTaskEdge


    def baseline_TaskGreedy(self):
        '''
        Baseline algorithm to compute task assignment. Greedily computes best assignment for each task
        RETURN:
            taskAssignment  : task assigment
        '''
        startTime = time.perf_counter()

        #Create empty task assigment matrix
        taskAssignment = np.zeros((self.n, self.m), dtype=np.int8)

        #Initialize current coverage list
        self.currentCoverageList = [0 for i in range(self.m)]
        #Initialize expert union skills list as an empty list of sets
        self.currentExpertUnionSkills = [set() for j in range(self.m)]
        
        for j, T_j in enumerate(self.tasks):
            #Create 1 copy of each expert, using a single list to keep track of copies
            expert_copies_list = [1 for i in range(self.n)]

            #Get first best expert-task assignment and delta coverage value
            deltaCoverage, best_ExpertTaskEdge = self.getBestExpertForTaskGreedy(taskAssignment, j)

            #Assign edges until there is no more coverage possible or no experts left 
            while (0.66 <= deltaCoverage) and (sum(expert_copies_list) != 0):
                #Add edge to assignment
                taskAssignment[best_ExpertTaskEdge['expert_index'], best_ExpertTaskEdge['task_index']] = 1

                ##Perform updates 
                #Decrement expert copy, Update current coverage list and expert union skills list
                expert_copies_list[best_ExpertTaskEdge['expert_index']] -= 1
                self.updateCurrentCoverageList(best_ExpertTaskEdge, deltaCoverage)
                self.updateExpertUnionSkillsList(best_ExpertTaskEdge)
                logging.debug("Current Coverage List = {}".format(self.currentCoverageList))

                #Get next best assignment and coverage value
                deltaCoverage, best_ExpertTaskEdge = self.getBestExpertForTaskGreedy(taskAssignment, j)

                logging.debug("deltaCoverage={:.1f}, best_ExpertTaskEdge={}".format(deltaCoverage, best_ExpertTaskEdge))


        runTime = time.perf_counter() - startTime
        logging.debug("Greedy Task Assignment computation time = {:.1f} seconds".format(runTime))

        return taskAssignment

    
    def createMaxHeap(self):
        '''
        Initialize the max heap for the lazy greedy evaluation, by evaluating coverage of all expert-task edges
        Creates and updates self.maxHeapOriginal
        '''
        self.maxHeapOriginal = []
        heapify(self.maxHeapOriginal)

        for i, E_i in enumerate(self.experts):
            for j, T_j in enumerate(self.tasks):
                expert_skills = set(E_i)    #Get expert E_i skills as set
                task_skills = set(T_j)    #Get task T_j skills as set
                
                #Compute initial coverage of E_i-T_J edge
                edge_coverage = len(expert_skills.intersection(task_skills))/len(task_skills)
                heapItem = (edge_coverage*-1, i, j)

                heappush(self.maxHeapOriginal, heapItem)
        
        logging.debug("Max Heap created for Lazy Greedy: {}".format(self.maxHeapOriginal))

        return  


    def initializeMaxHeap(self):
        '''
        Initialize the max heap for the specific threshold lazy greedy evaluation
        makes a copy of self.maxHeapOriginal and stores in self.maxHeap
        '''
        self.maxHeap = self.maxHeapOriginal.copy()
        logging.debug("Created Max Heap for iteration: {}".format(self.maxHeap))

        return        


    def getLazyExpertTaskEdge(self, taskAssignment, experts_copies):
        '''
        Lazy greedy compute the best expert-task edge (assigment) using self.maxHeap
        ARGS:
            taskAssignment : current task assigment
            experts_copies  : current list of number of copies available of experts
        
        RETURN:
            max_edge_coverage  : maximum value of change in task coverage by adding edge (i,j)
            bestExpertTaskEdge  : indices of expert and task as a dictionary
        '''
        bestExpertTaskEdge = {'expert_index':None, 'task_index':None}
        delta_best = 0

        while len(self.maxHeap) > 1:
            #Pop best edge from maxHeap
            best_edge = heappop(self.maxHeap)
            second_edge = self.maxHeap[0] #Check item now on top
            delta_second = second_edge[0]*-1

            #Compute coverage of top edge - Retrieve expert and task indices of top edge
            best_expert_i, best_task_j  = best_edge[1], best_edge[2]

            #Check if this expert is not yet assigned to T_j (A[i,j] = 0) AND copies are left
            if (taskAssignment[best_expert_i, best_task_j] == 0) and (experts_copies[best_expert_i] != 0):

                #Retrieve union of skills of all experts assigned to T_j and add expert E_i
                expert_skills = self.currentExpertUnionSkills[best_task_j].union(set(self.experts[best_expert_i]))
                task_skills = set(self.tasks[best_task_j])   #Get task skills as a set
                
                #Compute delta_coverage of current best edge
                best_edge_coverage = len(expert_skills.intersection(task_skills))/len(task_skills)
                delta_best = best_edge_coverage - self.currentCoverageList[best_task_j]

                #Return if better than 2nd
                if delta_best >= delta_second:
                    bestExpertTaskEdge['expert_index'] = best_expert_i
                    bestExpertTaskEdge['task_index'] = best_task_j
                    
                    return delta_best, bestExpertTaskEdge
   
                else:
                    #Update best edge and put back in maxHeap
                    updated_best_edge = (delta_best*-1, best_expert_i, best_task_j)
                    heappush(self.maxHeap, updated_best_edge)

        #if only 1 edge is left return it
        last_edge = self.maxHeap[0] #Check item now on top
        #Check if this expert is not yet assigned to T_j (A[i,j] = 0) AND copies are left
        last_expert_i, last_task_j  = last_edge[1], last_edge[2]

        if (taskAssignment[last_expert_i, last_task_j] == 0) and (experts_copies[last_expert_i] != 0):
            delta_best = last_edge[0]*-1
            bestExpertTaskEdge['expert_index'] = last_expert_i
            bestExpertTaskEdge['task_index'] = last_task_j

        return delta_best, bestExpertTaskEdge


    def lazyGreedyTaskAssignment(self, expert_copies_list):
        '''
        Lazy Greedy compute task assignment. Uses similar framework as greedyTaskAssignment() but with self.maxHeap to order edges
        ARGS:
            expert_copies_list : list of number of copies available of experts
        
        RETURN:
            taskAssignment  : greedy task assigment, with lazy evaluation
        '''
        startTime = time.perf_counter()
        #First initialize maxheap to store edge coverages in self.maxHeap
        self.initializeMaxHeap()

        #Create empty task assigment matrix
        taskAssignment_i = np.zeros((self.n, self.m), dtype=np.int8)

        #Initialize current coverage list and expert union skills list as an empty list of sets
        self.currentCoverageList = [0 for i in range(self.m)]
        self.currentExpertUnionSkills = [set() for j in range(self.m)]
        
        #Get first best expert-task assignment and delta coverage value
        deltaCoverage, best_ExpertTaskEdge = self.getLazyExpertTaskEdge(taskAssignment_i, expert_copies_list)

        #Assign edges until there is no more coverage possible or no expert copies left
        while (deltaCoverage > 0) and (sum(expert_copies_list) != 0):
            #Add edge to assignment
            taskAssignment_i[best_ExpertTaskEdge['expert_index'], best_ExpertTaskEdge['task_index']] = 1

            ##Perform updates 
            #Decrement expert copy, Update current coverage list and expert union skills list
            expert_copies_list[best_ExpertTaskEdge['expert_index']] -= 1
            self.updateCurrentCoverageList(best_ExpertTaskEdge, deltaCoverage)
            self.updateExpertUnionSkillsList(best_ExpertTaskEdge)
            logging.debug("Current Coverage List = {}".format(self.currentCoverageList))

            #Get next best assignment and coverage value
            deltaCoverage, best_ExpertTaskEdge = self.getLazyExpertTaskEdge(taskAssignment_i, expert_copies_list)
            logging.debug("deltaCoverage={:.1f}, best_ExpertTaskEdge={}".format(deltaCoverage, best_ExpertTaskEdge))
        

        runTime = time.perf_counter() - startTime
        logging.debug("Lazy Greedy Task Assignment computation time = {:.1f} seconds".format(runTime))

        return taskAssignment_i


    def computeTaskAssigment(self, algorithms=['lazy_greedy', 'random', 'no_update_greedy', 'task_greedy'], plot_flag=False):
        '''
        Compute a Task Assignment, of experts to tasks.
        Use m thresholds for the maximum load, and call a greedy method for each threshold
        Store this task assignment in self.taskAssignment
        ARGS:
            algorithms   : List of algorithms to run, must be a list consisting of one or more of: ['lazy_greedy','random','no_update_greedy','task_greedy]
            plot_flag   : Plot f vs. threshold, set False as default
        '''
        startTime = time.perf_counter()
        F_max = 0 #store maximum objective value
        best_T_i = None
        F_i_prev = 0 #variable to store previous objective
        
        #Track Runtime for Lazy Greedy, Random and No-Update-Greedy baseline
        runtimeDict = {'lazyGreedy': 0, 'noUpdateGreedy': 0, 'taskGreedy': 0, 'random': 0, 'total': 0}

        #Track Objective and Max Load values
        F_vals = {'lazyGreedy': None, 'noUpdateGreedy': None, 'taskGreedy': None, 'random': None}
        workLoad_vals = {'lazyGreedy': None, 'noUpdateGreedy': None, 'taskGreedy': None, 'random': None}

        #Pre-compute lambda
        self.createMaxHeap()
        lambda_val = 0
        experts_copy_list_lambda = [self.m for i in range(self.n)]
        taskAssignment_T_i = self.lazyGreedyTaskAssignment(experts_copy_list_lambda)
        
        max_load_val = self.m - min(experts_copy_list_lambda)
        F_max_val = sum(self.currentCoverageList)
        lambda_val = max_load_val/F_max_val
        logging.info("Pre-Computed Lambda value = {:.3f}".format(lambda_val))

        if 'lazy_greedy' in algorithms:
            logging.info('--------------------------Computing Greedy Task Assignment (Lazy Eval)------------------------------------')
            lazyGreedyi_start = time.perf_counter()

            for T_i in range(1, self.maxWorkloadThreshold+1):
                #Create T_i copies of each expert, using a single list to keep track of copies
                experts_copy_list_T_i = [T_i for i in range(self.n)]

                #Greedily assign experts to tasks using this expert list
                logging.debug("Computing Greedy Task Assignment (Lazy Eval) for max load, T_i={}".format(T_i))
                taskAssignment_T_i = self.lazyGreedyTaskAssignment(experts_copy_list_T_i)       
                logging.debug("Greedy Task Assignment (Lazy Eval): \n{}".format(taskAssignment_T_i))
                
                #Compute Objective: (lambda)*Coverage - T_i
                F_i = (lambda_val * sum(self.currentCoverageList)) - T_i
                logging.debug("F_i = {:.3f}".format(F_i))
                logging.info("Computed Greedy Task Assignment (Lazy Eval) for T_i={}, F_i={:.3f}".format(T_i, F_i))

                if F_i > F_max:
                    F_max = F_i
                    self.taskAssignment = taskAssignment_T_i
                    best_T_i = T_i

                # stop search if max is found
                if F_i < F_i_prev:
                    F_vals['lazyGreedy'] = F_max
                    workLoad_vals['lazyGreedy'] = best_T_i
                    break
                F_i_prev = F_i

            runtimeDict['lazyGreedy'] += time.perf_counter() - lazyGreedyi_start

        ### Run Baselines ###
        if 'random' in algorithms:
            randi_startTime = time.perf_counter()
            F_vals['random'], workLoad_vals['random'] = self.computeBaselineRandom(lambda_val)
            runtimeDict['random'] += time.perf_counter() - randi_startTime

        if 'no_update_greedy' in algorithms:
            NUGi_startTime = time.perf_counter()
            F_vals['noUpdateGreedy'], workLoad_vals['noUpdateGreedy'] = self.computeBaselineNoUpdateGreedy(lambda_val)
            runtimeDict['noUpdateGreedy'] += time.perf_counter() - NUGi_startTime


        #Task Greedy baseline
        if 'task_greedy' in algorithms:
            taskGreedy_startTime = time.perf_counter()
            F_vals['taskGreedy'], workLoad_vals['taskGreedy'] = self.computeBaselineTaskGreedy(lambda_val)
            runtimeDict['taskGreedy'] += time.perf_counter() - taskGreedy_startTime


        logging.debug("Best Task Assignment is for max workload threshold: {}, F_i(max)={:.3f} \n".format(best_T_i, F_max))

        runtimeDict['total'] = time.perf_counter() - startTime
        logging.info("\nAlgorithm Best Objective Values: Lazy Greedy = {}; No-Update-Greedy = {}; Task Greedy = {}; Random = {};\
            \n".format(F_vals['lazyGreedy'], F_vals['noUpdateGreedy'], F_vals['taskGreedy'], F_vals['random']))

        logging.info("\nAlgorithm Optimal Workload Values: Lazy Greedy = {}; No-Update-Greedy = {}; Task Greedy = {}; Random = {};\
            \n".format(workLoad_vals['lazyGreedy'], workLoad_vals['noUpdateGreedy'], workLoad_vals['taskGreedy'], workLoad_vals['random']))

        logging.info("\nAlgorithm Runtimes: Total = {:.3f}s; Lazy Greedy = {:.3f}s; No-Update-Greedy = {:.3f}s; Task Greedy = {:.3f}s; Random = {:.3f}s;\
            \n".format(runtimeDict['total'], runtimeDict['lazyGreedy'], runtimeDict['noUpdateGreedy'], runtimeDict['taskGreedy'], runtimeDict['random']))

        return runtimeDict, F_vals, workLoad_vals


    def computeBaselineNoUpdateGreedy(self, lambda_val):
        '''
        Compute No-Update-Greedy Baseline
        '''
        F_max = 0 #store maximum objective value
        best_T_i = None
        F_i_prev = 0 #variable to store previous objective

        self.createMaxHeap()

        for T_i in range(1, self.maxWorkloadThreshold+1):
            #Create T_i copies of each expert, using a single list to keep track of copies
            experts_copy_list_T_i = [T_i for i in range(self.n)]
            NUG_taskAssignment_T_i = self.baseline_NoUpdateGreedy(experts_copy_list_T_i)       
            logging.debug("Baseline No Update Greedy Task Assignment: \n{}".format(NUG_taskAssignment_T_i))

            #Compute Objective: Coverage - T_i
            NUG_F_i = (lambda_val * sum(self.currentCoverageList)) - T_i
            logging.info("Computed Baseline No-Update Greedy Task Assignment for T_i={}, F_i = {:.3f}".format(T_i, NUG_F_i))

            if NUG_F_i > F_max:
                F_max = NUG_F_i
                best_T_i = T_i

            # stop search if max is found
            if NUG_F_i < F_i_prev:
                return F_max, best_T_i            
            
            F_i_prev = NUG_F_i

        return F_max, best_T_i


    def computeBaselineRandom(self, lambda_val):
        '''
        Compute Random Baseline
        '''
        F_max = 0 #store maximum objective value
        best_T_i = None
        F_i_prev = -100 #variable to store previous objective

        for T_i in range(1, self.maxWorkloadThreshold+1):
            #Create T_i copies of each expert, using a single list to keep track of copies
            experts_copy_list_T_i = [T_i for i in range(self.n)]
            random_taskAssignment_T_i = self.baseline_Random(experts_copy_list_T_i)       
            logging.debug("Baseline Random Task Assignment: \n{}".format(random_taskAssignment_T_i))

            #Compute Objective: Coverage - T_i
            random_F_i = (lambda_val * sum(self.currentCoverageList)) - T_i
            logging.info("Computed Baseline Random Task Assignment for T_i={}, F_i = {:.3f}".format(T_i, random_F_i))

            if random_F_i > F_max:
                F_max = random_F_i
                best_T_i = T_i

            # stop search if max is found
            if random_F_i < F_i_prev:
                return F_max, best_T_i            
            
            F_i_prev = random_F_i

        return F_max, best_T_i

    
    def computeBaselineTaskGreedy(self, lambda_val):
        '''
        Compute Task-Greedy Baseline
        '''
        taskGreedy_taskAssignment = self.baseline_TaskGreedy()       
        logging.debug("Baseline Task Greedy Task Assignment: \n{}".format(taskGreedy_taskAssignment))

        max_load = self.maximumExpertLoad(taskGreedy_taskAssignment)
        logging.info("Computed Baseline Task Greedy Task Assignment, max load = {}".format(max_load))

        #Compute Objective: (lambda)*Coverage - max_load
        task_Greedy_F_i = (lambda_val * sum(self.currentCoverageList)) - max_load
        logging.info("Baseline Task Greedy F_i = {:.3f}".format(task_Greedy_F_i))

        return task_Greedy_F_i, max_load


    def testLambdaTaskAssignment(self, algorithms=['lazy_greedy', 'random', 'no_update_greedy', 'task_greedy']):
        '''
        Compute a Task Assignment, of experts to tasks.
        Use m thresholds for the maximum load, and call a greedy method for each threshold
        Store this task assignment in self.taskAssignment
        ARGS:
            algorithms   : List of algorithms to run, must be a list consisting of one or more of: ['lazy_greedy','random','no_update_greedy','task_greedy]
            plot_flag   : Plot f vs. threshold, set False as default
        '''
        startTime = time.perf_counter()
        
        #Track Runtime for Lazy Greedy, Random and No-Update-Greedy baseline
        runtimeDict = {'lazyGreedy': 0, 'noUpdateGreedy': 0, 'taskGreedy': 0, 'random': 0, 'total': 0}

        #Track Objective and Max Load values
        F_vals = {'lazyGreedy': None, 'noUpdateGreedy': None, 'taskGreedy': None, 'random': None}
        workLoad_vals = {'lazyGreedy': None, 'noUpdateGreedy': None, 'taskGreedy': None, 'random': None}

        #Pre-compute lambda
        self.createMaxHeap()
        lambda_val = 0
        experts_copy_list_lambda = [self.m for i in range(self.n)]
        taskAssignment_T_i = self.lazyGreedyTaskAssignment(experts_copy_list_lambda)
        
        max_load_val = self.m - min(experts_copy_list_lambda)
        F_max_val = sum(self.currentCoverageList)
        lambda_val = max_load_val/F_max_val
        logging.info("Pre-Computed Lambda value = {:.3f}".format(lambda_val))

        #Track lambda values for l, l/2, l/3... 
        lambda_vals = [1,2,3,4,5,6]
        lambdaFiDict = {}
        for l in lambda_vals:
            lambdaFiDict[lambda_val/l] = []

        if 'lazy_greedy' in algorithms:
            logging.info('--------------------------Computing Greedy Task Assignment (Lazy Eval)------------------------------------')
            lazyGreedyi_start = time.perf_counter()
            F_maxArr = []
            T_maxArr = []

            for lVal in lambdaFiDict.keys():
                F_max = 0 #store maximum objective value
                best_T_i = None

                for T_i in range(1, self.maxWorkloadThreshold+1):
                    #Create T_i copies of each expert, using a single list to keep track of copies
                    experts_copy_list_T_i = [T_i for i in range(self.n)]

                    #Greedily assign experts to tasks using this expert list
                    logging.debug("Computing Greedy Task Assignment (Lazy Eval) for max load, T_i={}".format(T_i))
                    taskAssignment_T_i = self.lazyGreedyTaskAssignment(experts_copy_list_T_i)       
                    logging.debug("Greedy Task Assignment (Lazy Eval): \n{}".format(taskAssignment_T_i))
                    
                    #Compute Objective: (lambda)*Coverage - T_i
                    F_i = (lVal * sum(self.currentCoverageList)) - T_i
                    logging.debug("F_i = {:.3f}".format(F_i))
                    logging.debug("Computed Greedy Task Assignment (Lazy Eval) for T_i={}, F_i={:.3f}".format(T_i, F_i))

                    if F_i > F_max:
                        F_max = F_i
                        self.taskAssignment = taskAssignment_T_i
                        best_T_i = T_i

                    lambdaFiDict[lVal].append(F_i)

                F_maxArr.append(F_max)
                T_maxArr.append(best_T_i)
                logging.info("Computed Lazy Greedy Task Assignment for Lambda={:.3f}, F_i={:.3f}, T_max={}".format(lVal, F_max, best_T_i))

            runtimeDict['lazyGreedy'] += time.perf_counter() - lazyGreedyi_start

        Ti_arr = [i for i in range(1, self.maxWorkloadThreshold+1)]

        ### Run Baselines ###
        if 'random' in algorithms:
            randi_startTime = time.perf_counter()
            F_vals['random'], workLoad_vals['random'] = self.computeBaselineRandom(lambda_val)
            runtimeDict['random'] += time.perf_counter() - randi_startTime

        if 'no_update_greedy' in algorithms:
            NUGi_startTime = time.perf_counter()
            F_vals['noUpdateGreedy'], workLoad_vals['noUpdateGreedy'] = self.computeBaselineNoUpdateGreedy(lambda_val)
            runtimeDict['noUpdateGreedy'] += time.perf_counter() - NUGi_startTime


        #Task Greedy baseline
        if 'task_greedy' in algorithms:
            taskGreedy_startTime = time.perf_counter()
            F_vals['taskGreedy'], workLoad_vals['taskGreedy'] = self.computeBaselineTaskGreedy(lambda_val)
            runtimeDict['taskGreedy'] += time.perf_counter() - taskGreedy_startTime


        logging.debug("Best Task Assignment is for max workload threshold: {}, F_i(max)={:.3f} \n".format(best_T_i, F_max))

        runtimeDict['total'] = time.perf_counter() - startTime
        logging.info("\nAlgorithm Best Objective Values: Lazy Greedy = {}; No-Update-Greedy = {}; Task Greedy = {}; Random = {};\
            \n".format(F_vals['lazyGreedy'], F_vals['noUpdateGreedy'], F_vals['taskGreedy'], F_vals['random']))

        logging.info("\nAlgorithm Optimal Workload Values: Lazy Greedy = {}; No-Update-Greedy = {}; Task Greedy = {}; Random = {};\
            \n".format(workLoad_vals['lazyGreedy'], workLoad_vals['noUpdateGreedy'], workLoad_vals['taskGreedy'], workLoad_vals['random']))

        logging.info("\nAlgorithm Runtimes: Total = {:.3f}s; Lazy Greedy = {:.3f}s; No-Update-Greedy = {:.3f}s; Task Greedy = {:.3f}s; Random = {:.3f}s;\
            \n".format(runtimeDict['total'], runtimeDict['lazyGreedy'], runtimeDict['noUpdateGreedy'], runtimeDict['taskGreedy'], runtimeDict['random']))

        return Ti_arr, lambdaFiDict, T_maxArr, F_maxArr
