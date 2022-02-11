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

    def __init__(self, m_tasks, n_experts, max_workload_threshold=2):
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
        logging.info('Team Formation Problem initialized with {} tasks and {} experts'.format(self.m,self.n))

    
    
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


    def initializeMaxHeap(self):
        '''
        Initialize the max heap for the lazy greedy evaluation, by evaluating coverage of all expert-task edges
        Creates and updates self.maxHeap
        '''
        self.maxHeap = []
        heapify(self.maxHeap)

        for i, E_i in enumerate(self.experts):
            for j, T_j in enumerate(self.tasks):
                expert_skills = set(E_i)    #Get expert E_i skills as set
                task_skills = set(T_j)    #Get task T_j skills as set
                
                #Compute initial coverage of E_i-T_J edge
                edge_coverage = len(expert_skills.intersection(task_skills))/len(task_skills)
                heapItem = (edge_coverage*-1, i, j)

                heappush(self.maxHeap, heapItem)
        
        logging.debug("Max Heap: {}".format(self.maxHeap))

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


    def removeLastAssigments(self, taskAssignment, expert_copies):
        '''
        Funtion to remove the last expert-task assignments for each expert
        ARGS:
            taskAssignment : task assignment
        
        RETURN:
            editedAssignments : Edited task assignments with last assignment removed
        '''
        editedAssignments = taskAssignment.copy()

        logging.debug("self.lastAssignmentList:{}".format(self.lastAssignmentList))
        for i, task_j in enumerate(self.lastAssignmentList):
            if task_j: #If there are tasks assigned to this expert
                if expert_copies[i] <= 0:
                    j = task_j[-1]
                    editedAssignments[i,j] = 0
                    self.lastAssignmentList[i].remove(task_j[-1])

        return editedAssignments, expert_copies
        

    def compute_CoverageList_ExpertSkills(self, taskAssignment_i, expertCopiesList):
        '''
        Compute the current coverage list and union of skills assigned to each task
        ARGS:
            taskAssignment_i : task assignment
        
        RETURN:
            coverageList  : task coverage value of each task
            expertUnionSkills   : union of expers skills to each task
            expert_copies_list  : Remaining copies of each expert
        '''
        coverageList = [0 for j in range(self.m)]
        expertUnionSkillsList = [set() for j in range(self.m)]
        expert_copies_list = expertCopiesList.copy()

        #Initialize max heap
        self.maxHeap = []
        heapify(self.maxHeap)

        for j, T_j in enumerate(self.tasks):
            task_skills = set(T_j) #Get task T_j skills as set
            task_j_assignment_vec = taskAssignment_i[:,j]
            
            for expert_index, is_assigned in enumerate(task_j_assignment_vec):
                
                expert_skills = set(self.experts[expert_index]) #Get expert E_i skills as set
                #If expert is assigned to task, update skills lists
                if is_assigned == 1:
                    expertUnionSkillsList[j] = expertUnionSkillsList[j].union(expert_skills)
                    expert_copies_list[expert_index] -= 1

            #If not assigned to task add to max heap     
            #Compute initial coverage of E_i-T_J edge
            task_j_cov = len(expertUnionSkillsList[j].intersection(task_skills))/len(task_skills)
            
            for expert_index, is_assigned in enumerate(task_j_assignment_vec):
                expert_skills = set(self.experts[expert_index]) #Get expert E_i skills as set
                if is_assigned == 0:                     
                    edge_coverage = len((expertUnionSkillsList[j].union(expert_skills)).intersection(task_skills))/len(task_skills) - task_j_cov
                    heapItem = (edge_coverage*-1, expert_index, j)
                    heappush(self.maxHeap, heapItem)
            
            #Update task coverage
            #task_j_cov = len(expertUnionSkillsList[j].intersection(task_skills))/len(task_skills)
            coverageList[j] += task_j_cov

        logging.debug("Max Heap: {}".format(self.maxHeap))
        logging.debug("Coverage list check:{}".format(coverageList))

        return coverageList, expertUnionSkillsList, expert_copies_list


    def reverseThresholdtaskAssignment(self, expert_copies_list, initialAssignment):
        '''
        Compute task assignment by going over thresholds in reverse order. 
        Uses similar framework as greedyTaskAssignment() but with self.maxHeap to order edges
        ARGS:
            expert_copies_list : list of number of copies available of experts
            experts_smallestThresholds  : 
            initialAssignment   : Initial assignment of experts to tasks
        
        RETURN:
            taskAssignment_i    : 
            taskAssignment_t_iminus1
        '''
        startTime = time.perf_counter()

        #Initialize task assignment and coverage list thus far
        threshold_flag = False

        taskAssignment_i = initialAssignment.copy()

        #Compute coverage, expert skills, expert copies at start
        #Also initializes the max heap
        self.currentCoverageList, self.currentExpertUnionSkills, expert_copies = self.compute_CoverageList_ExpertSkills(taskAssignment_i, expert_copies_list)
        
        deltaCoverage, best_ExpertTaskEdge = self.getLazyExpertTaskEdge(taskAssignment_i, expert_copies)

        #if minVal is already 0, then remove the last assigments for experts with value 0
        expertCopyMinVal = min(expert_copies)
        logging.debug("Minval: {}, expert copies: {}".format(expertCopyMinVal, expert_copies))

        if expertCopyMinVal <= 0:
            taskAssignment_t_iminus1, expert_copies = self.removeLastAssigments(taskAssignment_i, expert_copies)
            threshold_flag = True

        #Assign edges until there is no more coverage possible or no expert copies left
        while (deltaCoverage > 0) and (sum(expert_copies) > 0):

            #Add edge to assignment
            taskAssignment_i[best_ExpertTaskEdge['expert_index'], best_ExpertTaskEdge['task_index']] = 1

            ##Perform updates 
            #Decrement expert copy, Update current coverage list and expert union skills list
            expert_copies[best_ExpertTaskEdge['expert_index']] -= 1

            #check expert copies for minimum value to detect if threshold is fully used
            #Store task assigment
            expertCopyMinVal = min(expert_copies)
            if expertCopyMinVal == 0 and threshold_flag is False:
                threshold_flag = True
                taskAssignment_t_iminus1 = taskAssignment_i.copy()
        
                #Remove edge that was just added
                taskAssignment_t_iminus1[best_ExpertTaskEdge['expert_index'], best_ExpertTaskEdge['task_index']] = 0

            self.lastAssignmentList[best_ExpertTaskEdge['expert_index']].append(best_ExpertTaskEdge['task_index'])

            self.updateCurrentCoverageList(best_ExpertTaskEdge, deltaCoverage)
            self.updateExpertUnionSkillsList(best_ExpertTaskEdge)
            logging.debug("Current Coverage List = {}".format(self.currentCoverageList))

            #Get next best assignment and coverage value
            deltaCoverage, best_ExpertTaskEdge = self.getLazyExpertTaskEdge(taskAssignment_i, expert_copies)


        #If all experts weren't used then t_(i-1) assignment can be the same for next assignment
        if expertCopyMinVal > 0:
            taskAssignment_t_iminus1 = taskAssignment_i.copy()

        runTime = time.perf_counter() - startTime
        logging.debug("Reverse Threshold Greedy Task Assignment computation time = {:.1f} seconds".format(runTime))


        return taskAssignment_i, taskAssignment_t_iminus1



    def compute_reverseThreshold(self):
        '''
        Compute a Task Assignment, of experts to tasks - reuse computation by traversing thresholds in reverse order
        '''
        startTime = time.perf_counter()
        F_max = 0 #store maximum objective value
        best_T_i = None
        
        #Create empty task assigment matrix
        taskAssignment_T_iminusone = np.zeros((self.n, self.m), dtype=np.int8)

        #Keep track of last assignments of each expert
        self.lastAssignmentList = [[] for i in range(self.n)]


        #Run for subsequent T_(i-1) thresholds, reusing computation from T_i
        for T_i in range(self.maxWorkloadThreshold, 0, -1):
            logging.info("Computing Reverse Threshold Greedy Task Assignment (Lazy Eval) for max load, T_i={}".format(T_i))

            #Create T_i copies of each expert, using a single list to keep track of copies
            experts_copy_list_T_i = [T_i] * self.n

            initial_taskAssignment_i = taskAssignment_T_iminusone.copy()
            taskAssignment_T_i, taskAssignment_T_iminusone = self.reverseThresholdtaskAssignment(experts_copy_list_T_i, initial_taskAssignment_i)     

            logging.info("Task Assigment: \n{}".format(taskAssignment_T_i))
            logging.info("Task Assigment t_(i-1): \n{}".format(taskAssignment_T_iminusone))


            #Compute Objective: Coverage - T_i
            F_i = sum(self.currentCoverageList) - T_i
            logging.info("F_i = {:.3f}".format(F_i))

            if F_i > F_max:
                F_max = F_i
                self.taskAssignment = taskAssignment_T_i
                best_T_i = T_i

        logging.info("Best Task Assignment is for max workload threshold: {}, F_i(max)={:.3f} \n{}".format(best_T_i, F_max, self.taskAssignment))
        
        runTime = time.perf_counter() - startTime
        logging.info("\nTotal Computation time Reverse Threshold= {:.3f} seconds".format(runTime))
            
        return None



    def computeTaskAssigment(self, lazy_eval=True, baselines=['random', 'no_update_greedy'], plot_flag=False):
        '''
        Compute a Task Assignment, of experts to tasks.
        Use m thresholds for the maximum load, and call a greedy method for each threshold
        Store this task assignment in self.taskAssignment
        ARGS:
            lazy_eval  : Lazy Greedy evaluation, set True as default
            baselines   : List of baselines to run, must be a list consisting of one or more of: ['random', 'no_update_greedy']
            plot_flag   : Plot f vs. threshold, set False as default
        '''
        startTime = time.perf_counter()
        F_max = 0 #store maximum objective value
        best_T_i = None

        #Arrays to store objective and threshold values for plotting
        F_arr, F_random_arr, F_noupdate_arr = [], [], []
        T_arr = []

        for T_i in range(1, self.maxWorkloadThreshold+1):
            logging.info('--------------------------------------------------------------------------------------------------')
            #Create T_i copies of each expert, using a single list to keep track of copies
            experts_copy_list_T_i = [T_i for i in range(self.n)]

            #Greedily assign experts to tasks using this expert list
            if lazy_eval:
                logging.info("Computing Greedy Task Assignment (Lazy Eval) for max load, T_i={}".format(T_i))
                taskAssignment_T_i = self.lazyGreedyTaskAssignment(experts_copy_list_T_i)       
                logging.info("Greedy Task Assignment (Lazy Eval): \n{}".format(taskAssignment_T_i))

            else:
                logging.info("Computing Greedy Task Assignment, for max load, T_i={}".format(T_i))
                taskAssignment_T_i = self.greedyTaskAssignment(experts_copy_list_T_i) 
                logging.info("Greedy Task Assignment (regular): \n{}".format(taskAssignment_T_i))
            
            #Compute Objective: Coverage - T_i
            F_i = sum(self.currentCoverageList) - T_i
            logging.info("F_i = {:.3f}".format(F_i))

            F_arr.append(F_i)
            T_arr.append(T_i)

            if F_i < 0:
                break

            if F_i > F_max:
                F_max = F_i
                self.taskAssignment = taskAssignment_T_i
                best_T_i = T_i


            #Run baselines:
            for b in baselines:
                if b == 'random':
                    #Create T_i copies of each expert, using a single list to keep track of copies
                    experts_copy_list_T_i = [T_i for i in range(self.n)]

                    logging.info("Baseline Random Task Assignment for max load, T_i={}".format(T_i))
                    random_taskAssignment_T_i = self.baseline_Random(experts_copy_list_T_i)       
                    logging.info("Baseline Random Task Assignment: \n{}".format(random_taskAssignment_T_i))

                    #Compute Objective: Coverage - T_i
                    random_F_i = sum(self.currentCoverageList) - T_i
                    logging.info("Baseline Random F_i = {:.3f}".format(random_F_i))

                    F_random_arr.append(random_F_i)

                if b == 'no_update_greedy':
                    #Create T_i copies of each expert, using a single list to keep track of copies
                    experts_copy_list_T_i = [T_i for i in range(self.n)]

                    logging.info("Baseline No Update Greedy Task Assignment for max load, T_i={}".format(T_i))
                    NUG_taskAssignment_T_i = self.baseline_NoUpdateGreedy(experts_copy_list_T_i)       
                    logging.info("Baseline No Update Greedy Task Assignment: \n{}".format(NUG_taskAssignment_T_i))

                    #Compute Objective: Coverage - T_i
                    NUG_F_i = sum(self.currentCoverageList) - T_i
                    logging.info("Baseline No Update Greedy F_i = {:.3f}".format(NUG_F_i))
                    F_noupdate_arr.append(NUG_F_i)

        #Plotting logic
        if plot_flag:
            f_objectives_arr = [F_arr]
            f_objectives_labels = ['F']
            for b in baselines:
                if b == 'random':
                    f_objectives_arr.append(F_random_arr)
                    f_objectives_labels.append('Random')

                if b == 'no_update_greedy':
                    f_objectives_arr.append(F_noupdate_arr)
                    f_objectives_labels.append('No-Update Greedy')

            self.thresholdPlotter(T_arr, f_objectives_arr,f_objectives_labels)

        logging.info("Best Task Assignment is for max workload threshold: {}, F_i(max)={:.3f} \n{}".format(best_T_i, F_max, self.taskAssignment))
        
        runTime = time.perf_counter() - startTime
        logging.info("\nTotal Computation time = {:.3f} seconds".format(runTime))

        return None

