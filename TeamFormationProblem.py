## Implementation of Framework for Team Formation problem
## Balancing Task Coverage vs. Maximum Expert Load
## Karan Vombatkere, Dec 2021


#Import and logging config
import numpy as np
from heapq import heappop, heappush, heapify
import time
import logging
logging.basicConfig(format='%(asctime)s |%(levelname)s: %(message)s', level=logging.INFO)



class TeamFormationProblem:
    '''
    Class to implement base framework for Team Formation problem
    The goal is to optimally Balance Task Coverage vs. Maximum Expert Load

    Parameters
    ----------
        m_tasks : list of m tasks; each task is a list of skills
        n_experts : list of n experts; each expert is a list of skills
    ----------
    self.taskAssignment  : Created (n x m) matrix to store task assignment
                            n rows represent the n experts, m columns are for m tasks
    '''

    def __init__(self, m_tasks, n_experts, max_workload_threshold=3):
        '''
        Initialize problem instance with m tasks, n experts
        ARGS:
            m_tasks : list of m tasks; each task is a list of skills
            n_experts : list of n experts; each expert is a list of skills
            max_workload_threshold: maximum workload threshold to attempt to solve for, default=3
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
        Given a bestExpertTaskEdge, update the current coverage list
        ARGS:
            bestExpertTaskEdge  : best expert task edge
            delta_coverage  : increase in coverage
        '''
        self.currentCoverageList[bestExpertTaskEdge['task_index']] += delta_coverage
        return

    

    def updateExpertUnionSkillsList(self, bestExpertTaskEdge):
        '''
        Given a bestExpertTaskEdge, update the expert union skills list
        ARGS:
            bestExpertTaskEdge  : best expert task edge
        '''
        self.currentExpertUnionSkills[bestExpertTaskEdge['task_index']].union(self.experts[bestExpertTaskEdge['expert_index']])
        return



    def getBestExpertTaskEdge(self, taskAssignment, experts_copies):
        '''
        Greedily compute the best expert-task edge (assigment) that maximizes coverage in that iteration
        ARGS:
            taskAssignment : current task assigment
            experts_copies  : current list of number of copies available of experts
        
        RETURN:
            max_edge_coverage  : maximum value of change in task coverage by adding edge (i,j)
            bestExpertTaskEdge  : indices of expert and task as a tuple (i,j)
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



    def greedyTaskAssignment(self, expert_copies_list):
        '''
        Greedily compute a Task Assignment given a set of tasks and experts
        ARGS:
            expert_copies_list  : list of number of copies of each expert
        RETURN:
            taskAssignment  : greedy task assigment
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



    def initializeMaxHeap(self):
        '''
        Initialize the max heap for the lazy greedy evaluation, by evaluating coverage of all expert-task edges
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

        return



    def getLazyExpertTaskEdge(self, taskAssignment, experts_copies):
        '''
        Lazy greedy compute the best expert-task edge (assigment) that maximizes coverage in that iteration
        ARGS:
            taskAssignment : current task assigment
            experts_copies  : current list of number of copies available of experts
        
        RETURN:
            max_edge_coverage  : maximum value of change in task coverage by adding edge (i,j)
            bestExpertTaskEdge  : indices of expert and task as a tuple (i,j)
        '''
        bestExpertTaskEdge = {'expert_index':None, 'task_index':None}
        delta_best = 0

        while len(self.maxHeap)>1:
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
        Lazy Greedy compute best expert-task edge (assigment) that maximizes coverage in that iteration
        Use a max heap/priority queue to order edges
        ARGS:
            experts_copies  : current list of number of copies available of experts
        
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
        first_edge = heappop(self.maxHeap)

        deltaCoverage = first_edge[0]*-1
        best_ExpertTaskEdge = {'expert_index': first_edge[1], 'task_index': first_edge[2]}

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
        logging.debug("Greedy Task Assignment computation time = {:.1f} seconds".format(runTime))

        return taskAssignment_i
        


    def computeTaskAssigment(self, lazy_eval=True):
        '''
        Compute a Task Assignment, of experts to tasks.
        Use m thresholds for the maximum load, and call a greedy method for each threshold
        Store this task assignment in self.taskAssignment
        
        ARGS:
            lazy_eval  : Lazy Greedy evaluation, set True as default
        '''
        startTime = time.perf_counter()
        F_max = 0 #store maximum objective value
        best_T_i = None

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

            if F_i < 0:
                break

            if F_i > F_max:
                F_max = F_i
                self.taskAssignment = taskAssignment_T_i
                best_T_i = T_i


        logging.info("Best Task Assignment is for max workload threshold: {}, F_i(max)={:.3f} \n{}".format(best_T_i, F_max, self.taskAssignment))
        
        runTime = time.perf_counter() - startTime
        logging.info("\nTotal Computation time = {:.3f} seconds".format(runTime))

        return None


    def compareTest_Lazy_Regular_Assignments(self):
        '''
        Compare the performance and correctness of both lazy and regular implementations
        '''
        lazyRunTime = 0
        regularRunTime = 0

        equal_assignment_list = []
        equal_objective_list = []

        for T_i in range(1, self.maxWorkloadThreshold+1):
            #Create T_i copies of each expert, using a single list to keep track of copies
            experts_copy_list_T_i = [T_i for i in range(self.n)]

            logging.info("----------Computing Greedy Task Assignment, for max load, T_i={} ---------------".format(T_i))
            #Run Lazy Evaluation method
            startTime = time.perf_counter()
            lazy_taskAssignment_T_i = self.lazyGreedyTaskAssignment(experts_copy_list_T_i) 
            #Compute Objective: Coverage - T_i
            lazy_F_i = sum(self.currentCoverageList) - T_i
            runTime = time.perf_counter() - startTime
            lazyRunTime += runTime
            logging.info("Lazy F_i = {:.3f}".format(lazy_F_i))

            experts_copy_list_T_i = [T_i for i in range(self.n)]
            startTime = time.perf_counter()
            taskAssignment_T_i = self.greedyTaskAssignment(experts_copy_list_T_i) 

            #Compute Objective: Coverage - T_i
            F_i = sum(self.currentCoverageList) - T_i
            runTime = time.perf_counter() - startTime
            regularRunTime += runTime
            logging.info("Regular Greedy F_i = {:.3f}".format(F_i))

            if(np.array_equal(lazy_taskAssignment_T_i, taskAssignment_T_i)):
                equal_assignment_list.append(1)
            else:
                equal_assignment_list.append(0)

            if lazy_F_i == F_i:
                equal_objective_list.append(1)
            else:
                equal_objective_list.append(0)

            logging.info("============================================================================================")
        
        if sum(equal_objective_list) == self.maxWorkloadThreshold and sum(equal_assignment_list) == self.maxWorkloadThreshold:
            logging.info("\nAll {} Assignment Matrices Equal; All {} Objectives Equal".format(sum(equal_assignment_list), sum(equal_objective_list)))
        else:
            logging.info("\nAssignment Matrices NOT Equal: {}, Objectives NOT Equal".format(equal_assignment_list, equal_objective_list))

        logging.info("\nTotal Regular Greedy runtime = {:.3f} seconds".format(regularRunTime))
        logging.info("\nTotal Lazy Evaluation runtime = {:.3f} seconds".format(lazyRunTime))

        logging.info("\nLazy Evaluation runtime improvement = {:.1f}x".format(regularRunTime/lazyRunTime))

