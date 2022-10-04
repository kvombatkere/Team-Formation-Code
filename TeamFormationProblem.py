## Implementation for Team Formation problem
## Balancing Task Coverage vs. Maximum Expert Load
## Karan Vombatkere, Dec 2021

#Import and logging config
import numpy as np
from heapq import heappop, heappush, heapify
import time, random
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB
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

    def __init__(self, m_tasks, n_experts, max_workload_threshold=100):
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


    def getExpertSkillSet(self, expertIndices):
        '''
        Given a list of expert indices, return union of all expert skills
        ARGS:
            expertIndices   : list of indices of experts
        RETURN:
            expert_skillset    : Union of skills of all experts, returned as a set
        '''
        expert_skillset = set()
        for expert_index in expertIndices:
            expert_i_skills = set(self.experts[expert_index])
            expert_skillset = expert_skillset.union(expert_i_skills)

        return expert_skillset


    def singleTaskCoverage(self, taskAssignment, task_index):
        '''
        Given a taskAssignment, compute the task coverage of a single task 
        ARGS:
            taskAssignment  : current task assigment
            task_index  : index of task in original list m_tasks
        RETURN:
            task_coverage   : task coverage of a single task
        '''
        #Get union of skills of experts assigned to task
        task_assigned_experts_indices = self.getExpertsAssignedToTask(taskAssignment, task_index)
        all_expert_skills = self.getExpertSkillSet(task_assigned_experts_indices)

        task_skills = set(self.tasks[task_index])
        logging.debug("Single Task Coverage Debug: task index={}, task skills={}, expert skills={}".format(task_index, task_skills, all_expert_skills))
        
        #Compute coverage
        task_coverage = len(all_expert_skills.intersection(task_skills))/len(task_skills)
        
        return task_coverage


    def computeTotalTaskCoverage(self, taskAssignment, list_flag = False):
        '''
        Given a taskAssignment, compute the total task coverage of the assigment
        ARGS:
            taskAssignment  : (n x m) matrix task assigment of experts to tasks
            list_flag  : flag to determine whether to output total coverage or list
        RETURN:
            total_task_coverage_value   : total task coverage across all tasks
            total_task_coverage_list    : list of all m task coverage values
        '''
        task_assigment = taskAssignment
        
        total_task_coverage_value = 0
        total_task_coverage_list = []

        for index_task, t in enumerate(self.tasks):
            cov = self.singleTaskCoverage(task_assigment, index_task)
            total_task_coverage_value += cov
            total_task_coverage_list.append(cov)

        if list_flag:
            return total_task_coverage_list
        
        else:
            return total_task_coverage_value

    
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


    def baseline_Random(self):
        '''
        Baseline algorithm to compute task assignment. Uses random task assignment for each expert
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

        #Assign each expert to m/4 random tasks
        for i, E_i in enumerate(self.experts):
            counter = 0
            while counter < self.m:

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

                    #Add edge to assignment and update if deltaCoverage > x
                    if delta_coverage > 0.8:
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


    def baseline_NoUpdateGreedy(self):
        '''
        Baseline algorithm to compute task assignment. Uses max heap without any updates
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
        expert_copies_list = [self.maxWorkloadThreshold for i in range(self.n)]
        
        #Assign edges from heap until coverage stabilizes or everyone is assigned
        while len(self.maxHeap) > 0:
            #Pop top edge from maxHeap
            top_edge = heappop(self.maxHeap)
            top_ExpertTaskEdge = {'expert_index': top_edge[1], 'task_index': top_edge[2]}

            #Retrieve union of skills of all experts assigned to T_j and add expert E_i
            expert_skills = self.currentExpertUnionSkills[top_ExpertTaskEdge['task_index']].union(set(self.experts[top_ExpertTaskEdge['expert_index']]))
            task_skills = set(self.tasks[top_ExpertTaskEdge['task_index']])   #Get task skills as a set
            
            #Compute delta_coverage of current edge
            edge_coverage = len(expert_skills.intersection(task_skills))/len(task_skills)
            deltaCoverage = edge_coverage - self.currentCoverageList[top_ExpertTaskEdge['task_index']]

            #Add edge to assignment and update if deltaCoverage > 0.9 and task isn't already covered
            if deltaCoverage > 0.8 and expert_copies_list[top_ExpertTaskEdge['expert_index']] != 0:
                taskAssignment_i[top_ExpertTaskEdge['expert_index'], top_ExpertTaskEdge['task_index']] = 1
                #Decrement expert copy
                expert_copies_list[top_ExpertTaskEdge['expert_index']] -= 1

                #Update coverage list
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
            while (deltaCoverage > 0.85) and (sum(expert_copies_list) != 0):
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
    

    def createExpertTaskSkillMatrices(self):
        '''
        Create (n_experts, n_skills) and (m_tasks, n_skills) matrices from skill and expert lists
        RETURN:
            experts_mat : (n_experts, n_skills) binary matrix
            tasks_mat   : (m_tasks, n_skills) binary matrix
            tasks_not_coverable : list of tasks that are not fully coverable
        '''
        #First check if all tasks are coverable and get set of all skills from experts and tasks
        all_experts_skillset = set()
        all_tasks_skillset = set()

        for expert_i in self.experts:
            for skill in expert_i:
                all_experts_skillset = all_experts_skillset.union({skill})

        allTasksCoverable = True
        for task_j in self.tasks:
            for skill in task_j:
                all_tasks_skillset = all_tasks_skillset.union({skill})
                if skill not in all_experts_skillset:
                    allTasksCoverable = False

        all_skills_set = all_experts_skillset.union(all_tasks_skillset)
        numSkills = len(all_skills_set) #Get total number of skills

        logging.info("Extracted expert and tasks skillset, Total skills = {}".format(numSkills))

        partialCoverageBoundMatrix = np.ones((self.m, numSkills), dtype=np.float32)
        #Reindex skills for datasets with partial coverage and get partial coverage values
        if not allTasksCoverable:
            skillsNewIndexDict = {}
            for indx, val in enumerate(all_skills_set):
                skillsNewIndexDict[val] = indx

            for task_index, task_i in enumerate(self.tasks):
                for skill in task_i:
                    if skill not in all_experts_skillset:
                        skillIndex = skillsNewIndexDict[skill]
                        partialCoverageBoundMatrix[task_index, skillIndex] = 0
            
            logging.info("Full task coverage not possible, generated coverage upper bounds for each task")

        #Create (n_experts, n_skills) matrix
        experts_mat = np.zeros((self.n, numSkills), dtype=np.int8)
        for expert_index, expert_i in enumerate(self.experts):
            for skill in expert_i:
                skill_index = int(skill)
                if not allTasksCoverable:
                    skill_index = skillsNewIndexDict[skill]

                experts_mat[expert_index][skill_index] = 1

        logging.info("Generated expert-skill matrix, shape = {}".format(experts_mat.shape))
        
        #Create (m_tasks, n_skills) matrix
        tasks_mat = np.zeros((self.m, numSkills), dtype=np.int8)

        for task_index, task_i in enumerate(self.tasks):
            for skill in task_i:
                skill_index = int(skill)
                if not allTasksCoverable:
                    skill_index = skillsNewIndexDict[skill]
                tasks_mat[task_index][skill_index] = 1
        
        logging.info("Generated task-skill matrix, shape = {}".format(tasks_mat.shape))
        
        return experts_mat, tasks_mat, partialCoverageBoundMatrix

    
    def convertLPSolutionToMatrix(self, lp_model):
        '''
        Convert the lp_model output to a (n_experts x m_tasks) matrix
        Entries in n x m matrix represent probabilities of assigning expert i to task j
        ARGS:
            lp_model: Gurobi solved LP model
        RETURN:
            LP_soln_matrix: (n_experts x m_tasks) matrix with LP solution X_ji values as per Power in Unity paper
        '''
        v = lp_model.getVars()
        count = 0

        LP_soln_matrix = np.zeros((self.n, self.m), dtype=np.float32)
        
        for i in range(self.n):
            for j in range(self.m):
                LP_soln_matrix[i,j] = v[count].x
                count += 1

        return LP_soln_matrix


    def solve_LP(self, expertMatrix, taskMatrix, partialCoverageMatrix):
        '''
        Given (n_experts, n_skills) and (m_tasks, n_skills) matrices, solve the relaxed ILP and return 
        a (n_experts x m_tasks) matrix with LP solution
        ARGS:
            expertMatrix : (n_experts, n_skills) binary matrix of expert skills
            taskMatrix   : (m_tasks, n_skills) binary matrix of task skills
        RETURN:
            LP_solution_matrix: (n_experts x m_tasks) matrix with LP solution X_ji values as per Power in Unity paper
        '''
        #Create empty assignment matrix of shape (n_experts x m_tasks)
        X =  np.zeros((self.n, self.m), dtype=np.float32)

        #Create Gurobi LP Model
        m = gp.Model("TaskCoverageLP")

        #Add variables
        x = m.addVars(len(X), len(X[1]), vtype='S', ub=1.0, name="x")

        #Set objective function
        L = m.addVar(vtype='S', name = 'Load')
        obj = 1*L
        m.setObjective(obj, GRB.MINIMIZE)

        #Add constraints
        # c1 - Load of each expert is upper bounded by L
        c1 = m.addConstrs(x.sum(i,'*') <= L for i in range(len(X)))

        # c2 - Each task is (fully) covered
        c2 = m.addConstrs(gp.quicksum(expertMatrix[l][j]*x[l,i] for l in range(len(expertMatrix))) >= partialCoverageMatrix[i][j]*taskMatrix[i][j] 
                                            for i in range(len(taskMatrix)) for j in range(len(taskMatrix[0])) if taskMatrix[i][j] > 0)
            
        # Silence model output
        m.setParam('OutputFlag', 0)

        logging.info("Computing LP solution matrix...")

        #Solve LP model
        m.optimize()

        LP_solution_matrix = self.convertLPSolutionToMatrix(m)

        return LP_solution_matrix


    def setCoverLPTaskCoverage(self, numRounds):
        '''
        Adapted LP algorithm for the non-online setting of the Load minimization problem by Anagnostopoulos et al.
        ARGS:
            numRounds   : Number of rounds R to run algorithm for
        RETURN:
            taskAssignmentMatrixList    : list of task assignments over several rounds
        '''
        startTime = time.perf_counter()
        
        #Solve relaxed LP
        expertMatrix, taskMatrix, partialCovMat = self.createExpertTaskSkillMatrices()
        LP_solution = self.solve_LP(expertMatrix, taskMatrix, partialCovMat)

        #List of task assignment matrices for each round
        taskAssignmentMatrixList = []

        #List of total task coverage and max load for each round - stored as a list of tuples
        taskCoverageMaxLoadList = []

        #Create empty task assigment matrix for first round
        taskAssignment_0 = np.zeros((self.n, self.m), dtype=np.int8)    
        taskAssignmentMatrixList.append(taskAssignment_0)

        logging.info("Running Probabilistic Task Assignment using LP Solution Matrix")
        for round in range(numRounds):
            #Run a round of the probabilistic algorithm using LP_solution matrix
            for i in range(self.n):
                for j in range(self.m):
                    #Only consider non-zero values
                    if LP_solution[i,j] != 0:
                        randVal = random.random()
                        if randVal <= LP_solution[i,j]: #Assign expert to task if randVal <= prob
                            taskAssignmentMatrixList[round][i,j] = 1

            #Compute task coverage and maximum load
            roundTaskCoverageVal = self.computeTotalTaskCoverage(taskAssignmentMatrixList[round])
            roundMaxCoverageVal = self.maximumExpertLoad(taskAssignmentMatrixList[round])
            taskCoverageMaxLoadList.append((roundTaskCoverageVal, roundMaxCoverageVal))

            #Create next matrix using the last task assignment until second last round
            if round < (numRounds-1):
                nextRoundTaskAssignment = taskAssignmentMatrixList[round].copy()
                taskAssignmentMatrixList.append(nextRoundTaskAssignment)

        runTime = time.perf_counter() - startTime
        logging.info("LP solver Assignment computation time = {:.1f} seconds".format(runTime))

        return taskAssignmentMatrixList, taskCoverageMaxLoadList

    
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


    def computeTaskAssigment(self, algorithms=['lazy_greedy', 'random', 'no_update_greedy', 'task_greedy'], lambdaVal=1):
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
        F_vals = {'lazyGreedy': 0, 'noUpdateGreedy': 0, 'taskGreedy': 0, 'random': 0}
        workLoad_vals = {'lazyGreedy': None, 'noUpdateGreedy': None, 'taskGreedy': None, 'random': None}

        #Create max heap for greedy algos
        self.createMaxHeap()

        #Pre-compute lambda
        # lambda_val = 0
        # experts_copy_list_lambda = [self.m for i in range(self.n)]
        # taskAssignment_T_i = self.lazyGreedyTaskAssignment(experts_copy_list_lambda)
        
        # max_load_val = self.m - min(experts_copy_list_lambda)
        # F_max_val = sum(self.currentCoverageList)
        # lambda_val = max_load_val/F_max_val 
        # logging.info("Pre-Computed Lambda value = {:.3f}".format(lambda_val))

        lambda_val = lambdaVal

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
                C_i = sum(self.currentCoverageList)
                F_i = (lambda_val * C_i) - T_i

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
        logging.info("\nAlgorithm Best Objective Values: Lazy Greedy = {:.2f}; Random = {:.2f}; No-Update-Greedy = {:.2f}; Task Greedy = {:.2f}; \
            \n".format(F_vals['lazyGreedy'], F_vals['random'], F_vals['noUpdateGreedy'], F_vals['taskGreedy']))

        logging.info("\nAlgorithm Optimal Workload Values: Lazy Greedy = {}; Random = {}; No-Update-Greedy = {}; Task Greedy = {};\
            \n".format(workLoad_vals['lazyGreedy'], workLoad_vals['random'], workLoad_vals['noUpdateGreedy'], workLoad_vals['taskGreedy']))

        logging.info("\nAlgorithm Runtimes: Total = {:.2f}s; Lazy Greedy = {:.2f}s; Random = {:.2f}s; No-Update-Greedy = {:.2f}s; Task Greedy = {:.2f}s; \
            \n".format(runtimeDict['total'], runtimeDict['lazyGreedy'], runtimeDict['random'], runtimeDict['noUpdateGreedy'], runtimeDict['taskGreedy']))

        return runtimeDict, F_vals, workLoad_vals


    def computeBaselineNoUpdateGreedy(self, lambda_val):
        '''
        Compute No-Update-Greedy Baseline
        '''
        noUpdateGreedy_taskAssignment = self.baseline_NoUpdateGreedy()       

        max_load = self.maximumExpertLoad(noUpdateGreedy_taskAssignment)
        logging.info("Computed No-Update Greedy Task Assignment workload, L = {}".format(max_load))

        #Compute Objective: (lambda)*Coverage - max_load
        noUpdateGreedy_F = (lambda_val * sum(self.currentCoverageList)) - max_load
        logging.info("No-Update Greedy objective, F = {:.3f}".format(noUpdateGreedy_F))

        return noUpdateGreedy_F, max_load


    def computeBaselineRandom(self, lambda_val):
        '''
        Compute Smart Random Baseline
        '''
        random_taskAssignment = self.baseline_Random()       
        logging.debug("Baseline Random Task Assignment: \n{}".format(random_taskAssignment))

        maxLoad = self.maximumExpertLoad(random_taskAssignment)
        logging.info("Computed Random Task Assignment, max load = {}".format(maxLoad))

        #Compute Objective: Coverage - T_i
        random_F = (lambda_val * sum(self.currentCoverageList)) - maxLoad
        logging.info("Computed Random Task Assignment objective, F_i = {:.3f}".format(random_F))

        return random_F, maxLoad

    
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
        lambda_vals = [lambda_val, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]
        lambdaFiDict = {}
        for l in lambda_vals:
            lambdaFiDict[l] = []

        if 'lazy_greedy' in algorithms:
            logging.info('--------------------------Computing Greedy Task Assignment (Lazy Eval)------------------------------------')
            lazyGreedyi_start = time.perf_counter()
            F_maxArr = []
            T_maxArr = []

            for lVal in lambdaFiDict.keys():
                F_max = 0 #store maximum objective value
                best_T_i = None
                F_i_prev = 0 #variable to store previous objective
                C_max = 0

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
                        C_max = sum(self.currentCoverageList)
                        self.taskAssignment = taskAssignment_T_i
                        best_T_i = T_i

                    # stop search if max is found
                    if F_i < F_i_prev:
                        break
                    F_i_prev = F_i
                    lambdaFiDict[lVal].append(F_i)

                F_maxArr.append(C_max)
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


    def getCoverageValues(self):
        #Track coverage list
        coverageList = []
        self.createMaxHeap()
        
        logging.info("Computing Lazy Greedy coverage list")
        for T_i in range(1, self.maxWorkloadThreshold+1):
            #Create T_i copies of each expert, using a single list to keep track of copies
            experts_copy_list_T_i = [T_i for i in range(self.n)]
            taskAssignment_T_i = self.lazyGreedyTaskAssignment(experts_copy_list_T_i)       
            C_i = sum(self.currentCoverageList)
            coverageList.append(C_i)
            logging.info("Computed Threshold Greedy task assignment for T_i={}, C_i={:.3f}".format(T_i, C_i))

        return coverageList

    def getStepCoverageValues(self):
        #Track coverage list
        coverageDict = {}
        self.createMaxHeap()
        
        logging.info("Computing Lazy Greedy coverage list")
        for T_i in range(10, 211, 20):
            #Create T_i copies of each expert, using a single list to keep track of copies
            experts_copy_list_T_i = [T_i for i in range(self.n)]
            taskAssignment_T_i = self.lazyGreedyTaskAssignment(experts_copy_list_T_i)       
            C_i = sum(self.currentCoverageList)
            coverageDict[T_i] = C_i
            logging.info("Computed Threshold Greedy task assignment for T_i={}, C_i={:.3f}".format(T_i, C_i))

        return coverageDict
