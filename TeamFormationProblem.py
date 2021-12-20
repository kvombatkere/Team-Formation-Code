## Implementation of Framework for Team Formation problem
## Balancing Task Coverage vs. Maximum Expert Load
## Karan Vombatkere, Dec 2021


#Import and logging config
import numpy as np
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


    def computeTotalTaskCoverage(self, taskAssigment = None, list_flag = True):
        '''
        Given a taskAssignment, compute the total task coverage of the assigment
        ARGS:
            taskAssignment  : current task assigment
            list_flag  : flag to determine whether to output total coverage or list
        RETURN:
            total_task_coverage_value   : total task coverage across all tasks
            total_task_coverage_list    : list of all m task coverage values
        '''
        if taskAssigment is not None:
            task_assigment = taskAssigment
        else:
            task_assigment = self.taskAssignment
        

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



    def getBestExpertTaskEdge(self, taskAssignment, experts_array_indices, current_coverage_list):
        '''
        Greedily compute the best expert-task edge (assigment) that maximizes coverage in that iteration
        ARGS:
            taskAssignment : current task assigment
            experts_array_indices  : current list of available experts, as index values
            current_coverage_list    : current total task coverage of each task
        RETURN:
            max_edge_coverage  : maximum value of change in task coverage by adding edge (i,j)
            bestExpertTaskEdge  : indices of expert and task as a tuple (i,j)
        '''
        bestExpertTaskEdge = None
        max_edge_coverage = 0
        
        for i in experts_array_indices:
            E_i = self.experts[i]
            for j, T_j in enumerate(self.tasks):

                #Get experts currently assigned to T_j
                T_j_assigned_experts_indices = self.getExpertsAssignedToTask(taskAssignment, j)

                #Check if this expert is already assigned to T_j
                if i not in T_j_assigned_experts_indices:
                    #Compute union of skills of all experts assigned to T_j
                    expert_skills = self.getExpertSkillSet(T_j_assigned_experts_indices)
                    
                    expert_skills = expert_skills.union(set(E_i))       #Add expert E_i
                    task_skills = set(T_j)      #Get task skills as a set
                    
                    #Compute task coverage with expert added
                    T_j_coverage = len(expert_skills.intersection(task_skills))/len(task_skills)
                    delta_coverage = T_j_coverage - current_coverage_list[j]
                    logging.debug("deltaCoverage={:.1f}, T_j_coverage={}".format(delta_coverage, T_j_coverage))

                    if delta_coverage > max_edge_coverage:
                        max_edge_coverage = delta_coverage
                        bestExpertTaskEdge = (i,j)
                        
                        #Return immediately if all skills in a task are covered
                        if max_edge_coverage == 1:
                            return max_edge_coverage, bestExpertTaskEdge

        return max_edge_coverage, bestExpertTaskEdge



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



    def greedyTaskAssignment(self, experts_arr_indices):
        '''
        Greedily compute a Task Assignment given a set of tasks and experts
        ARGS:
            experts_arr_indices  : list of available experts, as index values
        RETURN:
            taskAssignment  : greedy task assigment
        '''
        startTime = time.perf_counter()

        #Create empty task assigment matrix
        taskAssignment_i = np.zeros((self.n, self.m), dtype=np.int8)

        #Initialize current coverage list
        currentCoverageList = self.computeTotalTaskCoverage(taskAssignment_i)
        
        #Get next best assignment and coverage value
        deltaCoverage, best_ExpertTaskEdge = self.getBestExpertTaskEdge(taskAssignment_i, experts_arr_indices, currentCoverageList)

        #Assign edges until there is no more coverage possible or no experts left
        while (deltaCoverage != 0) and (experts_arr_indices):

            #Add edge to assignment
            taskAssignment_i[best_ExpertTaskEdge[0], best_ExpertTaskEdge[1]] = 1
            #self.displayTaskAssignment(taskAssignment_i)

            #Remove expert from experts_arr_indices
            experts_arr_indices.remove(best_ExpertTaskEdge[0])
            
            #Calculate current coverage list
            currentCoverageList = self.computeTotalTaskCoverage(taskAssignment_i)
            logging.debug("Current Coverage List = {}".format(currentCoverageList))

            #Get next best assignment and coverage value
            deltaCoverage, best_ExpertTaskEdge = self.getBestExpertTaskEdge(taskAssignment_i, experts_arr_indices, currentCoverageList)
            logging.debug("deltaCoverage={:.1f}, best_ExpertTaskEdge={}".format(deltaCoverage, best_ExpertTaskEdge))
        

        runTime = time.perf_counter() - startTime
        logging.debug("Greedy Task Assignment computation time = {:.1f} seconds".format(runTime))

        return taskAssignment_i



    def computeTaskAssigment(self):
        '''
        Compute a Task Assignment, of experts to tasks.
        Use m thresholds for the maximum load, and call a greedy method for each threshold
        Store this task assignment in self.taskAssignment
        '''
        startTime = time.perf_counter()
        F_max = 0 #store maximum objective value
        best_T_i = None

        for T_i in range(1, self.maxWorkloadThreshold+1):
            logging.info('--------------------------------------------------------------------------------------------------')
            logging.info("Computing Greedy Task Assignment for threshold max load, T_i={}".format(T_i))

            #Create T_i copies of each expert, using index values
            experts_index_list_T_i = [indx for indx in range(self.n)]*T_i
            logging.info("Expert Index List: {}".format(experts_index_list_T_i))

            #Greedily assign experts to tasks using this expert list
            taskAssignment_T_i = self.greedyTaskAssignment(experts_index_list_T_i)       
            logging.info("Greedy Task Assignment = \n{}".format(taskAssignment_T_i))

            #Compute Objective: Coverage - T_i
            F_i = self.computeTotalTaskCoverage(taskAssignment_T_i, list_flag=False) - T_i
            logging.info("F_i = {:.3f}".format(F_i))

            if F_i > F_max:
                F_max = F_i
                self.taskAssignment = taskAssignment_T_i
                best_T_i = T_i

        logging.info("Best Task Assignment is for max workload threshold: {}, F_i(max)={:.3f} \n{}".format(best_T_i, F_max, self.taskAssignment))
        
        runTime = time.perf_counter() - startTime
        logging.info("\nTotal Computation time = {:.3f} seconds".format(runTime))

        return None

