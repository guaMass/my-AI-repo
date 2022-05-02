from numpy import zeros, float32, random
import numpy as np
#  pgmpy
import pgmpy
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
#You can also use following set of modules from 'pgmpy' Library to implement a Bayes Filter
#
# pgmpy.sampling.*
# pgmpy.factors.*
# pgmpy.estimators.*

def get_game_network():
    """Create a Bayes Net representation of the game problem.
    Name the nodes as "A","B","C","AvB","BvC" and "CvA".  """
    BayesNet = BayesianModel()
    BayesNet.add_node("A")
    BayesNet.add_node("B")
    BayesNet.add_node("C")
    BayesNet.add_node("AvB")
    BayesNet.add_node("BvC")    
    BayesNet.add_node("CvA")
    # TODO: fill this out
    BayesNet.add_edge("A","AvB")
    BayesNet.add_edge("B","AvB")
    BayesNet.add_edge("B","BvC")
    BayesNet.add_edge("C","BvC")
    BayesNet.add_edge("C","CvA")
    BayesNet.add_edge("A","CvA")
    cpd_A = TabularCPD('A', 4, values=[[0.15], [0.45], [0.3],[0.1]])
    cpd_B = TabularCPD('B', 4, values=[[0.15], [0.45], [0.3],[0.1]])
    cpd_C = TabularCPD('C', 4, values=[[0.15], [0.45], [0.3],[0.1]])
    cpdAvB = TabularCPD('AvB', 3, values=[[0.1, 0.2, 0.15, 0.05, 0.6, 0.1, 0.2, 0.15, 0.75, 0.6, 0.1, 0.2, 0.90, 0.75, 0.6, 0.1], \
                                          [0.1, 0.6, 0.75, 0.90, 0.2, 0.1, 0.6, 0.75, 0.15, 0.2, 0.1, 0.6, 0.05, 0.15, 0.2, 0.1], \
                                          [0.8, 0.2, 0.10, 0.05, 0.2, 0.8, 0.2, 0.10, 0.10, 0.2, 0.8, 0.2, 0.05, 0.10, 0.2, 0.8]], evidence=['A', 'B'], evidence_card=[4, 4])
    cpdBvC = TabularCPD('BvC', 3, values=[[0.1, 0.2, 0.15, 0.05, 0.6, 0.1, 0.2, 0.15, 0.75, 0.6, 0.1, 0.2, 0.90, 0.75, 0.6, 0.1], \
                                          [0.1, 0.6, 0.75, 0.90, 0.2, 0.1, 0.6, 0.75, 0.15, 0.2, 0.1, 0.6, 0.05, 0.15, 0.2, 0.1], \
                                          [0.8, 0.2, 0.10, 0.05, 0.2, 0.8, 0.2, 0.10, 0.10, 0.2, 0.8, 0.2, 0.05, 0.10, 0.2, 0.8]], evidence=['B', 'C'], evidence_card=[4, 4])
    cpdCvA = TabularCPD('CvA', 3, values=[[0.1, 0.2, 0.15, 0.05, 0.6, 0.1, 0.2, 0.15, 0.75, 0.6, 0.1, 0.2, 0.90, 0.75, 0.6, 0.1], \
                                          [0.1, 0.6, 0.75, 0.90, 0.2, 0.1, 0.6, 0.75, 0.15, 0.2, 0.1, 0.6, 0.05, 0.15, 0.2, 0.1], \
                                          [0.8, 0.2, 0.10, 0.05, 0.2, 0.8, 0.2, 0.10, 0.10, 0.2, 0.8, 0.2, 0.05, 0.10, 0.2, 0.8]], evidence=['C', 'A'], evidence_card=[4, 4])
    BayesNet.add_cpds(cpd_A,cpd_B,cpd_C,cpdAvB,cpdBvC,cpdCvA)   
    return BayesNet


def calculate_posterior(bayes_net):
    """Calculate the posterior distribution of the BvC match given that A won against B and tied C. 
    Return a list of probabilities corresponding to win, loss and tie likelihood."""
    posterior = [0,0,0]
    solver = VariableElimination(bayes_net)
    post_prob = solver.query(variables=['BvC'],evidence={'AvB':0,'CvA':2})    
    posterior = post_prob.values
    return posterior # list 

def cal_match_prob(bayes_net,node,ev1,ev2):
    match_prob = [0]*3
    match_table = bayes_net.get_cpds(node).values
    for i in range(len(match_table)):
        match_prob[i] = match_table[i][ev1][ev2]
        pass
    return match_prob

def cal_mul_match_prob(bayes_net,node,blankets,sampleDict):
    return cal_match_prob(bayes_net,blankets[3],sampleDict[blankets[1]],sampleDict[node])[sampleDict[blankets[3]]]*cal_match_prob(bayes_net,blankets[2],sampleDict[node],sampleDict[blankets[0]])[sampleDict[blankets[2]]]

def cal_team_prob(bayes_net,node,blankets,sampleDict):
    team_table = bayes_net.get_cpds(node).values
    team_prob = team_table.copy()
    for i in range(len(team_table)):
        sampleDict[node] = i
        a = team_table[i]*team_table[sampleDict[blankets[0]]]*team_table[sampleDict[blankets[1]]]*cal_mul_match_prob(bayes_net,node,blankets,sampleDict)
        team_prob[i] = a
    total_prob = sum(team_prob)
    for i in range(len(team_table)):
        team_prob[i] = team_prob[i]/total_prob
    return team_prob
  
  def Gibbs_sampler(bayes_net, initial_state):
    """Complete a single iteration of the Gibbs sampling algorithm 
    given a Bayesian network and an initial state value. 
    
    initial_state is a list of length 6 where: 
    index 0-2: represent skills of teams A,B,C (values lie in [0,3] inclusive)
    index 3-5: represent results of matches AvB, BvC, CvA (values lie in [0,2] inclusive)
    
    Returns the new state sampled from the probability distribution as a tuple of length 6.
    Return the sample as a tuple.    
    """
    sample = tuple(initial_state)   
    skill_level = [0,1,2,3]
    game_result = [0,1,2]       #[1st win,1st lose,tie]
    non_evidenceDict = {'A':0,'B':1,'C':2,'BvC':4}
    team_blanketDict = {'A':('B','C','AvB','CvA'),'B':('C','A','BvC','AvB'),'C':('A','B','CvA','BvC')}
    '''  
    for example: P(BvC|Ci,B) = cal_match_prob(bayes_net,blankets[3],sampleDict[blankets[1]],sampleDict[node])[sampleDict[blankets[3]]]
                 P(CvA|Ci,A) = cal_match_prob(bayes_net,blankets[2],sampleDict[node],sampleDict[blankets[0]])[sampleDict[blankets[2]]]
    '''
    var_flagStr = ""
    if initial_state:
        [A,B,C,AvB,BvC,CvA] = initial_state
    else:
        A = random.choice(skill_level)
        B = random.choice(skill_level)
        C = random.choice(skill_level)
        BvC = random.choice(game_result)
    AvB = 0
    CvA = 2
    prep_sampleDict = {'A':A,'B':B,'C':C,'AvB':AvB,'BvC':BvC,'CvA':CvA}
    prep_sample = [A,B,C,AvB,BvC,CvA]
    node = random.choice(list(non_evidenceDict.keys()))
    node_table = bayes_net.get_cpds(node).values

    if len(node_table) == 3:
        var_flagStr = "match"
    elif len(node_table) == 4:
        var_flagStr = "team"

    if var_flagStr == "match":
        match_prob = [0]*3
        for i in range(len(node_table)):
            match_prob[i] = node_table[i][B][C]
            pass
        prep_sample[non_evidenceDict[node]] = random.choice(game_result,p=match_prob)
    elif var_flagStr == "team":
        team_prob = cal_team_prob(bayes_net,node,team_blanketDict[node],prep_sampleDict)
        prep_sample[non_evidenceDict[node]] = random.choice(skill_level,p=team_prob)
        pass

    sample = tuple(prep_sample)
    return sample


def MH_sampler(bayes_net, initial_state):
    """Complete a single iteration of the MH sampling algorithm given a Bayesian network and an initial state value. 
    initial_state is a list of length 6 where: 
    index 0-2: represent skills of teams A,B,C (values lie in [0,3] inclusive)
    index 3-5: represent results of matches AvB, BvC, CvA (values lie in [0,2] inclusive)    
    Returns the new state sampled from the probability distribution as a tuple of length 6. 
    """
    sample = tuple(initial_state)    
    skill_level = [0,1,2,3]
    game_result = [0,1,2]       #[1st win,1st lose,tie]
    non_evidenceDict = {'A':0,'B':1,'C':2,'BvC':4}
    team_blanketDict = {'A':('B','C','AvB','CvA'),'B':('C','A','BvC','AvB'),'C':('A','B','CvA','BvC')}
    var_flagStr = ""
    if initial_state:
        [A,B,C,AvB,BvC,CvA] = initial_state
    else:
        A = random.choice(skill_level)
        B = random.choice(skill_level)
        C = random.choice(skill_level)
        BvC = random.choice(game_result)
    AvB = 0
    CvA = 2
    nA = random.choice(skill_level)
    nB = random.choice(skill_level)
    nC = random.choice(skill_level)
    nBvC = random.choice(game_result)
    A_cpd = bayes_net.get_cpds('A').values
    B_cpd = bayes_net.get_cpds('B').values
    C_cpd = bayes_net.get_cpds('C').values
    AvB_cpd = bayes_net.get_cpds('AvB').values
    BvC_cpd = bayes_net.get_cpds('BvC').values
    CvA_cpd = bayes_net.get_cpds('CvA').values
    prep_sampleDict = {'A':A,'B':B,'C':C,'AvB':AvB,'BvC':BvC,'CvA':CvA}
    ini_sample = [A,B,C,AvB,BvC,CvA]
    new_sample = [nA,nB,nC,0,nBvC,2]
    prob_new = A_cpd[nA] * B_cpd[nB] * C_cpd[nC] * AvB_cpd[0][nA][nB] * BvC_cpd[nBvC][nB][nC] * CvA_cpd[2][nC][nA] # Generate a uniform distribution
    prob_ini = A_cpd[A] * B_cpd[B] * C_cpd[C] * AvB_cpd[0][A][B] * BvC_cpd[BvC][B][C] * CvA_cpd[2][C][A]
    r = prob_new / prob_ini
    alpha = min(r,1)
    if random.random() < alpha:
        sample = tuple(new_sample)
    else:
        sample = tuple(ini_sample)   
    return sample


def compare_sampling(bayes_net, initial_state):
    """Compare Gibbs and Metropolis-Hastings sampling by calculating how long it takes for each method to converge."""    
    Gibbs_count = 0
    MH_count = 0
    MH_rejection_count = 0
    Gibbs_convergence = [0,0,0] # posterior distribution of the BvC match as produced by Gibbs 
    MH_convergence = [0,0,0] # posterior distribution of the BvC match as produced by MH
    Bwc = 0
    Blc = 0
    Btc = 0
    pw = 0
    pl = 0
    pt = 0
    delta = 0.00001
    N = 200000
    sample = initial_state
    for i in range(N):
        Gibbs_count = i
        sample = Gibbs_sampler(bayes_net,sample)
        if sample[4] == 0:
            Bwc += 1
        elif sample[4] == 1:
            Blc += 1
        elif sample[4] == 2:
            Btc += 1
        pw = Bwc/(Bwc+Blc+Btc)
        pl = Blc/(Bwc+Blc+Btc)
        pt = Btc/(Bwc+Blc+Btc)
        err0 = Gibbs_convergence[0]-pw
        err1 = Gibbs_convergence[1]-pl
        err2 = Gibbs_convergence[2]-pt
        #print("erro:",err0,err1,err2)
        if (abs(Gibbs_convergence[0]-pw) < delta) and (abs(Gibbs_convergence[1]-pl) < delta) and (abs(Gibbs_convergence[2]-pt) < delta):
            if i > 50:
                Gibbs_count = i
                break
        else:
            Gibbs_convergence = [pw,pl,pt]
    Bwc = 0
    Blc = 0
    Btc = 0
    pw = 0
    pl = 0
    pt = 0
    sample = initial_state
    for i in range(N):
        MH_count = i
        new_sample = MH_sampler(bayes_net,sample)
        if new_sample == sample:
            MH_rejection_count += 1
        else:
            sample = new_sample
        #print(sample)
        if len(sample) <= 4:
            print(new_sample)
            print(sample,"warn!")
        if sample[4] == 0:
            Bwc += 1
        elif sample[4] == 1:
            Blc += 1
        elif sample[4] == 2:
            Btc += 1
        pw = Bwc/(Bwc+Blc+Btc)
        pl = Blc/(Bwc+Blc+Btc)
        pt = Btc/(Bwc+Blc+Btc)
        err0 = MH_convergence[0]-pw
        err1 = MH_convergence[1]-pl
        err2 = MH_convergence[2]-pt
        if (abs(MH_convergence[0]-pw) < delta) and (abs(MH_convergence[1]-pl) < delta) and (abs(MH_convergence[2]-pt) < delta):
            if i > 50:
                MH_count = i
                break
        else:
            MH_convergence = [pw,pl,pt]       
    return Gibbs_convergence, MH_convergence, Gibbs_count, MH_count, MH_rejection_count
