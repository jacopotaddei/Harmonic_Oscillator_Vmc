
'''______________________________Modules________________________________________________________________'''

import numpy as np
import matplotlib.pyplot as plt
# Imported mathematical functions and constants  
from numpy import sqrt, exp, pi     


'''____________________________Definition of Functions___________________________________________________'''

# Trial wave function. Alfa = variational parameter
def wave_function(x,Alfa):
    wf=sqrt(Alfa)*pi**(-1/4)*exp(-0.5*(Alfa*x)**2)  
    return wf

# Local energy, i.e. Hamiltonian applied to the wf over the wf
def local_energy(x,Alfa):               
    el=Alfa**2+(1-Alfa**4)*x**2
    return el

 # Metropolis Algoritm
def Metropolis(Alfa, N_met, N_th=0):   
    
    e, e2, rej = 0, 0, 0
    # Metropolis step
    step=0.5                           
   
    # Three sigma, typical relevant lenght of the system
    L=3/(np.sqrt(2)*Alfa)        
    # Random coordinate generator      
    x=np.random.uniform(0,1)*2*L-L     
    
    # Number of moves of the Metropolis algoritm without computation of observables
    for k in range(N_th):       
        x_try= x + step*(np.random.uniform(0,1)*2*L-L)
        P_try=wave_function(x_try,Alfa)**2
        P_start=wave_function(x,Alfa)**2
        x_mute=np.random.uniform(0,1)
        if (P_try/P_start)>x_mute:
            x=x_try
        else: pass
            
    # Beginning of Metropolis sampling and computation of the local energy
    for i in range(N_met):      
        x_try= x + step*(np.random.uniform(0,1)*2*L-L)
        P_try=wave_function(x_try,Alfa)**2
        P_start=wave_function(x,Alfa)**2
        x_mute=np.random.uniform(0,1)
        if (P_try/P_start)>x_mute:
            x=x_try
        else: rej+=1
        
        e+=local_energy(x,Alfa)
        e2+=local_energy(x,Alfa)**2
        
    # Averages over the random walk and computation of the rejection ratio
    E=e/N_met         
    E2=e2/N_met
    RejRatio=100*rej/N_met
    return E, E2, RejRatio

def analytical_solutions(Alfa):
    Analitical_Energy=0.5*Alfa**2+1/(2*Alfa**2)
    Analitical_Error=(Alfa**4-1)/(sqrt(2)*Alfa**2)
    return Analitical_Energy, Analitical_Error

'''________________________________________Program____________________________________________________'''


# N_met = number of loops for metropolis, N_nel= number of non-evaluated loops
N_met, N_nel= 500, 0
# Parameter to be incrementated every loop
Alfa=0.4
# Number of points, namelly the number of different values of Alfa
N_pts=15
# Number of random walkers for each value of Alfa
N_walk = 50

#Empty arrays to be filled during loops
E_Vec=np.empty([0])
dE_Vec=np.empty([0])
Rej_Vec=np.empty([0])
Alfa_Vec=np.empty([0])

#Beginning of the variation over N_walk random walkers
for i in range(N_pts):
    E, E2, Rej=0,0,0
    for l in range(N_walk):
        e_m, e2_m, rej_m= Metropolis(Alfa, N_met, N_nel)
        E+=e_m/N_walk
        E2+=e2_m/N_walk
        Rej+=rej_m/N_walk
    E_Vec=np.append(E_Vec, E)
    dE_Vec=np.append(dE_Vec, sqrt(abs(E2-E**2)/N_walk))
    Rej_Vec=np.append(Rej_Vec, Rej)
    Alfa_Vec=np.append(Alfa_Vec, Alfa)
    #Increment for each loop of the variational parameter
    Alfa+=0.1
    
#Expectations from the theory
E_theory, dE_theory=analytical_solutions(Alfa_Vec)

#Plot of the results
plt.figure(1)
plt.xlabel('\u03B1')
plt.ylabel('Energy')
plt.grid(True)
#Simulated values and errors
plt.errorbar(Alfa_Vec, E_Vec, dE_Vec, None, '.', ecolor='g') 
#Analytical points
plt.plot(Alfa_Vec, E_theory, 'o', color='r') 
plt.legend(['Analytic Solution', 'VMC Estimation'])

print()
for i in range(N_pts):
    Energy=E_Vec[i]
    dEnergy=dE_Vec[i]
    Alfa=Alfa_Vec[i]
    E_th=E_theory[i]
    dE_th=dE_theory[i]
    print('\u03B1 = %.2f' %Alfa)
    print('VMC: ( %.2f'%Energy,' +- %.2f' %dEnergy,')')
    print('Theory: ( %.2f'%E_th,' +- %.2f' %dE_th,')')
    print()
    


    






















