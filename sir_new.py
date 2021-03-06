import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import math
#SIR with vital dynamics

# Initial Population
N = 221950
# Initial number of infected,recovered and dead (because of the desease)individuals, I0,R0, M0.
I0, R0, M0 = 0.1*N, 0, 0
# Susceptible people
S0 = N - I0 - R0
# BETA= Contact Rate, GAMMA = Recovery rate.
beta, gamma = 0.2, 0.1 
# X axis, representing time(in days)
t = np.linspace(0, 100, 100)
#Natality rate
m = 0.00338
#Death Rate (D1 represents natural death, D2=date of I.)
D1,D2 = 0.00243,0.026
es = 3 #S people entering the system
ei = 2 #I people entering the system
ss = 2#S people leaving.
si = 0.5 #I people leaving


# The SIR model differential equations.
def deriv(y, t, beta, gamma):
    S, I, R, M = y
    N = S + I + R
    dSdt = es*math.exp(-0.04*t) - ss*math.exp(-0.04*t) + (m*N) -beta * S * I / N - (D1*S)
    dIdt = ei*math.exp(-0.04*t) - si*math.exp(-0.04*t) +beta * S * I / N - gamma * I - (D2*I)
    dRdt = gamma * I - (D1*R)
    dMdt = D2*I
    return dSdt, dIdt, dRdt, dMdt

# Initial conditions vector
y0 = S0, I0, R0, M0
# Integrate the SIR equations over the time grid, t.
ret = odeint(deriv, y0, t, args=(beta, gamma))
S, I, R, M = ret.T

# Plot the data on three separate curves for S(t), I(t) and R(t)
fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
ax.plot(t, S/1000, 'b', alpha=0.5, lw=2, label='Suscetíveis')
ax.plot(t, I/1000, 'g', alpha=0.5, lw=2, label='Infectados')
ax.plot(t, R/1000, 'y', alpha=0.5, lw=2, label='Recuperados com imunidade')
ax.plot(t, M/1000, 'r', alpha=0.5, lw=2, label='Mortos pela doença')
ax.plot(t, (S+I+R)/1000, '--', alpha=0.5, lw=2, label='População Total')
ax.set_xlabel('Tempo (dias)')
ax.set_ylabel('Quantidade de pessoas (*1000)')
ax.set_ylim(0,400)
ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
ax.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax.legend()
legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(False)
plt.show()
