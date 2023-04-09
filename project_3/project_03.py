import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from scipy import optimize as opt

deca = np.loadtxt("dec_lengths.txt")
c = 299792458
maxx = np.amax(deca)
minn = np.amin(deca)

mu_k_new = 562.2512 #fitting from Cédric

tau_k = 1.238*10**(-8) #Source Wikipedia
Beta_pp_lab = 0.999998268
Beta_k_lab = mu_k_new/np.sqrt(tau_k**2*c**2+mu_k_new**2)


m_k = 493.677 #MeV/c^2
m_pp = 139.5704 #MeV/c^2
m_pn = 134.9768 #MeV/c^2

factor = m_k**4 + (m_pp**2-m_pn**2)**2#(MeV/c^2)^4
v_pp = np.sqrt((factor - 2*m_k**2*(m_pp**2+m_pn**2))/(factor - 2*m_k**2*(-m_pp**2+m_pn**2))) #eigentlich beta faktor, c := 1
v_pn = np.sqrt((factor - 2*m_k**2*(m_pp**2+m_pn**2))/(factor - 2*m_k**2*(m_pp**2-m_pn**2)))

gamma_pp = 1/np.sqrt(1-v_pp**2)
gamma_pn = 1/np.sqrt(1-v_pn**2)

p_pp = gamma_pp*m_pp*v_pp
p_pn = gamma_pn*m_pn*v_pn

def Mat(p_k, E_k):
    Beta = p_k/E_k
    Gamma = E_k/m_k
    global mat
    mat = np.array([[Gamma, 0, 0, Beta*Gamma], [0, 1, 0, 0], [0, 0, 1, 0], [Beta*Gamma, 0, 0, Gamma]])

Mat(Beta_k_lab*m_k/np.sqrt(1-Beta_k_lab**2), m_k/np.sqrt(1-Beta_k_lab**2))

def opt_detector(A):
    ct = 0
    for i in range(10000):
        K_vertex = np.random.exponential(mu_k_new)
        theta = np.random.uniform(0, np.pi)
        phi = np.random.uniform(-np.pi, np.pi)
        mom_pp = p_pp*np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
        mom_pn = -p_pn*np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])

        fourvec_pp = np.concatenate(([np.sqrt(m_pp**2+p_pp**2)], mom_pp))
        fourvec_pn = np.concatenate(([np.sqrt(m_pn**2+p_pn**2)], mom_pn))

        fourvec_lab_pp = np.matmul(mat, fourvec_pp)
        fourvec_lab_pn = np.matmul(mat, fourvec_pn)

        p_lab_pp = LA.norm(fourvec_lab_pp[1:])
        p_lab_pn = LA.norm(fourvec_lab_pn[1:])
        pz_lab_pp = fourvec_lab_pp[3]
        pz_lab_pn = fourvec_lab_pn[3]
        theta_p = np.arccos(pz_lab_pp/p_lab_pp)
        theta_n = np.arccos(pz_lab_pn/p_lab_pn)
        if theta_p <= np.arctan(2/(A-K_vertex)) and theta_n <= np.arctan(2/(A-K_vertex)) and A >= K_vertex:
            ct += 1
    return ct
#D Acceptance Rate wird akzeptabler. Han jetze es grobs Maximum binere Detektordistanz vo 315 Meter, wo immerhin 35% vode Teili registriert
