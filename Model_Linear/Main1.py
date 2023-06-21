#%% ---------- Load libraries -----------
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from data_generator import generate_case_I
from iteration_deep import Est_deep
from I_spline import I_S
from Least_FD import LFD
from iteration_linear import Est_linear
from iteration_additive import Est_additive

#%% ---------- define seed -------------
def set_seed(seed):
    np.random.seed(seed)  # Numpy module
    torch.manual_seed(seed) # torch module
#%% ---------- set seed ----------------
set_seed(8)
#%% ----------- set parameters ----------
tau = 10 # the end time of study
p = 3 # cubic integrated spline basis function
Set_n = np.array([1000, 2000]) # two sample sizes
corr = 0.5 # correlation coefficient
n_layer = 3 # the number of layers
n_node = 50 # layer-width
n_epoch = 200 # the number of epochs
n_lr = 4e-4 # learning rate
Beta = 1 # the true value of beta
B = 200 # repeat times
#%% ------ generate testing data (sample size 200)------
test_data = generate_case_I(200, corr, Beta)
X_test = test_data['X']
g_true = test_data['g_X']
dim_x = X_test.shape[0]
u_value = np.array(np.linspace(0, tau, 50), dtype="float32") # choose 50 points in [0, tau] and draw images of cumulative hazard function
Lambda_true = np.sqrt(u_value)/5 # the true baseline cumulative hazard function
m = 10 # the number of interior knot set of integrated spline basis functions
nodevec = np.array(np.linspace(0, tau, m+2), dtype="float32") # the knot set of integrated spline basis functions

m0 = 4 # the number of interior knot set of B-spline functions
nodevec0 = np.array(np.linspace(0, 2, m0+2), dtype="float32") # the knot set of B-spline basis functions
#%% --------Graphics parameter setting------
Markers = np.array(['s','o','^'])
lines = np.array([':','--','-.'])

#%% ---------------- Main results -----------------------
# --- fig1 is used to save the DNN estimation error graph under two samples (each graph has three methods) ---
fig1 = plt.figure()
ax1_1 = fig1.add_subplot(1, 2, 1)
plt.ylim(-2,2) # Set vertical coordinate scale
ax1_1.set_title("Case 1, n=1000",fontsize=10) # Set title and size
ax1_1.set_xlabel("Predictor",fontsize=8) # Set horizontal coordinate label and size
ax1_1.set_ylabel("Error",fontsize=8) # Set vertical coordinate label and size
ax1_1.tick_params(axis='both',labelsize=6) # Axis scale size

ax1_2 = fig1.add_subplot(1, 2, 2)
plt.ylim(-2,2) # Set vertical coordinate scale
ax1_2.set_title("Case 1, n=2000",fontsize=10) # Set title and size
ax1_2.set_xlabel("Predictor",fontsize=8) # Set horizontal coordinate label and size
ax1_2.set_ylabel("Error",fontsize=8) # Set vertical coordinate label and size
ax1_2.tick_params(axis='both',labelsize=6) # Axis scale size
plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=0.3,hspace=0.15)


# --- fig2 is used to save the spline function image under two samples (each graph has three methods)---
fig2 = plt.figure()
ax2_1 = fig2.add_subplot(1, 2, 1)
plt.ylim(0,1) # Set vertical coordinate scale
ax2_1.set_title("Case 1, n=1000", fontsize=10) # Set title and size
ax2_1.set_xlabel("Time",fontsize=8) # Set horizontal coordinate label and size
ax2_1.set_ylabel("Cumulative hazard function",fontsize=8) # Set vertical coordinate label and size
ax2_1.tick_params(axis='both',labelsize=6) # Axis scale size
ax2_1.plot(u_value, Lambda_true, color='k', label='True')
ax2_1.legend(loc='upper left', fontsize=6) # Display the position and size of the label

ax2_2 = fig2.add_subplot(1, 2, 2)
plt.ylim(0,1) # Set vertical coordinate scale
ax2_2.set_title("Case 1, n=2000", fontsize=10) # Set title and size
ax2_2.set_xlabel("Time",fontsize=8) # Set horizontal coordinate label and size
ax2_2.set_ylabel("Cumulative hazard function",fontsize=8) # Set vertical coordinate label and size
ax2_2.tick_params(axis='both',labelsize=6) # Axis scale size
ax2_2.plot(u_value, Lambda_true, color='k', label='True')
ax2_2.legend(loc='upper left', fontsize=6) # Display the position and size of the label
plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=0.3,hspace=0.15)


# --- Save Bias, SSE, ESE, CP in csv format ---
Bias_deep = []; Sse_deep = []; Ese_deep = []; Cp_deep = []; Re_deep = []; G_deep_sd = []
Bias_L = []; Sse_L = []; Ese_L = []; Cp_L = []; Re_L = []; G_L_sd = []
Bias_A = []; Sse_A = []; Ese_A = []; Cp_A = []; Re_A = []; G_A_sd = []
for i in range(len(Set_n)):
    n = Set_n[i]
    #%% ------------ Store the results of B loops ---------
    G_test_deep = []; C_deep=[]; beta_deep = []; Info_deep = []; re_deep = []
    G_test_L = []; C_L = []; beta_L = []; Info_L = []; re_L = []
    G_test_A = []; C_A = []; beta_A = []; Info_A = []; re_A = []
    for b in range(B):
        print('n=', n, 'b=', b)
        set_seed(12 + b)
        #%% ----------- Initial value settings -------------
        c0 = np.array(0.1*np.ones(m+p), dtype="float32") 
        Beta0 = np.array(0, dtype='float32')
        #%% ------------ Generate training data ------------
        train_data = generate_case_I(n, corr, Beta)
        Z_train = train_data['Z']
        U_train = train_data['U']
        De_train = train_data['De']
        g_train = train_data['g_X']
        #%% ======================================DPLCM Method========================================
        Est_hat = Est_deep(train_data=train_data,X_test=X_test,Beta=Beta,Beta0=Beta0,n_layer=n_layer,n_node=n_node,n_lr=n_lr,n_epoch=n_epoch,nodevec=nodevec,m=m,c0=c0)
        # -------- About g_test---------
        G_test_deep.append(Est_hat['g_test']) # vector to add row by row
        # -- Compute relative error and standard deviation of hat_g --
        re_deep.append(np.sqrt(np.mean((Est_hat['g_test']-np.mean(Est_hat['g_test'])-g_true)**2)/np.mean(g_true**2)))
        # -------- About Lambda_U ---------
        C_deep.append(Est_hat['c']) # Calculate the spline function value at u_value according to the estimated parameters
        # ------- About the statistical inference of \hat\beta -----
        a_b_deep = LFD(train_data,I_S(m,Est_hat['c'],U_train,nodevec),Est_hat['g_train'],Est_hat['Beta'],n_layer,n_node=50,n_lr=1e-3,n_epoch=200)
        # Calculate information matrix I(beta_0)
        h_v_deep = I_S(m,Est_hat['c'],U_train,nodevec) * np.exp(Z_train*Est_hat['Beta'] + Est_hat['g_train'])
        Q_y_deep = h_v_deep * (De_train * np.exp(-h_v_deep)/(1-np.exp(-h_v_deep)) - (1-De_train))
        beta_deep.append(Est_hat['Beta'])
        Info_deep.append(np.mean(Q_y_deep**2 * (Z_train-a_b_deep)**2))
        #%% ========================================CPH Method========================================
        Est_L = Est_linear(train_data,X_test,Beta0,nodevec,m,c0)
        # -------- About g_test ---------
        G_test_L.append(Est_L['g_test'])
        # -- Compute relative error and standard deviation of hat_g --
        re_L.append(np.sqrt(np.mean((Est_L['g_test']-np.mean(Est_L['g_test'])-g_true)**2)/np.mean(g_true**2)))
        # -------- About Lambda_U ---------
        C_L.append(Est_L['c']) # Calculate the spline function value at u_value according to the estimated parameters
        # ------- About the statistical inference of \hat\beta -----
        a_b_L = LFD(train_data,I_S(m,Est_L['c'],U_train,nodevec),Est_L['g_train'],Est_L['Beta'],n_layer,n_node=50,n_lr=1e-3,n_epoch=200)
        # Calculate information matrix I(beta_0)
        h_v_L = I_S(m,Est_L['c'],U_train,nodevec) * np.exp(Z_train*Est_L['Beta'] + Est_L['g_train'])
        Q_y_L = h_v_L * (De_train * np.exp(-h_v_L)/(1-np.exp(-h_v_L)) - (1-De_train)) 
        beta_L.append(Est_L['Beta'])
        Info_L.append(np.mean(Q_y_L**2 * (Z_train-a_b_L)**2))
        #%% ========================================PLACM Method========================================
        Est_A = Est_additive(train_data,X_test,Beta0,nodevec,m,c0,m0,nodevec0)
        # # -------- About g_test ---------
        G_test_A.append(Est_A['g_test'])
        # -- Compute relative error and standard deviation of hat_g --
        re_A.append(np.sqrt(np.mean((Est_A['g_test']-np.mean(Est_A['g_test'])-g_true)**2)/np.mean(g_true**2)))
        # -------- About Lambda_U ---------
        C_A.append(Est_A['c'])  # Calculate the spline function value at u_value according to the estimated parameters
        # ------- About the statistical inference of \hat\beta -----
        a_b_A = LFD(train_data,I_S(m,Est_A['c'],U_train,nodevec),Est_A['g_train'],Est_A['Beta'],n_layer,n_node=50,n_lr=1e-3,n_epoch=200)
        # Calculate information matrix I(beta_0)
        h_v_A = I_S(m,Est_A['c'],U_train,nodevec) * np.exp(Z_train*Est_A['Beta'] + Est_A['g_train'])
        Q_y_A = h_v_A * (De_train * np.exp(-h_v_A)/(1-np.exp(-h_v_A)) - (1-De_train)) 
        beta_A.append(Est_A['Beta'])
        Info_A.append(np.mean(Q_y_A**2 * (Z_train-a_b_A)**2))
        
        
    #%% ============Figures for DPLCM================
    Error_deep = np.mean(np.array(G_test_deep), axis=0) - g_true
    if (i == 0):
        ax1_1.scatter(np.arange(dim_x), Error_deep, s=4, marker='o', label='DPLCM')
        ax1_1.legend(loc='upper left', fontsize=4)
        ax2_1.plot(u_value, I_S(m,np.mean(np.array(C_deep), axis=0),u_value,nodevec), label='DPLCM', linestyle='--')
        ax2_1.legend(loc='upper left', fontsize=6)
    else:
        ax1_2.scatter(np.arange(dim_x), Error_deep, s=4, marker='o', label='DPLCM')
        ax1_2.legend(loc='upper left', fontsize=4)
        ax2_2.plot(u_value, I_S(m,np.mean(np.array(C_deep), axis=0),u_value,nodevec), label='DPLCM', linestyle='--')
        ax2_2.legend(loc='upper left', fontsize=6)
    # --------- Bias_deep, SSE_deep, ESE_deep, CP_deep of hat_Beta for DPLCM-----------
    Bias_deep.append(np.mean(np.array(beta_deep))-Beta)
    Sse_deep.append(np.sqrt(np.mean((np.array(beta_deep)-np.mean(np.array(beta_deep)))**2)))
    Ese_deep.append(1/np.sqrt(n*np.mean(np.array(Info_deep))))
    Cp_deep.append(np.mean((np.array(beta_deep)-1.96/np.sqrt(n*np.mean(np.array(Info_deep)))<=Beta)*(Beta<=np.array(beta_deep)+1.96/np.sqrt(n*np.mean(np.array(Info_deep))))))
    # ----- relative error and standard deviation of hat_g for DPLCM -----
    Re_deep.append(np.mean(re_deep))
    G_deep_sd.append(np.mean(np.sqrt((re_deep-np.mean(re_deep))**2)))
    
    #%% ============ Figures for CPH ===================
    Error_L = np.mean(np.array(G_test_L), axis=0) - g_true
    if (i == 0):
        ax1_1.scatter(np.arange(dim_x), Error_L, s=4, marker='s', label='CPH')
        ax1_1.legend(loc='upper left', fontsize=4)
        ax2_1.plot(u_value, I_S(m,np.mean(np.array(C_L), axis=0),u_value,nodevec), label='CPH', linestyle=':')
        ax2_1.legend(loc='upper left', fontsize=6)
    else:
        ax1_2.scatter(np.arange(dim_x), Error_L, s=4, marker='s', label='CPH')
        ax1_2.legend(loc='upper left', fontsize=4)
        ax2_2.plot(u_value, I_S(m,np.mean(np.array(C_L), axis=0),u_value,nodevec), label='CPH', linestyle=':')
        ax2_2.legend(loc='upper left', fontsize=6)
    # --------- Bias_L, SSE_L, ESE_L, CP_L of hat_Beta for CPH -----------
    Bias_L.append(np.mean(np.array(beta_L))-Beta)
    Sse_L.append(np.sqrt(np.mean((np.array(beta_L)-np.mean(np.array(beta_L)))**2)))
    Ese_L.append(1/np.sqrt(n*np.mean(np.array(Info_L))))
    Cp_L.append(np.mean((np.array(beta_L)-1.96/np.sqrt(n*np.mean(np.array(Info_L)))<=Beta)*(Beta<=np.array(beta_L)+1.96/np.sqrt(n*np.mean(np.array(Info_L))))))
    # ----- relative error and standard deviation of hat_g  for CPH -----
    Re_L.append(np.mean(re_L))
    G_L_sd.append(np.mean(np.sqrt((re_L-np.mean(re_L))**2)))
    
    # #%% ============ Figures for PLACM ===================
    Error_A = np.mean(np.array(G_test_A), axis=0) - g_true
    if (i == 0):
        ax1_1.scatter(np.arange(dim_x), Error_A, s=4, marker='^', label='PLACM')
        ax1_1.legend(loc='upper left', fontsize=4)
        ax2_1.plot(u_value, I_S(m,np.mean(np.array(C_A), axis=0),u_value,nodevec), label='PLACM', linestyle='-.')
        ax2_1.legend(loc='upper left', fontsize=6)
    else:
        ax1_2.scatter(np.arange(dim_x), Error_A, s=4, marker='^', label='PLACM')
        ax1_2.legend(loc='upper left', fontsize=4)
        ax2_2.plot(u_value, I_S(m,np.mean(np.array(C_A), axis=0),u_value,nodevec), label='PLACM', linestyle='-.')
        ax2_2.legend(loc='upper left', fontsize=6)
    # --------- Bias_A, SSE_A, ESE_A, CP_A of hat_Beta for PLACM -----------
    Bias_A.append(np.mean(np.array(beta_A))-Beta)
    Sse_A.append(np.sqrt(np.mean((np.array(beta_A)-np.mean(np.array(beta_A)))**2)))
    Ese_A.append(1/np.sqrt(n*np.mean(np.array(Info_A))))
    Cp_A.append(np.mean((np.array(beta_A)-1.96/np.sqrt(n*np.mean(np.array(Info_A)))<=Beta)*(Beta<=np.array(beta_A)+1.96/np.sqrt(n*np.mean(np.array(Info_A))))))
    # ----- relative error and standard deviation of hat_g  for PLACM -----
    Re_A.append(np.mean(re_A))
    G_A_sd.append(np.mean(np.sqrt((re_A-np.mean(re_A))**2)))
#%% -----------Save all results------------
# ================figures======================
fig1.savefig('fig_g_1.jpeg', dpi=400, bbox_inches='tight')
fig2.savefig('fig_Lambda_1.jpeg', dpi=400, bbox_inches='tight')

# =================tables=======================
dic_error = {"n": Set_n, "Bias_deep": np.array(Bias_deep), "SSE_deep": np.array(Sse_deep), "ESE_deep": np.array(Ese_deep), "CP_deep": np.array(Cp_deep), "Bias_L": np.array(Bias_L),  "SSE_L": np.array(Sse_L), "ESE_L": np.array(Ese_L), "CP_L": np.array(Cp_L), "Bias_A": np.array(Bias_A), "SSE_A": np.array(Sse_A), "ESE_A": np.array(Ese_A), "CP_A": np.array(Cp_A)}
result_error = pd.DataFrame(dic_error)
result_error.to_csv('result_error_linear.csv')

dic_re = {"n": Set_n, "Re_deep": np.array(Re_deep), "G_deep_sd": np.array(G_deep_sd), "Re_L": np.array(Re_L), "G_L_sd": np.array(G_L_sd), "Re_A": np.array(Re_A), "G_A_sd": np.array(G_A_sd)}
result_re = pd.DataFrame(dic_re)
result_re.to_csv('result_re_linear.csv')

