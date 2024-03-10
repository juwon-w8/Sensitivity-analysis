#!/usr/bin/env python
# coding: utf-8

# In[74]:


import numpy as np
import math as m
import matplotlib.pyplot as plt

def high_p(R1,C1,w):
    Tf=(R1/(R1+(-1j/w*C1)))
    Re=Tf.real
    Im=Tf.imag
    Mag=np.sqrt(Re**2 + Im**2)
    #Phase= m.degrees(np.arctan(Im/Re))
    #print(Tf)
    return [Tf, Re, Im, Mag]

def low_p(R2,C2,w):
    Tf=((-1j/w*C2)/(R2+(-1j/w*C2)))
    Re = Tf.real
    Im = Tf.imag
    Mag = np.sqrt(Re**2 + Im**2)
    #Phase= m.degrees(np.arctan(Im/Re))
    return [Tf, Re, Im, Mag]
    
def Nop_amp(R3):
    R4= 3*R3
    Tf=(1+ R4/R3)
    return[Tf]
    
    
def Transfer_fn(R1,R2,C1,C2,C,w):
    Tf=(R1/(R1+(-1j/w*C1)))*((-1j/w*C2)/(R2+(-1j/w*C2)))*C
    Re=Tf.real
    Im=Tf.imag
    Mag= np.sqrt(Re**2 + Im**2)
   # Phase= m.degrees(np.arctan(Im/Re))
    #print (Mag)
    return [Re, Im, Tf, Mag]

def freq_check(R1,R2,C1,C2):
    high_cut=(1/(2*m.pi*R1*C1))
    low_cut=(1/(2*m.pi*R2*C2))
    bandwidth=(high_cut-low_cut)
    center_f=bandwidth/2
    #print(bandwidth,center_f)
    return[bandwidth,center_f]

def R4_range(R3):
    R4= np.arange(2,100,0.1)*R3
    Tf=(1+ R4/R3)
    return[Tf]
    


def main():
        R1=50e-6
        R2=50e-6
        R3=2
        C1=15.98e-6
        C2=5*C1
        w=12560
        #A=high_p(R1,C1,w)[0]
        #B=low_p(R2,C2,w)[0]
        C=Nop_amp(R3)[0]
       
        w=np.arange(0.1,5,0.0011)
        Mag_h=high_p(R1,C1,w)[3]
        Mag_l=low_p(R1,C2,w)[3]
        Mag_tf=Transfer_fn(R1,R2,C1,C2,C,w)[3]
        #Sens=freq_check(R1,R2,C1)
        #print (Mag_tf)
        
        figure, axis= plt.subplots(nrows=2,ncols=3, layout='constrained')
        
        axis[0,0].plot(w,Mag_h)
        axis[0,0].set_title('High Pass Filter')
        axis[0,0].set_xlabel('Omega ,w')
        axis[0,0].set_ylabel('Magnitude, |Ts|')
        
        axis[0,1].plot(w,Mag_l)
        axis[0,1].set_title('Low Pass Filter')
        axis[0,1].set_xlabel('Omega ,w')
        axis[0,1].set_ylabel('Magnitude, |Ts|')
        
        axis[0,2].plot(w,Mag_tf)
        axis[0,2].set_title('Transfer function')
        axis[0,2].set_xlabel('Omega ,w')
        axis[0,2].set_ylabel('Magnitude, |Ts|')
        
        
        C1=np.arange(-0.001,0.01,0.0015)*1e-6
        w=12560
        #Mag_tf=Transfer_fn(R1,R2,C1,C2,C,w)[3]
        Sens1=freq_check(R1,R2,C1,C2)[0]
        axis[1,0].plot(C1,Sens1)
        axis[1,0].set_title('Bandwidth vs C1 ')
        axis[1,0].set_xlabel('Capacitance ')
        axis[1,0].set_ylabel('Bandwidth')
        
        Sens2=freq_check(R1,R2,C1,C2)[1]
        #B=Nop_amp(R3)[0]
        #Mag_tf=Transfer_fn(R1,R2,C1,C2,B,w)[3]
        axis[1,1].plot(C1,Sens2)
        axis[1,1].set_title('Center Frequency vs C1')
        axis[1,1].set_xlabel('C1 ')
        axis[1,1].set_ylabel('Center Frequency')
        
        C1=15.98e-6
        B=R4_range(R3)[0]
        #w=np.arange(10000,12500,100)
        Sens_tf=Transfer_fn(R1,R2,C1,C2,B,w)[3]
        axis[1,2].plot(B,Sens_tf)
        axis[1,2].set_title('Gain vs Ts')
        axis[1,2].set_xlabel('Gain ')
        axis[1,2].set_ylabel('Transfer fn, |Ts|')
        #figure.tight_layout()

        

        

if __name__=="__main__":
    main()


# In[75]:


import numpy as np
import math as m
import matplotlib.pyplot as plt

def high_p(R1,C1,w):
    Tf=(R1/(R1+(-1j/w*C1)))
    Re=Tf.real
    Im=Tf.imag
    Mag=np.sqrt(Re**2 + Im**2)
    #Phase= m.degrees(np.arctan(Im/Re))
    #print(Tf)
    return [Tf, Re, Im, Mag]

def low_p(R2,C2,w):
    Tf=((-1j/w*C2)/(R2+(-1j/w*C2)))
    Re = Tf.real
    Im = Tf.imag
    Mag = np.sqrt(Re**2 + Im**2)
    #Phase= m.degrees(np.arctan(Im/Re))
    return [Tf, Re, Im, Mag]
    
def Nop_amp(R3):
    R4= 3*R3
    Tf=(1+ R4/R3)
    return[Tf]
    
    
def Transfer_fn(R1,R2,C1,C2,C,w):
    Tf=(R1/(R1+(-1j/w*C1)))*((-1j/w*C2)/(R2+(-1j/w*C2)))*C
    Re=Tf.real
    Im=Tf.imag
    Mag= np.sqrt(Re**2 + Im**2)
   # Phase= m.degrees(np.arctan(Im/Re))
    #print (Mag)
    return [Re, Im, Tf, Mag]

def freq_check(R1,R2,C1):
    C2=5*C1
    high_cut=(1/(2*m.pi*R1*C1))
    low_cut=(1/(2*m.pi*R2*C2))
    bandwidth=(high_cut-low_cut)
    center_f=bandwidth/2
    #print(bandwidth,center_f)
    return[bandwidth,center_f]

def R4_range(R3):
    R4= np.arange(2,100,0.1)*R3
    Tf=(1+ R4/R3)
    return[Tf]
    


def main():
        R1=100e-6
        R2=100e-6
        R3=2
        C1=15.98e-6
        C2=5*C1
        w=12560
        #A=high_p(R1,C1,w)[0]
        #B=low_p(R2,C2,w)[0]
        C=Nop_amp(R3)[0]
       
        w=np.arange(0.1,5,0.01)
        Mag_h=high_p(R1,C1,w)[3]
        Mag_l=low_p(R1,C2,w)[3]
        Mag_tf=Transfer_fn(R1,R2,C1,C2,C,w)[3]
        Sens=freq_check(R1,R2,C1)
        #print (Mag_tf)
        
        figure, axis= plt.subplots(nrows=2,ncols=3, layout='constrained')
        
        axis[0,0].plot(w,Mag_h)
        axis[0,0].set_title('High Pass Filter')
        axis[0,0].set_xlabel('Omega ,w')
        axis[0,0].set_ylabel('Magnitude, |Ts|')
        
        axis[0,1].plot(w,Mag_l)
        axis[0,1].set_title('Low Pass Filter')
        axis[0,1].set_xlabel('Omega ,w')
        axis[0,1].set_ylabel('Magnitude, |Ts|')
        
        axis[0,2].plot(w,Mag_tf)
        axis[0,2].set_title('Transfer function')
        axis[0,2].set_xlabel('Omega ,w')
        axis[0,2].set_ylabel('Magnitude, |Ts|')
        
        
        C1=np.arange(-0.001,0.001,0.00015)*1e-6
        w=12560
        #Mag_tf=Transfer_fn(R1,R2,C1,C2,C,w)[3]
        Sens1=freq_check(R1,R2,C1)[0]
        axis[1,0].plot(C1,Sens1)
        axis[1,0].set_title('Bandwidth vs C1 ')
        axis[1,0].set_xlabel('Capacitance ')
        axis[1,0].set_ylabel('Bandwidth')
        
        Sens2=freq_check(R1,R2,C1)[1]
        #B=Nop_amp(R3)[0]
        #Mag_tf=Transfer_fn(R1,R2,C1,C2,B,w)[3]
        axis[1,1].plot(C1,Sens2)
        axis[1,1].set_title('Center Frequency vs C1')
        axis[1,1].set_xlabel('Tf ')
        axis[1,1].set_ylabel('C1')
        
        C1=15.98e-6
        B=R4_range(R3)[0]
        #w=np.arange(10000,12500,100)
        Sens_tf=Transfer_fn(R1,R2,C1,C2,B,w)[3]
        axis[1,2].plot(B,Sens_tf)
        axis[1,2].set_title('Gain vs Ts')
        axis[1,2].set_xlabel('Gain ')
        axis[1,2].set_ylabel('Transfer fn, |Ts|')
        #figure.tight_layout()

        

        

if __name__=="__main__":
    main()


# In[ ]:





# In[ ]:




