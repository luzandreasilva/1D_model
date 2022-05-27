import matplotlib.pylab as plt
import netCDF4
import numpy as np
import netCDF4 as nc


def wind_prof(time, lat, lon, kind, UA_v, VA_v, ti, tf,w):
    """This function allows to have different simple wind profiles to use in a 1D simple model as surface boundary condition:
    • time: time vector
    • lat: Vector of latitudes
    • lon: Vector of longitudes
    • kind: ST for step profile, SL  for profile with slope, INC for just one increasing, PER for periodic profile
    • UA_v: rated wind velocity for u component
    • VA_v: rated wind velocity for v component
    • ti, tf: at which time is desire wind different of zero
    • w: frequency at given latitude"""
    
    
  
    nt=len(time)
    
    # Initializing arrays
    UA = np.zeros((nt))
    VA = np.zeros((nt))
    
    offset_u = 0
    offset_v = 0
    
    # Definition of profiles
     
    if kind == 'ST':
        offset_u = 0
        offset_v = 0
    elif kind == 'SL':
        offset_u = 30
        offset_v = 30

    ti_o = ti +  offset_u
    tf_o = tf -  offset_v

    for i,d in enumerate(time):
        if ((d >= ti) & (d <= ti_o)):
            UA[i] = UA_v/(ti_o - ti) * (d-ti)
            VA[i] = VA_v/(ti_o - ti) * (d-ti)
        if ((d > ti_o) & (d < tf_o)):
            UA[i] = UA_v
            VA[i] = VA_v
        if ((d >= tf_o) & (d <= tf)):
            UA[i] =  UA_v - (UA_v/(tf - tf_o)) * (d-tf_o)
            VA[i] =  VA_v - (VA_v/(tf - tf_o)) * (d-tf_o)
            
    if kind == 'INC':
        for i,d in enumerate(time):
            if ((d >= ti) & (d <= ti_o)):
                UA[i] = UA_v/(ti_o - ti) * (d-ti)
                VA[i] = VA_v/(ti_o - ti) * (d-ti)
            if ((d > ti_o)):
                UA[i] = UA_v
                VA[i] = VA_v

    if kind == 'PER':
        for i,d in enumerate(time):
            if (d > ti):
                UA[i] = np.cos(w*50*(time[i]-ti))* UA_v
                VA[i] = np.sin(w*50*(time[i]-ti))* VA_v
                
    
    plt.plot(time-time[0], UA, c='g', lw=2, label='10m zonal wind')
    plt.plot(time-time[0], VA, c='r', lw=2, label='10m meridional wind')
    plt.grid('on')
    plt.ylabel('Wind (m/s)', fontsize=12)
    plt.xlabel('time (days)', fontsize=12)
    plt.legend(fontsize=12)
    
    
    return UA,VA


def mod_arr_1D(field, order, variable):
    
    """Define array with boundary conditions to compute partial derivatives along z axis for 1D model. 
    This 1D model is discretized using a finite volume approach with second order scheme for partial derivatives 
    along z.
    
    • field: field to derive (array to modify).
    • order=order of precision of derivatives. This code can be extended for more orders
    • variable = "independent" or "independent". In the case of order 2 the independent variable needs another 
    differences than for dependent variable"""

    ### for the computation of different the delta Z necessaries in this 1D model are generated 
    #2 sintetic arrows:N=N+(N-(N-1)) in the vector of delta Z for the computation of the extreme elements of 
    # the velocities (first and last one)###
    
    if order == 2:
        if variable == 'dependent':
            mod_arr_1D = np.roll(field,-1,axis = 0)
            mod_arr_1D[-1]= 0
            
        if variable == 'independent':
            
            mod_arr_1D = np.roll(field,-2,axis = 0)        

    
    return mod_arr_1D 



def derivative_1Dmodel(field1, field2, order, direction):
    
    """Computation of partial derivatives along along z axis for 1D model.This 1D model is discretized using a finite 
    volume approach with second order scheme for partial derivatives along z. This function allows to get the two 
    derivates required for the 1D model. 1 of them has a forward second order scheme and the other a backward second
    order scheme.
     
    • field1: Field to derive, this has to be a vertical vector !!
    • field2: Field to derive from, this has to be a vertical vector !!
    • order= order of precision of derivatives, this function can be expanded to a higher order of precision
    • Direction = forward or backward scheme"""
    
    ### for the computation of derivativers along Z necessaries in this 1D model are generated 
    #2 sintetic arrows:N=N+(N-(N-1)) in the vector of delta Z for the computation of the extreme elements of 
    # the velocities (first and last one)###
    
    Dzf = np.insert(field2, 0, 2*field2[0]-field2[1], axis=0) #This insert an additional first row
    Dz  = np.insert(Dzf,len(Dzf), 2*field2[-1]-field2[-2],axis=0) #This insert an additional last row
    
    
    if order == 2:
        if direction == 'forward':
            der_z = 2* ((field1 - mod_arr_1D(field1, 2, "dependent"))[0:field1.shape[0]-1]) / ((Dz - mod_arr_1D(Dz, 2, "independent"))[0:Dz.shape[0]-3])
        if direction == 'backward':
            der_z = 2* ((field1 - mod_arr_1D(field1, 2, "dependent"))[1::]) / (np.delete((Dz - mod_arr_1D(Dz, 2, "independent")),0,0)[0:Dz.shape[0]-3])

    return der_z  


def eddy_visc_aver(K,direction):
    """"Computation of the meaning of K for the different levels
    • K: Vector that contains eddy viscosity for each layer
    • Direction = definition if this K is using to compute with forward or backward scheme"""
    
    ### Since is required the average for each layer, for the extreme cases is duplicated the first and last row
    # to be able to do the computations
    
    Kf   = np.insert(K,1,K[0],axis=0)
    K_m = np.insert(Kf,len(Kf),Kf[-1],axis=0)

    
  
    if direction == 'forward':
        K_av = ( (K_m[:,None] + np.roll(K_m[:,None],-1,axis = 0))[0:K_m.shape[0]-2] ) / 2
        
    if direction == 'backward':
        K_av = ( np.delete((K_m[:,None] + np.roll(K_m[:,None],-1,axis = 0)),0,0) )[0:K_m.shape[0]-2]  / 2
    
    return K_av


#_________________________________________________________________________________________________________________________________

