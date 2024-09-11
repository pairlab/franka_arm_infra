from csc376_backend import MoveGroupPythonInterface
import numpy as np
def main():
    object_ = MoveGroupPythonInterface()
    ######################
    #Your values go here #
    T_S_A_T = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0.454, -0.453, 0.020, 1]])
    T_S_A = np.transpose(T_S_A_T)

    T_B_A_T = np.array( [[0.80577, 0.39562, -0.44069, 0], [0.01475, 0.73049, 0.68275, 0], [0.59203, -0.55648, 0.582780, 0], [0.091, 0.630, 0.501, 1]])   
    
    T_B_A = np.transpose(T_B_A_T) 
    
    T_A_S = np.linalg.inv(T_S_A)
    T_S_B = np.linalg.inv(np.matmul(T_B_A, T_A_S))
    
    T_A_C = np.array([[0, 1, 0, 0.124],[-1, 0, 0, -0.331],[0, 0, 1, 0],[0, 0, 0, 1]])
    
    T_S_C = np.matmul(T_S_A, T_A_C)

                                                        
    #######################
    object_.go_to_home()
    ip =  input("============ Press `Enter` to move end effector when ready ..." )
    if (ip==""):
        pass 
    else:
        exit()      
    object_.go_to_transformation(T_S_A)        
    ip =  input("============ Press `Enter` to move end effector when ready ..." )
    if (ip==""):
        pass 
    else:
        exit() 
    object_.go_to_home()
    object_.go_to_transformation(T_S_B)   
    ip =  input("============ Press `Enter` to move end effector when ready ..." )
    if (ip==""):
        pass 
    else:
        exit()           
    object_.go_to_home()
    object_.go_to_transformation(T_S_C) 
    ip =  input("============ Press `Enter` when done ..." )
    if (ip==""):
        pass 
    else:
        exit()  
    object_.go_to_home()                   
    

if __name__ == "__main__":
    main()
