from matplotlib import pyplot as plt
import numpy as np

def plot_line(dom:np.ndarray,result:np.ndarray)->None: 
    
    coor = np.where(dom[0] == 0)
    pos = dom[coor,:][0]
    x = pos[:,0].reshape(1,np.shape(x)[0])
    y = pos[:,1].reshape(1,np.shape(x)[0])
    
    for r in result:
        plt.plot(x,y,r)

    plt.legend()
    plt.show()
    
    return 
# def p_o_l()->None:
    
#     pass

if __name__ == " __mmain__":
    
    dom = np.load()
    result = np.load()
    plot_line(dom, result)