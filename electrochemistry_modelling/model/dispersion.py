import math
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 24})

class dispersion_class():
    '''
    This is a class to implement distributions of samples on electrochemical
    model parameters in order to simulate he phenomena of dispersion
    '''

    def __init__(self):
        pass

    def normal(self, nodes_num, std, mean):
        nodes, normal_weights =self.GH_setup(nodes_num)

        param_vals = np.zeros_like(nodes)

        node_length = nodes.shape
        node_length = node_length[0]
        for i in range(node_length):
            param_vals[i] = std*math.sqrt(2)*nodes[i] + mean

        # plt.plot(param_vals, normal_weights)
        # plt.show()

        return param_vals, normal_weights


    def log_normal(self, nodes_num, std, mean):
        
        std = math.log(std)
        mean = math.log(mean)
        # std = math.sqrt(math.log(1+(std**2/mean**2)))
        # mean = math.log((mean**2)/math.sqrt(mean**2+ std**2))
        print('mean: ', mean)
        print('std: ', std)
        param_vals, normal_weights = self.normal(nodes_num,std,mean)

        # param_vals = np.log(param_vals)
        param_vals = np.exp(param_vals)
        # normal_weights = np.exp(normal_weights)
        # print(param_vals.shape)
        # plt.plot(param_vals[:-4], normal_weights[:-4])
        # plt.show()

        return param_vals, normal_weights
    
    def skewed_normal_distribution(self, nodes_num, std, mean, skew):
        from scipy.stats import skewnorm

        param_mean=mean
        param_std=std
        param_skew=skew
        min_val=skewnorm.ppf(1e-4, param_skew, loc=param_mean, scale=param_std)
        max_val=skewnorm.ppf(1-1e-4, param_skew, loc=param_mean, scale=param_std)
        param_vals=np.linspace(min_val, max_val, nodes_num)
        param_weights=np.zeros(nodes_num)
        param_weights[0]=skewnorm.cdf(param_vals[0],param_skew, loc=param_mean, scale=param_std)
        param_midpoints=np.zeros(nodes_num)
        param_midpoints[0]=skewnorm.ppf((1e-4/2), param_skew, loc=param_mean, scale=param_std)
        for j in range(1, nodes_num):
            param_weights[j]=skewnorm.cdf(param_vals[j],param_skew, loc=param_mean, scale=param_std)-skewnorm.cdf(param_vals[j-1],param_skew, loc=param_mean, scale=param_std)
            param_midpoints[j]=(param_vals[j-1]+param_vals[j])/2
        return param_midpoints, param_weights




    def GH_setup(self, nodes):
        """
        We assume here that for n>1 normally dispersed parameters then the order of the integral
        will be the same for both
        return: nodes, normal_weights
        """

        nodes, weights=np.polynomial.hermite.hermgauss(nodes)
        normal_weights=np.multiply(1/math.sqrt(math.pi), weights)

        return nodes, normal_weights
    
if __name__ == "__main__":
    
    dipserison_fun = dispersion_class()

    nodes = 16
    std = math.exp(0.2) #math.log(20)
    mean = 100

    dipserison_fun.normal(nodes,std,mean)

    dipserison_fun.log_normal(nodes,std,mean)