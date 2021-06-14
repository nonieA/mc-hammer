import numpy as np
import math

def hypothesis_test(q_list,q_method):
    q_no_miss = [i for i in q_list if isinstance(i,str) == False]
    if (len(q_no_miss) == 0) or (math.isnan(q_list[-1])):
        return 0.99
    else:
        if q_method == 'DB':
            q_list = [max(q_no_miss) if i == 'one_cluster' else i for i in q_list]
        else:
            q_list = [max(q_no_miss) if i == 'one_cluster' else i for i in q_list]
        x_val = q_list[-1]
        q_arr = np.sort(np.array(q_list))
        p_val = (np.where(q_arr == x_val)[0][0] + 1)/len(q_list)
        if q_method in ['DB','CVNN','SD_score','S_Dbw']:
            if q_list[-1] == max(q_list):
                return 0.99
            else:
                return p_val

        else:
            if q_list[-1] == min(q_list):
                return 0.99
            else:
                return 1-p_val