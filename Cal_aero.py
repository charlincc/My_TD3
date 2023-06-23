import numpy as np
import os
import re
from func_timeout import func_set_timeout

@func_set_timeout(10) #设置定时器，超过5秒就报错
def cal_func(free_relative_path,index_loc,input_data,input_relative_path,exe_relative_path):#相对路径,对应参数所在索引位置，需要填入的参数,运行程序的相对路径

    file_pointer = open(free_relative_path,'r',encoding='gbk')
    para = file_pointer.readlines()
    file_pointer.close()#读取相应的赋值文件

    j = 0 #输入参数的索引值
    for i in index_loc:
        para[i] = str(input_data[j]) + para[i]
        j = j + 1

    file_pointer = open(input_relative_path,'w')
    file_pointer.writelines(para)
    file_pointer.close()#将改变过后的input更新

    os.system(exe_relative_path)#有待改进


    file_pointer = open('waverider/result.dat','r',encoding='gbk')
    aero_result = file_pointer.readlines()
    file_pointer.close()#读取结果
    result_value  = np.array([])#存结果的数组

    total_pressure_coe = aero_result[0] #升力系数
    total_pressure_coe = re.findall(r"\d+\.?\d*", total_pressure_coe)#提取系数

    if len(total_pressure_coe) == 0:
        return 0

    else:
        return total_pressure_coe



