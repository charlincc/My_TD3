import numpy as np
import pandas as pd
import func_timeout
from Cal_aero import cal_func
import os
from func_timeout import func_set_timeout


data = pd.read_csv('data/OLHS_input')

data_input = np.array(data)
data_input = data_input[:,1:]

data_output = np.array([])
data_input_o = np.array([])

free_relative_path = 'waverider/input_pass.dat'

input_relative_path = 'waverider/input.dat'

exe_relative_path = 'cd waverider && Waverider.exe' #cd转换当前工作路径，&&执行命令之后再执行

index_loc = np.array([14,15,16,17,18,19,20])

cnt = 0

for i in data_input:

    cnt = cnt + 1
    print('第',cnt,'次采集数据')
    try:
        total_pressure = cal_func(free_relative_path,index_loc,i,input_relative_path,exe_relative_path)#正常计算返回总压恢复系数

        if total_pressure == 0:
            continue
        else:

            data_output = np.append(data_output,float(total_pressure[0]) )#全部输出加入数组里面

            data_input_o = np.append(data_input_o,i)#全部输入加入数组里面


    except func_timeout.exceptions.FunctionTimedOut:

        os.system(r'taskkill /F /T /IM Waverider.exe')#杀掉程序重新执行

        continue

output_coln_name = np.array(['Lift_drag_coe'])#命名，总压恢复系数
input_coln_name = np.array(["beta_cu", "beta_cd", "beta_bm", "ucoey", "ucoez", "dcoey", "dcoez"])#命名设计参数

data_output = np.reshape(data_output,(-1,1))#转换为列向量
data_input_o = np.reshape(data_input_o,(-1,7))#转换

data_output = pd.DataFrame(data = data_output,columns = output_coln_name)
data_input_o = pd.DataFrame(data = data_input_o,columns= input_coln_name)

data_output.to_csv('data/OLHS_output')
data_input_o.to_csv('data/OLHS_input')