import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from dateutil.parser import parse
from collections import Counter
import copy
import random
from IPython.display import display
from sklearn import preprocessing
import math

# 向上取整
def max_list(x, col):
    list_x = x[col].split(',')
    int_x = [int(i) for i in list_x]
    return math.ceil(np.average(int_x))

def calculate_cmplnum(rcl_data, cmpl, weekly=False, time_interval = 28, days = 7):
    car_cmpl = cmpl.loc[(cmpl["MAKETXT"] == rcl_data[0]) & (cmpl["MODELTXT"] == rcl_data[1]) & (cmpl["YEARTXT"] == rcl_data[2])].reset_index()
    car_rcltime = rcl_data[3]
        
    if len(car_cmpl) == 0:
        return 0
    if not weekly:
        return len(car_cmpl)

def get_cmpl_num(rcl_data, cmpl_data):
	print("一共需要运算次数：", len(rcl_data), "\n")
	cmpl_num_list = []
	for i in range(len(rcl_data)): # len(new_sub_FLAT_RCL)
		if i % 1000 == 0:
			print('---- %i ----'%i)
		list_name = ["MAKETXT", "MODELTXT", "YEARTXT", "ODATE"]
		rcl = list(rcl_data.loc[i,list_name]) # "MAKETXT", "MODELTXT", "YEARTXT", "ODATE"
        # 全部投诉数据
		cmpl_num_i = calculate_cmplnum(rcl, cmpl_data, weekly=False)
		cmpl_num_list.append(cmpl_num_i)
	return cmpl_num_list


def get_cmpl_relate_variables(rcl_data, cmpl_data, time_interval = 28*24, days = 7):
    print("一共需要运算次数：", len(rcl_data), "\n")
    cmpl_data['count'] = 1
    compdesc_frame = pd.DataFrame(columns=['global_id', 'time', 'COMPDESC_SUMMARY', 'count'])
    state_frame = pd.DataFrame(columns=['global_id', 'time', 'STATE_SUMMARY', 'STATE', 'count'])
    CRASH_frame = pd.DataFrame(columns=['global_id', 'time', 'CRASH_sum'])
    INJURED_frame = pd.DataFrame(columns=['global_id', 'time', 'INJURED_sum'])
    DEATH_frame = pd.DataFrame(columns=['global_id', 'time', 'DEATHS_sum'])
    cmpl_num_weekly = pd.DataFrame(columns=['global_id', 'cmpl_num', 'twentyfour_month_weekly_cmpl'])
    CDESCR_frame = pd.DataFrame(columns=['global_id', 'time', 'cdescrs'])
    k=0
    t=0
    for i in range(len(rcl_data)):
        if i%100 ==0:
            print('--- %i ----'%i)
        # 获取单条召回数据
        list_name = ["MAKETXT", "MODELTXT", "YEARTXT", "ODATE", "global_id"]
        rcl = list(rcl_data.loc[i,list_name])
        rcl_id = rcl[4]
        # 获取召回数据对应的客户投诉数据
        car_cmpl = cmpl_data.loc[(cmpl_data["MAKETXT"] == rcl[0]) & (cmpl_data["MODELTXT"] == rcl[1]) & (cmpl_data["YEARTXT"] == rcl[2])].reset_index()

        car_rcltime = rcl[3]
        if len(car_cmpl) == 0:
            return '0'
        else:
            begin_time = car_rcltime - timedelta(days = time_interval)
            end_time = car_rcltime + timedelta(days = time_interval)
            car_cmpl_date = car_cmpl['LDATE']
            # 获取满足时间区间的数据
            satis_index = list(np.where((car_cmpl_date >= begin_time) & (car_cmpl_date <= end_time))[0])
            car_cmpl_inter = car_cmpl.iloc[satis_index,].reset_index(drop=True)
            
            # 计算每几天的投诉数量
            start = copy.deepcopy(begin_time)
    #         days_key = []
            days_value = []
            while start < end_time:
                end = start + timedelta(days = days)
                dic_car_cmpl_inter = dict(Counter((car_cmpl_inter['LDATE'] >= start) & (car_cmpl_inter['LDATE'] < end)))
                if True in dic_car_cmpl_inter:
                    true_val = dic_car_cmpl_inter[True]
                else:
                    true_val = 0         
    #             days_key.append(end)
                days_value.append(true_val)
                start = end
            # 处理为召回前n个月和召回后n个月每周客户投诉数量
            days_value = [str(i) for i in days_value]
            month_cmplnum_weekly = ",".join([','.join(days_value[i:(i+4)]) for i in range(0, len(days_value), 4)])
            cmpl_num_weekly.loc[k] = [rcl_id, len(car_cmpl), month_cmplnum_weekly]
            k += 1
            
            # 召回相关的投诉数据
            start2 = copy.deepcopy(begin_time)
            m = 1
            while start2 < end_time:
                if m<=24:
                    tm = 24 - m + 1
                    str_tm = 'Before%i'%tm
                else:
                    tm = m - 24
                    str_tm = 'After%i'%tm
                end = start2 + timedelta(days = 28)
                month_car_cmpl_inter = car_cmpl_inter[(car_cmpl_inter['LDATE']>=start2) & (car_cmpl_inter['LDATE']<end)]
                
                sub_state_frame = month_car_cmpl_inter[['STATE','STATE_SUMMARY', 'count']].groupby(['STATE_SUMMARY','STATE']).count().reset_index()
                sub_compdesc_frame = month_car_cmpl_inter[['COMPDESC_SUMMARY', 'count']].groupby('COMPDESC_SUMMARY').count().reset_index()
                sub_state_frame['time'] = str_tm
                sub_state_frame['global_id'] = rcl_id
                sub_compdesc_frame['time'] = str_tm
                sub_compdesc_frame['global_id'] = rcl_id
                sub_compdesc_frame = sub_compdesc_frame[['global_id', 'time', 'COMPDESC_SUMMARY', 'count']]
                sub_state_frame = sub_state_frame[['global_id', 'time', 'STATE_SUMMARY', 'STATE', 'count']]
                compdesc_frame = pd.concat([compdesc_frame, sub_compdesc_frame], axis=0)
                state_frame = pd.concat([state_frame, sub_state_frame], axis=0)
                
                # 客户每月投诉内容的集合
                cdescrs = ";".join(list(month_car_cmpl_inter['CDESCR']))
                CDESCR_frame = CDESCR_frame.append(pd.DataFrame({"global_id": rcl_id, "time": str_tm, "cdescrs": cdescrs}, index=range(1)))
                
                if len(month_car_cmpl_inter) ==0:
                    CRASH_frame.loc[t] = [rcl_id, str_tm, 0]
                    INJURED_frame.loc[t] = [rcl_id, str_tm, 0]
                    DEATH_frame.loc[t] = [rcl_id, str_tm, 0]
                else:
                    CRASH_frame.loc[t] = [rcl_id, str_tm, sum(month_car_cmpl_inter['CRASH'])]
                    INJURED_frame.loc[t] = [rcl_id, str_tm, sum(month_car_cmpl_inter['INJURED'])]
                    DEATH_frame.loc[t] = [rcl_id, str_tm, sum(month_car_cmpl_inter['DEATHS'])]
                m += 1
                t += 1
                start2 = end
    CDESCR_frame = CDESCR_frame.reset_index(drop=True)        
    return cmpl_num_weekly, compdesc_frame, state_frame, CRASH_frame, INJURED_frame, DEATH_frame, CDESCR_frame

def extract_data_between_time(data, frame_type, before=[6, 0], after=[0, 6]):
    # 注意：使用召回前后的时间区间还是召回前的，会影响结果
    time_extract = ['Before'+str(i) for i in list(range(before[0], before[1], -1))] + ['After'+str(i) for i in list(range(after[0]+1, after[1]+1, 1))]
    if frame_type == 'cdescrs':
        time_name = 'C'
    else:
        time_name = 'time'
    sub_data = data[data[time_name].isin(time_extract)]
    
    name_dict = {'CRASH':'CRASH_sum', 'INJURED':'INJURED_sum', 'DEATH':'DEATHS_sum'}
    name_dict2 = {'compdesc':'COMPDESC_SUMMARY', 'state':'STATE'}
    if frame_type in name_dict:       
        # CRASH, INJURED, DEATH
        data_for_merge = sub_data[[name_dict[frame_type], 'global_id']].groupby(['global_id']).sum().reset_index()
    elif frame_type == 'state':
        merged_data = sub_data[['global_id', name_dict2[frame_type], 'count']].groupby(['global_id', name_dict2[frame_type],]).sum().reset_index()
        group = merged_data[['global_id', name_dict2[frame_type], 'count']].groupby(['global_id', name_dict2[frame_type]]).sum().reset_index()
        data_for_merge = group[['global_id', 'count']].groupby('global_id').var().reset_index()
    elif frame_type == 'compdesc':
        merged_data = sub_data[['global_id', name_dict2[frame_type], 'count', 'relate_count']].groupby(['global_id', name_dict2[frame_type],]).sum().reset_index()
        group = merged_data[['global_id', name_dict2[frame_type], 'count', 'relate_count']].groupby(['global_id', name_dict2[frame_type]]).sum().reset_index()
        data_for_merge = group[[ 'count', 'relate_count']].groupby(group['global_id']).agg({'count':{'count_sum':sum, 'count_var':'var'}, 'relate_count':{'relate_count_sum':sum}}).reset_index()
    elif frame_type == 'cdescrs':
        sub_data = sub_data[['B', 'Analytic', 'Tone']]
        sub_data.columns = ['global_id', 'Analytic', 'Tone']
        data_for_merge = sub_data.groupby(['global_id']).mean().reset_index()
    return data_for_merge   

def cmpl_num_between_month(data, before=[6, 0]):  # , after=[0, 6]
    twentyfour_month_weekly_cmpl = list(data['twentyfour_month_weekly_cmpl'])
    int_cmpl_num_weekly = [[int(ii) for ii in i.split(',')] for i in twentyfour_month_weekly_cmpl]
    sub_cmpl = [ii[(96-4*before[0]):(96-4*before[1])] for ii in int_cmpl_num_weekly]  # +ii[(96+4*after[0]):(96+4*after[1])]
    sub_cmpl = [sum(i) for i in sub_cmpl]
    return sub_cmpl
    

def cmpl_num_monthly_NM(data):
    cmpl_num_frame = pd.DataFrame(columns=['global_id', 'time', 'cmpl_num_month'])
    m = 0
    for i in range(len(data)): # len(research_recall_with_cmpl)
        data_i = data.loc[i]
        global_id = data_i['global_id']
        twentyfour_month_weekly_cmpl = data_i['twentyfour_month_weekly_cmpl'].split(',')
        int_cmpl_num_weekly = [int(i) for i in twentyfour_month_weekly_cmpl]
        for j in range(48):
            if j<24:
                time = "Before"+str(24-j)
            else:
                time = "After"+str(j-23)
            cmpl_num_monthly = sum(int_cmpl_num_weekly[j*4:(j+1)*4])
            cmpl_num_frame.loc[m] = [global_id, time, cmpl_num_monthly]
            m += 1
    return cmpl_num_frame

def normalize(data, colname, method='max-min'):
    col_value = data[colname]
    if isinstance(data[colname][0], str):
        col_value = [float(i.replace(',', '')) for i in col_value]
    if method=='max-min':
        max_vlaue = np.max(col_value)
        min_vlaue = np.min(col_value)
        if max_vlaue!= min_vlaue:
            norm_value = (col_value - min_vlaue)/(max_vlaue-min_vlaue)
            data.loc[:,[colname]] = norm_value 
    else:
        mean_value = np.mean(col_value)
        std_value = np.std(col_value)
        data.loc[:,[colname]] = (col_value-mean_value)/std_value
        

def generate_y(datas, before_t, after_t, imf):
    imf_col_name = "imfs_B%i_B%i_A%i_A%i"%(before_t[0],before_t[1],after_t[0], after_t[1])
    residue_col_name = "residue_B%i_B%i_A%i_A%i"%(before_t[0],before_t[1],after_t[0], after_t[1])
    imfs = list(datas[imf_col_name])
    len_imfs = [len(imf.split(';')) for imf in imfs]
    datas.loc[:,'len_imfs'] = len_imfs
    data = datas[datas['len_imfs'] >= (imf+1)]
    
    imfs_above3 = list(data[imf_col_name])
    imfs0 = [i.split(';')[imf] for i in imfs_above3]   
    float_imfs0 = [[float(i0) for i0 in i.split(',')] for i in imfs0]

    # imfs1 = [i.split(';')[1] for i in imfs_above3]   
    # float_imfs1 = [[float(i0) for i0 in i.split(',')] for i in imfs1]

    # imfs2 = [i.split(';')[2] for i in imfs_above3]   
    # float_imfs2 = [[float(i0) for i0 in i.split(',')] for i in imfs2]

    # residue = list(data[residue_col_name])
    # residue_list = [i.split(',') for i in residue]
    # float_residue0 = [[float(i1) for i1 in i0 ]for i0 in residue_list]

    # sum_vale = np.array(float_imfs0)+np.array(float_imfs1)+np.array(float_imfs2)+np.array(float_residue0) #+np.array(float_residue1)+np.array(float_residue2)
    # 召回前后imf
    ind = (before_t[1]-before_t[0])*4
    float_imfs0_before = [i[:ind] for i in float_imfs0]
    float_imfs0_after = [i[ind:] for i in float_imfs0]

    # float_sum_value_bef = [sum(i[:ind]) for i in sum_vale]
    # float_sum_value_after = [sum(i[ind:]) for i in sum_vale]
    # 召回后-召回前
    month_float_imfs0_before = [[sum(a[i:i+4]) for i in range(0,len(a),4)] for a in float_imfs0_before]
    month_float_imfs0_after = [[sum(a[i:i+4]) for i in range(0,len(a),4)] for a in float_imfs0_after]

    # 每月平均客户投诉数量召回后-召回前
    # float_imfs0_before_avg = [np.average(i) for i in month_float_imfs0_before]
    # float_imfs0_after_avg = [np.average(i) for i in month_float_imfs0_after]

	# 每周平均客户投诉数量召回后-召回前
    float_imfs0_before_avg = [np.average(i) for i in float_imfs0_before]
    float_imfs0_after_avg = [np.average(i) for i in float_imfs0_after]

    float_diff = [float_imfs0_after_avg[i] - float_imfs0_before_avg[i] for i in range(len(float_imfs0_before))]
    # |召回后-召回前|
    abs_float_diff = [abs(float_imfs0_after_avg[i] - float_imfs0_before_avg[i]) for i in range(len(float_imfs0_before))]
    
    # data.loc[:, 'cmpl_num_after_before_sum'] = np.array(float_sum_value_after) - np.array(float_sum_value_bef)
    # data.loc[:, 'cmpl_num_before_sum'] = 
    data.loc[:, 'cmpl_num_between_N'] = [sum(i) for i in float_imfs0_before]
    data.loc[:, 'average_imf%i_before'%imf] = float_imfs0_before_avg
    data.loc[:, 'average_imf%i_after'%imf] = float_imfs0_after_avg
    data.loc[:, 'diffy_imf%i'%imf] = float_diff
    data.loc[:, 'absdiffy_imf%i'%imf] = abs_float_diff
    return data


def fi_index(imf):
    imf_array = np.array(imf)
    be_imf = imf_array[1:]
    af_imf = imf_array[0:-1]
    fi = np.average(np.abs(be_imf - af_imf))
    return fi

def Vc(imf):
    imf_array = np.array(imf)
    sigma = np.square(np.std(imf_array))
    mean = np.square(np.mean(imf_array))
    vc = sigma/mean
    return vc

def energy_entropy(imfs):
    np_imfs = np.array(imfs)+0.00001
    Ei = np.sum(np.square(np.array(np_imfs)))
    pi = np.square(np_imfs)/Ei
    log_pi = np.log(pi)
    Hi = -np.sum(pi*log_pi)
    return Hi

def generate_ys_imfi(datas, before_t, after_t, imf):
    imf_col_name = "imfs_B%i_B%i_A%i_A%i"%(before_t[0],before_t[1],after_t[0], after_t[1])
    cmpl_col_name = "cmpl_B%i_B%i_A%i_A%i"%(before_t[0],before_t[1],after_t[0], after_t[1])
    imfs = list(datas[imf_col_name])
    len_imfs = [len(imf.split(';')) for imf in imfs]
    datas.loc[:,'len_imfs'] = len_imfs
    data = datas[datas['len_imfs'] >= (imf+1)]

    imfs_above3 = list(data[imf_col_name])
    imfs0 = [i.split(';')[imf] for i in imfs_above3]   
    float_imfs0 = [[float(i0) for i0 in i.split(',')] for i in imfs0]   
    # 召回前后imf
    ind = (before_t[1]-before_t[0])*4
    float_imfs0_before = [i[:ind] for i in float_imfs0]
    float_imfs0_after = [i[ind:] for i in float_imfs0]
    # 计算召回前的客户投诉数量
    float_imfs0_before_sum = [sum(i) for i in float_imfs0_before]
    cmpl_weekly = list(data[cmpl_col_name])
    float_cmpl_weekly = [[float(i0) for i0 in i.split(",")] for i in cmpl_weekly]
    float_cmpl_before_sum = [sum(i[:ind]) for i in float_cmpl_weekly]
    float_cmpl_before_ratio = np.array(float_cmpl_before_sum)/sum(float_cmpl_before_sum)
    #data.loc[:, 'cmpl_num_between_N'] = [float_imfs0_before_sum[i]/float_cmpl_before_sum[i] if float_cmpl_before_sum[i]!=0 else 0 for i in range(len(float_cmpl_before_sum))]
    data.loc[:, 'cmpl_num_between_N'] = float_imfs0_before_sum 
    
    # 波动指数
    imf0_fi_before = [fi_index(i) for i in float_imfs0_before]
    imf0_fi_after = [fi_index(i) for i in float_imfs0_after]
    data.loc[:,'imf%s_findex_before'%str(imf)] = imf0_fi_before
    data.loc[:,'imf%s_findex_after'%str(imf)] = imf0_fi_after
    

    # 变化系数
    imf0_vc_before = [Vc(i) for i in float_imfs0_before]
    imf0_vc_after = [Vc(i) for i in float_imfs0_after]
    data.loc[:,'imf%s_vc_before'%str(imf)] = imf0_vc_before
    data.loc[:,'imf%s_vc_after'%str(imf)] = imf0_vc_after
    # 能量熵
    H_imfs0_before = [energy_entropy(i) for i in float_imfs0_before]
    H_imfs0_after = [energy_entropy(i) for i in float_imfs0_after]  
    data.loc[:,'H_imfs%s_before'%str(imf)] = H_imfs0_before
    data.loc[:,'H_imfs%s_after'%str(imf)] = H_imfs0_after
    # 均值
    Mean_imfs0 = [np.mean(np.array(i)) for i in float_imfs0]   
    data.loc[:,'Mean_imfs%s'%str(imf)] = Mean_imfs0
    # 标准差
    std_imfs0 = [np.std(np.array(i), ddof=1) for i in float_imfs0]   
    data.loc[:,'std_imfs%s'%str(imf)] = std_imfs0
    
    data.loc[:, 'imf%s_findex'%str(imf)] = data['imf%s_findex_after'%str(imf)] - data['imf%s_findex_before'%str(imf)]
    data.loc[:, 'imf%s_vc'%str(imf)] = data['imf%s_vc_after'%str(imf)] - data['imf%s_vc_before'%str(imf)]
    data.loc[:, 'H_imfs%s'%str(imf)] = data['H_imfs%s_after'%str(imf)] - data['H_imfs%s_before'%str(imf)]
    
    residue_col_name = "residue_B%i_B%i_A%i_A%i"%(before_t[0],before_t[1],after_t[0], after_t[1])
    residue = list(data[residue_col_name])
    float_residue = [[float(ii) for ii in i.split(",")] for i in residue]
    # 召回前后residue
    float_residue_before = [i[:ind] for i in float_residue]
    float_residue_after = [i[ind:] for i in float_residue]
    residue_fi_before = [fi_index(i) for i in float_residue_before]
    residue_fi_after = [fi_index(i) for i in float_residue_after]
    data.loc[:,'residue_findex_before'] = residue_fi_before
    data.loc[:,'residue_findex_after'] = residue_fi_after
    
    data.loc[:, "residue_findex"] = data['residue_findex_after'] - data['residue_findex_before']
    
    return data.reset_index(drop=True)
    
    
def get_sale_num(search_data, US_CarSales_byNameplate, after_time, before_time):
    # search_data的MODELTXT换成小写，同理US_CarSales_byNameplate的nameplate
    search_data['ODATE'] = pd.to_datetime(search_data['ODATE'])
    model_search_data = search_data['MODELTXT']
    lower_model_search_data = [models.lower() for models in model_search_data]
    search_data['MODELTXT'] = lower_model_search_data

    nameplat = US_CarSales_byNameplate['namplate']
    lower_nameplat = [models.lower() for models in nameplat]
    US_CarSales_byNameplate['namplate'] = lower_nameplat
    avg_sales = []
    sum_sales = []
    before_recall_sales = []
    ratio_sales = []
    not_in = []
    after_divide_befores = []
    for i in range(len(search_data)):
        odate = search_data.loc[i].ODATE
        # 如果获取的是销量的变化率，必须要是召回前后同一时间区间因此需要选择时间区间
        if after_time[1] < before_time[0]:
            t1 = after_time[1]
        e_t = odate + timedelta(days = 30*t1)
        b_t = odate - timedelta(days = 30*t1)
        subs1_after = US_CarSales_byNameplate[(US_CarSales_byNameplate['namplate'].str.contains(search_data.loc[i].MODELTXT))&(~US_CarSales_byNameplate['namplate'].str.contains('total')) & (US_CarSales_byNameplate['DateStamp']>odate)
                                      & (US_CarSales_byNameplate['DateStamp']<=e_t)]
        subs1_before = US_CarSales_byNameplate[(US_CarSales_byNameplate['namplate'].str.contains(search_data.loc[i].MODELTXT))&(~US_CarSales_byNameplate['namplate'].str.contains('total')) & (US_CarSales_byNameplate['DateStamp']>b_t)
                                      & (US_CarSales_byNameplate['DateStamp']<=odate)]                             
        
        begin_time = odate + timedelta(days = 30*after_time[0])
        end_time = odate + timedelta(days = 30*after_time[1])
        
        begin_time_before = odate - timedelta(days = 30*before_time[0])
        end_time_before = odate - timedelta(days = 30*before_time[1])        
        
        subs1 = US_CarSales_byNameplate[(US_CarSales_byNameplate['namplate'].str.contains(search_data.loc[i].MODELTXT))&(~US_CarSales_byNameplate['namplate'].str.contains('total')) & (US_CarSales_byNameplate['DateStamp']>begin_time)
                                      & (US_CarSales_byNameplate['DateStamp']<=end_time)]
                                      
        interval_sumsales = US_CarSales_byNameplate[(US_CarSales_byNameplate['DateStamp']>begin_time)& (US_CarSales_byNameplate['DateStamp']<=end_time)]

        not_i = US_CarSales_byNameplate[(US_CarSales_byNameplate['namplate'].str.contains(search_data.loc[i].MODELTXT))]
        if len(not_i) == 0:
            not_in.append(search_data.loc[i].MODELTXT)
        
        if len(subs1_before) == 0:
            avg_sale = 0
            sum_sale = 0
            ratio_sale = 0
            after_divide_before = 0
            before_recall_sale_value = 0
        else:
            if int(np.sum(subs1_before['Sales'])) == 0:
                after_divide_before = 0
            else:
                after_divide_before = (int(np.sum(subs1_after['Sales']))/int(np.sum(subs1_before['Sales']))) - 1
            
            avg_sale = int(np.sum(subs1['Sales'])/(after_time[1]-after_time[0]))
            sum_sale = int(np.sum(subs1['Sales']))
            interval_sum = int(np.sum(interval_sumsales['Sales']))
            before_recall_sale_value = int(np.sum(subs1_before['Sales']))
            if len(interval_sumsales) == 0:
                ratio_sale = 0
            else:
                ratio_sale = round(sum_sale/interval_sum, 4)
        before_recall_sales.append(before_recall_sale_value)
        avg_sales.append(avg_sale)
        sum_sales.append(sum_sale)
        ratio_sales.append(ratio_sale)
        after_divide_befores.append(after_divide_before)

    search_data['after_divide_befores'] = after_divide_befores
    search_data['avg_sales'] = avg_sales
    search_data['sum_sales'] = sum_sales
    search_data['ratio_sales'] = ratio_sales
    search_data['before_sales'] = before_recall_sales
    print(set(not_in))
    return search_data
    
def findcompy(x, make_mfr):
    maketxt = list(set(x['MAKETXT']))[0]
    yeartxt = list(set(x['YEARTXT']))[0]
    sub_mfr = make_mfr[(make_mfr['MAKETXT'] == maketxt) & (make_mfr['YEAR']== yeartxt)]
    int_year_1 = yeartxt
    while(len(sub_mfr)==0 and int_year_1<=2019):     
        int_year_1 += 1
        sub_mfr = make_mfr[(make_mfr['MAKETXT'].str.contains(maketxt)) & (make_mfr['YEAR']== int_year_1)]
    int_year_2 = yeartxt
    while(len(sub_mfr)==0 and int_year_2>=2009):   
        int_year_2 -= 1
        sub_mfr = make_mfr[(make_mfr['MAKETXT'].str.contains(maketxt)) & (make_mfr['YEAR']== int_year_2)]
    sub_mfr = sub_mfr[['total asset(million)', 'equity(million)', 'profit(million)', 'revenue(million)']]
    sub_mfr['MAKETXT'] = maketxt
    sub_mfr['YEARTXT'] = yeartxt
    return sub_mfr    
    
    
    
    