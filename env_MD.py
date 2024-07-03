#强化学习标准环境
#多重离散空间版本
import numpy as np
import gymnasium as gym

import psychrolib
import pandas as pd
from stable_baselines3.common.env_checker import check_env

#自定义环境

# 命名规范
#############################################################
# 状态输入：  
# 室外干球温度 temperature_outdoor_air_now
# 室外湿球温度 wet_bulb_temperature 
# 室外相对湿度 relative_humidity_outdoor_air_now
# 下一时刻预测负荷 cooling_load_fact

# 房间当前干球温度 temperature_indoor_air_now
# 房间当前相对湿度 relative_humidity_indoor_air_now

# 动作输入：
# 主机出水温度 chilled_water_supply_temperature
# 水泵频率 pump_frequency

# 环境输出：
# 下一时刻房间干球温度 room_air_temperature_change
# 下一时刻房间相对湿度 relative_humidity_indoor_air_change
#############################################################

class air_env(gym.Env):
    def __init__(self):
        super(air_env, self).__init__()
        #六个状态
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        #多重离散空间
        self.action_space = gym.spaces.MultiDiscrete([5,5])
        
        #训练数据
        self.data_train = pd.read_excel('环境对应数据.xlsx',sheet_name='Sheet2')
        self.state = np.zeros(8,dtype=np.float32)
        #建一个字典来表示动作序号和值的对应关系
        self.action_dict = {0: -2, 
                            1: -1, 
                            2:  0, 
                            3:  1, 
                            4:  2}
        #计数器
        self.count = 0
        self.count2 = 0
        self.power = 0
        #初始化
        self.reset()
        
        
    
    def step(self, action):
        #根据动作计算下一时刻的状态，以及这段时间的奖励
        #检查动作是否合格
        assert self.action_space.contains(action)
        #动作变量序号0-4,0表示-2，1表示-1，以此类推
        temperature_index = action[0]
        humidity_index = action[1]
        
        #为了方便容易看
        temp1 = self.state[6] + self.action_dict[temperature_index]
        temp2 = self.state[7] + self.action_dict[humidity_index]
        #限制温度和频率范围,注意这里还是当前时刻
        if temp1>6 and temp1<15:
            self.state[6] += self.action_dict[temperature_index]
        if temp2>35 and temp2<50:
            self.state[7] += self.action_dict[humidity_index]
        
        #计算执行该动作产生的能耗，下一时刻的室内温度，湿度
        self.P,T,H = self.calculate(temperature_outdoor_air_now=self.state[0],
                                        wet_bulb_temperature=self.state[1],
                                        relative_humidity_outdoor_air_now=self.state[2],
                                        cooling_load_fact=self.state[3],
                                        temperature_indoor_air_now=self.state[4],
                                        relative_humidity_indoor_air_now=self.state[5],
                                        chilled_water_supply_temperature=self.state[6],
                                        pump_frequency=self.state[7])

        self.power += self.P
        self.count2 += 1
        #得到下一时刻状态
        self.state[0] = self.batch_data['室外干球温度'].values[self.count2]
        self.state[1] = self.batch_data['室外湿球温度'].values[self.count2]
        self.state[2] = self.batch_data['室外相对湿度'].values[self.count2] 
        self.state[3] = self.batch_data['预测负荷'].values[self.count2]
        self.state[4] = T
        self.state[5] = H
        #检查状态
        assert self.observation_space.contains(self.state)
        #到一天的最后一个时间点状态，即结束
        if self.count2 >= 40:
            terminated = True
        else:
            terminated = False
        #计算奖励
        reward = self.reward_function2()
        #其他信息
        info = {}
        return self.state, reward, terminated, False, info
        
        
        
        
    def reset(self, seed=None, options=None):
        
        super().reset(seed=seed)
        #初始化状态
        #一天有41个时间点数据
        
        #随机选择一天的数据
        #随机从0到52随机一个整数
        # self.count = np.random.randint(0,52)
        self.count = 1
        #随机选择一天的数据
        i = 41*self.count
        self.batch_data = self.data_train.iloc[i:i + 41]
        
        self.state = np.array([self.batch_data['室外干球温度'].values[0], 
                               self.batch_data['室外湿球温度'].values[0], 
                               self.batch_data['室外相对湿度'].values[0], 
                               self.batch_data['预测负荷'].values[0], 
                               26.0,                                         #室内温度初始值为当天8：30的温度
                               60.0,                                         #室内湿度初始值为当天8：30的湿度
                               10.0,                                         #初始出水温度为10
                               43.0],                                        #初始水泵频率为43
                              dtype=np.float32)                                          
                                  
        #检查状态是否合规
        assert self.observation_space.contains(self.state)
        self.power = 0
        self.count2 = 0
        info = {}
        return self.state, info
    
    
    def calculate(self,
                  temperature_outdoor_air_now,
                  wet_bulb_temperature,
                  relative_humidity_outdoor_air_now,
                  cooling_load_fact,
                  temperature_indoor_air_now,
                  relative_humidity_indoor_air_now,
                  chilled_water_supply_temperature,
                  pump_frequency):
        """输入当前状态s和动作a，计算能耗，下一时刻室内数据

        Args:
            temperature_outdoor_air_now: 室外干球温度 
            wet_bulb_temperature: 室外湿球温度
            relative_humidity_outdoor_air_now: 室外相对湿度 
            cooling_load_fact: 预测负荷 
            temperature_indoor_air_now: 房间当前干球温度 
            relative_humidity_indoor_air_now: 房间当前相对湿度
            chilled_water_supply_temperature: 主机出水温度
            pump_frequency: 水泵频率

        Returns:
            all_system_energy_consumption: 能耗
            room_air_temperature_change: 下一时刻房间干球温度
            relative_humidity_indoor_air_change: 下一时刻房间相对湿度
        """
        # 室内的环境
        
        psychrolib.SetUnitSystem(psychrolib.SI)  # 使用国际单位制（SI）

        def calculate_specific_enthalpy(temperature, relative_humidity):
            # 获取空气的压力（假设为标准大气压 101325 Pa）
            pressure = 101325  # Pa
            # 计算空气的湿度比（kg 水蒸气/kg 干空气）
            humidity_ratio = psychrolib.GetHumRatioFromRelHum(temperature, relative_humidity, pressure)
            # 计算空气的焓值（kJ/kg）
            enthalpy = psychrolib.GetMoistAirEnthalpy(temperature, humidity_ratio)
            return enthalpy

        def calculate_humidity_ratio(temperature, relative_humidity):
            pressure = 101325  # Pa
            humidity_ratio = psychrolib.GetHumRatioFromRelHum(temperature, relative_humidity, pressure)
            return humidity_ratio

        def calculate_relative_humidity(temperature, humidity_ratio):
            pressure = 101325  # Pa
            relative_humidity = psychrolib.GetRelHumFromHumRatio(temperature, humidity_ratio, pressure)
            return relative_humidity

        # 环境参数设置
        
        
        distribution_losses = 0.5

        # 冷冻水供水温度设置
        
        chilled_water_flow_rate = 39.09675552-2.59387582*pump_frequency+0.06112368*pump_frequency**2-0.00046424 * pump_frequency**3
        pump_P_fact = 315 + ( 730 - 315 )* ( pump_frequency - 35 ) / 15

        # 时间设置
        set_time = 900
        all_time_ratio = set_time / 3600  

        # 输入空气的温度（摄氏度）和相对湿度（百分比）
        

        
        

        temperature_fresh_air = chilled_water_supply_temperature + 8
        relative_humidity_fresh_air = 90

        temperature_return_air = chilled_water_supply_temperature + 8
        relative_humidity_return_air = 90

        # 计算焓值
        enthalpy_indoor_air_now = calculate_specific_enthalpy(temperature_indoor_air_now, relative_humidity_indoor_air_now / 100) / 1000 # 室内当前空气焓值
        enthalpy_outdoor_air_now = calculate_specific_enthalpy(temperature_outdoor_air_now, relative_humidity_outdoor_air_now / 100) / 1000 # 室外当前空气焓值
        enthalpy_fresh_air = calculate_specific_enthalpy(temperature_fresh_air, relative_humidity_fresh_air / 100) / 1000   # 新风送风焓值
        enthalpy_return_air = calculate_specific_enthalpy(temperature_return_air, relative_humidity_return_air / 100) / 1000  # 风盘送风焓值

        # print(f"室内空气的焓值为: {enthalpy_indoor_air_now:.2f} kJ/kg")
        # print(f"室外空气的焓值为: {enthalpy_outdoor_air_now:.2f} kJ/kg")
        # print(f"新风送风空气的焓值为: {enthalpy_fresh_air:.2f} kJ/kg")
        # print(f"回风送风空气的焓值为: {enthalpy_return_air:.2f} kJ/kg")

        # 回水温度计算
        # 空气容积
        room_volume = 2000
        fresh_air_volume_rated = 300  # 新风机额定送风量
        return_air_volume_rated = 1000  # 回风机额定送风量
        fresh_air_volume = fresh_air_volume_rated * all_time_ratio  # 新风机送风量
        return_air_volume = return_air_volume_rated * all_time_ratio  # 回风机送风量

        # 空气质量
        air_density = 1.2
        air_specific_heat_capacity = 1.005
        fresh_air_quality = fresh_air_volume * air_density  # 新风机送风质量
        return_air_quality = return_air_volume * air_density  # 回风机送风质量

        # 计算湿度比
        humidity_ratio_indoor_air_now = calculate_humidity_ratio(temperature_indoor_air_now, relative_humidity_indoor_air_now / 100)
        humidity_ratio_fresh_air = calculate_humidity_ratio(temperature_fresh_air, relative_humidity_fresh_air / 100)
        humidity_ratio_return_air = calculate_humidity_ratio(temperature_return_air, relative_humidity_return_air / 100)

        # 计算混合空气的湿度比
        mixed_humidity_ratio = (fresh_air_quality * humidity_ratio_fresh_air + return_air_quality * humidity_ratio_return_air) / (fresh_air_quality + return_air_quality)

        # 转换为相对湿度
        relative_humidity_indoor_air_change = calculate_relative_humidity(temperature_indoor_air_now, mixed_humidity_ratio) * 100
        # print(f"混合后的室内相对湿度: {relative_humidity_indoor_air_change:.2f} %")

        # 空气焓差
        # 空气焓差
        if enthalpy_outdoor_air_now <= enthalpy_indoor_air_now:
            Q_fresh_air = 0
        else:
            Q_fresh_air = fresh_air_quality * (enthalpy_outdoor_air_now - enthalpy_fresh_air) / 3600 # 新风达到送风条件需要降低的焓值 # 新风达到送风条件需要降低的焓值

        Q_fresh_air_room = fresh_air_quality * ( enthalpy_indoor_air_now - enthalpy_fresh_air ) / 3600 # 算室内新风对室内温度降低贡献时用
        Q_return_air = return_air_quality * ( enthalpy_indoor_air_now - enthalpy_return_air ) / 3600 # 回风达到送风条件需要降低的焓值

        # 计算回水温度
        chilled_water_return_temperature = chilled_water_supply_temperature +(Q_fresh_air + Q_return_air)/(4.19 * chilled_water_flow_rate  * all_time_ratio) + distribution_losses
        chilled_water_temperature_change = ( chilled_water_return_temperature - chilled_water_supply_temperature )

        # print(f"新风需要的焓值为: {Q_fresh_air:.2f} kWh")
        # print(f"回风需要的焓值为: {Q_return_air:.2f} kWh")
        # print(f"冷冻水回水温度: {chilled_water_return_temperature:.2f} ℃")
        # print(f"冷冻水供回水温差: {chilled_water_temperature_change:.2f} ℃")

        # 室内温度计算
        net_air_heat_change =cooling_load_fact * all_time_ratio - ( Q_fresh_air_room + Q_return_air )
        room_air_temperature_change = temperature_indoor_air_now + 3600 * net_air_heat_change / ( (room_volume + fresh_air_volume + return_air_volume ) * air_density * 1.005 )

        # print(f"变化后的室内温度: {room_air_temperature_change:.2f} ℃")

        # 空调系统能耗计算
        chiller_Pref = 3500
        cooled_water_supply_temperature = wet_bulb_temperature + 3
        chiller_plr = 4.19 * chilled_water_flow_rate * chilled_water_temperature_change / 20

        chillercapftemp = 0.257896+0.0389016*chilled_water_supply_temperature-0.000217*chilled_water_supply_temperature**2+0.0468684*cooled_water_supply_temperature-0.000943*cooled_water_supply_temperature**2-0.000343*chilled_water_supply_temperature*cooled_water_supply_temperature
        chillerEIRFTemp = 0.933884-0.058212*chilled_water_supply_temperature+0.00450036*chilled_water_supply_temperature**2+0.00243*cooled_water_supply_temperature+0.000486*cooled_water_supply_temperature**2-0.001215*chilled_water_supply_temperature*cooled_water_supply_temperature
        chillerEIRFPLR = 0.222903+0.313387*chiller_plr+0.463710*chiller_plr**2
        chiller_P_fact = chiller_Pref * chillercapftemp * chillerEIRFTemp * chillerEIRFPLR
        cooled_water_return_temperature = cooled_water_supply_temperature + chilled_water_temperature_change + 3.6 * chiller_P_fact * all_time_ratio / ( 4.19 * chilled_water_flow_rate * 1000 )
        all_system_energy_consumption = ( chiller_P_fact + pump_P_fact ) * all_time_ratio / 1000

        # print(f"PLR: {chiller_plr * 100:.2f} %")
        # print(f"空调功率: {chiller_P_fact:.2f} W")
        # print(f"冷却水回水温度: {cooled_water_return_temperature:.2f} ℃")
        # print(f"冷冻水流量: {chilled_water_flow_rate:.2f} m³/h")
        # print(f"水泵功率: {pump_P_fact:.2f} W")
        # print(f"总能耗: {all_system_energy_consumption:.2f} kWh")
        
        return all_system_energy_consumption, room_air_temperature_change,relative_humidity_indoor_air_change
    
    def reward_function1(self):
        # 温度25.5℃~26.5℃之间，给一个恒定奖励
        # 在范围外给一个线性奖励会训练得比较好
        #约束如下设计能保证在范围内奖励为1，在范围外为0以下
        a1 = 0.3
        if self.state[4] > 25.5:
            r1 = a1 * (26.5-self.state[4])
        else:
            r1 = 0
        if self.state[4] < 26.5:
            r2 = a1 * (self.state[4]-25.5)
        else:
            r2 = 0        
        #另外关注重点是能耗，能耗越多意味着越差，也可以给几个超参数，自行修改
        r3 = -self.P
        reward = r1 + r2 + r3
        #返回全部奖励
        return reward
     
    def reward_function2(self):
        # 温度25.5℃~26.5℃之间，给一个恒定奖励
        # 在范围外给一个线性奖励会训练得比较好
        #约束如下设计能保证在范围内奖励为1，在范围外为0以下
        #智能体短期注意约束，长期注意能耗
        a1 = 0.3
        if self.state[4] > 25.5:
            r1 = a1 * (26.5-self.state[4])
        else:
            r1 = 0
        if self.state[4] < 26.5:
            r2 = a1 * (self.state[4]-25.5)
        else:
            r2 = 0        

        if self.count2 >=40:
            r3 = -self.power
        else:
            r3 = 0
        reward = r1 + r2 + r3
        #返回全部奖励
        return reward
    
    def reward_function3(self):
        #简单奖励
        a1 = 0.3
        if self.state[4] > 26.5 or self.state[4] < 25.5:
            r1 = -1
        else:
            r1 = 1      
        #另外关注重点是能耗，能耗越多意味着越差，也可以给几个超参数，自行修改
        if self.count2 >=40:
            r3 = -self.power
        else:
            r3 = 0
        reward = r1 + r3
        #返回全部奖励
        return reward




if __name__ == "__main__":
    # 测试代码
    env = air_env()
    check_env(env)
    # for i in range(50):
    #     env.reset()
    #     totol_reward = 0
    #     for j in range(40):
    #         time_step,reward,done,_,info = env.step([2,2])
    #         totol_reward += reward
    #         #print(reward)
    #     print(totol_reward)
    
    

