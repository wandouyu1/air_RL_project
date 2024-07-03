# air_project_RL
 一个烂尾外包，用强化学习做室内温度和能耗最优控制

import psychrolib


### 命名规范
状态输入：  
室外干球温度 temperature_outdoor_air_now
室外湿球温度 wet_bulb_temperature 
室外相对湿度 relative_humidity_outdoor_air_now
下一时刻预测负荷 cooling_load_fact

房间当前干球温度 temperature_indoor_air_now
房间当前相对湿度 relative_humidity_indoor_air_now

动作输入：
主机出水温度 chilled_water_supply_temperature
水泵频率 pump_frequency

环境输出：
下一时刻房间干球温度 room_air_temperature_change
下一时刻房间相对湿度 relative_humidity_indoor_air_change

奖励函数：
最小能耗，超出25.5-26.5度范围，-200







