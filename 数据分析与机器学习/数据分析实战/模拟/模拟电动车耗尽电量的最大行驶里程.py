import numpy as np
import simpy

class RechargeStation(object):
    '''
        定义充电站的类
    '''
    def __init__(self, env, loc):
        '''
            主函数

            @param env -- 环境 变量
            @param loc -- 充电站的位置
        '''
        # 定义一个指向环境的指针
        self.env = env
        self.LOCATION = loc

        # 设置充电速度
        self.RECHARGE_SPEED = 0.01 

    @staticmethod
    def generateRechargeStations(simTime):
        '''
            沿途随机放置充电站，距离范围 80 ~ 140英里

            @param simTime -- 模拟时间
        '''
        # 假设速度是匀速每小时35英里,计算模拟时间内最大的行驶距离
        maxDistance = simTime / 60 * 35 * 2

        # 创建充电站
        distCovered = 0
        rechargeStations = [RechargeStation(env, 0)]

        while(distCovered < maxDistance):
            nextStation = np.random.randint(80, 140)

            distCovered += nextStation
            rechargeStations.append(RechargeStation(env, distCovered))
        
        return rechargeStations


class Car(object):
    '''
        定义'车'的类
    '''
    def __init__(self, env, driver, rechargeStations):
        '''
            类的主函数

            @param env              -- 环境对象
            @param driver           -- 司机ID
            @param rechargeStations -- 充电站列表
        '''
        # 指向环境和充电站对象的指针
        self.env = env
        self.rechargeStations = rechargeStations
        self.driver = driver

        # 车的设定
        self.BATTERY_CAPACITY = np.random.choice([70, 85], p=[0.5, 0.5])   # 两种规格的电池可选，比重相同
        self.BATTERY_LEVEL = np.random.randint(80, 100) / 100    # 电池起始电量在 80% ~100% 
        self.AVG_SPEED = np.random.randint(364, 492) / 10  # 平均速度的范围（每小时36.4-49.2英里）

        # 原耗电情况每百英里34-38千瓦时（每英里耗电量）
        self.AVG_ECONOMY = np.random.randint(34, 38) / 100 

        # 初始位置为0
        self.LOCATION = 0 

        # 模拟从driving()内的过程开始
        self.action = self.env.process(self.driving())

    def driving(self):
        # 每15分钟更新
        interval = 15

        # 假设匀速，15分钟内行驶的路程
        distanceTraveled = self.AVG_SPEED / 60 * interval

        # 更新间隔的耗电
        batteryUsed = distanceTraveled * self.AVG_ECONOMY

        while True: 
            # 更新位置
            self.LOCATION += distanceTraveled

            # 剩余电量
            batteryLeft = self.BATTERY_LEVEL * self.BATTERY_CAPACITY - batteryUsed
            
            # 更新剩余电量占比
            self.BATTERY_LEVEL = batteryLeft / self.BATTERY_CAPACITY
            
            # 电量耗尽则停止
            if self.BATTERY_LEVEL <= 0.0:
                print()
                print('!~' * 15)
                print('RUN OUT OF JUICE...')
                print('!~' * 15)
                print()
                break

            # 检查最近两个充电站的距离
            nearestRechargeStations = [gs for gs in self.rechargeStations if gs.LOCATION > self.LOCATION][0:2]

            distanceToNearest = [rs.LOCATION - self.LOCATION for rs in nearestRechargeStations]

            # 判断是否正经过充电站（布尔变量）
            passingRechargeStation = self.LOCATION + distanceTraveled > nearestRechargeStations[0].LOCATION

            # 剩余电量能否支撑到下一个充电站
            willGetToNextOne = self.check(batteryLeft, nearestRechargeStations[-1].LOCATION)

            # 如果正经过充电站且余电不足以到达下个充电站
            if passingRechargeStation and not willGetToNextOne:
                '''注意try, except的使用。若不中断，try部分代码全部执行'''

                # 司机可以中断未充满的充电过程
                try:   
                    ### 先确定充满用时并开始充电，同时使用drive()方法调用Driver对象，开始决定是否中断充电过程
                    # 充满电所需用时
                    timeToFullRecharge = (1 - self.BATTERY_LEVEL) / nearestRechargeStations[0].RECHARGE_SPEED

                    # 开始充电
                    charging = self.env.process(self.charging(timeToFullRecharge, nearestRechargeStations[0].RECHARGE_SPEED))

                    # 决定是否提前中断充电过程
                    yield self.env.process(self.driver.drive(self, timeToFullRecharge))

                # 若中断充电过程
                except simpy.Interrupt:

                    print('Charging interrupted at {0}' .format(int(self.env.now)))
                    print('-' * 30)

                    charging.interrupt()

            # 更新车的状态
            toPrint = '{time}\t{level:2.2f}\t{loc:4.1f}'
            print(toPrint.format(time=int(self.env.now), level=self.BATTERY_LEVEL, loc=self.LOCATION))
            
            # 等待下次更新
            yield self.env.timeout(interval)


    def check(self, batteryLeft, nextRechargeStation):
        '''
            检查是否可以到达下个充电站
        '''
        distanceToNext = nextRechargeStation - self.LOCATION
        batteryToNext = distanceToNext / self.AVG_ECONOMY

        return batteryLeft > batteryToNext

    def charging(self, timeToFullRecharge, rechargeSpeed):
        '''
            给车充电
        '''
        # 开始充电
        try:
            # 分割以标出显示
            print('-' * 30)
            print('Charging at {0}'.format(self.env.now))

            # 每秒增量式更新电量，以应对中断
            for _ in range(int(timeToFullRecharge)):
                self.BATTERY_LEVEL += rechargeSpeed
                yield self.env.timeout(1)

            # 如果充电过程没有被提前中断
            print('Fully charged...')
            print('-' * 30)

        # interrupt异常会由driving()方法传到charging()，然后无需任何处理
        except simpy.Interrupt:
            pass

class Driver(object):
    '''
        定义'司机'的类
    '''
    def __init__(self, env):
        '''
            类的主函数
        '''
        self.env = env

    def drive(self, car, timeToFullRecharge):
        '''
            @param car -- 指向车的指针
            @timeToFullRecharge -- 充满电所需时间
        '''
        # 司机有权中断未充满的充电过程
        interruptTime = np.random.randint(50, 120)

        # 如果中断时间大于充满时间，就等到充满；若中断时间小于充满时间，则提前离去
        yield self.env.timeout(int(np.min([interruptTime, timeToFullRecharge])))

        if interruptTime < timeToFullRecharge:
            car.action.interrupt()   # interrupt()可以在Simpy中断进程


if __name__ == '__main__':
    # 模拟时间
    SIM_TIME = 10 * 60 * 60    # 10 hours

    # 创建环境（环境是模拟的基础，环境封装了时间，并处理模拟中代理人之间的互动）
    env = simpy.Environment()

    # 创建充电站(代理人)
    rechargeStations = RechargeStation.generateRechargeStations(SIM_TIME)

    # 创建司机和车
    driver = Driver(env)
    car = Car(env, driver, rechargeStations)

    # 打印表头
    print()
    print('-' * 30)
    print('Time\tBatt.\tDist.')
    print('-' * 30)

    # 开始模拟
    env.run(until = SIM_TIME)