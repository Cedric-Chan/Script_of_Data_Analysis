'''
使用代理人基模型（代理人基模型及其限制http://ijcsi.org/papers/IJCSI-9-1-3-115-119.pdf）
收集数据是费时费力的过程，收集过来的数据容易导致短视
当不可能观察每个局部，或者想在多种情况下测试模型时，或者要验证假设时，模拟就很有用了
代理人基最主要的局限：代理人之间互动的局限；错失一个重要行为可能导致错误的结论
'''
import numpy as np
import simpy
import itertools
import collections as col
'''
加油站的吞吐量由贮油器的数目限定，simpy.Resource()对象决定着系统资源的吞吐，因此本加油站模拟中将贮油器设为Resource对象
所有贮油器都连接着大油箱，使用.Container()对这种情形建模
来车时，使用.Resource()的request()访问一个贮油器；若所有贮油器都被占用，则需等待

'''
class FuelPump(object):
    '''
        定义贮油器的类
    '''
    def __init__(self, env, count):
        '''
            类的主构造函数
            @param env   -- 环境对象
            @param count -- 特定资源（油）有多少个吞吐口（贮油器）  
            （加油站的吞吐量由贮油器的数目限定）   
        '''
        # 限制资源的吞吐口
        self.resource = simpy.Resource(env, count)

    def request(self):
        '''
            有车时创建对可用贮油器的的请求
            （使用.Resource()的request()访问一个贮油器）
        '''
        return self.resource.request()

class GasStation(object):
    '''
        加油站的类 
    '''
    def __init__(self, env):    # 类都有__init__(self, ……)方法，self参数是指向实例对象自身的引用
        '''
            应该在此列出所有实例对象都会拥有的内部属性
        '''
        # 定义一个指向环境的指针
        self.env = env

        # 油箱（加仑）和贮油器
        self.CAPACITY = {'PETROL': 8000, 'DIESEL': 3000}   # 容量
        self.RESERVOIRS = {}       # RESERVOIRS属性有两种.Container()，对应这两种燃料。.Container()对象就是许多处理过程都会用到的资源，直到模拟结束或者.Container()中用光
        self.generateReservoirs(self.env, self.CAPACITY)

        # 每种燃料的贮油器数
        self.PUMPS_COUNT = {'PETROL': 3, 'DIESEL': 1}
        self.PUMPS = {}
        self.generatePumps(self.env, self.CAPACITY, self.PUMPS_COUNT)

        # 抽出的速度
        self.SPEED_REFUEL = 0.3    # 每秒0.3加仑

        # 补充前的最小油量
        self.MINIMUM_FUEL = {'PETROL': 300, 'DIESEL': 100}

        # 补充耗时      
        self.TRUCK_TIME = 200   # 油车耗时200s
        self.SPEED_REPLENISH = 5   # 重装油每秒5加仑

        # 记账
        self.sellPrice = {'PETROL': 2.45, 'DIESEL': 2.23}  # 售出价格
        self.buyPrice  = {'PETROL': 1.95, 'DIESEL': 1.67}  # 购入价格
        self.cashIn = 0  # 初始收入为0
        self.cashOut = np.sum([ self.CAPACITY[ft] * self.buyPrice[ft] for ft in self.CAPACITY])  # 支出累加
        self.cashLost = 0  # 初始流失利润为0（等待时限）

        # 控制可供销售的油量
        self.control = self.env.process(self.controlLevels())

        print('Gas station generated...')

    def generatePumps(self, env, fuelTypes, noOfPumps):
        '''
            贮油器逻辑的实现
        '''
        for fuelType in fuelTypes:
                self.PUMPS[fuelType] = FuelPump(env, noOfPumps[fuelType])

    def generateReservoirs(self, env, levels):
        '''
            大油箱逻辑的实现
            （所有贮油器都连接着大油箱）
        '''
        for fuel in levels:
            self.RESERVOIRS[fuel] = simpy.Container(env, levels[fuel], init=levels[fuel])  # .Container()以指向Environment()对象的指针作为一参，贮油器容量作为二参，初始量作为三参

    def replenish(self, fuelType):
        '''
            重装补满的逻辑
        '''
        # 显示分割以区分补油阶段
        print('-' * 70)
        print('CALLING TRUCK AT {0:4.0f}s.'.format(self.env.now))
        print('-' * 70)

        # 等油车到来
        yield self.env.timeout(self.TRUCK_TIME)   # 先挂起过程等油车到来
        print('-' * 70)
        print('TRUCK ARRIVING AT {0:4.0f}s'.format(self.env.now))   # 油车到达时间

        # 过程恢复，要补多少油
        toReplenish = self.RESERVOIRS[fuelType].capacity - self.RESERVOIRS[fuelType].level    # 容量capacity减去当前的存量level
        print('TO REPLENISH {0:4.0f} GALLONS OF {1}'.format(toReplenish, fuelType))
        print('-' * 70)

        # 等油车重装补满
        yield self.env.timeout(toReplenish / self.SPEED_REPLENISH)   # 重装补满的速度已知，需补油量已知

        # 将油量加到油箱对应的种类fuelType
        yield self.RESERVOIRS[fuelType].put(toReplenish)

        # 向油车(供应商)支付款项
        self.pay(toReplenish * self.buyPrice[fuelType])  # .pay()将金额加到self.cashOut变量上

        print('-' * 70)
        print('FINISHED REPLENISHING AT {0:4.0f}s.'.format(self.env.now))   # 显示补满的时刻
        print('-' * 70)

    def controlLevels(self):    # 将controlLevels()放到环境中
        '''
            每五秒查看大油箱油量，低于最小值时加装
        '''
        while True:
            # 循环所有油箱
            for fuelType in self.RESERVOIRS:
                # 如果油量小于最小值
                if self.RESERVOIRS[fuelType].level < self.MINIMUM_FUEL[fuelType]:
                    # 重装补满
                    yield env.process(self.replenish(fuelType))  # 整个模拟期间，各代理人都会创建与挂起过程。yield命令挂起过程，并在之后由另一个过程唤起。若油量不足，挂起过程并叫来油车。补满后触发新的遍历
                # 每5秒循环检查
                yield env.timeout(5)  # 循环的过程挂起5秒，5秒后触发另一次遍历

    def getPump(self, fuelType):
        '''
            返回贮油器对象
        '''
        return self.PUMPS[fuelType]

    def getReservoir(self, fuelType):
        '''
            返回大油箱对象
        '''
        return self.RESERVOIRS[fuelType]

    def getRefuelSpeed(self):
        '''
            定义注入油的速度
        '''
        return self.SPEED_REFUEL

    def getFuelPrice(self, fuelType):
        return self.sellPrice[fuelType]

    def sell(self, amount):
        self.cashIn += amount

    def pay(self, amount):
        self.cashOut += amount

    def lost(self, amount):
        self.cashLost += amount

    def printCashRegister(self):
        print('\nTotal cash in:   ${0:8.2f}'.format(self.cashIn))
        print('Total cash out:  ${0:8.2f}'.format(self.cashOut))
        print('Total cash lost: ${0:8.2f}'.format(self.cashLost))
        print('\nProfit: ${0:8.2f}'.format(self.cashIn - self.cashOut))
        print('Profit (if no loss of customers): ${0:8.2f}'.format(self.cashIn - self.cashOut + self.cashLost))


class Car(object):
    '''
        定义'车'类
    '''
    def __init__(self, i, env, gasStation):
        '''
            类的主构造函数
            @param i          -- 车辆ID编号
            @param env        -- 环境对象
            @param gasStation -- 加油站对象
        '''
        # 指向环境和加油站对象的指针
        self.env = env
        self.gasStation = gasStation

        # 单个车辆油箱容量（加仑）
        self.TANK_CAPACITY = np.random.randint(12, 23)   # 用random模拟各车之间的油箱容积差异   
        
        # 单个车的剩余油量
        self.FUEL_LEFT = self.TANK_CAPACITY * np.random.randint(10, 40) / 100

        # 需要的燃油种类
        self.FUEL_TYPE = np.random.choice(['PETROL', 'DIESEL'], p=[0.7, 0.3])   # 汽油车相对较多,p为权重

        # 车的ID
        self.CAR_ID = i

        # 开始加油过程
        self.action = env.process(self.refuel())

    def refuel(self):
        '''
            车辆加油过程
        '''
        # 确定燃油种类，以请求相应的泵
        fuelType = self.FUEL_TYPE

        # 对应的泵对象
        pump = gasStation.getPump(fuelType) 

        # 请求空闲的泵
        with pump.request() as req:
            # 到达加油站的时刻
            arrive = self.env.now

            # 挂起过程等待分配空闲的泵，一旦有可用的泵立刻恢复执行
            yield req

            #  需加装油量
            required = self.TANK_CAPACITY - self.FUEL_LEFT  # 车油箱容量减去剩余

            # 等了多久！
            waitedTooLong = self.env.now - arrive > 5 * 60  # 假设五分钟即满足布尔变量“等太久”

            if waitedTooLong:  # 若满足“等太久”
                # 离开
                print('-' * 70)
                print('CAR {0} IS LEAVING -- WAIT TOO LONG'.format(self.CAR_ID))
                print('-' * 70)
                gasStation.lost(required * self.gasStation.getFuelPrice(fuelType))
            else:
                # 开始加油
                start = self.env.now
                yield self.gasStation.getReservoir(fuelType).get(required)

                # 记录油量
                petrolLevel = self.gasStation.getReservoir('PETROL').level
                dieselLevel = self.gasStation.getReservoir('DIESEL').level

                # 挂起等待加装完毕
                yield env.timeout(required / gasStation .getRefuelSpeed())

                # 加满的时刻
                fin = self.env.now

                # 支付油钱
                toPay = required * self.gasStation.getFuelPrice(fuelType)
                self.gasStation.sell(toPay)  # .sell()将金额加到self.cashin变量上

                yield env.timeout(np.random.randint(15, 90))  # 付钱耗时

                # 显示加油过程明细
                refuellingDetails  = '{car}\t{tm}\t{start}'
                refuellingDetails += '\t{fin}'
                refuellingDetails += '\t{gal:2.2f}\t{fuel}'
                refuellingDetails += '\t{petrol}\t{diesel}'
                refuellingDetails += '\t${pay:3.2f}'

                print(
                    refuellingDetails.format(
                        car=self.CAR_ID, tm=arrive, 
                        start=int(start), 
                        fin=int(self.env.now), 
                        gal=required, fuel=fuelType, 
                        petrol=int(petrolLevel), 
                        diesel=int(dieselLevel),
                        pay=toPay
                    )
                )

    @staticmethod
    def generate(env, gasStation):    # generate()静态方法在模拟过程中随机生成车，静态方法不同于对象方法，不需要self关键字
        '''
            静态方法随机生成车
        '''
        # 在模拟过程中随机生成车
        for i in itertools.count():
            # 随机 5 ~ 45 秒一辆
            yield env.timeout(np.random.randint(5, 45))
            
            # 生成新的车
            Car(i, env, gasStation)

if __name__ == '__main__':
    # 模拟的时限(以秒计数)
    SIM_TIME = 20 * 60 * 60    # 20小时

    # 创建环境（环境是模拟的基础，环境封装了时间，并处理模拟中代理人之间的互动）
    env = simpy.Environment()

    # 创建加油站(代理人)
    gasStation = GasStation(env)

    # 打印表头
    print('\t\t\t\t\t\t     Left')
    print('CarID\tArrive\tStart\tFinish\tGal\tType\tPetrol\tDiesel\tPaid')
    print('-' * 70)

    # 将生成车的过程添加到环境中去！
    env.process(Car.generate(env, gasStation))

    # 开始模拟
    env.run(until = SIM_TIME)

    gasStation.printCashRegister()   # 打印模拟记录