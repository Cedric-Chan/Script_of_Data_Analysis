'''
一个著名的代理人基模型就是羊-狼捕食的例子
先构建一定区域，初始化一定数量的羊与狼。羊群食草补充能量，狼群捕食羊群得到能量；动物在区域内移动消耗能量。
能量低于0的动物就有灭绝风险
'''
import numpy as np
import simpy
import collections as col

rint = np.random.randint
choice = np.random.choice

# 模拟参数
SIM_TIME = 1000  # 时间跨度
LAND = (300,300)  # 区域为300*300
GRASS_COVERAGE = 0.5  # 草地覆盖情况

# 初始化动物数量
INITIAL_SHEEP = 6000
INITIAL_WOLF = 200

# 羊吃草获能
ENERGY_FROM_GRASS = 5

# 新生动物能量值
ENERGY_AT_BIRTH = [10, 20]

# 生育率
SHEEP_REPRODUCE = 0.04
WOLF_REPRODUCE = 0.006

class Animal(object):
    '''
        定义动物的类
    '''
    def __init__(self, i, env, energy, pos, plane):
        '''
            构造函数
        '''
        # 属性
        self.energy = energy
        self.pos = pos          # 当前位置

        # 是否存活
        self.alive = True
        self.causeOfDeath = None

        # 上次进食时刻
        self.lastTimeEaten = 0

        # 移动范围
        self.movements = [i for i in range(-50,51)]

        # 指向环境和区域的指针
        self.env = env
        self.plane = plane
        self.id = i

    def move(self):
        '''
            改变行进方向
        '''
        # 决定水平轴和垂直轴
        h = choice(self.movements)
        v = choice(self.movements)

        # 调整位置
        self.pos[0] += h
        self.pos[1] += v

        # 确保没有越界
        self.pos[0] = np.min([np.max([0, self.pos[0] - 1]), LAND[0] - 1])
        self.pos[1] = np.min([np.max([0, self.pos[1] - 1]), LAND[1] - 1])

        # 减去移动消耗的能量
        self.energy -= (h+v) / 4

    def getPosition(self):
        '''
            返回动物的当前位置
        '''
        return self.pos

    def die(self, cause):
        '''
            返回动物的死亡状态
        '''
        self.alive = False
        self.causeOfDeath = cause

    def isAlive(self):
        '''
            返回动物的存活状态
        '''
        return self.alive

    def getCauseOfDeath(self):
        '''
            范围动物死因
        '''
        return self.causeOfDeath

    def getEnergy(self):
        '''
            获取能量
        '''
        return self.energy

class Sheep(Animal):
    '''
        羊属于动物大类
    '''
    def __init__(self, i, env, energy, pos, plane):
        '''
            调用animal的构造函数
        '''
        Animal.__init__(self, i, env, energy, pos, plane)

    def eatGrass(self):
        '''
            羊吃草
        '''
        if self.plane.hasGrass(self.pos):
            # 食草获能 
            self.energy += ENERGY_FROM_GRASS
            self.lastTimeEaten = self.env.now

            # 标记此处草已被吃
            self.plane.grassEaten(self.pos)

        if self.energy > 200:   # 羊的能量不大于200，即不能无限增长
            self.energy = 200

       
class Wolf(Animal):
    '''
        狼属于动物大类
    '''
    def __init__(self, i, env, energy, pos, plane):
        '''
            调用animal的构造函数
        '''
        Animal.__init__(self, i, env, energy, pos, plane)

    def eatSheep(self):
        '''
            狼吃羊
        '''
        # 获取当前位置所有的羊
        sheep = self.plane.getSheep(self.pos)
        
        # 决定吃多少只
        howMany = np.random.randint(1, np.max([len(sheep), 2]))

        # 吃羊
        for i, s in enumerate(sheep):
            # 先检查当前位置活羊数
            if s.isAlive() and i < howMany:
                self.energy += s.getEnergy() / 20
                s.die('eaten') 
                
        if self.energy > 200:  # 羊的能量不大于200
            self.energy = 200

        # 更新上次进食时刻（用来决定生存概率）
        self.lastTimeEaten = self.env.now

class Plane(object):
    '''
        定义'区域'的类
    '''
    def __init__(self, env, bounds, grassCoverage, sheep, wolves):
        '''
            构造函数
        '''        
        # 指向环境的指针
        self.env = env

        # 区域的边界
        self.bounds = bounds

        # 创建草
        self.grassCoverage = grassCoverage
        self.grass = [[0  for _ in range(self.bounds[0])] for _ in range(self.bounds[1])]  # 先都设为0 即无草

        # 跟踪吃草情况
        self.grassEatenIndices = col.defaultdict(list)

        # 创建动物
        self.noOfSheep  = sheep
        self.noOfWolves = wolves

        self.sheep = []
        self.wolves = []

        # 跟踪动物的数量变化
        self.counts = {
            'sheep': {
                'count': 0,
                'died': {
                    'energy': 0,
                    'eaten': 0,
                    'age': 0,
                },
                'born': 0
            },
            'wolves': {
                'count': 0,
                'died': {
                    'energy': 0,
                    'age': 0,
                },
                'born': 0
            } 
        }

        # 生成草和动物
        self.generateGrass()
        self.generateSheep()
        self.generateWolves()

        # 开始监控和模拟过程
        self.monitor = self.env.process(self.monitorPopulation())
        self.action = self.env.process(self.run())

    def run(self):
        '''
            模拟的主循环
        '''
        while True:
            # 首先，动物在区域内移动
            self.updatePositions()

            # 其次，各自进食获能
            self.eat()

            # 然后，繁殖
            self.reproduceAnimals()

            # 记录新长出的草
            self.env.process(self.regrowGrass())

            # 打印日志
            toPrint = '{tm}\t{sheep_alive}\t{sheep_born}'
            toPrint += '\t{sheep_died_energy}'
            toPrint += '\t{sheep_died_eaten}'
            toPrint += '\t{wolves_alive}\t{wolves_born}'
            toPrint += '\t{wolves_died_energy}'


            print(toPrint.format(
                tm=int(self.env.now), 
                sheep_alive=int(len(self.sheep)), 
                sheep_born=self.counts['sheep']['born'],
                sheep_died_energy= self.counts['sheep']['died']['energy'],
                sheep_died_eaten= self.counts['sheep']['died']['eaten'],
                sheep_died_age= self.counts['sheep']['died']['age'],
                wolves_alive=int(len(self.wolves)), 
                wolves_born=self.counts['wolves']['born'],
                wolves_died_energy= self.counts['wolves']['died']['energy'],
                wolves_died_age= self.counts['wolves']['died']['age'])
            )

            # 等下一次迭代
            yield self.env.timeout(1)

    def generateGrass(self):
        '''
            生成区域并种上草
        '''
        # 区域面积
        totalSize = self.bounds[0] * self.bounds[1]

        # 多少面积有草
        totalGrass = int(totalSize * self.grassCoverage)

        # 在区域内随机种草
        grassIndices = sorted(choice(totalSize, totalGrass, replace=False))

        for index in grassIndices:
            row = int(index / self.bounds[0])
            col = index - (self.bounds[1] * row)

            self.grass[row][col] = 1  # 用0-1表示是否种草

    def hasGrass(self, pos):
        '''
            检查某位置是否有草
        '''
        if self.grass[pos[0]][pos[1]] == 1:
            return True
        else:
            return False

    def grassEaten(self, pos):
        '''
            更新被吃过的草地
        '''
        self.grass[pos[0]][pos[1]] = 0
        self.grassEatenIndices[self.env.now].append(pos)

    def regrowGrass(self):
        '''
            长出新草
        '''
        # 长草时间
        regrowTime = 2
        yield self.env.timeout(regrowTime)
        
        # 该位置重新有草
        for pos in self.grassEatenIndices[self.env.now - regrowTime]:
            self.grass[pos[0]][pos[1]] = 1

    def generateSheep(self):
        '''
            生成羊
        '''
        # 随机在区域内生成羊（坐标，初始能量）
        for _ in range(self.noOfSheep):
            pos_x = rint(0, LAND[0])
            pos_y = rint(0, LAND[1])
            energy = rint(*ENERGY_AT_BIRTH)

            self.sheep.append(
                Sheep(
                    self.counts['sheep']['count'], 
                    self.env, energy, [pos_x, pos_y], self)
                )
            self.counts['sheep']['count'] += 1

    def generateWolves(self):
        '''
            生成狼
        '''
        # 随机在区域内生成狼（坐标，初始能量）
        for _ in range(self.noOfWolves):
            pos_x = rint(0, LAND[0])
            pos_y = rint(0, LAND[1])
            energy = rint(*ENERGY_AT_BIRTH)

            self.wolves.append(
                Wolf(
                    self.counts['wolves']['count'], 
                    self.env, energy, [pos_x, pos_y], self)
                )

            self.counts['wolves']['count'] += 1

    def updatePositions(self):
        '''
            更新动物坐标
        '''
        for s in self.sheep:
            s.move()

        for w in self.wolves:
            w.move()
            
    def eat(self):
        '''
            动物获能方法
        '''
        for s in self.sheep:
            s.eatGrass()

        for w in self.wolves:
            w.eatSheep()

    def getSheep(self, pos):
        '''
            获取当前位置所有的羊
        '''
        return [s for s in self.sheep if s.getPosition() == pos]

    def reproduceAnimals(self):
        '''
            生育动物幼崽
        '''
        # 计算新生情况
        births = {'sheep': 0, 'wolves': 0}

        # 新生羊
        for s in self.sheep:
            # 是否会繁殖
            willReproduce = np.random.rand() < (SHEEP_REPRODUCE * 3 / (self.env.now - s.lastTimeEaten + 1))  # 随机决定，且距上次进食时间越久越不可能繁殖

            # 若顺利繁殖
            if willReproduce and s.isAlive():
                energy = rint(*ENERGY_AT_BIRTH)
                self.sheep.append(Sheep(self.counts['sheep']['count'], self.env, energy, s.getPosition(), self))  # 原地创建新羊

                # 羊群总数加一
                self.counts['sheep']['count'] += 1

                # 出生数加一
                births['sheep'] += 1

        # 新生狼
        for w in self.wolves:
            # 是否会繁殖
            willReproduce = np.random.rand() < ( WOLF_REPRODUCE / (self.env.now - w.lastTimeEaten  + 1))  # 随机决定，且距上次进食时间越久越不可能繁殖
            # 若顺利繁殖
            if willReproduce and w.isAlive():
                energy = rint(*ENERGY_AT_BIRTH)
                self.wolves.append(
                    Wolf(self.counts['wolves']['count'], self.env, energy, w.getPosition(), self))   # 原地创建新狼
                
                # 狼群总数加一
                self.counts['wolves']['count'] += 1

                # 出生数加一
                births['wolves'] += 1

        # 更新数量
        for animal in births:
            self.counts[animal]['born'] = births[animal]

    def monitorPopulation(self):
        '''
            移除所有能量耗尽的动物
        '''
        # 移除所有能量耗尽的动物
        while True:
            for s in self.sheep:
                if s.energy < 0:
                    s.die('energy')

            for w in self.wolves:
                if w.energy < 0:
                    w.die('energy')
            
            # 清理
            self.removeAnimalsThatDied()
                
            yield self.env.timeout(1)

    def removeAnimalsThatDied(self):
        '''
            清理死亡的动物
        '''
        # 清理死亡的动物
        sheepDied = []
        wolvesDied = []

        sheepAlive = []
        wolvesAlive = []

        for s in self.sheep:
            if s.isAlive():
                sheepAlive.append(s)
            else:
                sheepDied.append(s)

        for w in self.wolves:
            if w.isAlive():
                wolvesAlive.append(w)
            else:
                wolvesDied.append(w)

        # 只保留存活的动物
        self.sheep = sheepAlive
        self.wolves = wolvesAlive
        
        # 添加死因
        cod = {'energy': 0, 'eaten': 0, 'age': 0}
        for s in sheepDied:
            cod[s.getCauseOfDeath()] += 1

        for cause in cod:
            self.counts['sheep']['died'][cause] = cod[cause]

        cod = {'energy': 0, 'age': 0}
        for w in wolvesDied:
            cod[w.getCauseOfDeath()] += 1

        for cause in cod:
            self.counts['wolves']['died'][cause] = cod[cause]
 
        # 删除死亡对象以释放内存
        for s in sheepDied:
            del s

        for w in wolvesDied:
            del w
        
if __name__ == '__main__':
    # 创建环境
    env = simpy.Environment()

    # 创建区域
    plane = Plane(env, LAND, GRASS_COVERAGE, INITIAL_SHEEP, INITIAL_WOLF)

    # 打印表头
    print('\tSheep\t\tDied\t\tWolves\t\tDied\t')
    print('T\tLive\tBorn\tEnergy\tEaten\tLive\tBorn\tEnergy')

    # 开始模拟
    env.run(until = SIM_TIME)