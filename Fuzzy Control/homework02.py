import numpy as np
import skfuzzy as fuzzy
import skfuzzy.control as ctrl
import matplotlib.pyplot as plt

from pprint import pprint
from mpl_toolkits.mplot3d import Axes3D


def show(system):
    universe = system['universe']
    # Visualize these universes and membership functions
    fig, ax = plt.subplots(nrows=3, figsize=(8, 9))
    ax[0].plot(universe['temperature'],
               fuzzy.trapmf(universe['temperature'], [32, 32, 50, 68]), 'blue',
               linewidth=1.5, label='cold')
    ax[0].plot(universe['temperature'],
               fuzzy.trimf(universe['temperature'], [50, 68, 86]), 'cyan',
               linewidth=1.5, label='cool')
    ax[0].plot(universe['temperature'],
               fuzzy.trimf(universe['temperature'], [59, 77, 95]), 'yellow',
               linewidth=1.5, label='normal')
    ax[0].plot(universe['temperature'],
               fuzzy.trimf(universe['temperature'], [68, 86, 104]), 'orange',
               linewidth=1.5, label='warm')
    ax[0].plot(universe['temperature'],
               fuzzy.trapmf(universe['temperature'], [86, 104, 122, 122]),
               'red', linewidth=1.5, label='hot')
    ax[0].set_title('air temperature')
    ax[0].legend()

    ax[1].plot(universe['moisture'],
               fuzzy.trapmf(universe['moisture'], [0, 0, 16.5, 49.5]), 'red',
               linewidth=1.5, label='dry')
    ax[1].plot(universe['moisture'],
               fuzzy.trapmf(universe['moisture'], [16.5, 40.5, 62.5, 88.6]),
               'cyan', linewidth=1.5, label='moist')
    ax[1].plot(universe['moisture'],
               fuzzy.trapmf(universe['moisture'], [65.5, 88.6, 100, 100]),
               'blue', linewidth=1.5, label='wet')
    ax[1].set_title('soil moisture')
    ax[1].legend()

    ax[2].plot(universe['duration'],
               fuzzy.trapmf(universe['duration'], [0, 0, 2, 8]), 'b',
               linewidth=1.5, label='short')
    ax[2].plot(universe['duration'],
               fuzzy.trapmf(universe['duration'], [2, 10, 15, 23]), 'g',
               linewidth=1.5, label='medium')
    ax[2].plot(universe['duration'],
               fuzzy.trapmf(universe['duration'], [16, 23, 30, 30]), 'r',
               linewidth=1.5, label='long')
    ax[2].set_title('watering duration')
    ax[2].legend()

    # plot 2d figures
    for fig in ax:
        fig.spines['top'].set_visible(False)
        fig.spines['right'].set_visible(False)
        fig.get_xaxis().tick_bottom()
        fig.get_yaxis().tick_left()

    plt.tight_layout()

    def funz(x, y):
        system['simulation'].input['air temperature'] = y
        system['simulation'].input['soil moisture'] = x
        system['simulation'].compute()
        z = system['simulation'].output['watering duration']
        return z

    fig1 = plt.figure()  # 建立一個繪圖物件
    ax = Axes3D(fig1)  # 用這個繪圖物件建立一個Axes物件(有3D座標)

    X, Y = np.meshgrid(universe['moisture'], universe['temperature'])
    Z = funz(X, Y)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                    cmap=plt.cm.coolwarm)  # 用取樣點(x,y,z)去構建曲面
    ax.set_xlabel('soil moisture', color='r')
    ax.set_ylabel('air temperature', color='g')
    ax.set_zlabel('watering duration', color='b')  # 給三個座標軸註明
    plt.show()  # 顯示模組中的所有繪圖物件


def main():
    np.set_printoptions(threshold=7)

    # create a system to store data
    system = {
        # define the universe of discourse for each label
        'universe': {
            'temperature': np.arange(32, 122.1, 1, np.float32),
            'moisture': np.arange(0, 100.1, 1, np.float32),
            'duration': np.arange(0, 30.1, 1, np.float32)
        },
    }

    # set antecedents (input) and consequents (output)
    system['temperature'] = ctrl.Antecedent(system['universe']['temperature'],
                                            'air temperature')
    system['moisture'] = ctrl.Antecedent(system['universe']['moisture'],
                                         'soil moisture')
    system['duration'] = ctrl.Consequent(system['universe']['duration'],
                                         'watering duration')

    # set membership functions; trapmf == trapezoid, trimf = triangle
    # TODO adjust temperature mf
    temp = system['temperature']
    universe = system['universe']['temperature']
    temp['cold'] = fuzzy.trapmf(universe, [32, 32, 50, 68])
    temp['cool'] = fuzzy.trimf(universe, [50, 68, 86])
    temp['normal'] = fuzzy.trimf(universe, [59, 77, 95])
    temp['warm'] = fuzzy.trimf(universe, [68, 86, 104])
    temp['hot'] = fuzzy.trapmf(universe, [86, 104, 122, 122])

    # TODO adjust moisture mf
    moisture = system['moisture']
    universe = system['universe']['moisture']
    moisture['dry'] = fuzzy.trapmf(universe, [0, 0, 16.5, 49.5])
    moisture['moist'] = fuzzy.trapmf(universe, [16.5, 40.5, 62.5, 88.6])
    moisture['wet'] = fuzzy.trapmf(universe, [65.5, 88.6, 100, 100])

    # TODO adjust duration mf
    time = system['duration']
    universe = system['universe']['duration']
    time['short'] = fuzzy.trapmf(universe, [0, 0, 2, 8])
    time['medium'] = fuzzy.trapmf(universe, [2, 10, 15, 23])
    time['long'] = fuzzy.trapmf(universe, [16, 23, 30, 30])

    # set the rules
    time.defuzzify_method = 'centroid'
    system['rules'] = {
        'short': ctrl.Rule(
            antecedent=(
                    temp['cold'] | moisture['wet']
                    | (temp['cold'] & moisture['dry'])),
            consequent=time['short'],
            label='short'
        ),
        'medium': ctrl.Rule(
            antecedent=(
                    ((temp['cool'] | temp['normal']) & moisture['dry'])
                    | ((temp['normal'] | temp['warm'] | temp['hot']) &
                       moisture['moist'])),
            consequent=time['medium'],
            label='medium'
        ),
        'long': ctrl.Rule(
            antecedent=(
                ((temp['warm'] | temp['hot']) & moisture['dry'])),
            consequent=time['long'],
            label='long'
        ),
    }

    # initialize system
    system['system'] = ctrl.ControlSystem(system['rules'].values())
    system['simulation'] = ctrl.ControlSystemSimulation(system['system'])

    # # input conditions
    # system['simulation'].input['air temperature'] \
    #     = int(input('Please enter the temperature (℉): '))
    # system['simulation'].input['soil moisture'] \
    #     = int(input('Please enter the soil moisture: '))
    #
    # # compute the output
    # system['simulation'].compute()
    #
    # # show the output
    # print(
    #     'watering duration:',
    #     system['simulation'].output['watering duration'],
    #     end='\n\n'
    # )

    pprint(system)

    show(system)


if __name__ == '__main__':
    main()
    plt.show()
