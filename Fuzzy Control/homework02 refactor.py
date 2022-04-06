import numpy as np

from typing import Union
from pprint import pprint

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import skfuzzy as fuzzy
import skfuzzy.control as ctrl


class FuzzySystem:
    def __init__(self):
        self._universe: dict = {}
        self._antecedents: dict = {}
        self._consequents: dict = {}
        self._rules: dict = {}
        self._system: Union[ctrl.ControlSystem, None] = None
        self._simulation: Union[ctrl.ControlSystemSimulation, None] = None

    def print(self):
        print('universe:')
        pprint(self._universe)
        print('\nantecedents:')
        pprint(self._antecedents)
        print('\nconsequents:')
        pprint(self._consequents)
        print('\nrules:')
        pprint(self._rules)

    @property
    def universe(self):
        return self._universe

    @universe.deleter
    def universe(self):
        self._universe = {}

    @property
    def antecedents(self):
        return self._antecedents

    def set_antecedent(self, antecedent, label):
        try:
            self._antecedents[antecedent] \
                = ctrl.Antecedent(self._universe[antecedent], label)
        except KeyError:
            return Exception(f'The universe of discord {antecedent} does '
                             f'not exist.')

    @antecedents.deleter
    def antecedents(self):
        self._antecedents = {}

    @property
    def consequents(self):
        return self._consequents

    def set_consequent(self, consequent, label):
        try:
            self._consequents[consequent] \
                = ctrl.Consequent(self._universe[consequent], label)
        except KeyError:
            return Exception(f'The universe of discord {consequent} does '
                             f'not exist.')

    @consequents.deleter
    def consequents(self):
        self._consequents = {}

    @staticmethod
    def get_mf(points):
        # trimf == triangle, trapmf == trapezoid
        if len(points) == 3:
            mf = fuzzy.trimf
        elif len(points) == 4:
            mf = fuzzy.trapmf
        else:
            raise Exception(f'Incorrect number of points passed. '
                            f'(3 or 4 expected, got {len(points)} instead.)')
        return mf

    def antecedent_mf(self, antecedent, label, points):
        mf = self.get_mf(points)
        self._antecedents[antecedent][label] \
            = mf(self._universe[antecedent], points)

    def consequent_mf(self, consequent, label, points):
        mf = self.get_mf(points)
        self._consequents[consequent][label] \
            = mf(self._universe[consequent], points)

    def defuzzify_method(self, consequent, method='centroid'):
        self._consequents[consequent].defuzzify_method = method

    @property
    def rules(self):
        return self._rules

    def set_rules(self, label, rule):
        # TODO parse rule input, or find a way to write rules outside the class
        self._rules[label] = rule

    @rules.deleter
    def rules(self):
        self._rules = {}

    def initialize(self):
        self._system = ctrl.ControlSystem(self._rules.values())
        self._simulation = ctrl.ControlSystemSimulation(self._system)

    @property
    def system(self):
        return self._system

    @property
    def simulation(self):
        return self._simulation

    def input(self, *antecedents):
        for antecedent in antecedents:
            self._simulation.input[antecedent] \
                = int(input(f'Please enter the {antecedent}: '))

    def compute(self):
        self._simulation.compute()

    def output(self):
        for v in self._consequents.values():
            print(f'{v.label}: {self._simulation.output[v.label]:.5f}')


def define_universe(system):
    # define the universe of discourse for each label
    system.universe['temperature'] = np.arange(0, 50.1, 0.25, np.float32)
    system.universe['moisture'] = np.arange(0, 100.1, 0.5, np.float32)
    system.universe['duration'] = np.arange(0, 30.1, 0.5, np.float32)


def antecedents(system):
    # set antecedents (inputs)
    system.set_antecedent(antecedent='temperature', label='air temperature')
    system.set_antecedent(antecedent='moisture', label='soil moisture')


def consequents(system):
    # set consequents (outputs)
    system.set_consequent(consequent='duration', label='watering duration')


def member_functions(system):
    # set membership functions
    mfs = {
        'temperature': [
            ('cold', [0, 0, 5, 14]),
            ('cool', [6, 15, 24]),
            ('normal', [16, 25, 34]),
            ('warm', [26, 35, 44]),
            ('hot', [36, 45, 50, 50])
        ],
        'moisture': [
            ('dry', [0, 0, 16, 35]),
            ('moist', [20, 40, 60, 80]),
            ('wet', [65, 84, 100, 100])
        ],
        'duration': [
            ('short', [0, 0, 4, 8]),
            ('medium', [5, 10, 18, 23]),
            ('long', [19, 24, 30, 30])
        ]
    }

    for antecedent in system.antecedents:
        for label, points in mfs[antecedent]:
            system.antecedent_mf(antecedent, label, points)

    for consequent in system.consequents:
        for label, points in mfs[consequent]:
            system.consequent_mf(consequent, label, points)

    return mfs


def rules(system):
    # set the rules of the system
    system.defuzzify_method('duration', 'centroid')

    temp = system.antecedents['temperature']
    soil = system.antecedents['moisture']
    time = system.consequents['duration']
    system.set_rules(
        label='short',
        rule=ctrl.Rule(
            antecedent=(
                    temp['cold']
                    | soil['wet']
                    | (temp['cool'] & soil['moist'])
            ),
            consequent=time['short'],
            label='short')
    )
    system.set_rules(
        label='medium',
        rule=ctrl.Rule(
            antecedent=(
                    ((temp['cool'] | temp['normal'] | temp['warm'])
                     & soil['dry'])
                    | (~temp['cold'] & soil['moist'])
            ),
            consequent=time['medium'],
            label='medium')
    )
    system.set_rules(
        label='long',
        rule=ctrl.Rule(
            antecedent=(
                    (temp['warm'] & soil['dry'])
                    | (temp['hot'] & ~soil['wet'])
            ),
            consequent=time['long'],
            label='long')
    )


def show_mfs(system, mfs):
    # Visualize these universes and membership function
    titles = [r'air temperature ($^{\circ}$C)', 'soil moisture (%)',
              'watering duration (minutes)']
    assert len(mfs) == len(titles)

    fig, axs = plt.subplots(nrows=len(mfs), figsize=(8, 9))
    # consider using ctrl.Rule.view() instead
    for i, (ax, title, (key, mf_list)) \
            in enumerate(zip(axs, titles, mfs.items())):
        for mf_label, mf in mf_list:
            ax.plot(system.universe[key],
                    FuzzySystem.get_mf(mf)(system.universe[key], mf),
                    linewidth=1.5, label=mf_label)
        ax.set_title(title)
        ax.legend()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
    plt.tight_layout()
    return fig, axs


def show_surface(system):
    def f(x, y):
        system.simulation.input['air temperature'] = y
        system.simulation.input['soil moisture'] = x
        system.simulation.compute()
        return system.simulation.output['watering duration']

    fig = plt.figure(figsize=(9, 8))
    ax = Axes3D(fig)

    X, Y = np.meshgrid(
        system.universe['moisture'],
        system.universe['temperature']
    )
    Z = f(X, Y)

    ax.plot_surface(X, Y, Z, rstride=2, cstride=2, cmap=plt.cm.jet)
    # ax.plot_wireframe(X, Y, Z, rstride=2, cstride=2)
    ax.set_xlabel('soil moisture (%)', color='r')
    ax.set_ylabel(r'air temperature ($^{\circ}$C)', color='g')
    ax.set_zlabel('watering duration (minutes)', color='b')
    return fig, ax


def main():
    np.set_printoptions(threshold=7)  # maximum number of array values to print

    sprinkler = FuzzySystem()  # create a fuzzy sprinkler

    define_universe(sprinkler)  # define the universes of discourse
    antecedents(sprinkler)  # set the antecedent variables
    consequents(sprinkler)  # set the consequent variable
    mfs = member_functions(sprinkler)  # set the member functions
    rules(sprinkler)  # set the rules of the sprinkler

    sprinkler.initialize()  # initialize sprinkler
    sprinkler.input('air temperature', 'soil moisture')  # input conditions
    sprinkler.compute()  # compute the output
    sprinkler.output()  # print the output

    sprinkler.print()  # print the internal states of the sprinkler

    fig, _ = show_mfs(sprinkler, mfs)  # plot the member functions
    plt.savefig('hw02_membership_functions.png')
    fig1, _ = show_surface(sprinkler)  # plot the performance surface
    plt.savefig('hw02_performance_surface.png')
    plt.show()  # show the plot


if __name__ == '__main__':
    main()
