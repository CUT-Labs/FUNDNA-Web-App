from gui.classes.fsm.FunctionTypes import *
import numpy as np


class State:
    def __init__(self, name, output):
        self.name = name
        self.output = output


class Transition:
    def __init__(self, fromState, toState, transitionOn):
        assert type(fromState) is State and type(toState) is State

        self.fromState = fromState
        self.toState = toState
        self.condition = transitionOn


class FSM:
    def __init__(self, name, function, inputType, outputType):
        assert Types.isIn(inputType) and Types.isIn(outputType)

        self.name = name
        self.states = []
        self.transitions = []
        self.inputType = inputType
        self.outputType = outputType
        self.initialState = None

        self.func = function

        self.quadrantI = False
        self.quadrantII = False
        self.quadrantIII = False
        self.quadrantIV = False

        self.error = None

        self.testQuadrants()

    def addState(self, state):
        assert type(state) is State

        for s in self.states:
            if str.upper(s.name) == str.upper(state.name):
                print("\nERROR!!!\n")
                raise FSMError(self, f"You already have a state by the name of {s.name} for {self.name}!")

        print(f"Adding state {state.name} to {self.name}")
        self.states.append(state)

    def setInitial(self, state):
        assert type(state) is State

        for s in self.states:
            if str.upper(s.name) == str.upper(state.name):
                print(f"Setting {self.name} initial state to {state.name}")
                self.initialState = state
                return

        print("\nERROR!!!\n")
        raise FSMError(self, f"There is no state in {self.name} identified by {state.name}!")

    def addTransition(self, transition):
        assert type(transition) is Transition

        for t in self.transitions:
            if transition.condition == t.condition and transition.fromState == t.fromState:
                print("\nERROR!!!\n")
                raise FSMError(self, f"You already have a transition from state {t.fromState.name} with the condition {t.condition} for {self.name}!")

        print(f"Adding transition from {transition.fromState.name} -> {transition.toState.name} via {transition.condition} to {self.name}")

        self.transitions.append(transition)

    def getStateTransitions(self, state):
        transitions = []

        for t in self.transitions:
            if t.fromState.name is state.name:
                transitions.append(t)

        return transitions

    def getState(self, state_name):
        for s in self.states:
            if str.upper(s.name) == str.upper(state_name):
                return s

        print("\nERROR!!!\n")
        raise FSMError(self, f"No state in {self.name} identified by {state_name}")

    def processInput(self, input_stream):
        if self.initialState is None:
            print("\nERROR!!!\n")
            raise FSMError(self, f"Please indicate your initial state for {self.name}!")

        current_state = self.initialState
        output_stream = []

        for bit in input_stream:
            current_state = self.getNextState(current_state, bit)
            output_stream.append(current_state.output)

        return np.array(output_stream)

    def getNextState(self, current, cond):
        for t in self.transitions:
            if t.fromState.name == current.name:
                if t.condition == cond:
                    return t.toState

        return current

    def printFSM(self):
        print(f"Name: {self.name}\n"
              f"Input: {self.inputType.value}\n"
              f"Output: {self.outputType.value}\n"
              f"Initial: {self.initialState.name}\n"
              f"\nStates:")

        transMap = {}

        for s in self.states:
            transMap[s.name] = []
            print(f"  - {s.name}: {s.output}")

        print(f"\nTransitions:")

        for t in self.transitions:
            transMap[t.fromState.name].append(f"{t.condition}: {t.toState.name}")

        for f, cond in transMap.items():
            print(f"  - {f}:")

            for t in cond:
                print(f"    {t}")

    def testQuadrants(self):
        def getYRange(func, xRange):
            yRange = []
            for x in xRange:
                yRange.append(func(x))

            return yRange

        x_range1 = np.linspace(-1, 0, 10)
        x_range2 = np.linspace(0, 1, 10)

        for y in getYRange(self.func, x_range1):  # negative x
            if -1 <= y <= 1:
                if -1 <= y < 0:
                    self.quadrantIII = True
                if 0 < y < 1:
                    self.quadrantII = True

        for y in getYRange(self.func, x_range2):  # positive x
            if -1 <= y <= 1:
                if -1 <= y < 0:
                    self.quadrantIV = True
                if 0 < y < 1:
                    self.quadrantI = True


class FSMError(Exception):
    def __init__(self, fsm, message):
        self.message = message
        fsm.printFSM()

    def __str__(self):
        return self.message
