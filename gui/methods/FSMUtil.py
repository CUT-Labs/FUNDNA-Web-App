from math import *
import numpy as np
from gui.classes.fsm.FunctionTypes import Types


def parse_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    # Extracting FSM information
    print("\n\nInitializing FSM...\n\n")
    fsm_name = lines[0].split(': ')[1].strip()
    variable_name = lines[1].split(': ')[1].strip()
    original_func = lines[2].split(': ')[1].strip()

    lExpr = f"lambda {variable_name}: {original_func}"
    lFunc = eval(lExpr)

    input_type = Types(lines[3].split(': ')[1].strip())
    output_type = Types(lines[4].split(': ')[1].strip())
    initial_state = lines[5].split(': ')[1].strip()

    # Creating FSM object
    fsm = FSM(fsm_name, lFunc, input_type, output_type)

    print("\n\nAdding States...\n\n")

    # Adding states and transitions
    state_lines = lines[lines.index('States:\n') + 1:lines.index('Transitions:\n') - 1]
    for state_line in state_lines:
        if state_line.strip() == '':
            continue

        print(state_line)
        state_data = state_line.strip().split(': ')
        state_name, output_val = state_data[0].strip().removeprefix('- '), state_data[1].strip()
        fsm.addState(State(state_name, float(output_val)))

    print("\n\nAdding Transitions...\n\n")

    transition_lines = lines[lines.index('Transitions:\n') + 1:]
    current_from_state = None
    for trans_line in transition_lines:
        if trans_line.strip() == '':
            continue

        if trans_line.startswith('  - '):
            current_from_state = trans_line.split(': ')[0].strip().removeprefix('- ').removesuffix(':')
        else:
            cond, to_state = trans_line.strip().split(': ')

            print("---")
            print(f"From State: {current_from_state}\n"
                  f"To State: {to_state}\n"
                  f"On Condition: {cond}")

            tFrom = fsm.getState(current_from_state)
            tTo = fsm.getState(to_state)

            fsm.addTransition(Transition(tFrom, tTo, cond))

    fsm.setInitial(fsm.getState(initial_state))

    print("\n\nParsed FSM:\n")

    fsm.printFSM()

    return lFunc, fsm
