from math import *
import numpy as np
from gui.classes.fsm import *
import random
from io import BytesIO
import base64


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


def solverToFSM(solver):
    # Initialize FSM object
    fsm = FSM(
        solver.name,
        solver.func,
        solver.testFunctionType()[0],
        solver.testFunctionType()[1]
    )

    # Add states to FSM
    states = {}
    for i, val in enumerate(solver.b_vector, 0):
        state_name = f"S{i}"
        output_val = round(float(val), 4)
        state = State(state_name, output_val)
        fsm.addState(state)
        states[state_name] = state

    # Set initial state
    fsm.setInitial(states[next(iter(states))])

    # Add transitions to FSM
    state_names = list(states.keys())
    for idx, state_name in enumerate(state_names):
        from_state = states[state_name]
        if idx == 0:  # First state
            fsm.addTransition(Transition(from_state, from_state, "0"))  # Transition to self on 0
            fsm.addTransition(Transition(from_state, states[state_names[idx + 1]], "1"))  # Transition to next state on 1
        elif idx == len(state_names) - 1:  # Last state
            fsm.addTransition(Transition(from_state, states[state_names[idx - 1]], "0"))  # Transition to previous state on 0
            fsm.addTransition(Transition(from_state, from_state, "1"))  # Transition to self on 1
        else:  # Intermediate states
            fsm.addTransition(Transition(from_state, states[state_names[idx - 1]], "0"))  # Transition to previous state on 0
            fsm.addTransition(Transition(from_state, states[state_names[idx + 1]], "1"))  # Transition to next state on 1

    return fsm

def simulateFSM(fsm, func):
    x_values = []
    expected_values = []
    y_values = []

    error = []

    bipolarIssue = False

    if not bipolarIssue:
        # Force Unipolar, until bipolar is fixed
        fsm.inputType = Types.UNIPOLAR
        fsm.outputType = Types.UNIPOLAR

    nsteps = 1000  # how many data points to collect
    lbound = -1 if fsm.inputType is Types.BIPOLAR else 0
    ubound = 1
    step = (ubound - lbound) / nsteps

    x_value = lbound

    while x_value <= ubound:
        expected = func(x_value)
        expected_values.append(expected)

        input_bits = generateStream(abs(x_value), 10000)
        input_fraction = np.count_nonzero(input_bits == '1') / len(input_bits)

        # Compute FSM approximation
        fsm_output = fsm.processInput(input_bits)
        y_value = np.sum(fsm_output) / len(fsm_output)  # f_p(x)

        if (fsm.quadrantIII and x_value < 0) or (fsm.quadrantIV and x_value > 1):
            if fsm.outputType == Types.BIPOLAR:
                y_value = (2 * y_value) - 1  # compute f(x) from f_p(x)

        x_values.append(x_value)
        y_values.append(y_value)

        x_value += step

    table_values = [x_values, y_values, expected_values]
    print(tabulate(table_values, ["IN", "OUT", "f(x)"], tablefmt="pretty"))

    for i, y_value in enumerate(y_values):
        bottom_bound = 0 if fsm.outputType == Types.UNIPOLAR else 1

        if bottom_bound <= expected_values[i] <= 1:
            error.append(expected_values[i] - y_value)
        elif 1 <= expected_values[i]:
            error.append(1 - y_value)
        elif expected_values[i] <= bottom_bound:
            error.append(bottom_bound - y_value)

    fsm.error = error

    return fsm, x_values, y_values, expected_values


def generatePlot(x_values, y_values, expected_values, fsm):
    averageError, std_dev, ratioUnder, mse, mae = analyzeError(fsm.error)

    error_labels = [
        'Average Error',
        'Error Std. Dev',
        #'%neg',
        'MSE',
        'MAE'
    ]
    error_values = [
        averageError,
        std_dev,
        #ratioUnder*100,
        mse,
        mae
    ]

    # Create a figure with a main plot and a subplot for the caption
    fig, (ax_main, ax_caption) = plt.subplots(nrows=2, gridspec_kw={'height_ratios': [3, 1]}, figsize=(10, 8))

    # Plot the results in the main subplot
    ax_main.plot(x_values, expected_values, label='Original Function', color='blue')
    ax_main.plot(x_values, y_values, label='FSM Approximation', linestyle='--', color='orange')
    ax_main.set_xlabel('x', fontsize=16)
    ax_main.set_ylabel('f(x)', fontsize=16)
    ax_main.set_title(f'FSM Approximation for {fsm.name}', fontsize=20)
    ax_main.legend(fontsize=16)
    ax_main.tick_params(axis='both', which='major', labelsize=14)  # Increase font size of axis ticks
    ax_main.grid(True)

    # Adding the figure caption in the caption subplot
    caption = ";   ".join([f"{label}: {value:.6f}" for label, value in zip(error_labels, error_values)])
    ax_caption.text(0.5, 0.5, caption, ha='center', va='center', fontsize=14)
    ax_caption.axis('off')  # Hide axis for the caption subplot

    plt.tight_layout()  # Adjust layout to prevent overlap

    # Save plot to a BytesIO object
    buf = BytesIO()
    fig.savefig(buf, format='svg')  # Save as SVG for better scalability
    buf.seek(0)

    plt.close(fig)  # Close the figure to free up memory

    return buf


def generateStream(ratio, length):
    num_ones = int(ratio * length)
    num_zeros = length - num_ones

    # Create a list with the specified number of ones and zeros
    stream = ['1'] * num_ones + ['0'] * num_zeros

    # Shuffle the list to randomize the order of bits
    random.shuffle(stream)

    return stream


def analyzeError(error):
    # Average Error
    averageError = np.mean(error)

    # Calculate standard deviation
    std_dev = np.std(error)

    # Percent negative errors
    ratioUnder = np.count_nonzero(np.array(error) < 0) / len(error)

    # Calculate mean squared error (MSE)
    mse = np.mean(np.square(error))

    # Calculate mean absolute error (MAE)
    mae = np.mean(np.abs(error))

    print(f"Average Error: \t{averageError}")
    print(f"Standard deviation of errors: \t{std_dev}")
    print(f"Ratio of negative errors: \t{ratioUnder}")
    print(f"Mean Squared Error (MSE): \t{mse}")
    print(f"Mean Absolute Error (MAE): \t{mae}")

    return averageError, std_dev, ratioUnder, mse, mae


def FSMPlot(fsm, function):
    fsm_with_error, x_values, y_values, expected_values = simulateFSM(fsm, function)

    graph_io = generatePlot(x_values, y_values, expected_values, fsm_with_error)

    return "data:image/svg+xml;base64," + base64.b64encode(graph_io.getvalue()).decode()


def matrix_to_latex(matrix):
    """
    Converts a 2D numpy array or list of lists to LaTeX matrix representation.
    """
    rows = [" & ".join(map(str, row)) for row in matrix]
    latex_matrix = " \\ ".join(rows)
    return f"\\begin{{bmatrix}} {latex_matrix} \\end{{bmatrix}}"


def vector_to_latex(vector, transpose=False):
    """
    Converts a 1D numpy array or list to LaTeX vector representation.
    If transpose is True, the vector will be represented as a row vector.
    """
    if transpose:
        latex_vector = " & ".join(map(str, vector))
        return f"\\begin{{bmatrix}} {latex_vector} \\end{{bmatrix}}"
    else:
        latex_vector = " \\ ".join(map(str, vector))
        return f"\\begin{{bmatrix}} {latex_vector} \\end{{bmatrix}}"


def generate_objective_function_latex(h_matrix, b_vector, c_vector):
    """
    Generates the LaTeX code for the objective function with the filled-in matrices and vectors.
    """
    h_matrix_latex = matrix_to_latex(h_matrix)
    b_vector_latex = vector_to_latex(b_vector)
    b_vector_transpose_latex = vector_to_latex(b_vector, transpose=True)
    c_vector_transpose_latex = vector_to_latex(c_vector, transpose=True)

    objective_function = (
        f"T(x) = \\"
        f"\\frac{{1}}{{2}} \cdot {b_vector_transpose_latex} \cdot {h_matrix_latex} \cdot {b_vector_latex} + {c_vector_transpose_latex} \cdot {b_vector_latex}"
    )

    return objective_function