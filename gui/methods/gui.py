import shlex
import sys
usingExe = False

try:
    from PyInstaller import *
    import pyi_splash

    usingExe = True
except:
    print("Unable to import PyInstaller... continuing")
finally:
    from pathlib import Path

    from tkinter import *
    import tkinter as tk
    import tkinter.messagebox as messagebox
    from tkinter.scrolledtext import ScrolledText
    from tkinter import ttk

    import re

    from math import *
    from mpmath import *

    import sympy
    import sympy as sp
    from sympy import sympify

    from PIL import Image, ImageTk

    from FuncTypes import *
    from FUNDNAUtil import *
    from RearrangeType import *
    from Function import *

OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path("assets")

variable = "x"
point = 0
power = 6
functionStr = "exp(-x)"
lExpress = ""
lFunc = None

hasNuskell = False
useNuskell = False
verify = False
scheme = "soloveichik2010.ts"

hasPiperine = False
usePiperine = False
pipOptions = "--candidates 3 --energy 7.7 --maxspurious 0.4 --deviation 0.5"

crn = None


try:
    if usingExe:
        print("Using Pyinstall-Generated EXE Files")
    else:
        print("Using Python GUI Files")
        usingExe = False
except:
    print("Using Python GUI Files")
    usingExe = False

import io, os, subprocess, shutil

nuskellSchemes = [
    # development
    "cardelli_2domain_fixed_dev.ts", "cardelli_2domain_fixed_noGC_dev.ts", "lakin2016_3D_fix_dev.ts",
    "mehta_3domain_dev.ts", "soloveichik_cooperative_dev.ts", "soloveichik_opt_dev.ts",
    "thachuk_xchange_dev.ts", "thachuk_xchange_dev.ts",

    # implementations
    "zhang2007_autocatalyst.pil", "zhang2007_catalyst.pil",

    # literature
    "cardelli2011_FJ.ts", "cardelli2011_NM.ts", "cardelli2013_2D.ts", "cardelli2013_2D_2TGC.ts",
    "cardelli2013_2D_3I.ts", "chen2013_2D_JF.ts", "lakin2012_3D.ts", "lakin2016_2D_3I.ts",
    "qian2011_3D.ts", "soloveichik2010.ts", "srinivas2015.ts",

    # variants
    "cardelli2011_FJ_noGC.ts", "cardelli2011_NM_noGC.ts", "cardelli2013_2D_3I_noGC.ts",
    "cardelli2013_2D_noGC.ts", "chen2013_2D_JF_var1.ts", "chen2013_2D_JF_var2.ts", "lakin2012_3D_var1.ts",
    "qian2011_3D_var1.ts"
]

try:
    print("Testing for Nuskell...")
    import nuskell.dsdcompiler

    fcrn, fs = nuskell.dsdcompiler.crn_parser.parse_crn_string('A + B -> C')

    print("\tHas Nuskell")
    hasNuskell = True
finally:
    pass

try:
    print("Testing for Piperine...")
    import piperine.designer

    print("\tHas Piperine")
    hasPiperine = True
finally:
    pass


def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)


# entry_10 is the CRN
# entry_9 is the traced value
# entry_8 is the rearrangement
# entry_7 is the maclaurin
# entry_6 is the input function
# entry_5 is the nuskell true/false
# entry_4 is the nuskell scheme
# entry_3 is the variable
# entry_2 is the point
# entry_1 is the power
def updateVariables():
    global functionStr
    global variable
    global point
    global power
    global lExpress
    global lFunc
    global hasNuskell
    global useNuskell
    global hasPiperine
    global usePiperine
    global pipOptions
    global scheme
    global verify
    global usingExe

    functionStr = entry_6.get().replace("^", "**")
    variable = entry_3.get()
    point = float(entry_2.get())
    power = int(entry_1.get()) + 1
    lExpress = "lambda " + variable + ": " + functionStr
    # lambda x: sin(x)
    lFunc = eval(lExpress)

    print("-=( + Updated Variables (Local) + )=-")
    print("Function String: " + functionStr)
    print("Variable Selected: " + variable)
    print("Around Point: " + point.__str__())
    print("Degree Est.: " + power.__str__())
    print("Lambda Expression: " + lExpress)
    print("-" * 100)
    print("Has Nuskell: " + hasNuskell.__str__())
    print("Use Nuskell: " + useNuskell.__str__())
    print("Translation Scheme: " + scheme.__str__())
    print("Verify Results: " + verify.__str__())
    print("-" * 100)
    print("Has Piperine: " + hasPiperine.__str__())
    print("Use Piperine: " + usePiperine.__str__())
    print("Piperine CLI Options: " + pipOptions.__str__())
    print("-----------")


def clearEq():
    entry_6.delete(0, END)


def insertVar():
    updateVariables()

    global variable
    entry_6.insert(INSERT, variable)


def insertButton(button):
    if button == "0":
        entry_6.insert(INSERT, "0")
    elif button == "1":
        entry_6.insert(INSERT, "1")
    elif button == "2":
        entry_6.insert(INSERT, "2")
    elif button == "3":
        entry_6.insert(INSERT, "3")
    elif button == "4":
        entry_6.insert(INSERT, "4")
    elif button == "5":
        entry_6.insert(INSERT, "5")
    elif button == "6":
        entry_6.insert(INSERT, "6")
    elif button == "7":
        entry_6.insert(INSERT, "7")
    elif button == "8":
        entry_6.insert(INSERT, "8")
    elif button == "9":
        entry_6.insert(INSERT, "9")
    elif button == "/":
        entry_6.insert(INSERT, "/")
    elif button == "*":
        entry_6.insert(INSERT, "*")
    elif button == "-":
        entry_6.insert(INSERT, "-")
    elif button == "+":
        entry_6.insert(INSERT, "+")
    elif button == "^":
        entry_6.insert(INSERT, "^")
    else:
        entry_6.insert(INSERT, button)
        entry_6.icursor(entry_6.index(INSERT) - 1)  # move cursor back 1


def generateNuskell(messagebox, reactions):
    global verify, scheme, crn

    messagebox.showwarning("Beginning Nuskell Simulation", "Please note that the Nuskell simulation is "
                                                           "beginning and may take a few minutes depending "
                                                           "on how many gates were generated by FUNDNA. "
                                                           "Please report any errors or crashes to our "
                                                           "GitHub page.\n\nALL GENERATED INFORMATION CAN "
                                                           "BE FOUND IN: ~/tests/nuskell\n\nA message will "
                                                           "show when the task finished!")
    print("\n\nBeginning Nuskell Simulation.")
    # GENERATE NUSKELL COMPATIBLE CRN
    print("\nNUSKELL CRN STRING:\n")
    crn = reactions
    inputCRNStr = crn.NuskellString().replace('0.', 'c')
    print(inputCRNStr)

    print("\n\n\nPython Executable: " + sys.executable)

    # Get the path of the Python executable
    #python_executable_path = sys.executable

    # Extract the directory and filename from the path
    #directory, filename = os.path.split(python_executable_path)

    # Replace "python3" with "nuskell" in the filename
    #nuskellModule = filename.replace("python3", "nuskell")

    # Create the new path by joining the directory and the new filename
    #nuskellPath = os.path.join(directory, nuskellModule)
    #print("Nuskell Path: " + nuskellPath + "\n\n\n")

    cwd = os.getcwd()
    print(f"Current working directory: {cwd}")
    nuskellTestPath = os.path.join(cwd, "tests/nuskell")
    nuskellPath = "nuskell"

    # # GENERATE CLI STRING
    # if usingExe:
    #     cmd = ["echo", f'{input_crn}', "|", "nuskell", "--ts", f'{scheme}', "--pilfile", "-vv",
    #            "--enum-detailed", "--enumerate", "--logfile", "nuskellCLI.txt"]
    # else:
    cmd = ["echo", f'"{inputCRNStr}"', "|", f"{nuskellPath}", "--ts",
           f'{scheme}', "--pilfile", "-vv",
           "--enum-detailed", "--enumerate", "--logfile", "nuskellCLI.txt"]

    if verify:
        cmd.append("--verify")
        cmd.append("crn-bisimulation")

    cliString = " ".join(cmd)

    # GENERATE TESTS FOLDER IN ~/TESTS
    if not os.path.exists(nuskellTestPath):
        print("Tests directory does not exist, making one now...")
        os.makedirs(nuskellTestPath)

    # WRITE COMMAND BEING EXECUTED
    with open(nuskellTestPath + "/cli_command.txt", "w+") as file:
        file.write(cliString)

    # OPEN SHELL TERMINAL AND RUN NUSKELL COMMAND
    stream = subprocess.check_output(cliString, shell=True)
    # result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    # stream = result.stdout  # Access the output of the command

    # ENSURE THE NUSKELL-GENERATED FILES ARE IN TESTS FOLDER
    if os.path.exists("domainlevel_enum.pil"):
        os.replace("domainlevel_enum.pil", nuskellTestPath + "/domainlevel_enum.pil")

        with open(nuskellTestPath + "/domainlevel_enum.pil", "r") as file:
            data = file.read()
            data = data.replace("c", "0.")

        with open(nuskellTestPath + "/domainlevel_enum.pil", "w") as file:
            file.write(data)

    if os.path.exists("domainlevel_sys.pil"):
        os.replace("domainlevel_sys.pil", nuskellTestPath + "/domainlevel_sys.pil")

        with open(nuskellTestPath + "/domainlevel_sys.pil", "r") as file:
            data = file.read()
            data = data.replace("c", "0.")

        with open(nuskellTestPath + "/domainlevel_sys.pil", "w") as file:
            file.write(data)

    if os.path.exists("nuskellCLI.txt"):
        os.replace("nuskellCLI.txt", nuskellTestPath + "/nuskellCLI.txt")

        with open(nuskellTestPath + "/nuskellCLI.txt", "r") as file:
            data = file.read()
            data = data.replace("c", "0.")

        with open(nuskellTestPath + "/nuskellCLI.txt", "w") as file:
            file.write(data)

    print("Nuskell Simulation Complete.")

    file_to_open = os.getcwd() + "/tests/nuskell/nuskellCLI.txt"

    if hasattr(os, "startfile"):
        os.startfile(file_to_open)
    elif shutil.which("xdg-open"):
        subprocess.call(["xdg-open", file_to_open])
    elif "EDITOR" in os.environ:
        subprocess.call([os.environ["EDITOR"], file_to_open])

    messagebox.showinfo("Nuskell Simulation Completed!",
                        "Congratulations!\n The Nuskell DSD Simulation has been completed. Please "
                        "open the files in ~/tests/nuskell.")


import subprocess
import sys
import shlex
import re


def CheckPiperineExecutionStatus(output_file, error_string, success_string, polling_interval=5, max_attempts=10):
    """
    Recursively checks the content of an output file for specific strings.

    Parameters:
        - output_file: The path to the output file.
        - error_string: The string indicating an error in the output.
        - success_string: The string indicating successful execution in the output.
        - polling_interval: The interval (in seconds) between each check.
        - max_attempts: The maximum number of attempts before giving up.

    Returns:
        - True if the success string is found.
        - False if the error string is found or the maximum attempts are reached.
    """
    if max_attempts == 0:
        print("Max attempts reached. Giving up.")
        return False

    try:
        with open(output_file, 'r') as file:
            content = file.read()

            if error_string in content:
                print("Error string found in the output. Aborting.")
                return False

            if success_string in content:
                print("Success string found in the output. Continuing.")
                return True

    except FileNotFoundError:
        print(f"Output file not found: {output_file}. Waiting for it to be created.")

    # Wait for the specified interval before the next check
    import time
    time.sleep(polling_interval)

    # Recursive call
    return CheckPiperineExecutionStatus(output_file, error_string, success_string, polling_interval, max_attempts - 1)


def runPiperine(function, cliString, messagebox, output_file):
    global pipOptions
    global crn

    # Print the command for debugging purposes
    print("Executing command:", cliString)

    # Determine the platform
    system = sys.platform.lower()

    if system == 'windows':
        command = ['start', 'cmd', '/k'] + shlex.split(cliString)
    elif system == 'linux':
        command = ['xterm', '-e'] + shlex.split(cliString)
    elif system == 'darwin':
        command = ['osascript', '-e', f'tell app "Terminal" to do script "{cliString} > \'{output_file}\'"']
    else:
        raise OSError("Unsupported operating system")

    # Print the command for debugging purposes
    print("Full command:", " ".join(command))

    # Execute the command
    result = subprocess.run(command, capture_output=True, text=True, check=True)

    import time
    time.sleep(15)

    # Check the command output for debugging
    print("Command output:", result.stdout)

    if system == 'darwin':
        # Activate the Terminal application
        subprocess.run(['osascript', '-e', 'tell app "Terminal" to activate'])

    status = CheckPiperineExecutionStatus(output_file,
                                          "Try target energy",
                                          "Winning sequence set is",
                                          max_attempts=1000,
                                          polling_interval=15)

    if not status:
        # Read and print the content of the output file
        with open(output_file, 'r') as file:
            output_content = file.read()
            print("Command output:", output_content)

        match = re.search(r"Try target energy:(\S+), maxspurious:(\S+), deviation:(\S+),", output_content)

        if match:
            suggested_energy = str(match.group(1))
            suggested_maxspurious = str(match.group(2))
            suggested_deviation = str(match.group(3))

            print(
                f"Suggested parameters: energy={suggested_energy}, maxspurious={suggested_maxspurious}, deviation={suggested_deviation}... trying again...\n\n")

            # Modify the pipOptions string with suggested parameters
            original_candidates_match = re.search(r"--candidates (\d+)", pipOptions)
            original_candidates = str(original_candidates_match.group(1)) if original_candidates_match else "3"

            pipOptions = f"--candidates {original_candidates} --energy {suggested_energy} --maxspurious {suggested_maxspurious} --deviation {suggested_deviation}"

            cliString, crn, input_crn, output_file = DeterminePiperineCLI(function, pipOptions)

            # WRITE COMMAND BEING EXECUTED
            with open("tests/piperine/cli_command.txt", "w+") as file:
                file.write(cliString)

            # Retry Piperine with suggested parameters
            runPiperine(function, cliString, messagebox, output_file)

        else:
            # Display an error message
            messagebox.showerror("Error!",
                                 f"Error!\n\nPlease look at the file in {output_file} to see piperine errors.\n\nFor further assistance, please reach out to us on our GitHub page: https://github.com/CUT-Labs/FUNDNA.")


def DeterminePiperineCLI(reactions, options):
    global crn
    # GENERATE Piperine COMPATIBLE CRN
    # print("\nPIPERINE CRN STRING:\n")
    crn = reactions
    inputCRNStr = crn.PiperineString().replace('0.', 'c')
    # print(input_crn)

    print("\nPython Executable: " + sys.executable)

    # Get the path of the Python executable
    python_executable_path = sys.executable

    # Extract the directory and filename from the path
    directory, filename = os.path.split(python_executable_path)

    # Replace "python3" with "piperine-design" in the filename
    piperineModule = filename.replace("python3", "piperine-design")

    # Create the new path by joining the directory and the new filename
    piperinePath = os.path.join(directory, piperineModule)
    print("Piperine Path: " + piperinePath + "\n")

    # Get crn file path
    # Set the directory path
    current_directory = os.path.abspath(os.path.dirname(__file__))

    # Set the destination folder
    destination_folder = os.path.join(current_directory, "tests/piperine")

    cmd = [f"{piperinePath}", f"\'{destination_folder}/my.crn\'"] + options.split(" ")

    output_file = f"{destination_folder}/cliLog.txt"

    # print(cmd)

    cliString = " ".join(cmd)

    return cliString, crn, inputCRNStr, output_file


def generatePiperine(messagebox, reactions):
    global pipOptions
    global crn

    messagebox.showwarning("Beginning Piperine Simulation", "Please note that the Piperine simulation is "
                                                            "beginning and may take a few minutes depending "
                                                            "on how many gates were generated by FUNDNA. "
                                                            "Please report any errors or crashes to our "
                                                            "GitHub page.\n\nALL GENERATED INFORMATION CAN "
                                                            "BE FOUND IN: ~/tests/piperine\n\nA message will "
                                                            "show when the task finished!")

    print("\n\nBeginning Piperine Simulation.")

    cliString, crn, input_crn, output_file = DeterminePiperineCLI(reactions, pipOptions)

    # Set the directory path
    current_directory = os.path.abspath(os.path.dirname(__file__))

    # Set the destination folder
    destination_folder = os.path.join(current_directory, "tests/piperine")

    # Create the destination folder if it doesn't exist
    os.makedirs(destination_folder, exist_ok=True)

    crn.file_reference = open("tests/piperine/my.crn", "w+")
    crn.file_reference.write(input_crn)
    crn.CloseFile()

    # Execue Piperine
    # WRITE COMMAND BEING EXECUTED
    with open("tests/piperine/cli_command.txt", "w+") as file:
        file.write(cliString)

    runPiperine(reactions, cliString, messagebox, output_file)

    # Move generated files to tests/piperine
    # All files that end in .seq,
    # All files that end in _strands.txt
    # my_score_report.txt
    # my_scores.csv

    # List all files in the current directory
    files = os.listdir(current_directory)

    # Specify the file endings to move
    file_endings_to_move = ['.seq', '_strands.txt', 'my_score_report.txt', 'my_scores.csv']

    for file_name in files:
        # Check if the file ends with any of the specified endings
        if any(file_name.endswith(ending) for ending in file_endings_to_move):
            # Construct the source and destination paths
            source_path = os.path.join(current_directory, file_name)
            destination_path = os.path.join(destination_folder, file_name)

            # Move the file to the destination folder
            shutil.move(source_path, destination_path)

    # Find the best candidate from cliLog.txt
    if os.path.exists("tests/piperine/cliLog.txt"):
        with open("tests/piperine/cliLog.txt", "r") as file:
            data = file.read()

            # Use regex to find the relevant information in the string
            match = re.search(r"Winning sequence set is index (\S)", data)

            if match:
                winningCandidate = int(match.group(1))

                # Present best strands to user
                messagebox.showerror("Success!",
                                     f"Great News!\nPiperine was able to generate DNA strands for this "
                                     f"CRN! Please look in the ./tests/piperine/ folder and look at "
                                     f"strand {winningCandidate}.")
            else:
                #simulation didn't finnish, tere was an error.
                messagebox.showerror("Error!",
                                     "Error!\nPiperine encountered an error in it's execution. "
                                     "Please go to the directory './tests/piperine/', open the "
                                     "terminal, and execute the command provided in cli_command.txt. "
                                     "For further assistance, please reach out to us with information "
                                     "provided on our GitHub page: https://github.com/CUT-Labs/FUNDNA.")

    else:
        #simulation didn't finnish, tere was an error.
        messagebox.showerror("Error!",
                             "Error!\nPiperine encountered an error in it's execution. "
                             "Please go to the directory './tests/piperine/', open the "
                             "terminal, and execute the command provided in cli_command.txt. "
                             "For further assistance, please reach out to us with information "
                             "provided on our GitHub page: https://github.com/CUT-Labs/FUNDNA.")


def calculate():
    global image_4
    global lFunc
    global hasNuskell
    global useNuskell
    global hasPiperine
    global usePiperine

    # Update Variables
    updateVariables()

    print("-=( + Updated Variables (Global) + )=-")
    print("Function String: " + functionStr)
    print("Variable Selected: " + variable)
    print("Contains Variable: " + functionStr.__contains__(variable).__str__())
    print("Around Point: " + point.__str__())
    print("Degree Est.: " + power.__str__())
    print("Lambda Expression: " + lExpress)
    print("Lambda Function" + lFunc.__str__())
    print("-" * 100)
    print("Has Nuskell: " + hasNuskell.__str__())
    print("Use Nuskell: " + useNuskell.__str__())
    print("Translation Scheme: " + scheme.__str__())
    print("Verify Results: " + verify.__str__())
    print("-" * 100)
    print("Has Piperine: " + hasPiperine.__str__())
    print("Use Piperine: " + usePiperine.__str__())
    print("Piperine CLI Options: " + pipOptions.__str__())
    print("-" * 100)

    # Convert to Function Parameter Types
    funcType = FuncTypes.SINUSOIDAL
    if functionStr.__contains__("log"):
        funcType = FuncTypes.LOGARITHMIC
    if functionStr.__contains__("exp"):
        funcType = FuncTypes.EXPONENTIAL
    if functionStr.__contains__("sin") or \
            functionStr.__contains__("cos") or \
            functionStr.__contains__("tan") or \
            functionStr.__contains__("csc") or \
            functionStr.__contains__("cot") or \
            functionStr.__contains__("sec") or \
            functionStr.__contains__("cosh") or \
            functionStr.__contains__("sinh") or \
            functionStr.__contains__("csch") or \
            functionStr.__contains__("coth") or \
            functionStr.__contains__("sech") or \
            functionStr.__contains__("tanh"):
        funcType = FuncTypes.SINUSOIDAL
    else:
        funcType = FuncTypes.POLYNOMIAL

    function = Function(lFunc, point, power, funcType, functionStr, variable)

    print("Function Type: " + function.functype.__str__())

    # Generate Function Results
    function.generateCoeffs()
    if function.rearrangeType != RearrangeType.UNKNOWN:
        function.generateCircuit()
        function.generateReactions()

        # known issue, need to fix LaTeX creation of Maclaurin Series Approx.
        # # Set Labels
        # # convert taylor polynomial coeff dictionary to expression - set label (entry_7)
        # x = "x"
        # expr = "$\displaystyle " + sympy.latex(sympify(function.taylorString)) + "$"
        #
        # # This creates a PNG file and saves there the output of sympy.preview
        # bg_color = "{196, 196, 196}"
        # sp.preview(expr, euler=False, preamble=r"\documentclass{standalone}"
        #                                        r"\usepackage{pagecolor}"
        #                                        r"\definecolor{background}{RGB}" + bg_color +
        #                                        r"\pagecolor{background}"
        #                                        r"\begin{document}",
        #            viewer="file", filename=relative_to_assets("taylor.png"), dvioptions=["-D 1200"])
        # # Open the image as if it were a file. This works only for .ps!
        # img = Image.open(relative_to_assets("taylor.png"))
        # img.load()
        # img = img.resize((393, int((393 * img.size[1] / img.size[0]))), Image.BILINEAR)
        # photo = ImageTk.PhotoImage(img)
        # entry_7.config(image=photo)
        # entry_7.image = photo

        entry_7.delete(0, END)
        entry_7.insert(INSERT, function.taylorString)

        # rearrangement - set label (entry_8)
        entry_8.delete(0, END)
        entry_8.insert(INSERT, function.rearrangeString)

        # trace equation - set label (entry_9)
        function.generateTrace()
        entry_9.delete(0, END)
        entry_9.insert(INSERT, str(function.traceValue))

        # generate crn - set label (entry_10)
        entry_10.delete('1.0', END)
        entry_10.insert(INSERT, function.GUIReactionTable)

        # Update Circuit Diagram
        baseWidth = 908
        baseHeight = 268

        # from svglib.svglib import svg2rlg
        # from reportlab.graphics import renderPM
        # print("Rendering generated circuit diagram...")
        # drawing = svg2rlg("assets/result.svg")
        # print("\t...rendered!")
        # print("Converting to PNG...")
        #
        # with open('assets/result.png', 'wb') as f:
        #     renderPM.drawToFile(drawing, f, fmt="PNG")
        #
        # print("\t...converted!")

        img = Image.open("assets/result.png")

        # adjust size based on width (for bigger circuits)
        wpercent = (baseWidth / float(img.size[0]))
        hsize = int(float(img.size[1]) * float(wpercent))

        # adjust size based on height (for smaller circuits)
        hpercent = (baseHeight / float(img.size[1]))
        wsize = int(float(img.size[0]) * float(hpercent))

        # if the adjusted size is taller than the frame, use size based on height instead
        if hsize > baseHeight:
            img = img.resize((baseHeight, wsize), Image.LANCZOS)
        else:
            img = img.resize((baseWidth, hsize), Image.LANCZOS)

        photo = ImageTk.PhotoImage(img)

        print(f"PNG Rendered Dimensions: {img.size[0]} x {img.size[1]}")

        image_4_updater.config(image=photo)
        image_4_updater.image = photo

        # Run Nuskell Protocol
        if useNuskell is True:
            if hasNuskell is True:
                generateNuskell(messagebox=messagebox, reactions=function.CRN)

            else:  # NUSKELL NOT INSTALLED
                messagebox.showerror("Error! Nuskell Not Installed",
                                     "Error!\nNuskell is not installed. Please follow the "
                                     "installation steps provided in the documentation at "
                                     "https://github.com/CUT-Labs/FUNDNA.")

        # Run Piperine Protocol
        if usePiperine is True:
            if hasPiperine is True:
                generatePiperine(messagebox=messagebox, reactions=function.CRN)

            else:  # PIPERINE NOT INSTALLED
                messagebox.showerror("Error! Piperine Not Installed",
                                     "Error!\nPiperine is not installed. Please follow the "
                                     "installation steps provided in the documentation at "
                                     "https://github.com/CUT-Labs/FUNDNA.")

    else:  # INPUT FUNCTION NOT COMPATIBLE
        messagebox.showerror("UK Function To Circuit Designer",
                             "Error! The function you entered is not supported for these parameters:\n" +
                             "-----------------------------------\n"
                             "Function: " + functionStr.replace("**", "^") + "\n" +
                             "Variable: " + variable + "\n" +
                             "Degree: " + str(power - 1) + "\n" +
                             "Point Estimation: " + str(point))


def TranslationSettingsPopup():
    popup = Toplevel()
    popup.title("DSD and DNA Translation")
    popup.geometry("445x175")

    inner_useNuskell = BooleanVar()
    inner_scheme = StringVar()

    inner_usePiperine = BooleanVar()

    popup_canvas = Canvas(
        popup,
        bg="#DCDDDE",
        height=175,
        width=445,
        bd=0,
        highlightthickness=0,
        relief="ridge"
    )
    popup_canvas.place(x=0, y=0)

    from tkinter import ttk

    schemeDropdown = ttk.Combobox(
        popup,
        values=nuskellSchemes
    )
    schemeDropdown.place(
        x=185,
        y=38,
        width=250,
        height=30
    )
    inner_scheme = scheme
    schemeDropdown.set(inner_scheme)

    entry_pipOptions = Entry(
        popup,
        bd=0,
        bg="#DCDDDE",
        fg="#000716",
        highlightthickness=0,
        font=("BitterRoman ExtraBold", 15)
    )
    entry_pipOptions.place(
        x=185,
        y=100,
        width=250,
        height=30
    )
    entry_pipOptions.insert(INSERT, pipOptions)

    nuskellCheckbutton = tk.Checkbutton(
        popup,
        variable=inner_useNuskell,
        onvalue=True,
        offvalue=False,
        width=25,
    )
    nuskellCheckbutton.place(
        x=185,
        y=8,
        width=25.0,
        height=25.0
    )

    if useNuskell:
        nuskellCheckbutton.select()

    piperineCheckbutton = tk.Checkbutton(
        popup,
        variable=inner_usePiperine,
        onvalue=True,
        offvalue=False,
        width=25,
    )
    piperineCheckbutton.place(
        x=185,
        y=72,
        width=25.0,
        height=25.0
    )

    if usePiperine:
        piperineCheckbutton.select()

    save_button_image = PhotoImage(
        file=relative_to_assets("save_button.png"))
    save_button = Button(
        popup,
        image=save_button_image,
        borderwidth=0,
        highlightthickness=0,
        command=lambda: SaveTranslationConfig(
            inner_useNuskell.get(),
            schemeDropdown.get(),
            inner_usePiperine.get(),
            entry_pipOptions.get(),
            popup),
        relief="flat"
    )
    save_button.place(
        x=10.0,
        y=130.0,
        width=167.0,
        height=38.0
    )

    popup_canvas.create_text(
        10,
        10,
        anchor="nw",
        text="Translate to DSD?",
        fill="#1F2C5E",
        font=("BitterRoman ExtraBold", 20 * -1)
    )

    popup_canvas.create_text(
        10,
        40,
        anchor="nw",
        text="DSD Scheme:",
        fill="#1F2C5E",
        font=("BitterRoman ExtraBold", 20 * -1)
    )

    popup_canvas.create_text(
        10,
        70,
        anchor="nw",
        text="Translate to DNA?",
        fill="#1F2C5E",
        font=("BitterRoman ExtraBold", 20 * -1)
    )

    popup_canvas.create_text(
        10,
        100,
        anchor="nw",
        text="Piperine Options:",
        fill="#1F2C5E",
        font=("BitterRoman ExtraBold", 20 * -1)
    )

    popup.resizable(True, True)
    popup.mainloop()


def SaveTranslationConfig(useN, selectedScheme, useP, pOptions, popup):
    global useNuskell
    global scheme
    global usePiperine
    global pipOptions

    useNuskell = useN
    scheme = selectedScheme

    usePiperine = useP
    pipOptions = pOptions

    print("-" * 100)
    print("Use Nuskell: " + useNuskell.__str__())
    print("Translation Scheme: " + scheme.__str__())
    print("-" * 100)
    print("Use Piperine: " + usePiperine.__str__())
    print("Piperine CLI Options: " + pipOptions.__str__())
    print("-" * 100)

    popup.destroy()


window = Tk()

scheme = StringVar()
scheme = "soloveichik2010.ts"
useNuskell = BooleanVar()

favicon = PhotoImage(file=relative_to_assets("UK logo-white.png"))

# Setting icon of master window
#window.iconphoto(True, favicon)
window.title("UK DNA Function Designer")

window.geometry("1400x750")
window.configure(bg="#DCDDDE")

canvas = Canvas(
    window,
    bg="#DCDDDE",
    height=750,
    width=1400,
    bd=0,
    highlightthickness=0,
    relief="ridge"
)

canvas.place(x=0, y=0)
canvas.create_rectangle(
    0.0,
    0.0,
    1400.0,
    77.77777862548828,
    fill="#1B365D",
    outline="")

canvas.create_rectangle(
    0.0,
    700.0,
    1400.0,
    751.0,
    fill="#4D567C",
    outline="")

canvas.create_rectangle(
    0.0,
    466.6666564941406,
    1400.0,
    700.0,
    fill="#1B365D",
    outline="")

# +
button_image_1 = PhotoImage(
    file=relative_to_assets("button_1.png"))
button_1 = Button(
    image=button_image_1,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: insertButton("+"),
    relief="flat"
)
button_1.place(
    x=1330.0,
    y=637.7777709960938,
    width=52.5,
    height=46.66668701171875
)

# ^
button_image_2 = PhotoImage(
    file=relative_to_assets("button_2.png"))
button_2 = Button(
    image=button_image_2,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: insertButton("^"),
    relief="flat"
)
button_2.place(
    x=1264.0,
    y=638.0,
    width=52.5,
    height=46.66668701171875
)

# -
button_image_3 = PhotoImage(
    file=relative_to_assets("button_3.png"))
button_3 = Button(
    image=button_image_3,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: insertButton("-"),
    relief="flat"
)
button_3.place(
    x=1330.0,
    y=583.3333129882812,
    width=52.5,
    height=46.66668701171875
)

# *
button_image_4 = PhotoImage(
    file=relative_to_assets("button_4.png"))
button_4 = Button(
    image=button_image_4,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: insertButton("*"),
    relief="flat"
)
button_4.place(
    x=1330.0,
    y=528.888916015625,
    width=52.5,
    height=46.66668701171875
)

# /
button_image_5 = PhotoImage(
    file=relative_to_assets("button_5.png"))
button_5 = Button(
    image=button_image_5,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: insertButton("/"),
    relief="flat"
)
button_5.place(
    x=1330.0,
    y=474.4444580078125,
    width=52.5,
    height=46.66668701171875
)

# (-)
button_image_6 = PhotoImage(
    file=relative_to_assets("button_6.png"))
button_6 = Button(
    image=button_image_6,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: insertButton("-"),
    relief="flat"
)
button_6.place(
    x=1198.75,
    y=637.7777709960938,
    width=52.5,
    height=46.66668701171875
)

# 0
button_image_7 = PhotoImage(
    file=relative_to_assets("button_7.png"))
button_7 = Button(
    image=button_image_7,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: insertButton("0"),
    relief="flat"
)
button_7.place(
    x=1133.125,
    y=637.7777709960938,
    width=52.5,
    height=46.66668701171875
)

# 1
button_image_8 = PhotoImage(
    file=relative_to_assets("button_8.png"))
button_8 = Button(
    image=button_image_8,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: insertButton("1"),
    relief="flat"
)
button_8.place(
    x=1133.125,
    y=583.3333129882812,
    width=52.5,
    height=46.66668701171875
)

# 2
button_image_9 = PhotoImage(
    file=relative_to_assets("button_9.png"))
button_9 = Button(
    image=button_image_9,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: insertButton("2"),
    relief="flat"
)
button_9.place(
    x=1198.75,
    y=583.3333129882812,
    width=52.5,
    height=46.66668701171875
)

# 3
button_image_10 = PhotoImage(
    file=relative_to_assets("button_10.png"))
button_10 = Button(
    image=button_image_10,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: insertButton("3"),
    relief="flat"
)
button_10.place(
    x=1264.375,
    y=583.3333129882812,
    width=52.5,
    height=46.66668701171875
)

# 4
button_image_11 = PhotoImage(
    file=relative_to_assets("button_11.png"))
button_11 = Button(
    image=button_image_11,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: insertButton("4"),
    relief="flat"
)
button_11.place(
    x=1133.125,
    y=528.888916015625,
    width=52.5,
    height=46.66668701171875
)

# 5
button_image_12 = PhotoImage(
    file=relative_to_assets("button_12.png"))
button_12 = Button(
    image=button_image_12,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: insertButton("5"),
    relief="flat"
)
button_12.place(
    x=1198.75,
    y=528.888916015625,
    width=52.5,
    height=46.66668701171875
)

# 6
button_image_13 = PhotoImage(
    file=relative_to_assets("button_13.png"))
button_13 = Button(
    image=button_image_13,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: insertButton("6"),
    relief="flat"
)
button_13.place(
    x=1264.375,
    y=528.888916015625,
    width=52.5,
    height=46.66668701171875
)

# 7
button_image_14 = PhotoImage(
    file=relative_to_assets("button_14.png"))
button_14 = Button(
    image=button_image_14,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: insertButton("7"),
    relief="flat"
)
button_14.place(
    x=1133.125,
    y=474.4444580078125,
    width=52.5,
    height=46.66668701171875
)

# 8
button_image_15 = PhotoImage(
    file=relative_to_assets("button_15.png"))
button_15 = Button(
    image=button_image_15,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: insertButton("8"),
    relief="flat"
)
button_15.place(
    x=1198.75,
    y=474.4444580078125,
    width=52.5,
    height=46.66668701171875
)

# 9
button_image_16 = PhotoImage(
    file=relative_to_assets("button_16.png"))
button_16 = Button(
    image=button_image_16,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: insertButton("9"),
    relief="flat"
)
button_16.place(
    x=1264.375,
    y=474.4444580078125,
    width=52.5,
    height=46.66668701171875
)

# exp()
button_image_17 = PhotoImage(
    file=relative_to_assets("button_17.png"))
button_17 = Button(
    image=button_image_17,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: insertButton("exp()"),
    relief="flat"
)
button_17.place(
    x=989.125,
    y=480.0,
    width=112.875,
    height=46.66668701171875
)

# log()
button_image_18 = PhotoImage(
    file=relative_to_assets("button_18.png"))
button_18 = Button(
    image=button_image_18,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: insertButton("log()"),
    relief="flat"
)
button_18.place(
    x=858.75,
    y=480.0,
    width=112.875,
    height=46.66668701171875
)

# tan()
button_image_19 = PhotoImage(
    file=relative_to_assets("button_19.png"))
button_19 = Button(
    image=button_image_19,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: insertButton("tanh()"),
    relief="flat"
)
button_19.place(
    x=728.375,
    y=480.0,
    width=112.875,
    height=46.66668701171875
)

# cos()
button_image_20 = PhotoImage(
    file=relative_to_assets("button_20.png"))
button_20 = Button(
    image=button_image_20,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: insertButton("cos()"),
    relief="flat"
)
button_20.place(
    x=598.0,
    y=480.0,
    width=112.875,
    height=46.66668701171875
)

# sin()
button_image_21 = PhotoImage(
    file=relative_to_assets("button_21.png"))
button_21 = Button(
    image=button_image_21,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: insertButton("sin()"),
    relief="flat"
)
button_21.place(
    x=472.0,
    y=480.0,
    width=112.875,
    height=46.66668701171875
)

# sec()
button_image_22 = PhotoImage(
    file=relative_to_assets("button_22.png"))
button_22 = Button(
    image=button_image_22,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: insertButton("sec()"),
    relief="flat"
)
button_22.place(
    x=472.25,
    y=534.5555419921875,
    width=112.875,
    height=46.66668701171875
)

# sech()
button_image_23 = PhotoImage(
    file=relative_to_assets("button_23.png"))
button_23 = Button(
    image=button_image_23,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: insertButton("sech()"),
    relief="flat"
)
button_23.place(
    x=472.25,
    y=588.5555419921875,
    width=112.875,
    height=46.66668701171875
)

# csc()
button_image_24 = PhotoImage(
    file=relative_to_assets("button_24.png"))
button_24 = Button(
    image=button_image_24,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: insertButton("csc()"),
    relief="flat"
)
button_24.place(
    x=598.25,
    y=534.5555419921875,
    width=112.875,
    height=46.66668701171875
)

# csch()
button_image_25 = PhotoImage(
    file=relative_to_assets("button_25.png"))
button_25 = Button(
    image=button_image_25,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: insertButton("csch()"),
    relief="flat"
)
button_25.place(
    x=598.25,
    y=588.5555419921875,
    width=112.875,
    height=46.66668701171875
)

# coth()
button_image_26 = PhotoImage(
    file=relative_to_assets("button_26.png"))
button_26 = Button(
    image=button_image_26,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: insertButton("coth()"),
    relief="flat"
)
button_26.place(
    x=728.25,
    y=588.5555419921875,
    width=112.875,
    height=46.66668701171875
)

# tanh()
button_image_27 = PhotoImage(
    file=relative_to_assets("button_27.png"))
button_27 = Button(
    image=button_image_27,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: insertButton("tanh()"),
    relief="flat"
)
button_27.place(
    x=859.25,
    y=588.5555419921875,
    width=112.875,
    height=46.66668701171875
)

# cot()
button_image_28 = PhotoImage(
    file=relative_to_assets("button_28.png"))
button_28 = Button(
    image=button_image_28,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: insertButton("cot()"),
    relief="flat"
)
button_28.place(
    x=728.25,
    y=534.5555419921875,
    width=112.875,
    height=46.66668701171875
)

# sinh()
button_image_29 = PhotoImage(
    file=relative_to_assets("button_29.png"))
button_29 = Button(
    image=button_image_29,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: insertButton("sinh()"),
    relief="flat"
)
button_29.place(
    x=859.25,
    y=534.5555419921875,
    width=112.875,
    height=46.66668701171875
)

# cosh()
button_image_30 = PhotoImage(
    file=relative_to_assets("button_30.png"))
button_30 = Button(
    image=button_image_30,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: insertButton("cosh()"),
    relief="flat"
)
button_30.place(
    x=989.25,
    y=534.5555419921875,
    width=112.875,
    height=46.66668701171875
)

# var()
button_image_31 = PhotoImage(
    file=relative_to_assets("button_31.png"))
button_31 = Button(
    image=button_image_31,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: insertVar(),
    relief="flat"
)
button_31.place(
    x=473.25,
    y=643.5555419921875,
    width=112.875,
    height=46.66668701171875
)

# clear
button_image_32 = PhotoImage(
    file=relative_to_assets("button_32.png"))
button_32 = Button(
    image=button_image_32,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: clearEq(),
    relief="flat"
)
button_32.place(
    x=27.0,
    y=474.0,
    width=112.875,
    height=46.66668701171875
)

# calculate()
button_image_33 = PhotoImage(
    file=relative_to_assets("button_33.png"))
button_33 = Button(
    image=button_image_33,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: calculate(),
    relief="flat"
)
button_33.place(
    x=1133.0,
    y=431.0,
    width=267.0,
    height=38.0
)

button_image_34 = PhotoImage(
    file=relative_to_assets("button_34.png"))
button_34 = Button(
    image=button_image_34,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: TranslationSettingsPopup(),
    relief="flat"
)
button_34.place(
    x=224.0,
    y=594.0,
    width=220.0,
    height=59.0
)

# uk logo
image_image_1 = PhotoImage(
    file=relative_to_assets("image_1.png"))
image_1 = canvas.create_image(
    114.0,
    47.0,
    image=image_image_1
)

# nsf logo
image_image_2 = PhotoImage(
    file=relative_to_assets("image_2.png"))
image_2 = canvas.create_image(
    266.0,
    726.0,
    image=image_image_2
)

# divider line
image_image_3 = PhotoImage(
    file=relative_to_assets("image_3.png"))
image_3 = canvas.create_image(
    241.5,
    38.22222137451172,
    image=image_image_3
)

# schemdraw background
image_4_updater = Label(
    bd=0,
    bg="#fff",
    highlightthickness=0,
    #    state="disabled",
    justify="center",
    font=("BitterRoman ExtraBold", 15)
)
image_4_updater.place(
    x=1002,
    y=78,
    width=398.0,
    height=354.0
)

# fundna logo
image_image_5 = PhotoImage(
    file=relative_to_assets("image_5.png"))
image_5 = canvas.create_image(
    420.0,
    38.0,
    image=image_image_5
)

# degree field
entry_image_1 = PhotoImage(
    file=relative_to_assets("entry_1.png"))
entry_bg_1 = canvas.create_image(
    174.75,
    641.7222290039062,
    image=entry_image_1
)
entry_1 = Entry(
    bd=0,
    bg="#DCDDDE",
    fg="#000716",
    highlightthickness=0,
    font=("BitterRoman ExtraBold", 15)
)
entry_1.place(
    x=137.25,
    y=630.4444580078125,
    width=75.0,
    height=20.5555419921875
)
entry_1.insert(INSERT, "5")

# point est field
entry_image_2 = PhotoImage(
    file=relative_to_assets("entry_2.png"))
entry_bg_2 = canvas.create_image(
    174.75,
    605.9444580078125,
    image=entry_image_2
)
entry_2 = Entry(
    bd=0,
    bg="#DCDDDE",
    fg="#000716",
    highlightthickness=0,
    font=("BitterRoman ExtraBold", 15)
)
entry_2.place(
    x=137.25,
    y=594.6666870117188,
    width=75.0,
    height=20.5555419921875
)
entry_2.insert(INSERT, "0")

# var field
entry_image_3 = PhotoImage(
    file=relative_to_assets("entry_3.png"))
entry_bg_3 = canvas.create_image(
    174.375,
    570.0,
    image=entry_image_3
)
entry_3 = Entry(
    bd=0,
    bg="#DCDDDE",
    fg="#000716",
    highlightthickness=0,
    font=("BitterRoman ExtraBold", 15)
)
entry_3.place(
    x=136.875,
    y=559.0,
    width=75.0,
    height=20.0
)
entry_3.insert(INSERT, "x")

# function input field
entry_image_6 = PhotoImage(
    file=relative_to_assets("entry_6.png"))
entry_bg_6 = canvas.create_image(
    566.5,
    449.5,
    image=entry_image_6
)
entry_6 = Entry(
    bd=0,
    bg="#B1C9E8",
    fg="#000716",
    highlightthickness=0,
    font=("BitterRoman ExtraBold", 15)
)
entry_6.place(
    x=0.0,
    y=432.0,
    width=1133.0,
    height=33.0
)
entry_6.insert(INSERT, "exp(-x)")

# maclaurin series field
entry_image_7 = PhotoImage(
    file=relative_to_assets("entry_7.png"))

entry_bg_7 = canvas.create_image(
    223.5,
    164.0,
    image=entry_image_7
)
# known issue, need to fix LaTeX creation of Maclaurin Series Approx.
# entry_7 = Label(
#     bd=0,
#     bg="#C4C4C4",
#     highlightthickness=0,
#     #    state="disabled",
#     justify="center",
#     font=("BitterRoman ExtraBold", 15)
# )

entry_7 = Entry(
    bd=0,
    bg="#C4C4C4",
    fg="#000716",
    highlightthickness=0,
    font=("BitterRoman ExtraBold", 15)
)

entry_7.place(
    x=27.0,
    y=128.0,
    width=393.0,
    height=70.0
)

# rearrangement field
entry_image_8 = PhotoImage(
    file=relative_to_assets("entry_8.png"))
entry_bg_8 = canvas.create_image(
    220.5,
    288.0,
    image=entry_image_8
)
entry_8 = Entry(
    bd=0,
    bg="#C4C4C4",
    fg="#000716",
    highlightthickness=0,
    font=("BitterRoman ExtraBold", 15)
)
entry_8.place(
    x=24.0,
    y=252.0,
    width=393.0,
    height=70.0
)

# traced value field
entry_image_9 = PhotoImage(
    file=relative_to_assets("entry_9.png"))
entry_bg_9 = canvas.create_image(
    347.0,
    374.5,
    image=entry_image_9
)
entry_9 = Entry(
    bd=0,
    bg="#C4C4C4",
    fg="#000716",
    highlightthickness=0,
    font=("BitterRoman ExtraBold", 15)
)
entry_9.place(
    x=279.0,
    y=360.0,
    width=140.0,
    height=31.0
)

# CRN
canvas.create_rectangle(
    444.0,
    121.0,
    975.0,
    417.0,
    fill="#C4C4C4",
    outline="")
entry_10 = ScrolledText(
    # width=64, Scrolled Text Size
    # height=19, Scrolled Text Size
    bd=0,
    bg="#C4C4C4",
    fg="#1F2C5E",
    highlightthickness=0,
    font=("BitterRoman ExtraBold", 13)
    #    state="disabled",
)

entry_10.place(
    x=444.0,
    y=121.0,
    width=531.0,
    height=294.0
)
entry_10.insert(INSERT, "Enter a function in the calculator!")

canvas.create_text(
    296.0,
    715.0,
    anchor="nw",
    text="This project is supported by the National Science Foundation (NSF) and the University of Kentucky.",
    fill="#FFFFFF",
    font=("Caladea Regular", 20 * -1)
)
canvas.create_text(
    41.5,
    558.888916015625,
    anchor="nw",
    text="Variable:",
    fill="#FFFFFF",
    font=("BitterRoman ExtraBold", 20 * -1)
)

canvas.create_text(
    10.0,
    594.6666870117188,
    anchor="nw",
    text="Est. @ Point:",
    fill="#FFFFFF",
    font=("BitterRoman ExtraBold", 20 * -1)
)

canvas.create_text(
    53.0,
    630.4444580078125,
    anchor="nw",
    text="Degree:",
    fill="#FFFFFF",
    font=("BitterRoman ExtraBold", 20 * -1)
)

canvas.create_text(
    224.0,
    558.0,
    anchor="nw",
    text="Translate to DSD or DNA?",
    fill="#FFFFFF",
    font=("BitterRoman ExtraBold", 20 * -1)
)

canvas.create_text(
    27.0,
    90.0,
    anchor="nw",
    text="Maclaurin Series:",
    fill="#1F2C5E",
    font=("BitterRoman ExtraBold", 24 * -1)
)

canvas.create_text(
    27.0,
    215.0,
    anchor="nw",
    text="Rearranged Estimate:",
    fill="#1F2C5E",
    font=("BitterRoman ExtraBold", 24 * -1)
)

canvas.create_text(
    24.0,
    361.0,
    anchor="nw",
    text="Traced Value at Point:",
    fill="#1F2C5E",
    font=("BitterRoman ExtraBold", 24 * -1)
)

canvas.create_text(
    444.0,
    90.0,
    anchor="nw",
    text="Chemical Reaction Network (CRN):",
    fill="#1F2C5E",
    font=("BitterRoman ExtraBold", 24 * -1)
)

def MainGUI():
    window.resizable(True, True)
    window.mainloop()


def HasCRN():
    # Function to check if the user has a CRN
    answer = messagebox.askyesno("CRN Check", "Do you already have a CRN?")

    return answer  # True if they do


def CRNPopup():
    def parseInput(n=False, p=False):
        crn_text = text_widget.get("1.0", tk.END).strip()  # Get the CRN from the ScrolledText widget
        if not crn_text:
            messagebox.showerror("Error", "Please enter a CRN.")
            return

        crn_lines = crn_text.split("\n")
        crn_instance = CRN(crn_array=crn_lines)

        if n:
            generate_nuskell(crn_instance)
        if p:
            generate_piperine(crn_instance)

        popup.destroy()

    def generate_nuskell(input_crn):
        print("Generating Nuskell for CRN:", input_crn.crn_array)
        generateNuskell(messagebox, input_crn)

    def generate_piperine(input_crn):
        # Add your logic for generating Piperine here
        print("Generating Piperine for CRN:", input_crn.crn_array)
        generatePiperine(messagebox, input_crn)

    def open_github_link(event):
        import webbrowser
        webbrowser.open_new("https://github.com/CUTLabs/FUNDNA")

    popup = tk.Toplevel()
    popup.title("CRN Popup")

    # Centered heading text
    heading_label = tk.Label(popup, text="Input your Chemical Reaction Network (CRN):", font=("Helvetica", 12, "bold"))
    heading_label.pack(pady=10)

    # ScrolledText for CRN
    text_widget = ScrolledText(popup, wrap=tk.WORD, width=60, height=20)
    text_widget.pack(padx=20, pady=10)

    # Footnote with clickable link
    footnote_text = (
        "Please separate reactions by new lines. Forward reactions are indicated with ' -> ' and reversible\n"
        "reactions are indicated as ' <=> '. If any errors occur, please reach out to us at our GitHub:"
    )

    footnote_label = tk.Label(popup, text=footnote_text)
    footnote_label.pack(pady=5, padx=10)

    github_link = tk.Label(
        popup,
        text="https://github.com/CUT-Labs/FUNDNA",
        fg="#1E8AFF",
        cursor="hand2",
        font=("Helvetica", 12, "underline")
    )
    github_link.pack(pady=5)
    github_link.bind("<Button-1>", open_github_link)

    # Buttons for generating Nuskell and Piperine (placed horizontally)
    button_frame = tk.Frame(popup)
    button_frame.pack(pady=10)

    nuskell_button = tk.Button(button_frame, text="Generate Nuskell", command=lambda: parseInput(n=True))
    nuskell_button.pack(side=tk.LEFT, padx=5)

    piperine_button = tk.Button(button_frame, text="Generate Piperine", command=lambda: parseInput(p=True))
    piperine_button.pack(side=tk.LEFT, padx=5)

    # Start the Tkinter event loop for the popup window
    popup.mainloop()

def StartMenu():
    try:
        pyi_splash.close()
    finally:
        if HasCRN():
            CRNPopup()
        else:
            MainGUI()


StartMenu()