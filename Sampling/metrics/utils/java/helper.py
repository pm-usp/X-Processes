import json
import subprocess
from typing import Dict, Any
import os


def _run_bpmn_calculator(metric: str, file_path: str) -> Dict[str, Any]:
    """
    Internal helper to run the BPMNCalculator and return the parsed JSON result.

    This function constructs and executes a command to call the external
    `jbpt_mini.jar` library. It captures the output, handles potential
    errors, and parses the resulting JSON string into a Python dictionary.

    Args:
        metric (str): The name of the metric to calculate (e.g., "STRUCTUREDNESS").
        file_path (str): The path to the BPMN model file.

    Returns:
        Dict[str, Any]: A dictionary containing the parsed JSON response from
        the Java process.

    Raises:
        subprocess.CalledProcessError: If the external Java process returns an
            error or writes to stderr.
        json.JSONDecodeError: If the output from the Java process is not valid JSON.
    """

    helper_script_dir = os.path.dirname(os.path.abspath(__file__))
    jar_file = os.path.join(helper_script_dir, 'jbpt_mini', 'jbpt_mini.jar')
    lib_folder = os.path.join(helper_script_dir, 'jbpt_mini', 'lib', '*')
    class_path = f"{jar_file}{os.pathsep}{lib_folder}"

    base_cmd = [
        "java",
        "-cp",
        class_path,
        "bpmn.BPMNCalculator",
        metric,
        file_path,
    ]

    result = subprocess.run(
        base_cmd,
        capture_output=True,
        text=True,
        check=False
    )

    if result.stderr:
        raise subprocess.CalledProcessError(
            returncode=result.returncode,
            cmd=result.args,
            stderr=result.stderr
        )
    #endif

    # Try to parse and return the JSON output
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as e:
        # Raise a new exception that includes the problematic text
        raise json.JSONDecodeError(
            f"Failed to decode JSON: '{result.stdout}'",
            e.doc,
            e.pos
        )
    #endtry
#enddef

def calculate_structuredness(file_path: str) -> Dict[str, Any]:
    """
    Calculates the structuredness of a BPMN model.

    Args:
        file_path (str): The path to the BPMN model file to be analyzed.

    Returns:
        Dict[str, Any]: A dictionary containing the structuredness metric results.

    Raises:
        subprocess.CalledProcessError: If the underlying Java process fails.
        json.JSONDecodeError: If the process output is not valid JSON.
    """
    return _run_bpmn_calculator("STRUCTUREDNESS", file_path)
#enddef

def calculate_separability(file_path: str) -> Dict[str, Any]:
    """
    Calculates the separability of a BPMN model.

    Args:
        file_path (str): The path to the BPMN model file to be analyzed.

    Returns:
        Dict[str, Any]: A dictionary containing the separability metric results.

    Raises:
        subprocess.CalledProcessError: If the underlying Java process fails.
        json.JSONDecodeError: If the process output is not valid JSON.
    """
    return _run_bpmn_calculator("SEPARABILITY", file_path)
#enddef