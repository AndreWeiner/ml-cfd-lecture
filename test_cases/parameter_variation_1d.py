#!/usr/bin/python3

from os import environ, makedirs, system
from os.path import join
from shutil import copytree
from typing import Tuple, List
from subprocess import Popen
from multiprocessing import Pool
import torch as pt


def create_modified_copy(base_case: str, new_case: str,
                         to_replace: Tuple[Tuple[str, str, str]]):
    """Copy a base case and modify parameters

    :param base_case: path to base case
    :type base_case: str
    :param new_case: path to new case
    :type new_case: str
    :param to_replace: tuple of files, to modify, text to replace,
        and new text
    :type to_replace: Tuple[Tuple[str, str, str]]
    """
    copytree(base_case, new_case, dirs_exist_ok=True)
    for tup in to_replace:
        f, old, new = tup
        cmd = f"sed -i 's/{old}/{new}/' {join(new_case, f)}"
        system(cmd)


def run_simulation(path: str):
    """Execute a simulation by running the *Allrun* script.

    :param path: path to simulation folder
    :type path: str
    """
    return_code = Popen(["./Allrun"], cwd=path).wait()
    if return_code == 0:
        print(f"Simulation {path} completed successfully.")
    else:
        print(f"Warning: simulation {path} failed.")


def lhs_sampling(x_min: List[float], x_max: List[float],
                 n_samples: int) -> pt.Tensor:
    """Latin hypercube sampling

    :param x_min: lower bounds for each parameter
    :type x_min: List[float]
    :param x_max: upper bounds for each parameter
    :type x_max: List[float]
    :param n_samples: total number of samples
    :type n_samples: int
    :return: tensor with sampled parameters; each column
        form a parameter combintation
    :rtype: pt.Tensor
    """
    assert len(x_min) == len(x_max)
    n_parameters = len(x_min)
    samples = pt.zeros((n_parameters, n_samples))
    for i, (lower, upper) in enumerate(zip(x_min, x_max)):
        bounds = pt.linspace(lower, upper, n_samples+1)
        rand = bounds[:-1] + pt.rand(n_samples) * (bounds[1:]-bounds[:-1])
        samples[i, :] = rand[pt.randperm(n_samples)]
    return samples


def main():
    print("Starting parameter variation")

    # definition of basse case and target folder
    ML_CFD_BASE = environ.get("ML_CFD_BASE", "")
    if ML_CFD_BASE == "":
        raise ValueError(
            "Environment variable 'ML_CFD_BASE' not defined; run\n" +
            "   source setup-env\n" +
            "before executing this script"
        )
    base_simulation = join(ML_CFD_BASE, "test_cases", "boundary_layer_1D")
    base_parameter_study = join(ML_CFD_BASE, "exercises", "boundary_layer_1D_variation")

    # creating copies of the base case with modified properties
    makedirs(base_parameter_study, exist_ok=True)
    ubar = lhs_sampling([0.1], [1.0], 16).squeeze()
    cases = []
    for ub in ubar:
        replace = (
            ("system/controlDict", "^endTime.*", "endTime" + " "*9 + "{:1.0f};".format(50.0/ub.item())),
            ("system/controlDict", "^deltaT.*", "deltaT" + " "*10 + "{:1.1e};".format(1.0e-3/ub.item())),
            ("system/fvOptions", ".*Ubar.*", " "*4 + "Ubar" + " "*12 + "({:1.4f} 0 0);".format(ub.item()))
        )
        new_case = join(base_parameter_study, "Ub_{:1.4f}".format(ub.item()))
        create_modified_copy(base_simulation, new_case, replace)
        cases.append(new_case)
    
    # run all cases using 8 workers operating in parallel
    pool = Pool(8)
    with pool:
        pool.map(run_simulation, cases)

    print("Parameter variation finished")


if __name__ == "__main__":
    main()
