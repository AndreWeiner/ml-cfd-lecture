"""
    This file to run trajectory, hence handling OpenFOAM files and executing them in machine

    called in : replay_buffer.py
"""

import _thread
import os
import queue
import subprocess
import time

import numpy as np


class env:
    """
        This Class is to run trajectory, hence handling OpenFOAM files and executing them in machine
    """
    def __init__(self, n_worker, buffer_size):
        """

        Args:
            n_worker: no of trajectories at the same time (worker)
            buffer_size: total number of trajectories
        """
        self.n_worker = n_worker
        self.buffer_size = buffer_size

    def process_waiter(self, proc, job_name, que):
        """
             This method is to wait for the executed process till it is completed
         """
        try:
            proc.wait()
        finally:
            que.put((job_name, proc.returncode))

    def run_trajectory(self, buffer_counter, proc, results, sample, action_bounds):
        """
        To run the trajectories

        Args:
            buffer_counter: which trajectory to run (n -> traj_0, traj_1, ... traj_n)
            proc: array to hold process waiting flag
            results: array to hold process finish flag
            sample: number of iteration of main ppo
            action_bounds: min and max omega value

        Returns: execution of OpenFOAM Allrun file in machine

        """
        # some hardcoded trajectory settings
        core_count = 2

        # make dir for new trajectory
        traj_path = f"./env/sample_{sample}/trajectory_{buffer_counter}"

        print(f"\n starting trajectory : {buffer_counter} \n")
        os.makedirs(traj_path, exist_ok=True)
        # copy files form base_case
        # change of ending time -> system/controlDict
        os.popen(
            f'cp -r ./env/base_case/agentRotatingWallVelocity/* {traj_path}/ &&'
            f'sed -i "s/timeStart.*/timeStart       4.01;/g" {traj_path}/system/controlDict &&'
            f'sed -i "/^endTime/ s/endTime.*/endTime         5.0;/g" {traj_path}/system/controlDict'
        )
        
        for i in range(core_count):
            os.popen(
                f'sed -i "s/startTime.*/startTime       4.009999;/g" {traj_path}/processor{i}/4/U &&'
                f'sed -i "s/absOmegaMax.*/absOmegaMax       {action_bounds[1]};/g" {traj_path}/processor{i}/4/U'
            )

        # executing Allrun to start trajectory
        proc[buffer_counter] = subprocess.Popen(['./Allrun.singularity'], cwd=f'{traj_path}/')
        _thread.start_new_thread(self.process_waiter,
                                 (proc[buffer_counter], f"trajectory_{buffer_counter}", results))

    def sample_trajectories(self, sample, action_bounds):
        """

        Args:
            sample: main ppo iteration counter
            action_bounds: min and max omega value

        Returns: execution of n number of trajectory (n = buffer_size)

        """
        # set the counter to count the numbre of trajectory
        buffer_counter = 0

        # list for the status of trajectory running or finished
        proc = []

        # set the n_workers
        for t in range(int(max(self.buffer_size, self.n_worker))):
            item = "proc_" + str(t)
            proc.append(item)

        # get status of trajectory
        results = queue.Queue()
        process_count = 0

        # execute the n = n_workers trajectory simultaneously
        for n in np.arange(self.n_worker):
            self.run_trajectory(buffer_counter, proc, results, sample, action_bounds)
            process_count += 1
            # increase the counter of trajectory number
            buffer_counter += 1

        # check for any worker is done. if so give next trajectory to that worker
        while process_count > 0:
            job_name, rc = results.get()
            print("job : ", job_name, "finished with rc =", rc)
            if self.buffer_size > buffer_counter:
                self.run_trajectory(buffer_counter, proc, results, sample, action_bounds)
                process_count += 1
                buffer_counter += 1
            process_count -= 1


if __name__ == "__main__":
    n_worker = 2
    buffer_size = 2
    sample = 0
    env = env(n_worker, buffer_size)
    env.sample_trajectories(sample)
