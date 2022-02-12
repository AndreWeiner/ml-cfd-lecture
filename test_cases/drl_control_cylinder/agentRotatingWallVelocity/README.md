# agentRotatingWallVelocity

## Files and folders

- **Make**: files and options to compile the boundary condition with *wmake*
- **agentRotatingWallVelocityFvPatchVectorField.***: source files
- **test**: cylinder test case demonstrating how to apply the boundary condition
- **create_test_policy.py**: script to create a random policy for testing
- **read_example.py**: script demonstrating how to read *trajectory.csv* and *coefficients.dat* with Python Pandas

## Compiling the boundary condition
- If you're on a cluster, the first step is to load the singularity module
```
module load singularity/3.6.0rc2
```
- Start a Singularity shell, source the OF environment variables, navigate to the source files, and compile
```
# top level folder of repository
singularity shell of_v2012.sif
# now we are operating from inside the container
source /usr/lib/openfoam/openfoam2012/etc/bashrc
cd DRL_py/agentRotatingWallVelocity/
wmake
```
- the test simulation expects the library file *libAgentRotatingWallVelocity.so* to be in the parent folder; if this behavior needs adaptation, change the path in *test/system/controlDict*
```
libs ("../libAgentRotatingWallVelocity.so");
```

## Running a simulation

The test simulation can be started by executing the *Allrun* script. The parameters in the boundary dictionary are:
```
cylinder
    {
        type            agentRotatingWallVelocity;
        // center of cylinder
        origin          (0.2 0.2 0.0);
        // axis of rotation; normal to 2D domain
        axis            (0 0 1);
        // name of the policy network; must be a torchscript file
        policy          "policy.pt";
        // when to start controlling
        startTime       0.01;
        // how often to evaluate policy
        interval        20;
        // if true, the angular velocity is sampled from a Gaussian distribution
        // if false, the mean value predicted by the policy is used
        train           true;
        // currently ignored
        absOmegaMax     0.05;
        // limit log of standard deviation
        logStdMax       2.0;
    }
```
The boundary condition is currently not fully implemented. E.g., **restart is not supported**.
