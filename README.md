![funding_logo](logo_stiftung_hochschullehre.png)

This lecture was funded previously by the [foundation for innovation in higher education](https://stiftung-hochschullehre.de/).

# Machine learning in computational fluid dynamics

This repository contains resources accompanying the lecture [machine learning in fluid dynamics](https://tu-dresden.de/ing/maschinenwesen/ism/psm/studium/lehrveranstaltungen/maschinelles-lernen-in-der-stroemungsmechanik/index) provided by the Institute of Fluid Mechanics at TU Dresden. **Note that slides, notebooks, and other resources will be regularly updated throughout the term.**

## Lectures

If equations in the lecture notebooks do not get rendered properly on Github, download the notebook and open it using `jupyter-lab` (refer to the first exercise session for an overview of dependencies and installation instructions).

| # | topic | slides | notebook |
|--:|:------|:------:|:---------|
| 1 | Course overview and motivation | [link](https://andreweiner.github.io/ml-cfd-slides/ml_cfd_intro.html) | [view](./notebooks/ml_cfd_intro.ipynb) |
| 2 | Finite-volume-based simulations in a nutshell | [link](https://andreweiner.github.io/ml-cfd-slides/cfd_intro.html) | [view](./notebooks/cfd_intro.ipynb) |
| 3 | Introduction to machine learning | [link](https://andreweiner.github.io/ml-cfd-slides/ml_intro.html) | [view](./notebooks/ml_intro.ipynb) |
| 4 | Surrogate modeling for discrete predictions | [link](https://andreweiner.github.io/ml-cfd-slides/bubble_path_classification.html) | [view](./notebooks/bubble_path_classification.ipynb) |
| 5 | Surrogate modeling for continuous predictions | [link](https://andreweiner.github.io/ml-cfd-slides/mass_transfer_regression.html) |[view](./notebooks/mass_transfer_regression.ipynb) |
| 6 | Analyzing coherent structures| [link](https://andreweiner.github.io/ml-cfd-slides/coherent_structures_dim_reduction.html) | [view](./notebooks/coherent_structures_dim_reduction.ipynb) |
| 7 | Reduced-order modeling of flow fields | [link](https://andreweiner.github.io/ml-cfd-slides/cylinder_rom.html) | [view](./notebooks/cylinder_rom.ipynb) |
| 8 | Optimal open-loop control | [link](https://andreweiner.github.io/ml-cfd-slides/cylinder_bayesian_opt.html) | [view](./notebooks/cylinder_bayesian_opt.ipynb) |
| 9 | Closed-loop control using DRL | [link](https://andreweiner.github.io/ml-cfd-slides/cylinder_drl.html) | [view](./notebooks/cylinder_drl.ipynb) |

## Exercises

### Prerequisites

The exercises are designed for native Linux operating systems like Ubuntu (recommended). They may also work on Windows Subsystem for Linux (WSL). To set up your system for the exercises, refer to the notebook accompanying exercise session 1.

### Exercise sessions

| # | topic | notebook |
|--:|:------|:---------|
| 0 | Course-specific Python refresher | [view](./notebooks/python_intro.ipynb) |
| 1 | Setting up your system | [view](./notebooks/system_setup.ipynb) |
| 2 | End-to-end simulations in OpenFOAM and Basilisk | [view](./notebooks/cfd_intro_exercise.ipynb) |
| 3 | End-to-end machine learning project in PyTorch | [view](./notebooks/ml_intro_exercise.ipynb) |
| 4 | Building a robust path regime classification model | [view](./notebooks/bubble_path_classification_exercise.ipynb)|
| 5 | Computing highly accurate mass transfer at rising bubbles | [view](./notebooks/mass_transfer_regression_exercise.ipynb) |
| 6 | Analyzing coherent structures with POD and DMD| [view](./notebooks/coherent_structures_dim_reduction_exercise.ipynb) |
| 7 | Creating a reduced-order model using CNM | [view](./notebooks/cylinder_rom_exercise.ipynb) |
| 8 | Optimal open-loop control of the flow past a cylinder| [view](./notebooks/cylinder_bayesian_opt_exercise.ipynb) |
| 9 | Closed-loop control of the flow past a cylinder | [view](./notebooks/cylinder_drl_exercise.ipynb) |

## Datasets

Both exercises and lectures sometimes require datasets. Usually, there are instructions how to create or extract the data yourself. For convenience, a downloadable snapshot of the [latest data (20. Dec 2021)](https://cloudstorage.tu-braunschweig.de/getlink/fiYPP9HwVaypRziqMCjZuQVx/datasets_20_Dec_2021.zip) is provided, too.

## Getting and providing help and feedback

If you
- get stuck solving an exercise problem
- have technical issues
- have theoretical questions about math or programming
- think that some instructions or explanations need improvement
- want to report typos or logical errors
- want to provide feedback and suggestions about the course

the easiest way to get in touch is to open a [new issue](https://github.com/AndreWeiner/ml-cfd-lecture/issues/new) in this repository. Before opening a new issue, please use the **search function** to see if a related issue was reported previously. 

If you are a student at TU Dresden enrolled in the course *Machine Learning in Fluid Dynamics*, you may also get in touch via the [OPAL](https://bildungsportal.sachsen.de/opal/auth/RepositoryEntry/41285910529/CourseNode/84033834447509?6) platform or via mail.

## Glossary

The following list of acronyms may help you when exploring notebooks and slides:

- **CFD** - computational fluid dynamics
- **CNM** - cluster-based network modeling
- **DL** - deep learning
- **DRL** - deep reinforcement learning
- **GPU** - graphics processing unit
- **IEEE** - Institute of Electrical and Electronics Engineers
- **IEEE 754** - IEEE standard for floating-point arithmetics
- **JIT** - just in time (compiler)
- **LES** - large eddy simulation
- **LHS** - latin hypercube sampling
- **MAE** - mean absolute error
- **ML** - machine learning
- **MPI** - message passing interface
- **MSE** - mean squared error
- **PINN** - physics-informed neural network
- **RANS** - Reynolds-averaged Navier Stokes
- **RL** - reinforcement learning
- **TPU** - tensor processing unit

## References and other resources

### Book recommendations

- books for computational fluid dynamics
  - [The OpenFOAM technology primer](https://zenodo.org/record/4630596#.YXBgepuxVH4) by T. Marić, J. Höpken, and K. G. Mooney
  - *The finite volume method in computational fluid dynamics* by F. Moukalled, L. Mangani, and M. Darwish
  - *An introduction to computational fluid dynamics: the finite volume method* by H. K. Versteeg and W. Malalasekera
- books for linear algebra
  - *Introduction to linear algebra* by G. Strang
- books for data-driven modelling and control
  - *Data-driven science and engineering: machine learning, dynamical systems, and control* by S. L. Brunton and J. N. Kutz
  - *Dynamic mode decomposition: data-driven modeling of complex systems* by J. N. Kutz, S. L. Brunton, B. W. Brunton, and J. L. Proctor
  - *Grokking deep reinforcement learning* by M. Morales
  - *Deep learning with PyTorch* by E. Stevens, L. Antiga, and T. Viehmann
- books for programming
  - *Python crash course* by E. Matthes
  - *C++ crash course: a fast-paced introduction* by J. Lospinoso
  - [The Linux command line](https://linuxcommand.org/tlcl.php) by W. Shotts

### Video content

- [YouTube channel of Steve Brunton](https://www.youtube.com/c/Eigensteve)
- [YouTube channel of József Nagy](https://www.youtube.com/channel/UCjdgpuxuAxH9BqheyE82Vvw)
- [YouTube channel Fluid Mechanics 101](https://www.youtube.com/channel/UCcqQi9LT0ETkRoUu8eYaEkg)