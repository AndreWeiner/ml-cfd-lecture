# Machine learning in computational fluid dynamics

This repository contains resources accompanying the lecture [machine learning in computational fluid dynamics](https://www.tu-braunschweig.de/en/ism/teaching/courses/fluid-mechanics/translate-to-english-maschinelles-lernen-in-der-numerischen-stroemungsmechanik) provided by the Institute of Fluid Mechanics at TU Braunschweig. **Note that slides, notebooks, and other resources will be regularly updated throughout the term.**

## Lectures

| # | topic | slides | notebook |
|--:|:------|:------:|:---------|
| 1 | Course overview and motivation | [link](https://andreweiner.github.io/ml-cfd-slides/lecture_1.html) | [view online](https://nbviewer.org/github/AndreWeiner/ml-cfd-lecture/blob/main/notebooks/lecture_1.ipynb) |
| 2 | Finite-volume-based CFD in a nutshell | [link](https://andreweiner.github.io/ml-cfd-slides/lecture_2.html) | [view online](https://nbviewer.org/github/AndreWeiner/ml-cfd-lecture/blob/main/notebooks/lecture_2.ipynb) |
| 3 | Introduction to machine learning | [link](https://andreweiner.github.io/ml-cfd-slides/lecture_3.html) | [view online](https://nbviewer.org/github/AndreWeiner/ml-cfd-lecture/blob/main/notebooks/lecture_3.ipynb) |
| 4 | Predicting the stability regime of rising bubbles | [link](https://andreweiner.github.io/ml-cfd-slides/lecture_4.html) | [view online](https://nbviewer.org/github/AndreWeiner/ml-cfd-lecture/blob/main/notebooks/lecture_4.ipynb) |
| 5 | Computing highly accurate mass transfer at rising bubbles I | [link](https://andreweiner.github.io/ml-cfd-slides/lecture_5_6.html) |[view online](https://nbviewer.org/github/AndreWeiner/ml-cfd-lecture/blob/main/notebooks/lecture_5_6.ipynb) |
| 6 | Computing highly accurate mass transfer at rising bubbles II | [link](https://andreweiner.github.io/ml-cfd-slides/lecture_5_6.html) | [view online](https://nbviewer.org/github/AndreWeiner/ml-cfd-lecture/blob/main/notebooks/lecture_5_6.ipynb) |
| 7 | Analyzing coherent structures in flows displaying transonic buffets I | [link](https://andreweiner.github.io/ml-cfd-slides/lecture_7_8.html) | [view online](https://nbviewer.org/github/AndreWeiner/ml-cfd-lecture/blob/main/notebooks/lecture_7_8.ipynb) |
| 8 | Analyzing coherent structures in flows displaying transonic buffets II | [link](https://andreweiner.github.io/ml-cfd-slides/lecture_7_8.html) | [view online](https://nbviewer.org/github/AndreWeiner/ml-cfd-lecture/blob/main/notebooks/lecture_7_8.ipynb) |
| 9 | Reduced-order modeling of the flow past a cylinder | [link](https://andreweiner.github.io/ml-cfd-slides/lecture_9.html) | [view online](https://nbviewer.org/github/AndreWeiner/ml-cfd-lecture/blob/main/notebooks/lecture_9.ipynb) |
| 10 | Controlling the flow past a cylinder I | [link](https://andreweiner.github.io/ml-cfd-slides/lecture_10_11.html) | [view online](https://nbviewer.org/github/AndreWeiner/ml-cfd-lecture/blob/main/notebooks/lecture_10_11.ipynb) |
| 11 | Controlling the flow past a cylinder II | [link](https://andreweiner.github.io/ml-cfd-slides/lecture_10_11.html) | [view online](https://nbviewer.org/github/AndreWeiner/ml-cfd-lecture/blob/main/notebooks/lecture_10_11.ipynb) |

## Exercises

### Prerequisites

The exercises are designed for native Linux operating systems like Ubuntu (recommended). They may also work on Windows Subsystem for Linux (WSL). To set up your system for the exercises, refer to the notebook accompanying exercise session 1.

### Exercise sessions

| # | topic | slides | notebook |
|--:|:------|:------:|:---------|
| 1 | Setting up your system | - | [view online](https://nbviewer.org/github/AndreWeiner/ml-cfd-lecture/blob/main/notebooks/exercise_1.ipynb) |
| 2 | End-to-end simulations in OpenFOAM and Basilisk | - | [view online](https://nbviewer.org/github/AndreWeiner/ml-cfd-lecture/blob/main/notebooks/exercise_2.ipynb) |
| 3 | End-to-end machine learning project in PyTorch | - | [view online](https://nbviewer.org/github/AndreWeiner/ml-cfd-lecture/blob/main/notebooks/exercise_3.ipynb) |
| 4 | Building a robust path regime classification model in PyTorch | - | [view online](https://nbviewer.org/github/AndreWeiner/ml-cfd-lecture/blob/main/notebooks/exercise_4.ipynb)|
| 5 | Generation and processing of training data for regression | - | [view online](https://nbviewer.org/github/AndreWeiner/ml-cfd-lecture/blob/main/notebooks/exercise_5_6.ipynb) |
| 6 | Model creation in PyTorch and mass transfer simulations in OpenFOAM | - | [view online](https://nbviewer.org/github/AndreWeiner/ml-cfd-lecture/blob/main/notebooks/exercise_5_6.ipynb) |
| 7 | Buffet simulation and data extraction | - | [view online](https://nbviewer.org/github/AndreWeiner/ml-cfd-lecture/blob/main/notebooks/exercise_7_8.ipynb) |
| 8 | Dynamic mode decomposition of buffeting flow | - | [view online](https://nbviewer.org/github/AndreWeiner/ml-cfd-lecture/blob/main/notebooks/exercise_7_8.ipynb) |
| 9 | Creating a reduced-order model in flowTorch | - | [view online](https://nbviewer.org/github/AndreWeiner/ml-cfd-lecture/blob/main/notebooks/exercise_9.ipynb) |
| 10 | Open-loop control of the flow past a cylinder in OpenFOAM | - | [view online](https://nbviewer.org/github/AndreWeiner/ml-cfd-lecture/blob/main/notebooks/exercise_10_11.ipynb) |
| 11 | Closed-loop control of the flow past a cylinder with OpenFOAM and PyTorch | - | [view online](https://nbviewer.org/github/AndreWeiner/ml-cfd-lecture/blob/main/notebooks/exercise_10_11.ipynb) |

## Getting and providing help and feedback

If you
- get stuck solving an exercise problem
- have technical issues
- have theoretical questions about math or programming
- think that some instructions or explanations might need improvement
- want to report typos or logical errors
- want to provide feedback and suggestions about the course

the easiest way to get in touch is to open a [new issue](https://github.com/AndreWeiner/ml-cfd-lecture/issues/new) in this repository. Before opening a new issue, please use the **search function** to see if a related issue was reported previously. It also helps greatly if you label your issue using one or more of the predefined labels (lecture, exercise, OpenFOAM, ...), and if you take some time to state your problem as clearly as possible. **Note that everyone is welcome to participate in discussion and solving issues.**

If you are a student at TU Braunschweig enrolled in the course *Machine Learning in Computational Fluid Dynamics*, you may also get in touch via the [studIP](https://studip.tu-braunschweig.de/dispatch.php/course/overview?cid=f79375e64fd07fe6606d810ab17496e7) platform or via mail. However, the issue-workflow described above is the preferred method.

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