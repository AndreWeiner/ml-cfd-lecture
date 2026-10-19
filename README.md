# Machine Learning in Fluid Mechanics

**[Read the course online](https://andreweiner.github.io/ml-cfd-lecture/)** — lecture notes, exercises, and links to slides.

This repository contains resources accompanying the course [Machine Learning in Fluid Mechanics](https://tu-dresden.de/ing/maschinenwesen/ism/psm/studium/lehrveranstaltungen/maschinelles-lernen-in-der-stroemungsmechanik/index) offered by the Institute of Fluid Mechanics at TU Dresden. **Note that slides, Python chapters, and other resources will be regularly updated throughout the term.**

![Funding logo](logo_stiftung_hochschullehre.png#gh-light-mode-only)
![Funding logo](logo_stiftung_hochschullehre_white.svg#gh-dark-mode-only)

This course previously received funding from the [Foundation for Innovation in Higher Education](https://stiftung-hochschullehre.de/).

## Lectures and exercises

| # | Topic | Lecture | Exercise | Slides |
|--:|:------|:--------|:---------|:-------|
| 0 | How to learn with generative AI | [Read](https://andreweiner.github.io/ml-cfd-lecture/notebooks/lecture_00.html) | — | — |
| 1 | Course overview and motivation | [Read](https://andreweiner.github.io/ml-cfd-lecture/notebooks/ml_cfd_intro.html) | [Exercises](https://andreweiner.github.io/ml-cfd-lecture/notebooks/system_setup.html) | [Slides](https://andreweiner.github.io/ml-cfd-slides/ml_cfd_intro.html) |
| 2 | Introduction to machine learning | [Read](https://andreweiner.github.io/ml-cfd-lecture/notebooks/ml_intro.html) | | [Slides](https://andreweiner.github.io/ml-cfd-slides/ml_intro.html) |
| 3 | Surrogate models for discrete predictions | | | |
| 4 | Surrogate models for continuous predictions | | | |
| 5 | Analysis of coherent structures | | | |
| 6 | Reduced-order modeling | | | |
| 7 | Bayesian optimization and exploration | | | |
| 8 | Data assimilation | | | |
| 9 | Active flow control | | | |

Later lecture and exercise materials will be revised before they are linked.

## Datasets

The course datasets are currently being revised. A download link will be published here soon.

## References and other resources

### Book recommendations

- Books on computational fluid dynamics
  - [The OpenFOAM technology primer](https://zenodo.org/record/4630596#.YXBgepuxVH4) by T. Marić, J. Höpken, and K. G. Mooney
  - *The finite volume method in computational fluid dynamics* by F. Moukalled, L. Mangani, and M. Darwish
  - *An introduction to computational fluid dynamics: the finite volume method* by H. K. Versteeg and W. Malalasekera
- Books on linear algebra
  - *Introduction to linear algebra* by G. Strang
- Books on data-driven modeling and control
  - *Data-driven science and engineering: machine learning, dynamical systems, and control* by S. L. Brunton and J. N. Kutz
  - *Dynamic mode decomposition: data-driven modeling of complex systems* by J. N. Kutz, S. L. Brunton, B. W. Brunton, and J. L. Proctor
  - *Grokking deep reinforcement learning* by M. Morales
  - *Deep learning with PyTorch* by E. Stevens, L. Antiga, and T. Viehmann
- Books on programming
  - *Python crash course* by E. Matthes
  - *C++ crash course: a fast-paced introduction* by J. Lospinoso
  - [The Linux command line](https://linuxcommand.org/tlcl.php) by W. Shotts

### Video content

- [YouTube channel of Steve Brunton](https://www.youtube.com/c/Eigensteve)
- [YouTube channel of József Nagy](https://www.youtube.com/channel/UCjdgpuxuAxH9BqheyE82Vvw)
- [YouTube channel Fluid Mechanics 101](https://www.youtube.com/channel/UCcqQi9LT0ETkRoUu8eYaEkg)

## For lecturers and tutors

The [course overview](index.qmd) organizes the lectures, exercises, and slides. Materials are linked as they are revised. Chapters are editable Python scripts in `notebooks/`, with Markdown cells. These Python scripts are the course sources.

See [Working with the course sources](authoring.md) for installation, running scripts, capturing figures and outputs, local preview, and GitHub Pages publishing.

```bash
quarto render
quarto preview --port 8002
```

The website is generated in `docs/`. Whole-site builds reuse `_freeze/`; explicitly render a chapter to execute it and update its results.
