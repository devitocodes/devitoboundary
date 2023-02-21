## Devitoboundary

[![CI-Core](https://github.com/devitocodes/devitoboundary/actions/workflows/pytest_core.yml/badge.svg)](https://github.com/devitocodes/devitoboundary/actions/workflows/pytest_core.yml)

[![CI-1st-Order](https://github.com/devitocodes/devitoboundary/actions/workflows/pytest_1st_order.yml/badge.svg)](https://github.com/devitocodes/devitoboundary/actions/workflows/pytest_1st_order.yml)

Note: this repo has been superseded by [Schism](https://github.com/EdCaunt/schism).

Devitoboundary is a set of utilities used for the implementation of
immersed boundaries in Devito. The intention is to build useful
abstractions to simplify the process of imposing boundary conditions
on non-grid-conforming topographies. By making a suitably versatile,
generic tool for constructing immersed boundaries from unstructured
topography data, the integration of immersed boundary methods into
higher level applications with minimal additional complexity is
made possible.

This repository is currently a WIP prototype.

In order to download, install and use Devito follow the instructions
listed [here](https://github.com/devitocodes/devito).


## Quickstart
In order to install Devitoboundary:
*Requirements:* A working Devito installation.

```
source activate devito
git clone https://github.com/devitocodes/devitoboundary.git
cd devitoboundary
pip install -e .
```

## Get in touch

If you're using Devitoboundary or Devito, we would like to hear from
you. Whether you are facing issues or just trying it out, join the
[conversation](https://opesci-slackin.now.sh).
