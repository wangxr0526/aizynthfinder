# AiZynthFinder

[![License](https://img.shields.io/github/license/MolecularAI/aizynthfinder)](https://github.com/MolecularAI/aizynthfinder/blob/master/LICENSE)
[![Tests](https://github.com/MolecularAI/aizynthfinder/workflows/tests/badge.svg)](https://github.com/MolecularAI/aizynthfinder/actions?workflow=tests)
[![codecov](https://codecov.io/gh/MolecularAI/aizynthfinder/branch/master/graph/badge.svg)](https://codecov.io/gh/MolecularAI/aizynthfinder)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)
[![version](https://img.shields.io/github/v/release/MolecularAI/aizynthfinder)](https://github.com/MolecularAI/aizynthfinder/releases)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MolecularAI/aizynthfinder/blob/master/contrib/notebook.ipynb)

AiZynthFinder is a tool for retrosynthetic planning. The default algorithm is based on a Monte Carlo tree search that recursively breaks down a molecule to purchasable precursors. The tree search is guided by a policy that suggests possible precursors by utilizing a neural network trained on a library of known reaction templates. This setup is completely customizable as the tool
supports multiple search algorithms and expansion policies.

An introduction video can be found here: [https://youtu.be/r9Dsxm-mcgA](https://youtu.be/r9Dsxm-mcgA)

## Prerequisites

Before you begin, ensure you have met the following requirements:

* Linux, Windows or macOS platforms are supported - as long as the dependencies are supported on these platforms.

* You have installed [anaconda](https://www.anaconda.com/) or [miniconda](https://docs.conda.io/en/latest/miniconda.html) with python 3.10 - 3.12

The tool has been developed on a Linux platform, but the software has been tested on Windows 10 and macOS Catalina.

## Installation

### For end-users

First time, execute the following command in a console or an Anaconda prompt

    conda create "python>=3.10,<3.13" -n aizynth-env

To install, activate the environment and install the package using pypi

    conda activate aizynth-env
    python -m pip install aizynthfinder[all]

for a smaller package, without all the functionality, you can also type

    python -m pip install aizynthfinder

### For developers

First clone the repository using Git.

Then execute the following commands in the root of the repository

    conda env create -f env-dev.yml
    conda activate aizynth-dev
    poetry install --all-extras

the `aizynthfinder` package is now installed in editable mode.


## Usage

The tool will install the `aizynthcli` and `aizynthapp` tools
as interfaces to the algorithm:

    aizynthcli --config config_local.yml --smiles smiles.txt
    aizynthapp --config config_local.yml


Consult the documentation [here](https://molecularai.github.io/aizynthfinder/) for more information.

### FastAPI service

The repository also exposes the local models through FastAPI. After installing
the project dependencies, start the single-worker service on port `8001`:

    aizynthapi

Or run it directly from this checkout:

    ./.venv/bin/aizynthapi

The service listens on `127.0.0.1:8001` by default. Set
`AIZYNTHFINDER_API_HOST` or `AIZYNTHFINDER_API_PORT` to override this. Keep one
worker per service process, because each worker loads the model and stock data.
You can also pass `--host` and `--port` directly to `aizynthapi`.

Models and stock data are loaded lazily by the first planning request for an
algorithm and retained for the lifetime of the API process. Requests that only
change `smiles` reuse both the loaded data and the existing finder
configuration; keep the service process running between requests to benefit
from this cache.

    curl -sS http://127.0.0.1:8001/health
    curl -sS -X POST http://127.0.0.1:8001/aizynthfinder_plan \
      -H 'Content-Type: application/json' \
      -d '{"smiles":"CCOC(=O)c1ccccc1","iterations":100,"expansion_topk":50}'

`algorithm` accepts `mcts` (the default/original setup), `original` (an alias
for MCTS), or `retrostar`. `model` accepts `uspto`, `ringbreaker`, `reaxys`, or
`multi` (the combined USPTO and RingBreaker strategy). Reaxys is independent of
`multi`. The response contains search statistics, stock information, and
serialised routes.

Use the following payload for AiZynthFinder's Retro* search tree with the USPTO
single-step policy and only the configured ZINC building-block stock:

    curl -sS -X POST http://127.0.0.1:8001/aizynthfinder_plan \
      -H 'Content-Type: application/json' \
      -d '{"smiles":"CCOC(=O)c1ccccc1","algorithm":"retrostar","model":"uspto","stocks":["zinc"]}'

Select the Reaxys single-step policy without changing the Retro* search tree or
ZINC stock by setting `model` to `reaxys`:

    curl -sS -X POST http://127.0.0.1:8001/aizynthfinder_plan \
      -H 'Content-Type: application/json' \
      -d '{"smiles":"CCOC(=O)c1ccccc1","algorithm":"retrostar","model":"reaxys","stocks":["zinc"]}'

The same payload is accepted inside `request.json` by the disk-backed
`/aizynthfinder_plan_async` endpoint.

The filter policy is disabled by default (`use_filter=false`) for both MCTS and
Retro*. Pass `"use_filter": true` explicitly only when a filter-model comparison
is required.

#### Reaxys model conversion

The Reaxys policy is licensed separately under CC BY-NC 4.0; see
`LICENSE_REAXYS_MODEL`. Its TorchServe archive is converted once to a native
AiZynthFinder ONNX policy and compressed template table. Conversion verifies the
known archive checksum, template ordering, ONNX numerical output, and a fixed
top-10 regression example before atomically installing the assets.

The conversion-only packages are intentionally not project dependencies. On a
host with the template-relevance image available, run the converter in a
disposable container from the project root:

    docker run --rm --user "$(id -u):$(id -g)" \
      --entrypoint /bin/bash \
      -v "$PWD:/workspace" \
      -v /path/to/reaxys.mar:/input/reaxys.mar:ro \
      -w /workspace \
      registry.gitlab.com/mlpds_mit/askcosv2/askcos2_core/retro/template_relevance:1.0-gpu \
      -lc '/opt/conda/bin/python3.9 -m pip install --quiet \
        --target /tmp/reaxys-convert "numpy<1.23" onnx==1.14.1 onnxruntime==1.16.3 \
        && PYTHONPATH=/tmp/reaxys-convert /opt/conda/bin/python3.9 \
        aizynthfinder/tools/convert_reaxys_model.py \
        --source-mar /input/reaxys.mar --output-dir /workspace/contrib/data'

The generated `reaxys.mar`, `reaxys_model.onnx`,
`reaxys_templates.csv.gz`, and `reaxys_manifest.json` remain under the ignored
`contrib/data` directory. Configure the policy with `chiral_fingerprints: true`
and `rescale_prior: false`, because the exported ONNX model already returns
softmax probabilities.

To use the tool you need

    1. A stock file
    2. A trained expansion policy network
    3. A trained filter policy network (optional)

Such files can be downloaded from [figshare](https://figshare.com/articles/AiZynthFinder_a_fast_robust_and_flexible_open-source_software_for_retrosynthetic_planning/12334577) and [here](https://figshare.com/articles/dataset/A_quick_policy_to_filter_reactions_based_on_feasibility_in_AI-guided_retrosynthetic_planning/13280507) or they can be downloaded automatically using

```
download_public_data my_folder
```

where ``my_folder`` is the folder that you want download to.
This will create a ``config.yml`` file that you can use with either ``aizynthcli`` or ``aizynthapp``.

## Development

### Testing

Tests uses the ``pytest`` package, and is installed by `poetry`

Run the tests using:

    pytest -v

The full command run on the CI server is available through an `invoke` command

    invoke full-tests

 ### Documentation generation

The documentation is generated by Sphinx from hand-written tutorials and docstrings

The HTML documentation can be generated by

    invoke build-docs

## Contributing

We welcome contributions, in the form of issues or pull requests.

If you have a question or want to report a bug, please submit an issue.


To contribute with code to the project, follow these steps:

1. Fork this repository.
2. Create a branch: `git checkout -b <branch_name>`.
3. Make your changes and commit them: `git commit -m '<commit_message>'`
4. Push to the remote branch: `git push`
5. Create the pull request.

Please use ``black`` package for formatting, and follow ``pep8`` style guide.


## Contributors

* [@SGenheden](https://www.github.com/SGenheden)
* [@lakshidaa](https://github.com/Lakshidaa)
* [@helenlai](https://github.com/helenlai)
* [@EBjerrum](https://www.github.com/EBjerrum)
* [@A-Thakkar](https://www.github.com/A-Thakkar)
* [@benteb](https://www.github.com/benteb)

The contributors have limited time for support questions, but please do not hesitate to submit an issue (see above).

## License

The software is licensed under the MIT license (see LICENSE file), and is free and provided as-is.

## References

1. Thakkar A, Kogej T, Reymond J-L, et al (2019) Datasets and their influence on the development of computer assisted synthesis planning tools in the pharmaceutical domain. Chem Sci. https://doi.org/10.1039/C9SC04944D
2. Genheden S, Thakkar A, Chadimova V, et al (2020) AiZynthFinder: a fast, robust and flexible open-source software for retrosynthetic planning. ChemRxiv. Preprint. https://doi.org/10.26434/chemrxiv.12465371.v1
