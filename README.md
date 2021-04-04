# Gemini: Dynamic Bias Correction for Autonomous Experimentation and Molecular Simulation

[![Build Status](https://travis-ci.com/rileyhickman/gemini.svg?token=N6jxEx5YcAwy6gVv2bNh&branch=release)](https://travis-ci.com/rileyhickman/gemini)
[![codecov](https://codecov.io/gh/rileyhickman/gemini/branch/release/graph/badge.svg?token=Bdt22mbq31)](https://codecov.io/gh/rileyhickman/gemini)
[![GitHub license](https://img.shields.io/github/license/Naereen/StrapDown.js.svg)](https://github.com/Naereen/StrapDown.js/blob/master/LICENSE)
<!-- [![PyPI pyversions](https://img.shields.io/pypi/pyversions/ansicolortags.svg)](https://pypi.python.org/pypi/ansicolortags/) -->

Gemini is an open-source Python package which provides scalable multi-fidelity machine learning targeting
the design and discovery of functional molecules and advanced materials. (https://arxiv.org/abs/2103.03391v1)

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install Gemini.

```bash
pip install matter-gemini
```

Alternatively, you can install it from source.

```bash
git clone https://github.com/rileyhickman/gemini.git
cd matter-gemini
pip install -e .
```

GPU use is optional. We recommend using the following

```bash
tensorflow-gpu            2.4.1
CUDA Version: 11.1
cuda-toolkit-11-1         11.1.1-1
Latest cuDNN
```

## Usage

### Supervised learning tasks with multi-fidelity data

Gemini can be easily trained given 2D (# samples, # dimensions) NumPy arrays containing
features (_x_) and targets (_y_) for _exp_ and _cheap_ datasets. Predictions using Gemini are
furnished with frequentist uncertainty estimates.

```python
from gemini import Gemini

gemini = Gemini()

gemini.train(x_exp, y_exp,
	 x_cheap, y_cheap)

pred_mu, pred_std = gemini.predict(x_exp_test)

```

### Scalable multi-fidelity Bayesian optimization

Gemini's predictions of expensive-to-evaluate objective functions can be used to
reduce the number of expensive black-box evaluations necessary to
achieve a desired target value.

The deep Bayesian optimizer Gryffin currently supports Gemini as a built-in predictive model.
After installing Gemini and Gryffin,

```python
from gryffin import Gryffin

# instantiate Gryffin
gryffin = Gryffin('config_file.json')

# optimization loop
while num_eval < budget:

    samples = gryffin.recommend(observations,
				proxy_observations)
```

The Gryffin config file must include a section specifying the predictive model, i.e.

```bash
...
"predictive_model": {
		"model_kind": "gemini"
},
...
```

Alternatively, you can train Gemini in an external manner, this gives the user
greater flexibility in their expreiment. Gryffin allows for the optional passing of
a callable object to its `recommend` method.

```python
from gryffin import Gryffin
from gemini import GeminiOpt as Gemini

# instantiate Gryffin
gryffin = Gryffin('config_file.json')

# instantiate Gemini
gemini = Gemini()

# optimization loop
while num_eval < budget:

    if len(observations) >= 2 and len(proxy_observations) >= 2:

        # construct training set with current observations
        training_set = gryffin.construct_training_set(observations, proxy_observations)

        # train Gemini
        gemini.train(training_set['train_features'], training_set['train_targets'],
                     training_set['proxy_train_features'], training_set['proxy_train_targets'],
                     num_folds=3)

    # pass callable when asking Gryffin for new samples
    samples = gryffin.recommend(observations,
                                predictive_model=gemini)
```

In this external trianing case, you need only provide a Gryffin config file (i.e. no predictive
model entries)


## Applications of Gemini (so far...)

* Inverse design of hybrid organic inorganic perovskites
* Inverse design of multi-component metal-oxide catalysts for the oxygen evolution reaction
* Inverse design of non-fullerene acceptor molecules for light harvesting applications


## Datasets

We provide methods for facile multi-fidelity data preprocessing/testing for 4 datasets reported in
the literature.

* `dataset_perovskites`
* `dataset_freesolv` (10.1007/s10822-014-9747-x)
* `dataset_photobleaching`
* `dataset_cat_oer_1_4`

## Contributing

Academic collaborations and extensions/improvements to the code are encouraged. Please reach out to Riley via email if you have questions/concerns.

## Developers

* Riley J. Hickman (riley.hickman@mail.utoronto.ca)
* Florian Häse
* Matteo Aldeghi

## Citation

Gemini is an open-source research software. If you use Gemini in a scientific report, please cite the
following article

```
@misc{gemini,
      title={Gemini: Dynamic Bias Correction for Autonomous Experimentation},
      author={Riley J. Hickman and Florian Häse and Loïc M. Roch and Alán Aspuru-Guzik},
      year={2021},
      eprint={2103.03391},
      archivePrefix={arXiv},
      primaryClass={stat.ML}
}
```

## License
[MIT](https://choosealicense.com/licenses/mit/)
