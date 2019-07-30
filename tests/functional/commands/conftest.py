#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Common fixtures and utils for unittests and functional tests."""
import os

from pymongo import MongoClient
import pytest
import yaml

from orion.algo.base import (BaseAlgorithm, OptimizationAlgorithm)
import orion.core.cli
from orion.core.io.database import Database
from orion.core.io.experiment_builder import ExperimentBuilder
from orion.core.worker.trial import Trial
from orion.storage.base import Storage, get_storage
from orion.core.io.database.pickleddb import PickledDB
from orion.storage.legacy import Legacy


PICKLE_DB_HARDCODED = '/tmp/unittests.pkl'


def remove(db):
    try:
        os.remove(db)
    except FileNotFoundError:
        pass


class DumbAlgo(BaseAlgorithm):
    """Stab class for `BaseAlgorithm`."""

    def __init__(self, space, value=5,
                 scoring=0, judgement=None,
                 suspend=False, done=False, **nested_algo):
        """Configure returns, allow for variable variables."""
        self._times_called_suspend = 0
        self._times_called_is_done = 0
        self._num = None
        self._points = None
        self._results = None
        self._score_point = None
        self._judge_point = None
        self._measurements = None
        super(DumbAlgo, self).__init__(space, value=value,
                                       scoring=scoring, judgement=judgement,
                                       suspend=suspend,
                                       done=done,
                                       **nested_algo)

    def suggest(self, num=1):
        """Suggest based on `value`."""
        self._num = num
        return [self.value] * num

    def observe(self, points, results):
        """Log inputs."""
        self._points = points
        self._results = results

    def score(self, point):
        """Log and return stab."""
        self._score_point = point
        return self.scoring

    def judge(self, point, measurements):
        """Log and return stab."""
        self._judge_point = point
        self._measurements = measurements
        return self.judgement

    @property
    def should_suspend(self):
        """Cound how many times it has been called and return `suspend`."""
        self._times_called_suspend += 1
        return self.suspend

    @property
    def is_done(self):
        """Cound how many times it has been called and return `done`."""
        self._times_called_is_done += 1
        return self.done


# Hack it into being discoverable
OptimizationAlgorithm.types.append(DumbAlgo)
OptimizationAlgorithm.typenames.append(DumbAlgo.__name__.lower())


@pytest.fixture(scope='session')
def dumbalgo():
    """Return stab algorithm class."""
    return DumbAlgo


@pytest.fixture()
def exp_config():
    """Load an example database."""
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
              'experiment.yaml')) as f:
        exp_config = list(yaml.safe_load_all(f))
    return exp_config


@pytest.fixture(scope='session')
def database():
    """Return Mongo database object to test with example entries."""
    client = PickledDB(host=PICKLE_DB_HARDCODED)
    yield client


@pytest.fixture()
def clean_db(database, db_instance):
    """Clean insert example experiment entries to collections."""
    print('cleaning')
    db = Storage.instance._db
    db.remove('experiments', {})
    db.remove('lying_trials', {})
    db.remove('trials', {})
    db.remove('workers', {})
    db.remove('resources', {})
    remove(PICKLE_DB_HARDCODED)


@pytest.fixture()
def db_instance(null_db_instances):
    """Create and save a singleton database instance."""
    try:
        config = {
            'database': {
                'type': 'PickledDB',
                'host': PICKLE_DB_HARDCODED
            }
        }
        db = Storage('legacy', config=config)
    except ValueError:
        db = get_storage()

    return db


@pytest.fixture
def only_experiments_db(clean_db, database, exp_config):
    """Clean the database and insert only experiments."""
    get_storage().create_experiment(exp_config[0])


def ensure_deterministic_id(name, db_instance, update=None):
    """Change the id of experiment to its name."""

    experiments = db_instance.fetch_experiments(dict(name=name))
    assert len(experiments) == 1

    experiment = experiments[0]
    print(experiment['_id'])
    experiment['_id'] = name
    print(experiment['_id'])

    if experiment['refers']['parent_id'] is None:
        experiment['refers']['root_id'] = name

    if update is not None:
        experiment.update(update)

    db_instance.create_experiment(experiment)


# Experiments combinations fixtures
@pytest.fixture
def one_experiment(monkeypatch, db_instance):
    """Create an experiment without trials."""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    orion.core.cli.main(['init_only', '-n', 'test_single_exp',
                         './black_box.py', '--x~uniform(0,1)'])
    ensure_deterministic_id('test_single_exp', db_instance)


@pytest.fixture
def broken_refers(one_experiment, db_instance):
    """Create an experiment with broken refers."""
    ensure_deterministic_id('test_single_exp', db_instance, update=dict(refers={'oups': 'broken'}))


@pytest.fixture
def single_without_success(one_experiment):
    """Create an experiment without a succesful trial."""
    statuses = list(Trial.allowed_stati)
    statuses.remove('completed')

    exp = ExperimentBuilder().build_from({'name': 'test_single_exp'})
    x = {'name': '/x', 'type': 'real'}

    x_value = 0
    for status in statuses:
        x['value'] = x_value
        trial = Trial(experiment=exp.id, params=[x], status=status)
        x_value += 1
        get_storage().register_trial(trial)


@pytest.fixture
def single_with_trials(single_without_success):
    """Create an experiment with all types of trials."""
    exp = ExperimentBuilder().build_from({'name': 'test_single_exp'})

    x = {'name': '/x', 'type': 'real', 'value': 100}
    results = {"name": "obj", "type": "objective", "value": 0}
    trial = Trial(experiment=exp.id, params=[x], status='completed', results=[results])
    get_storage().register_trial(trial)


@pytest.fixture
def two_experiments(monkeypatch, db_instance):
    """Create an experiment and its child."""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    orion.core.cli.main(['init_only', '-n', 'test_double_exp',
                         './black_box.py', '--x~uniform(0,1)'])
    ensure_deterministic_id('test_double_exp', db_instance)

    orion.core.cli.main(['init_only', '-n', 'test_double_exp',
                         '--branch', 'test_double_exp_child', './black_box.py',
                         '--x~+uniform(0,1,default_value=0)', '--y~+uniform(0,1,default_value=0)'])
    ensure_deterministic_id('test_double_exp_child', db_instance)


@pytest.fixture
def family_with_trials(two_experiments):
    """Create two related experiments with all types of trials."""
    exp = ExperimentBuilder().build_from({'name': 'test_double_exp'})
    exp2 = ExperimentBuilder().build_from({'name': 'test_double_exp_child'})
    x = {'name': '/x', 'type': 'real'}
    y = {'name': '/y', 'type': 'real'}

    x_value = 0
    for status in Trial.allowed_stati:
        x['value'] = x_value
        y['value'] = x_value
        trial = Trial(experiment=exp.id, params=[x], status=status)
        x['value'] = x_value
        trial2 = Trial(experiment=exp2.id, params=[x, y], status=status)
        x_value += 1
        Storage().register_trial(trial)
        Storage().register_trial(trial2)


@pytest.fixture
def unrelated_with_trials(family_with_trials, single_with_trials):
    """Create two unrelated experiments with all types of trials."""
    exp = ExperimentBuilder().build_from({'name': 'test_double_exp_child'})
    get_storage()._db.remove('trials', {'experiment': exp.id})
    get_storage()._db.remove('experiments', {'_id': exp.id})


@pytest.fixture
def three_experiments(two_experiments, one_experiment):
    """Create a single experiment and an experiment and its child."""
    pass


@pytest.fixture
def three_experiments_with_trials(family_with_trials, single_with_trials):
    """Create three experiments, two unrelated, with all types of trials."""
    pass


@pytest.fixture
def three_experiments_family(two_experiments, db_instance):
    """Create three experiments, one of which is the parent of the other two."""
    orion.core.cli.main(['init_only', '-n', 'test_double_exp',
                         '--branch', 'test_double_exp_child2', './black_box.py',
                         '--x~+uniform(0,1,default_value=0)', '--z~+uniform(0,1,default_value=0)'])
    ensure_deterministic_id('test_double_exp_child2', db_instance)


@pytest.fixture
def three_family_with_trials(three_experiments_family, family_with_trials):
    """Create three experiments, all related, two direct children, with all types of trials."""
    exp = ExperimentBuilder().build_from({'name': 'test_double_exp_child2'})
    x = {'name': '/x', 'type': 'real'}
    z = {'name': '/z', 'type': 'real'}

    x_value = 0
    for status in Trial.allowed_stati:
        x['value'] = x_value
        z['value'] = x_value * 100
        trial = Trial(experiment=exp.id, params=[x, z], status=status)
        x_value += 1
        get_storage().register_trial(trial)


@pytest.fixture
def three_experiments_family_branch(two_experiments, db_instance):
    """Create three experiments, each parent of the following one."""
    orion.core.cli.main(['init_only', '-n', 'test_double_exp_child',
                         '--branch', 'test_double_exp_grand_child', './black_box.py',
                         '--x~+uniform(0,1,default_value=0)', '--y~uniform(0,1,default_value=0)',
                         '--z~+uniform(0,1,default_value=0)'])
    ensure_deterministic_id('test_double_exp_grand_child', db_instance)


@pytest.fixture
def three_family_branch_with_trials(three_experiments_family_branch, family_with_trials):
    """Create three experiments, all related, one child and one grandchild,
    with all types of trials.

    """
    exp = ExperimentBuilder().build_from({'name': 'test_double_exp_grand_child'})
    x = {'name': '/x', 'type': 'real'}
    y = {'name': '/y', 'type': 'real'}
    z = {'name': '/z', 'type': 'real'}

    x_value = 0
    for status in Trial.allowed_stati:
        x['value'] = x_value
        y['value'] = x_value * 10
        z['value'] = x_value * 100
        trial = Trial(experiment=exp.id, params=[x, y, z], status=status)
        x_value += 1
        get_storage().register_trial(trial)
