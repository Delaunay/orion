#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Example usage and tests for :mod:`orion.core.io.experiment_builder`."""
import copy
import datetime

import pytest

from orion.algo.base import BaseAlgorithm
from orion.algo.space import Space
from orion.core.evc.adapters import BaseAdapter
import orion.core.io.experiment_builder as experiment_builder
import orion.core.utils.backward as backward
from orion.core.utils.tests import OrionState
from orion.core.utils.exceptions import NoConfigurationError, RaceCondition
from orion.storage.base import get_storage


def count_experiments():
    """Count experiments in storage"""
    return len(get_storage().fetch_experiments({}))


# TODO: Remove and use OrionState instead
@pytest.fixture
def init_storage(clean_db, test_config):
    """Create the storage singleton."""
    experiment_builder.setup_storage(
        storage={
            'type': 'legacy',
            'database': {
                'type': 'mongodb',
                'name': 'orion_test',
                'host': 'mongodb://user:pass@localhost'}})


@pytest.fixture
def space():
    return {'x': 'uniform(-50,50)'}


@pytest.fixture()
def python_api_config():
    """Create a configuration without the cli fluff."""
    # TODO: replace metadata[priors] by space when space in DB
    new_config = dict(
        name='supernaekei',
        version=1,
        metadata={'user': 'tsirif',
                  'orion_version': 'XYZ',
                  'VCS': {"type": "git",
                          "is_dirty": False,
                          "HEAD_sha": "test",
                          "active_branch": None,
                          "diff_sha": "diff"},
                  'priors': {'x':'uniform(0,10)'}},
        max_trials=1000,
        working_dir='',
        algorithms={
            'dumbalgo': {
                'done': False,
                'judgement': None,
                'scoring': 0,
                'seed': None,
                'suspend': False,
                'value': 5}},
        producer={'strategy': 'NoParallelStrategy'},
        _id='fasdfasfa',
        something_to_be_ignored='asdfa',
        refers=dict(
            root_id='supernaekei',
            parent_id=None,
            adapter=[])
        )

    return new_config


@pytest.fixture()
def new_config(random_dt, script_path):
    """Create a configuration that will not hit the database."""
    new_config = dict(
        name='supernaekei',
        metadata={'user': 'tsirif',
                  'orion_version': 'XYZ',
                  'user_script': script_path,
                  'user_config': 'abs_path/hereitis.yaml',
                  'user_args': ['--mini-batch~uniform(32, 256, discrete=True)'],
                  'VCS': {"type": "git",
                          "is_dirty": False,
                          "HEAD_sha": "test",
                          "active_branch": None,
                          "diff_sha": "diff"}},
        version=1,
        pool_size=10,
        max_trials=1000,
        working_dir='',
        algorithms={
            'dumbalgo': {
                'done': False,
                'judgement': None,
                'scoring': 0,
                'seed': None,
                'suspend': False,
                'value': 5}},
        producer={'strategy': 'NoParallelStrategy'},
        # attrs starting with '_' also
        _id='fasdfasfa',
        # and in general anything which is not in Experiment's slots
        something_to_be_ignored='asdfa',
        refers=dict(
            root_id='supernaekei',
            parent_id=None,
            adapter=[])
        )

    backward.populate_priors(new_config['metadata'])

    return new_config





@pytest.fixture
def parent_version_config():
    """Return a configuration for an experiment."""
    config = dict(
        _id='parent_config',
        name="old_experiment",
        version=1,
        algorithms='random',
        metadata={'user': 'corneauf', 'datetime': datetime.datetime.utcnow(),
                  'user_args': ['--x~normal(0,1)']})

    backward.populate_priors(config['metadata'])

    return config


@pytest.fixture
def child_version_config(parent_version_config):
    """Return a configuration for an experiment."""
    config = copy.deepcopy(parent_version_config)
    config['_id'] = 'child_config'
    config['version'] = 2
    config['refers'] = {'parent_id': 'parent_config'}
    config['metadata']['datetime'] = datetime.datetime.utcnow()
    config['metadata']['user_args'].append('--y~+normal(0,1)')
    backward.populate_priors(config['metadata'])
    return config


def test_get_cmd_config(config_file):
    """Test local config (cmdconfig, cmdargs)"""
    cmdargs = {"config": config_file}
    local_config = experiment_builder.get_cmd_config(cmdargs)

    assert local_config['algorithms'] == 'random'
    assert local_config['producer'] == {'strategy': 'NoParallelStrategy'}
    assert local_config['max_trials'] == 100
    assert local_config['name'] == 'voila_voici'
    assert local_config['pool_size'] == 1
    assert local_config['storage'] == {
        'type': 'legacy',
        'database': {
            'host': 'mongodb://user:pass@localhost',
            'name': 'orion_test',
            'type': 'mongodb'}}


def test_get_cmd_config_from_incomplete_config(incomplete_config_file):
    """Test local config with incomplete user configuration file
    (default, env_vars, cmdconfig, cmdargs)

    This is to ensure merge_configs update properly the subconfigs
    """
    cmdargs = {"config": incomplete_config_file}
    local_config = experiment_builder.get_cmd_config(cmdargs)

    assert 'algorithms' not in local_config
    assert 'max_trials' not in local_config
    assert 'pool_size' not in local_config
    assert 'name' not in local_config['storage']['database']
    assert local_config['storage']['database']['host'] == 'mongodb://user:pass@localhost'
    assert local_config['storage']['database']['type'] == 'incomplete'
    assert local_config['name'] == 'incomplete'


@pytest.mark.usefixtures('init_storage')
def test_fetch_config_from_db_no_hit(config_file, random_dt):
    """Verify that fetch_config_from_db returns an empty dict when the experiment is not in db"""
    db_config = experiment_builder.fetch_config_from_db(name='supernaekei')
    assert db_config == {}


@pytest.mark.usefixtures('with_user_tsirif')
def test_fetch_config_from_db_hit(new_config):
    """Verify db config when experiment is in db"""
    with OrionState(experiments=[new_config], trials=[]) as cfg:
        db_config = experiment_builder.fetch_config_from_db(name='supernaekei')

    assert db_config['name'] == new_config['name']
    assert db_config['refers'] == new_config['refers']
    assert db_config['metadata'] == new_config['metadata']
    assert db_config['pool_size'] == new_config['pool_size']
    assert db_config['max_trials'] == new_config['max_trials']
    assert db_config['algorithms'] == new_config['algorithms']


@pytest.mark.usefixtures("clean_db", "null_db_instances", "with_user_tsirif")
def test_build_view_from_args_no_hit(config_file, create_db_instance):
    """Try building experiment view when not in db"""
    cmdargs = {'name': 'supernaekei', 'config': config_file}

    with OrionState(experiments=[], trials=[]) as cfg:
        with pytest.raises(ValueError) as exc_info:
            experiment_builder.build_view_from_args(cmdargs)
        assert "No experiment with given name 'supernaekei' and version '*'" in str(exc_info.value)


@pytest.mark.usefixtures("with_user_tsirif")
def test_build_view_from_args_hit(config_file, random_dt, new_config):
    """Try building experiment view when in db"""
    cmdargs = {'name': 'supernaekei', 'config': config_file}

    with OrionState(experiments=[new_config], trials=[]) as cfg:
        exp_view = experiment_builder.build_view_from_args(cmdargs)

    assert exp_view._experiment._init_done is True
    assert exp_view._id == new_config['_id']
    assert exp_view.name == new_config['name']
    assert exp_view.configuration['refers'] == new_config['refers']
    assert exp_view.metadata == new_config['metadata']
    assert exp_view.pool_size == new_config['pool_size']
    assert exp_view.max_trials == new_config['max_trials']
    assert exp_view.algorithms.configuration == new_config['algorithms']


@pytest.mark.usefixtures("with_user_dendi")
def test_build_from_args_no_hit(config_file, random_dt, script_path, new_config):
    """Try building experiment when not in db"""
    cmdargs = {'name': 'supernaekei', 'config': config_file,
               'user_args': [script_path,
                             'x~uniform(0,10)']}

    with OrionState(experiments=[], trials=[]) as cfg:
        db_config = experiment_builder.fetch_config_from_db(name='supernaekei')

        with pytest.raises(ValueError) as exc_info:
            experiment_builder.build_view_from_args(cmdargs)
        assert "No experiment with given name 'supernaekei' and version '*'" in str(exc_info.value)

        exp = experiment_builder.build_from_args(cmdargs)

    assert exp._init_done is True
    assert exp.name == cmdargs['name']
    assert exp.configuration['refers'] == {'adapter': [], 'parent_id': None, 'root_id': exp._id}
    assert exp.metadata['datetime'] == random_dt
    assert exp.metadata['user'] == 'dendi'
    assert exp.metadata['user_script'] == cmdargs['user_args'][0]
    assert exp.metadata['user_args'] == cmdargs['user_args'][1:]
    assert exp.pool_size == 1
    assert exp.max_trials == 100
    assert exp.algorithms.configuration == {'random': {'seed': None}}


@pytest.mark.usefixtures("version_XYZ", "with_user_tsirif", "mock_infer_versioning_metadata")
def test_build_from_args_hit(old_config_file, script_path, new_config):
    """Try building experiment when in db (no branch)"""
    cmdargs = {'name': 'supernaekei',
               'config': old_config_file,
               'user_args': [script_path,
                             '--mini-batch~uniform(32, 256, discrete=True)']}

    with OrionState(experiments=[new_config], trials=[]) as cfg:
        # Test that experiment already exists
        experiment_builder.build_view_from_args(cmdargs)

        exp = experiment_builder.build_from_args(cmdargs)

    assert exp._init_done is True
    assert exp._id == new_config['_id']
    assert exp.name == new_config['name']
    assert exp.configuration['refers'] == new_config['refers']
    assert exp.metadata == new_config['metadata']
    assert exp.max_trials == new_config['max_trials']
    assert exp.algorithms.configuration == new_config['algorithms']


@pytest.mark.usefixtures("clean_db", "with_user_bouthilx")
def test_build_from_args_force_user(old_config_file, random_dt):
    """Try building experiment view when in db"""
    cmdargs = {'name': 'supernaedo2', 'config': old_config_file}
    cmdargs['user'] = 'tsirif'
    exp_view = experiment_builder.build_from_args(cmdargs)
    assert exp_view.metadata['user'] == 'tsirif'


@pytest.mark.usefixtures("with_user_tsirif")
def test_build_no_hit(config_file, random_dt, script_path):
    """Try building experiment from config when not in db"""
    name = 'supernaekei'
    space = {'x': 'uniform(0,10)'}
    max_trials = 100

    with OrionState(experiments=[], trials=[]) as cfg:

        with pytest.raises(ValueError) as exc_info:
            experiment_builder.build_view(name)
        assert "No experiment with given name 'supernaekei' and version '*'" in str(exc_info.value)

        exp = experiment_builder.build(name, space=space, max_trials=max_trials)

    assert exp._init_done is True
    assert exp.name == name
    assert exp.configuration['refers'] == {'adapter': [], 'parent_id': None, 'root_id': exp._id}
    assert exp.metadata['datetime'] == random_dt
    assert exp.metadata['user'] == 'tsirif'
    # TODO: Replace when space in db
    assert 'space' not in exp.configuration
    # assert exp.configuration['space'] == space
    assert exp.max_trials == max_trials
    assert not exp.is_done
    assert exp.algorithms.configuration == {'random': {'seed': None}}


def test_build_no_commandline_config():
    """Try building experiment with no commandline configuration."""
    with OrionState(experiments=[], trials=[]) as cfg:
        with pytest.raises(NoConfigurationError):
            experiment_builder.build('supernaekei')


@pytest.mark.usefixtures("with_user_dendi", "mock_infer_versioning_metadata", "version_XYZ")
def test_build_hit(python_api_config):
    """Try building experiment from config when in db (no branch)"""
    name = 'supernaekei'

    with OrionState(experiments=[python_api_config], trials=[]) as cfg:

        # Test that experiment already exists (this should fail otherwise)
        experiment_builder.build_view(name=name)

        exp = experiment_builder.build(space=python_api_config['metadata']['priors'],
                                       **python_api_config)

    assert exp._init_done is True
    assert exp._id == python_api_config['_id']
    assert exp.name == python_api_config['name']
    assert exp.configuration['refers'] == python_api_config['refers']
    assert exp.metadata == python_api_config['metadata']
    assert exp.max_trials == python_api_config['max_trials']
    assert exp.algorithms.configuration == python_api_config['algorithms']


@pytest.mark.usefixtures("with_user_dendi")
def test_build_without_config_hit(python_api_config):
    """Try building experiment without commandline config when in db (no branch)"""
    name = 'supernaekei'

    with OrionState(experiments=[python_api_config], trials=[]) as cfg:

        # Test that experiment already exists (this should fail otherwise)
        experiment_builder.build_view(name=name)

        exp = experiment_builder.build(name=name)

    assert exp._init_done is True
    assert exp._id == python_api_config['_id']
    assert exp.name == python_api_config['name']
    assert exp.configuration['refers'] == python_api_config['refers']
    assert exp.metadata == python_api_config['metadata']
    assert exp.max_trials == python_api_config['max_trials']
    assert exp.algorithms.configuration == python_api_config['algorithms']


@pytest.mark.usefixtures("with_user_tsirif", "version_XYZ")
def test_build_from_args_without_cmd(old_config_file, script_path, new_config):
    """Try building experiment without commandline when in db (no branch)"""
    name = 'supernaekei'

    cmdargs = {'name': name,
               'config': old_config_file}

    with OrionState(experiments=[new_config], trials=[]) as cfg:
        # Test that experiment already exists (this should fail otherwise)
        experiment_builder.build_view_from_args(cmdargs)

        exp = experiment_builder.build_from_args(cmdargs)

    assert exp._init_done is True
    assert exp._id == new_config['_id']
    assert exp.name == new_config['name']
    assert exp.configuration['refers'] == new_config['refers']
    assert exp.metadata == new_config['metadata']
    assert exp.max_trials == new_config['max_trials']
    assert exp.algorithms.configuration == new_config['algorithms']


@pytest.mark.usefixtures("create_db_instance", "with_user_tsirif")
class TestExperimentVersioning(object):
    """Create new Experiment with auto-versioning."""

    def test_new_experiment_wout_version(self, space):
        """Create a new and never-seen-before experiment without a version."""
        exp = experiment_builder.build(name="exp_wout_version", space=space)
        assert exp.version == 1

    def test_new_experiment_w_version(self, space):
        """Create a new and never-seen-before experiment with a version."""
        exp = experiment_builder.build(name="exp_wout_version", version=1, space=space)
        assert exp.version == 1

    def test_backward_compatibility_no_version(self, create_db_instance, parent_version_config,
                                               child_version_config):
        """Branch from parent that has no version field."""
        parent_version_config.pop('version')
        create_db_instance.write('experiments', parent_version_config)
        create_db_instance.write('experiments', child_version_config)

        exp = experiment_builder.build(name="old_experiment")
        assert exp.version == 2

    def test_old_experiment_wout_version(self, create_db_instance, parent_version_config,
                                         child_version_config):
        """Create an already existing experiment without a version."""
        create_db_instance.write('experiments', parent_version_config)
        create_db_instance.write('experiments', child_version_config)

        exp = experiment_builder.build(name="old_experiment")
        assert exp.version == 2

    def test_old_experiment_w_version(self, create_db_instance, parent_version_config,
                                      child_version_config):
        """Create an already existing experiment with a version."""
        create_db_instance.write('experiments', parent_version_config)
        create_db_instance.write('experiments', child_version_config)

        exp = experiment_builder.build(name="old_experiment", version=1)
        assert exp.version == 1

    def test_old_experiment_w_version_bigger_than_max(self, create_db_instance,
                                                      parent_version_config, child_version_config):
        """Create an already existing experiment with a too large version."""
        create_db_instance.write('experiments', parent_version_config)
        create_db_instance.write('experiments', child_version_config)

        exp = experiment_builder.build(name="old_experiment", version=8)
        assert exp.version == 2


@pytest.mark.usefixtures("create_db_instance", "with_user_tsirif")
class TestBuild(object):

    def test_good_set_before_init_hit_no_diffs_exc_max_trials(self, new_config):
        """Trying to set, and NO differences were found from the config pulled from db.

        Everything is normal, nothing changes. Experiment is resumed,
        perhaps with more trials to evaluate (an exception is 'max_trials').
        """
        with OrionState(experiments=[new_config], trials=[]) as cfg:

            space = new_config['metadata']['priors']
            new_config['max_trials'] = 5000

            exp = experiment_builder.build(space=space, **new_config)

        # Deliver an external configuration to finalize init
        new_config['algorithms']['dumbalgo']['done'] = False
        new_config['algorithms']['dumbalgo']['judgement'] = None
        new_config['algorithms']['dumbalgo']['scoring'] = 0
        new_config['algorithms']['dumbalgo']['suspend'] = False
        new_config['algorithms']['dumbalgo']['value'] = 5
        new_config['algorithms']['dumbalgo']['seed'] = None
        new_config['producer']['strategy'] = "NoParallelStrategy"
        new_config.pop('_id')
        new_config.pop('something_to_be_ignored')
        assert exp.configuration == new_config

    def test_good_set_before_init_no_hit(self, random_dt, new_config):
        """Trying to set, overwrite everything from input."""

        with OrionState(experiments=[], trials=[]) as cfg:
            space = new_config['metadata']['priors']
            exp = experiment_builder.build(space=space, **new_config)
            assert exp._init_done is True
            found_config = list(get_storage().fetch_experiments({'name': 'supernaekei',
                                                                 'metadata.user': 'tsirif'}))

        new_config['metadata']['datetime'] = exp.metadata['datetime']

        assert len(found_config) == 1
        _id = found_config[0].pop('_id')
        assert _id != 'fasdfasfa'
        assert exp._id == _id
        new_config['refers'] = {}
        new_config.pop('_id')
        new_config.pop('something_to_be_ignored')
        new_config['algorithms']['dumbalgo']['done'] = False
        new_config['algorithms']['dumbalgo']['judgement'] = None
        new_config['algorithms']['dumbalgo']['scoring'] = 0
        new_config['algorithms']['dumbalgo']['suspend'] = False
        new_config['algorithms']['dumbalgo']['value'] = 5
        new_config['algorithms']['dumbalgo']['seed'] = None
        new_config['refers'] = {'adapter': [], 'parent_id': None, 'root_id': _id}
        assert found_config[0] == new_config
        assert exp.name == new_config['name']
        assert exp.configuration['refers'] == new_config['refers']
        assert exp.metadata == new_config['metadata']
        assert exp.pool_size == new_config['pool_size']
        assert exp.max_trials == new_config['max_trials']
        assert exp.working_dir == new_config['working_dir']
        assert exp.version == new_config['version']
        assert exp.algorithms.configuration == new_config['algorithms']

    def test_working_dir_is_correctly_set(self, new_config):
        """Check if working_dir is correctly changed."""
        with OrionState(experiments=[], trials=[]) as cfg:
            space = new_config['metadata']['priors']
            new_config['working_dir'] = './'
            exp = experiment_builder.build(space=space, **new_config)
            assert exp._init_done is True
            storage = get_storage()
            found_config = list(storage.fetch_experiments({'name': 'supernaekei',
                                                           'metadata.user': 'tsirif'}))

        found_config = found_config[0]
        space = found_config['metadata']['priors']
        exp = experiment_builder.build(space=space, **found_config)
        assert exp.working_dir == './'

    def test_working_dir_works_when_db_absent(self, database, new_config):
        """Check if working_dir is correctly when absent from the database."""
        with OrionState(experiments=[], trials=[]) as cfg:
            space = new_config['metadata']['priors']
            exp = experiment_builder.build(space=space, **new_config)
            assert exp._init_done is True
            storage = get_storage()
            found_config = list(storage.fetch_experiments({'name': 'supernaekei',
                                                           'metadata.user': 'tsirif'}))

        found_config = found_config[0]
        space = found_config['metadata']['priors']
        exp = experiment_builder.build(space=space, **found_config)
        assert exp.working_dir == ''

    def test_configuration_hit_no_diffs(self, new_config):
        """Return a configuration dict according to an experiment object.

        Before initialization is done, it can be the case that the pair (`name`,
        user's name) has not hit the database. return a yaml compliant form
        of current state, to be used with :mod:`orion.core.cli.esolve_config`.
        """
        with OrionState(experiments=[new_config], trials=[]) as cfg:
            experiment_count_before = count_experiments()
            space = new_config['metadata']['priors']
            exp = experiment_builder.build(space=space, **new_config)
            assert experiment_count_before == count_experiments()

        assert exp._init_done is True
        new_config['algorithms']['dumbalgo']['done'] = False
        new_config['algorithms']['dumbalgo']['judgement'] = None
        new_config['algorithms']['dumbalgo']['scoring'] = 0
        new_config['algorithms']['dumbalgo']['suspend'] = False
        new_config['algorithms']['dumbalgo']['value'] = 5
        new_config['algorithms']['dumbalgo']['seed'] = None
        new_config['producer']['strategy'] = "NoParallelStrategy"
        new_config.pop('_id')
        new_config.pop('something_to_be_ignored')
        assert exp.configuration == new_config

    def test_instantiation_after_init(self, new_config):
        """Verify that algo, space and refers was instanciated properly"""
        with OrionState(experiments=[new_config], trials=[]) as cfg:
            space = new_config['metadata']['priors']
            exp = experiment_builder.build(space=space, **new_config)

        assert exp._init_done is True
        assert isinstance(exp.algorithms, BaseAlgorithm)
        assert isinstance(exp.space, Space)
        assert isinstance(exp.refers['adapter'], BaseAdapter)

    def test_try_set_after_race_condition(self, new_config, monkeypatch):
        """Cannot set a configuration after init if it looses a race
        condition.

        The experiment from process which first writes to db is initialized
        properly. The experiment which looses the race condition cannot be
        initialized and needs to be rebuilt.
        """ 

        with OrionState(experiments=[new_config], trials=[]) as cfg:
            experiment_count_before = count_experiments()

            def insert_race_condition(*args, **kwargs):
                if insert_race_condition.count == 0:
                    data = {}
                else:
                    data = new_config

                insert_race_condition.count += 1

                return data

            insert_race_condition.count = 0

            monkeypatch.setattr(experiment_builder, 'fetch_config_from_db', insert_race_condition)

            space = new_config['metadata']['priors']
            exp = experiment_builder.build(space=space, **new_config)

            assert experiment_count_before == count_experiments()

        # Should be called
        # - once in build(), 
        # - then register fails, 
        # - then called once again in build,
        # - then called in build_view to evaluate the conflicts
        assert insert_race_condition.count == 3

    def test_algorithm_config_with_just_a_string(self):
        """Test that configuring an algorithm with just a string is OK."""

        name = 'supernaedo3'
        space = {'x': 'uniform(0,10)'}
        algorithms = 'dumbalgo'

        with OrionState(experiments=[], trials=[]) as cfg:
            exp = experiment_builder.build(name=name, space=space, algorithms=algorithms)

        assert exp.configuration['algorithms'] == {
            'dumbalgo': {
                'done': False,
                'judgement': None,
                'scoring': 0,
                'suspend': False,
                'value': 5,
                'seed': None}}

    def test_new_child_with_branch(self):
        """Check that experiment is not incremented when branching with a new name."""
        name = 'parent'
        space = {'x': 'uniform(0, 10)'}
        metadata = {'priors': space}  # TODO: remove when space in db

        with OrionState(experiments=[], trials=[]) as cfg:
            parent = experiment_builder.build(name=name, space=space, metadata=metadata)

            assert parent.name == name
            assert parent.version == 1

            child_name = 'child'

            child = experiment_builder.build(name=name, branching={'branch_to': child_name})

            assert child.name == child_name
            assert child.version == 1
            assert child.refers['parent_id'] == parent.id

            child_name = 'child2'

            child = experiment_builder.build(name=child_name, branching={'branch_from': name})

            assert child.name == child_name
            assert child.version == 1
            assert child.refers['parent_id'] == parent.id

    def test_no_increment_when_child_exist(self):
        """Check that experiment cannot be incremented when asked for v1 while v2 exists."""
        name = 'parent'
        space = {'x': 'uniform(0,10)'}
        metadata = {'priors': space}  # TODO: remove when space in db

        with OrionState(experiments=[], trials=[]) as cfg:
            parent = experiment_builder.build(name=name, space=space, metadata=metadata)
            child = experiment_builder.build(name=name, space={'x': 'loguniform(1,10)'})
            assert child.name == parent.name
            assert parent.version == 1
            assert child.version == 2

            with pytest.raises(ValueError) as exc_info:
                experiment_builder.build(name=name, version=1, space={'x': 'loguniform(1,10)'})
            assert 'Configuration is different and generates a branching' in str(exc_info.value)

    def test_race_condition_wout_version(self, monkeypatch):
        """Test that an experiment loosing the race condition during version increment raises
        RaceCondition if version number was not specified.
        """
        name = 'parent'
        space = {'x': 'uniform(0,10)'}
        metadata = {'priors': space}  # TODO: remove when space in db

        with OrionState(experiments=[], trials=[]) as cfg:
            parent = experiment_builder.build(name, space=space, metadata=metadata)
            child = experiment_builder.build(name=name, space={'x': 'loguniform(1,10)'})
            assert child.name == parent.name
            assert parent.version == 1
            assert child.version == 2

            # Either
            # 1. 
            #     fetch_config_from_db only fetch parent
            #     test_version finds other child
            #     -> Detect race condition looking at conflicts
            # 2.
            #     fetch_config_from_db only fetch parent
            #     test_version do not find other child
            #     -> DuplicateKeyError

            def insert_race_condition_1(self, query):
                is_auto_version_query = (query == {'name': name, 'refers.parent_id': parent.id})
                if is_auto_version_query:
                    data = [child.configuration]
                # First time the query returns no other child
                elif insert_race_condition_1.count < 1:
                    data = [parent.configuration]
                else:
                    data = [parent.configuration, child.configuration]

                insert_race_condition_1.count += int(is_auto_version_query)

                return data

            insert_race_condition_1.count = 0

            monkeypatch.setattr(get_storage().__class__, 'fetch_experiments',
                                insert_race_condition_1)

            with pytest.raises(RaceCondition) as exc_info:
                child2 = experiment_builder.build(name=name, space={'x': 'loguniform(1,10)'})
            assert 'There was likely a race condition during version' in str(exc_info.value)

            def insert_race_condition_2(self, query):
                is_auto_version_query = (query == {'name': name, 'refers.parent_id': parent.id})
                # First time the query returns no other child
                if is_auto_version_query:
                    data = []
                elif insert_race_condition_2.count < 1:
                    data = [parent.configuration]
                else:
                    data = [parent.configuration, child.configuration]

                insert_race_condition_2.count += int(is_auto_version_query)

                return data

            insert_race_condition_2.count = 0

            monkeypatch.setattr(get_storage().__class__, 'fetch_experiments',
                                insert_race_condition_2)

            with pytest.raises(RaceCondition) as exc_info:
                child2 = experiment_builder.build(name=name, space={'x': 'loguniform(1,10)'})
            assert 'There was a race condition during branching.' in str(exc_info.value)

    def test_race_condition_w_version(self, monkeypatch):
        """Test that an experiment loosing the race condition during version increment cannot
        be resolved automatically if a version number was specified.

        Note that if we would raise RaceCondition, the conflict would still occur since
        the version number fetched will not be the new one from the resolution but the requested
        one. Therefore raising and handling RaceCondition would lead to infinite recursion in
        the experiment builder.
        """
        name = 'parent'
        space = {'x': 'uniform(0,10)'}
        metadata = {'priors': space}  # TODO: remove when space in db

        with OrionState(experiments=[], trials=[]) as cfg:
            parent = experiment_builder.build(name, space=space, metadata=metadata)
            child = experiment_builder.build(name=name, space={'x': 'loguniform(1,10)'})
            assert child.name == parent.name
            assert parent.version == 1
            assert child.version == 2

            # Either
            # 1. 
            #     fetch_config_from_db only fetch parent
            #     test_version finds other child
            #     -> Detect race condition looking at conflicts
            # 2.
            #     fetch_config_from_db only fetch parent
            #     test_version do not find other child
            #     -> DuplicateKeyError

            def insert_race_condition_1(self, query):
                is_auto_version_query = (query == {'name': name, 'refers.parent_id': parent.id})
                if is_auto_version_query:
                    data = [child.configuration]
                # First time the query returns no other child
                elif insert_race_condition_1.count < 1:
                    data = [parent.configuration]
                else:
                    data = [parent.configuration, child.configuration]

                insert_race_condition_1.count += int(is_auto_version_query)

                return data

            insert_race_condition_1.count = 0

            monkeypatch.setattr(get_storage().__class__, 'fetch_experiments',
                                insert_race_condition_1)

            with pytest.raises(ValueError) as exc_info:
                child2 = experiment_builder.build(name=name, version=1,
                                                  space={'x': 'loguniform(1,10)'})
            assert 'Configuration is different and generates' in str(exc_info.value)

            def insert_race_condition_2(self, query):
                is_auto_version_query = (query == {'name': name, 'refers.parent_id': parent.id})
                # First time the query returns no other child
                if is_auto_version_query:
                    data = []
                elif insert_race_condition_2.count < 1:
                    data = [parent.configuration]
                else:
                    data = [parent.configuration, child.configuration]

                insert_race_condition_2.count += int(is_auto_version_query)

                return data

            insert_race_condition_2.count = 0

            monkeypatch.setattr(get_storage().__class__, 'fetch_experiments',
                                insert_race_condition_2)

            with pytest.raises(RaceCondition) as exc_info:
                child2 = experiment_builder.build(name=name, version=1, space={'x': 'loguniform(1,10)'})
            assert 'There was a race condition during branching.' in str(exc_info.value)


class TestInitExperimentView(object):
    """Create new ExperimentView instance."""

    def test_empty_experiment_view(self):
        """Hit user name, but exp_name does not hit the db."""
        with OrionState(experiments=[], trials=[]) as cfg:
            with pytest.raises(ValueError) as exc_info:
                experiment_builder.build_view('supernaekei')
            assert ("No experiment with given name 'supernaekei' and version '*'"
                    in str(exc_info.value))

    def test_existing_experiment_view(self, new_config):
        """Hit exp_name + user's name in the db, fetch most recent entry."""
        with OrionState(experiments=[new_config], trials=[]) as cfg:
            exp = experiment_builder.build_view(name='supernaekei')

        assert exp._experiment._init_done is True

        assert exp._id == new_config['_id']
        assert exp.name == new_config['name']
        assert exp.configuration['refers'] == new_config['refers']
        assert exp.metadata == new_config['metadata']
        assert exp.pool_size == new_config['pool_size']
        assert exp.max_trials == new_config['max_trials']
        assert exp.version == new_config['version']
        assert isinstance(exp.refers['adapter'], BaseAdapter)
        assert exp.algorithms.configuration == new_config['algorithms']

        with pytest.raises(AttributeError):
            exp.this_is_not_in_config = 5

        # Test that experiment.update_completed_trial indeed exists
        exp._experiment.update_completed_trial
        with pytest.raises(AttributeError):
            exp.update_completed_trial

        with pytest.raises(AttributeError):
            exp.register_trial

        with pytest.raises(AttributeError):
            exp.reserve_trial
