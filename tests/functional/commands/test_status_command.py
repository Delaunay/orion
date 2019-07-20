#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Perform a functional test of the status command."""
import os

import orion.core.cli


def test_no_experiments(clean_db, monkeypatch, capsys):
    """Test status with no experiments."""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    orion.core.cli.main(['status'])

    captured = capsys.readouterr().out

    assert captured == ""


def test_experiment_without_trials_wout_ac(clean_db, one_experiment, capsys):
    """Test status with only one experiment and no trials."""
    orion.core.cli.main(['status'])

    captured = capsys.readouterr().out

    expected = """\
test_single_exp
===============
empty


"""
    assert captured == expected


def test_experiment_wout_success_wout_ac(clean_db, single_without_success, capsys):
    """Test status with only one experiment and no successful trial."""
    orion.core.cli.main(['status'])

    captured = capsys.readouterr().out

    expected = """\
test_single_exp
===============
status         quantity
-----------  ----------
broken                1
interrupted           1
new                   1
reserved              1
suspended             1


"""
    assert captured == expected


def test_experiment_w_trials_wout_ac(clean_db, single_with_trials, capsys):
    """Test status with only one experiment and all trials."""
    orion.core.cli.main(['status'])

    captured = capsys.readouterr().out

    expected = """\
test_single_exp
===============
status         quantity    min obj
-----------  ----------  ---------
broken                1
completed             1          0
interrupted           1
new                   1
reserved              1
suspended             1


"""
    assert captured == expected


def test_two_unrelated_w_trials_wout_ac(clean_db, unrelated_with_trials, capsys):
    """Test two unrelated experiments, with all types of trials."""
    orion.core.cli.main(['status'])

    captured = capsys.readouterr().out

    expected = """\
test_double_exp
===============
status         quantity
-----------  ----------
broken                1
completed             1
interrupted           1
new                   1
reserved              1
suspended             1


test_single_exp
===============
status         quantity    min obj
-----------  ----------  ---------
broken                1
completed             1          0
interrupted           1
new                   1
reserved              1
suspended             1


"""

    assert captured == expected


def test_two_related_w_trials_wout_ac(clean_db, family_with_trials, capsys):
    """Test two related experiments, with all types of trials."""
    orion.core.cli.main(['status'])

    captured = capsys.readouterr().out

    expected = """\
test_double_exp
===============
status         quantity
-----------  ----------
broken                1
completed             1
interrupted           1
new                   1
reserved              1
suspended             1


  test_double_exp_child
  =====================
  status         quantity
  -----------  ----------
  broken                1
  completed             1
  interrupted           1
  new                   1
  reserved              1
  suspended             1


"""

    assert captured == expected


def test_three_unrelated_wout_ac(clean_db, three_experiments_with_trials, capsys):
    """Test three unrelated experiments with all types of trials."""
    orion.core.cli.main(['status'])

    captured = capsys.readouterr().out

    expected = """\
test_double_exp
===============
status         quantity
-----------  ----------
broken                1
completed             1
interrupted           1
new                   1
reserved              1
suspended             1


  test_double_exp_child
  =====================
  status         quantity
  -----------  ----------
  broken                1
  completed             1
  interrupted           1
  new                   1
  reserved              1
  suspended             1


test_single_exp
===============
status         quantity    min obj
-----------  ----------  ---------
broken                1
completed             1          0
interrupted           1
new                   1
reserved              1
suspended             1


"""

    assert captured == expected


def test_three_related_wout_ac(clean_db, three_family_with_trials, capsys):
    """Test three related experiments with all types of trials."""
    orion.core.cli.main(['status'])

    captured = capsys.readouterr().out

    expected = """\
test_double_exp
===============
status         quantity
-----------  ----------
broken                1
completed             1
interrupted           1
new                   1
reserved              1
suspended             1


  test_double_exp_child
  =====================
  status         quantity
  -----------  ----------
  broken                1
  completed             1
  interrupted           1
  new                   1
  reserved              1
  suspended             1


  test_double_exp_child2
  ======================
  status         quantity
  -----------  ----------
  broken                1
  completed             1
  interrupted           1
  new                   1
  reserved              1
  suspended             1


"""

    assert captured == expected


def test_three_related_branch_wout_ac(clean_db, three_family_branch_with_trials, capsys):
    """Test three related experiments with all types of trials."""
    orion.core.cli.main(['status'])

    captured = capsys.readouterr().out

    expected = """\
test_double_exp
===============
status         quantity
-----------  ----------
broken                1
completed             1
interrupted           1
new                   1
reserved              1
suspended             1


  test_double_exp_child
  =====================
  status         quantity
  -----------  ----------
  broken                1
  completed             1
  interrupted           1
  new                   1
  reserved              1
  suspended             1


    test_double_exp_grand_child
    ===========================
    status         quantity
    -----------  ----------
    broken                1
    completed             1
    interrupted           1
    new                   1
    reserved              1
    suspended             1


"""

    assert captured == expected


def test_one_wout_trials_w_a_wout_c(clean_db, one_experiment, capsys):
    """Test experiments, without trials, with --all."""
    orion.core.cli.main(['status', '--all'])

    captured = capsys.readouterr().out

    expected = """\
test_single_exp
===============
id    status    best objective
----  --------  ----------------


"""

    assert captured == expected


def test_one_w_trials_w_a_wout_c(clean_db, single_with_trials, capsys):
    """Test experiment, with all trials, with --all."""
    orion.core.cli.main(['status', '--all'])

    captured = capsys.readouterr().out

    expected = """\
test_single_exp
===============
id                                status         min obj
--------------------------------  -----------  ---------
ec6ee7892275400a9acbf4f4d5cd530d  broken
c4c44cb46d075546824e2a32f800fece  completed            0
2b5059fa8fdcdc01f769c31e63d93f24  interrupted
7e8eade99d5fb1aa59a1985e614732bc  new
507496236ff94d0f3ad332949dfea484  reserved
caf6afc856536f6d061676e63d14c948  suspended


"""

    assert captured == expected


def test_one_wout_success_w_a_wout_c(clean_db, single_without_success, capsys):
    """Test experiment, without success, with --all."""
    orion.core.cli.main(['status', '--all'])

    captured = capsys.readouterr().out

    expected = """\
test_single_exp
===============
id                                status
--------------------------------  -----------
ec6ee7892275400a9acbf4f4d5cd530d  broken
2b5059fa8fdcdc01f769c31e63d93f24  interrupted
7e8eade99d5fb1aa59a1985e614732bc  new
507496236ff94d0f3ad332949dfea484  reserved
caf6afc856536f6d061676e63d14c948  suspended


"""

    assert captured == expected


def test_two_unrelated_w_a_wout_c(clean_db, unrelated_with_trials, capsys):
    """Test two unrelated experiments with --all."""
    orion.core.cli.main(['status', '--all'])

    captured = capsys.readouterr().out

    expected = """\
test_double_exp
===============
id                                status
--------------------------------  -----------
a8f8122af9e5162e1e2328fdd5dd75db  broken
ab82b1fa316de5accb4306656caa07d0  completed
c187684f7c7d9832ba953f246900462d  interrupted
1497d4f27622520439c4bc132c6046b1  new
bd0999e1a3b00bf8658303b14867b30e  reserved
b9f1506db880645a25ad9b5d2cfa0f37  suspended


test_single_exp
===============
id                                status         min obj
--------------------------------  -----------  ---------
ec6ee7892275400a9acbf4f4d5cd530d  broken
c4c44cb46d075546824e2a32f800fece  completed            0
2b5059fa8fdcdc01f769c31e63d93f24  interrupted
7e8eade99d5fb1aa59a1985e614732bc  new
507496236ff94d0f3ad332949dfea484  reserved
caf6afc856536f6d061676e63d14c948  suspended


"""

    assert captured == expected


def test_two_related_w_a_wout_c(clean_db, family_with_trials, capsys):
    """Test two related experiments with --all."""
    orion.core.cli.main(['status', '--all'])

    captured = capsys.readouterr().out

    expected = """\
test_double_exp
===============
id                                status
--------------------------------  -----------
a8f8122af9e5162e1e2328fdd5dd75db  broken
ab82b1fa316de5accb4306656caa07d0  completed
c187684f7c7d9832ba953f246900462d  interrupted
1497d4f27622520439c4bc132c6046b1  new
bd0999e1a3b00bf8658303b14867b30e  reserved
b9f1506db880645a25ad9b5d2cfa0f37  suspended


  test_double_exp_child
  =====================
  id                                status
  --------------------------------  -----------
  45c359f1c753a10f2cfeca4073a3a7ef  broken
  e79761fe3fc24dcbb7850939ede84b68  completed
  69928939792d67f6fe30e9b8459be1ec  interrupted
  5f4a9c92b8f7c26654b5b37ecd3d5d32  new
  58c4019fb2f92da88a0e63fafb36b3da  reserved
  82f340cb9d90cbf024169926b60aeef2  suspended


"""

    assert captured == expected


def test_three_unrelated_w_a_wout_c(clean_db, three_experiments_with_trials, capsys):
    """Test three unrelated experiments with --all."""
    orion.core.cli.main(['status', '--all'])

    captured = capsys.readouterr().out

    expected = """\
test_double_exp
===============
id                                status
--------------------------------  -----------
a8f8122af9e5162e1e2328fdd5dd75db  broken
ab82b1fa316de5accb4306656caa07d0  completed
c187684f7c7d9832ba953f246900462d  interrupted
1497d4f27622520439c4bc132c6046b1  new
bd0999e1a3b00bf8658303b14867b30e  reserved
b9f1506db880645a25ad9b5d2cfa0f37  suspended


  test_double_exp_child
  =====================
  id                                status
  --------------------------------  -----------
  45c359f1c753a10f2cfeca4073a3a7ef  broken
  e79761fe3fc24dcbb7850939ede84b68  completed
  69928939792d67f6fe30e9b8459be1ec  interrupted
  5f4a9c92b8f7c26654b5b37ecd3d5d32  new
  58c4019fb2f92da88a0e63fafb36b3da  reserved
  82f340cb9d90cbf024169926b60aeef2  suspended


test_single_exp
===============
id                                status         min obj
--------------------------------  -----------  ---------
ec6ee7892275400a9acbf4f4d5cd530d  broken
c4c44cb46d075546824e2a32f800fece  completed            0
2b5059fa8fdcdc01f769c31e63d93f24  interrupted
7e8eade99d5fb1aa59a1985e614732bc  new
507496236ff94d0f3ad332949dfea484  reserved
caf6afc856536f6d061676e63d14c948  suspended


"""

    assert captured == expected


def test_three_related_w_a_wout_c(clean_db, three_family_with_trials, capsys):
    """Test three related experiments with --all."""
    orion.core.cli.main(['status', '--all'])

    captured = capsys.readouterr().out

    expected = """\
test_double_exp
===============
id                                status
--------------------------------  -----------
a8f8122af9e5162e1e2328fdd5dd75db  broken
ab82b1fa316de5accb4306656caa07d0  completed
c187684f7c7d9832ba953f246900462d  interrupted
1497d4f27622520439c4bc132c6046b1  new
bd0999e1a3b00bf8658303b14867b30e  reserved
b9f1506db880645a25ad9b5d2cfa0f37  suspended


  test_double_exp_child
  =====================
  id                                status
  --------------------------------  -----------
  45c359f1c753a10f2cfeca4073a3a7ef  broken
  e79761fe3fc24dcbb7850939ede84b68  completed
  69928939792d67f6fe30e9b8459be1ec  interrupted
  5f4a9c92b8f7c26654b5b37ecd3d5d32  new
  58c4019fb2f92da88a0e63fafb36b3da  reserved
  82f340cb9d90cbf024169926b60aeef2  suspended


  test_double_exp_child2
  ======================
  id                                status
  --------------------------------  -----------
  d0f4aa931345bfd864201b7dd93ae667  broken
  5005c35be98025a24731d7dfdf4423de  completed
  c9fa9f0682a370396c8c4265c4e775dd  interrupted
  3d8163138be100e37f1656b7b591179e  new
  790d3c4c965e0d91ada9cbdaebe220cf  reserved
  6efdb99952d5f80f55adbba9c61dc288  suspended


"""

    assert captured == expected


def test_three_related_branch_w_a_wout_c(clean_db, three_family_branch_with_trials, capsys):
    """Test three related experiments in a branch with --all."""
    orion.core.cli.main(['status', '--all'])

    captured = capsys.readouterr().out

    expected = """\
test_double_exp
===============
id                                status
--------------------------------  -----------
a8f8122af9e5162e1e2328fdd5dd75db  broken
ab82b1fa316de5accb4306656caa07d0  completed
c187684f7c7d9832ba953f246900462d  interrupted
1497d4f27622520439c4bc132c6046b1  new
bd0999e1a3b00bf8658303b14867b30e  reserved
b9f1506db880645a25ad9b5d2cfa0f37  suspended


  test_double_exp_child
  =====================
  id                                status
  --------------------------------  -----------
  45c359f1c753a10f2cfeca4073a3a7ef  broken
  e79761fe3fc24dcbb7850939ede84b68  completed
  69928939792d67f6fe30e9b8459be1ec  interrupted
  5f4a9c92b8f7c26654b5b37ecd3d5d32  new
  58c4019fb2f92da88a0e63fafb36b3da  reserved
  82f340cb9d90cbf024169926b60aeef2  suspended


    test_double_exp_grand_child
    ===========================
    id                                status
    --------------------------------  -----------
    994602c021c470989d6f392b06cb37dd  broken
    24c228352de31010d8d3bf253604a82d  completed
    a3c8a1f4c80c094754c7217a83aae5e2  interrupted
    d667f5d719ddaa4e1da2fbe568e11e46  new
    a40748e487605df3ed04a5ac7154d4f6  reserved
    229622a6d7132c311b7d4c57a08ecf08  suspended


"""

    assert captured == expected


def test_two_unrelated_w_c_wout_a(clean_db, unrelated_with_trials, capsys):
    """Test two unrelated experiments with --collapse."""
    orion.core.cli.main(['status', '--collapse'])

    captured = capsys.readouterr().out

    expected = """\
test_double_exp
===============
status         quantity
-----------  ----------
broken                1
completed             1
interrupted           1
new                   1
reserved              1
suspended             1


test_single_exp
===============
status         quantity    min obj
-----------  ----------  ---------
broken                1
completed             1          0
interrupted           1
new                   1
reserved              1
suspended             1


"""

    assert captured == expected


def test_two_related_w_c_wout_a(clean_db, family_with_trials, capsys):
    """Test two related experiments with --collapse."""
    orion.core.cli.main(['status', '--collapse'])

    captured = capsys.readouterr().out

    expected = """\
test_double_exp
===============
status         quantity
-----------  ----------
broken                1
completed             1
interrupted           1
new                   2
reserved              1
suspended             1


"""

    assert captured == expected


def test_three_unrelated_w_c_wout_a(clean_db, three_experiments_with_trials, capsys):
    """Test three unrelated experiments with --collapse."""
    orion.core.cli.main(['status', '--collapse'])

    captured = capsys.readouterr().out

    expected = """\
test_double_exp
===============
status         quantity
-----------  ----------
broken                1
completed             1
interrupted           1
new                   2
reserved              1
suspended             1


test_single_exp
===============
status         quantity    min obj
-----------  ----------  ---------
broken                1
completed             1          0
interrupted           1
new                   1
reserved              1
suspended             1


"""

    assert captured == expected


def test_three_related_w_c_wout_a(clean_db, three_family_with_trials, capsys):
    """Test three related experiments with --collapse."""
    orion.core.cli.main(['status', '--collapse'])

    captured = capsys.readouterr().out

    expected = """\
test_double_exp
===============
status         quantity
-----------  ----------
broken                1
completed             1
interrupted           1
new                   3
reserved              1
suspended             1


"""

    assert captured == expected


def test_three_related_branch_w_c_wout_a(clean_db, three_family_branch_with_trials, capsys):
    """Test three related experiments with --collapse."""
    orion.core.cli.main(['status', '--collapse'])

    captured = capsys.readouterr().out

    expected = """\
test_double_exp
===============
status         quantity
-----------  ----------
broken                1
completed             1
interrupted           1
new                   3
reserved              1
suspended             1


"""

    assert captured == expected


def test_two_unrelated_w_ac(clean_db, unrelated_with_trials, capsys):
    """Test two unrelated experiments with --collapse and --all."""
    orion.core.cli.main(['status', '--collapse', '--all'])

    captured = capsys.readouterr().out

    expected = """\
test_double_exp
===============
id                                status
--------------------------------  -----------
a8f8122af9e5162e1e2328fdd5dd75db  broken
ab82b1fa316de5accb4306656caa07d0  completed
c187684f7c7d9832ba953f246900462d  interrupted
1497d4f27622520439c4bc132c6046b1  new
bd0999e1a3b00bf8658303b14867b30e  reserved
b9f1506db880645a25ad9b5d2cfa0f37  suspended


test_single_exp
===============
id                                status         min obj
--------------------------------  -----------  ---------
ec6ee7892275400a9acbf4f4d5cd530d  broken
c4c44cb46d075546824e2a32f800fece  completed            0
2b5059fa8fdcdc01f769c31e63d93f24  interrupted
7e8eade99d5fb1aa59a1985e614732bc  new
507496236ff94d0f3ad332949dfea484  reserved
caf6afc856536f6d061676e63d14c948  suspended


"""

    assert captured == expected


def test_two_related_w_ac(clean_db, family_with_trials, capsys):
    """Test two related experiments with --collapse and --all."""
    orion.core.cli.main(['status', '--collapse', '--all'])

    captured = capsys.readouterr().out

    expected = """\
test_double_exp
===============
id                                status
--------------------------------  -----------
a8f8122af9e5162e1e2328fdd5dd75db  broken
ab82b1fa316de5accb4306656caa07d0  completed
c187684f7c7d9832ba953f246900462d  interrupted
1497d4f27622520439c4bc132c6046b1  new
ad6ea2decff2f298594b948fdaea03b2  new
bd0999e1a3b00bf8658303b14867b30e  reserved
b9f1506db880645a25ad9b5d2cfa0f37  suspended


"""

    assert captured == expected


def test_three_unrelated_w_ac(clean_db, three_experiments_with_trials, capsys):
    """Test three unrelated experiments with --collapse and --all."""
    orion.core.cli.main(['status', '--collapse', '--all'])

    captured = capsys.readouterr().out

    expected = """\
test_double_exp
===============
id                                status
--------------------------------  -----------
a8f8122af9e5162e1e2328fdd5dd75db  broken
ab82b1fa316de5accb4306656caa07d0  completed
c187684f7c7d9832ba953f246900462d  interrupted
1497d4f27622520439c4bc132c6046b1  new
ad6ea2decff2f298594b948fdaea03b2  new
bd0999e1a3b00bf8658303b14867b30e  reserved
b9f1506db880645a25ad9b5d2cfa0f37  suspended


test_single_exp
===============
id                                status         min obj
--------------------------------  -----------  ---------
ec6ee7892275400a9acbf4f4d5cd530d  broken
c4c44cb46d075546824e2a32f800fece  completed            0
2b5059fa8fdcdc01f769c31e63d93f24  interrupted
7e8eade99d5fb1aa59a1985e614732bc  new
507496236ff94d0f3ad332949dfea484  reserved
caf6afc856536f6d061676e63d14c948  suspended


"""

    assert captured == expected


def test_three_related_w_ac(clean_db, three_family_with_trials, capsys):
    """Test three related experiments with --collapse and --all."""
    orion.core.cli.main(['status', '--collapse', '--all'])

    captured = capsys.readouterr().out

    expected = """\
test_double_exp
===============
id                                status
--------------------------------  -----------
a8f8122af9e5162e1e2328fdd5dd75db  broken
ab82b1fa316de5accb4306656caa07d0  completed
c187684f7c7d9832ba953f246900462d  interrupted
1497d4f27622520439c4bc132c6046b1  new
ad6ea2decff2f298594b948fdaea03b2  new
f357f8c185ccab3037c65dcf721b9e71  new
bd0999e1a3b00bf8658303b14867b30e  reserved
b9f1506db880645a25ad9b5d2cfa0f37  suspended


"""

    assert captured == expected


def test_three_related_branch_w_ac(clean_db, three_family_branch_with_trials, capsys):
    """Test three related experiments in a branch with --collapse and --all."""
    orion.core.cli.main(['status', '--collapse', '--all'])

    captured = capsys.readouterr().out

    expected = """\
test_double_exp
===============
id                                status
--------------------------------  -----------
a8f8122af9e5162e1e2328fdd5dd75db  broken
ab82b1fa316de5accb4306656caa07d0  completed
c187684f7c7d9832ba953f246900462d  interrupted
1497d4f27622520439c4bc132c6046b1  new
ad6ea2decff2f298594b948fdaea03b2  new
8f763d441db41d0f56e4e6aa40cc2321  new
bd0999e1a3b00bf8658303b14867b30e  reserved
b9f1506db880645a25ad9b5d2cfa0f37  suspended


"""

    assert captured == expected


def test_experiment_wout_child_w_name(clean_db, unrelated_with_trials, capsys):
    """Test status with the name argument and no child."""
    orion.core.cli.main(['status', '--name', 'test_single_exp'])

    captured = capsys.readouterr().out

    expected = """test_single_exp
===============
status         quantity    min obj
-----------  ----------  ---------
broken                1
completed             1          0
interrupted           1
new                   1
reserved              1
suspended             1


"""

    assert captured == expected


def test_experiment_w_child_w_name(clean_db, three_experiments_with_trials, capsys):
    """Test status with the name argument and one child."""
    orion.core.cli.main(['status', '--name', 'test_double_exp'])

    captured = capsys.readouterr().out

    expected = """\
test_double_exp
===============
status         quantity
-----------  ----------
broken                1
completed             1
interrupted           1
new                   1
reserved              1
suspended             1


"""

    assert captured == expected
