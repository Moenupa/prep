"""CLI tests for the ppls command (exact stdout comparison)."""

from prep.cli import app

LIST_FORMAT_EXPECTED = """\
sft
verl
eval
clip
cls
others"""

TABLE_EXPECTED = """\
Showing results under 'out'
                                  sft
┏━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━┳━━━━━━━┳━━━━━┳━━━━━━┓
┃ ID    ┃ Default Source     ┃ Local Path ┃ Size ┃ train ┃ val ┃ test ┃
┡━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━╇━━━━━━━╇━━━━━╇━━━━━━┩
│ auto  │                    │            │      │ ✖     │ ✖   │ ✖    │
│ geo3k │ hiyouga/geometry3k │            │      │ ✖     │     │ ✖    │
└───────┴────────────────────┴────────────┴──────┴───────┴─────┴──────┘
                                 verl
┏━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━┳━━━━━━━┳━━━━━┳━━━━━━┓
┃ ID    ┃ Default Source     ┃ Local Path ┃ Size ┃ train ┃ val ┃ test ┃
┡━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━╇━━━━━━━╇━━━━━╇━━━━━━┩
│ auto  │                    │            │      │ ✖     │ ✖   │ ✖    │
│ geo3k │ hiyouga/geometry3k │            │      │ ✖     │     │ ✖    │
└───────┴────────────────────┴────────────┴──────┴───────┴─────┴──────┘
                               eval
┏━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━┳━━━━━━━┳━━━━━┳━━━━━━┓
┃ ID   ┃ Default Source ┃ Local Path ┃ Size ┃ train ┃ val ┃ test ┃
┡━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━╇━━━━━━━╇━━━━━╇━━━━━━┩
│ auto │                │            │      │ ✖     │ ✖   │ ✖    │
└──────┴────────────────┴────────────┴──────┴───────┴─────┴──────┘

                               cls
┏━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━┳━━━━━━━┳━━━━━┳━━━━━━┓
┃ ID   ┃ Default Source ┃ Local Path ┃ Size ┃ train ┃ val ┃ test ┃
┡━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━╇━━━━━━━╇━━━━━╇━━━━━━┩
│ auto │                │            │      │ ✖     │ ✖   │ ✖    │
└──────┴────────────────┴────────────┴──────┴───────┴─────┴──────┘"""


def test_ppls_list_format_stdout(runner, assert_stdout):
    result = runner.invoke(app, ["ppls", "--list-format"])

    assert_stdout(result, LIST_FORMAT_EXPECTED)


def test_ppls_table_stdout(runner, assert_stdout):
    result = runner.invoke(app, ["ppls"])

    assert_stdout(result, TABLE_EXPECTED)
