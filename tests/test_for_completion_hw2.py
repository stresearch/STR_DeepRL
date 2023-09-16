import pytest
from multigrid.scripts.train_ppo_cleanrl import main, parse_args

def test_train_ppo_clanrl_execution():
    args = parse_args([]) 
    try:
        main(args)
        assert True
    except Exception as e:
        pytest.fail(f"Execution failed with error: {e}")
