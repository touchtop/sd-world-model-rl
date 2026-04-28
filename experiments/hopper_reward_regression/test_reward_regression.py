import unittest

from experiments.hopper_reward_regression.reward_regression_suite import (
    STATE_CASES,
    TRANSITION_CASES,
    evaluate_state_case,
    evaluate_transition_case,
)


class RewardRegressionTest(unittest.TestCase):
    def test_state_cases_unittest(self):
        for case in STATE_CASES:
            result = evaluate_state_case(case)
            self.assertTrue(result["passed"], msg=str(result))

    def test_transition_cases_unittest(self):
        for case in TRANSITION_CASES:
            result = evaluate_transition_case(case)
            self.assertTrue(result["passed"], msg=str(result))


def test_state_cases():
    for case in STATE_CASES:
        result = evaluate_state_case(case)
        assert result["passed"], result


def test_transition_cases():
    for case in TRANSITION_CASES:
        result = evaluate_transition_case(case)
        assert result["passed"], result


if __name__ == "__main__":
    unittest.main()
