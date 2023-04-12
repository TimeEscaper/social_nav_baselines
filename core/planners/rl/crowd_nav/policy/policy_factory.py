from core.planners.rl.crowd_nav.policy.sarl import SARL

def none_policy():
    return None


policy_factory = dict()
# policy_factory['linear'] = Linear
# policy_factory['orca'] = ORCA
policy_factory['none'] = none_policy
# policy_factory['subgoal_expert'] = SubgoalExpert
#
# policy_factory['cadrl'] = CADRL
# policy_factory['lstm_rl'] = LstmRL
policy_factory['sarl'] = SARL
