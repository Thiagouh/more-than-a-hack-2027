"""Experimental goalkeeper team hypotheses from reviewed match context.

Left/right mean defended ends of the pitch in the reviewed wide view, never
unqualified screen x coordinates. No ATD, identity tracking or possession truth.
"""
POLICY_VERSION = 'bar-ath-defending-ends-v1'
CONTEXT_PROFILE = 'contextual-goalkeepers-v1'


def validate_policy(policy):
    if (not isinstance(policy,dict) or policy.get('version') != POLICY_VERSION
            or policy.get('match') != 'BAR-ATH' or policy.get('confirmed_by') != 'user'
            or policy.get('first_half') != {'left':'H','right':'A'}
            or policy.get('second_half') != {'left':'A','right':'H'}
            or policy.get('side_semantics') != 'reviewed_defended_pitch_end'):
        raise ValueError('Reviewed BAR-ATH defending-end policy required')


def infer_goalkeeper_team(policy, evidence):
    """A heuristic hypothesis, not a verified uniform reference or probability.

    Role candidacy needs a distinctive kit AND proximity to a defended goal;
    green/teal alone cannot identify a goalkeeper or a team. Facts must describe
    the pictured play, including on replays, rather than the playback clock.
    """
    validate_policy(policy)
    result=dict(classification='uncertain',team=None,role='unknown',
        method='goalkeeper_context',policy_version=POLICY_VERSION,
        evidence_kind='contextual_heuristic',calibrated_probability=False,
        visual_reference_verified=False)
    checks = (
        ('goalkeeper_candidate', 'goalkeeper_candidate_not_established'),
        ('distinctive_kit', 'distinctive_kit_not_established'),
        ('near_defended_goal', 'defended_goal_proximity_unknown'),
        ('period_verified', 'period_unknown'),
        ('pitch_end_verified', 'pitch_end_unknown'),
        ('orientation_verified', 'view_orientation_unknown'),
    )
    for flag,reason in checks:
        if evidence.get(flag) is not True:
            return dict(result,reason=reason)
    if type(evidence.get('half')) is not int or evidence['half'] not in (1,2):
        return dict(result,reason='period_unknown')
    if evidence.get('pitch_end') not in ('left','right'):
        return dict(result,reason='pitch_end_unknown')
    if evidence.get('view') != 'wide_pitch' or evidence.get('play_origin_verified') is not True:
        return dict(result,reason='unverified_view_or_replay_origin')
    if evidence.get('kit_family') not in ('green','teal'):
        return dict(result,reason='kit_outside_user_hypothesis')
    if not isinstance(evidence.get('notes'),str) or not evidence['notes'].strip():
        return dict(result,reason='context_evidence_notes_required')
    half_key = 'first_half' if evidence['half']==1 else 'second_half'
    team = policy[half_key][evidence['pitch_end']]
    return dict(result,classification=team,team=team,role='goalkeeper',
                reason='distinctive_kit_and_reviewed_defending_end')
