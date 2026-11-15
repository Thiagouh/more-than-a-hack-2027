"""Explicit visual review contract for the complete step-2 reference bundle."""
import re
from .goalkeeper_context import CONTEXT_PROFILE, validate_policy

PROFILE = 'complete-v1'
REQUIRED = ('H/player/sun', 'H/player/shade', 'A/player/sun',
            'A/player/shade', 'referee', 'H/goalkeeper', 'A/goalkeeper')


def text(value):
    return isinstance(value,str) and bool(value.strip())


def slot(row):
    if row['role'] == 'referee':
        return 'referee'
    if row['role'] == 'goalkeeper':
        return f"{row['team']}/goalkeeper"
    return f"{row['team']}/player/{row.get('lighting')}"


def completeness(spec):
    """Return actionable blockers without approving anything or opening a model."""
    approved = [r for r in spec['references'] if r.get('approved') is True]
    covered = {slot(r) for r in approved}
    profile = spec.get('reference_profile', PROFILE)
    required = REQUIRED[:5] if profile == CONTEXT_PROFILE else REQUIRED
    missing = [s for s in required if s not in covered]
    blockers = [f'missing_reference:{s}' for s in missing]
    if profile not in (PROFILE,CONTEXT_PROFILE):
        blockers.append('unknown_reference_profile')
    if profile == CONTEXT_PROFILE:
        try:
            validate_policy(spec.get('goalkeeper_context_policy'))
        except ValueError:
            blockers.append('reviewed_goalkeeper_context_policy_required')
    if spec.get('mask_reviewed') is not True or not text(spec.get('reviewer')):
        blockers.append('explicit_manifest_review_required')
    for row in approved:
        label = row['label']
        review = row.get('visual_review',{})
        reasons = []
        if (review.get('decision') != 'approve' or not text(review.get('reviewer'))
                or not text(review.get('notes'))):
            reasons.append('explicit_visual_review_required')
        for key in ('image_sha256','mask_sha256','team','role','lighting','source_sha256','source_segment_id'):
            if review.get(key) != row.get(key):
                reasons.append('review_asset_hash_mismatch:'+key)
        for key in ('torso_verified','lighting_verified','team_and_role_verified'):
            if review.get(key) is not True:
                reasons.append(key+'_required')
        if review.get('halo_visible') is not False:
            reasons.append('halo_not_cleared')
        if (not isinstance(row.get('source_sha256'),str)
                or not re.fullmatch('[0-9a-f]{64}',row['source_sha256'])
                or not text(row.get('source_segment_id'))):
            reasons.append('reviewed_source_segment_required')
        if row.get('lighting') not in ('sun','shade'):
            reasons.append('sun_or_shade_required')
        if row['role'] == 'player' and row.get('origin') != 'pre-kickoff':
            reasons.append('pre_kickoff_required')
        if row['role'] == 'goalkeeper':
            if row.get('goalkeeper_verified') is not True or not text(review.get('goalkeeper_evidence')):
                reasons.append('own_goalkeeper_verification_required')
        blockers.extend(f'{label}:{reason}' for reason in reasons)
    return dict(profile=profile,ready=not blockers,missing=missing,blockers=blockers,
                goalkeeper_strategy='contextual_heuristic' if profile==CONTEXT_PROFILE else 'own_verified_references',
                approved_count=len(approved),rejected_count=sum(r.get('visual_review',{}).get('decision')=='reject' for r in spec['references']),
                pending_count=sum(r.get('visual_review',{}).get('decision','pending')=='pending' for r in spec['references']),coverage={s:s in covered for s in required})


def require_complete(spec):
    report = completeness(spec)
    if not report['ready']:
        raise ValueError('Incomplete reviewed references: '+'; '.join(report['blockers']))
