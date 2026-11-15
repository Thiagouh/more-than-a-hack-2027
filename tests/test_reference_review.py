import copy
import tempfile
import unittest
from pathlib import Path
from PIL import Image
from tests import test_team_pilot_safety as safety
from src.team_identification.pilot import sha256
from src.team_identification.manifests import freeze,load_frozen,write_json,read_json
from src.team_identification.reference_review import PROFILE,completeness
from scripts.review_team_references import review


class ReferenceReviewTests(unittest.TestCase):
    def candidate(self,root):
        helper=safety.PilotSafetyTests();helper.setUp()
        path,spec=helper.make_candidate(root)
        for i,team in enumerate(('H','A')):
            image=root/f'gk-{team}.png';Image.new('RGB',(40,80),(2,180,i*70+10)).save(image)
            mask=root/f'gk-mask-{team}.png';Image.new('L',(40,80),255).save(mask)
            spec['references'].append(dict(label=f'GK-{team}',team=team,role='goalkeeper',
                lighting='sun',approved=True,goalkeeper_verified=True,image=image.name,mask=mask.name,
                image_sha256=sha256(image),mask_sha256=sha256(mask)))
        spec['reference_profile']=PROFILE
        for r in spec['references']:
            r.update(source_sha256='1'*64,source_segment_id='reviewed-reference-segment',box=[0,0,40,80])
            r['visual_review']=dict(decision='approve',reviewer='synthetic-test',notes='synthetic fixture only',
                torso_verified=True,lighting_verified=True,team_and_role_verified=True,halo_visible=False,
                goalkeeper_evidence='Synthetic role-coded fixture; not a real goalkeeper')
            for k in ('image_sha256','mask_sha256','team','role','lighting','source_sha256','source_segment_id'):
                r['visual_review'][k]=r[k]
        write_json(path,spec)
        return path,spec

    def test_complete_roundtrip_and_portable_copy(self):
        import shutil
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path,spec=self.candidate(root)
            self.assertTrue(completeness(spec)['ready'])
            freeze(path,root/'frozen',require_complete=True)
            shutil.copytree(root/'frozen',root/'portable')
            _,loaded=load_frozen(root/'portable/frozen-references.json')
            self.assertEqual(loaded['reference_profile'],PROFILE)
            self.assertEqual(len(loaded['references']),7)

    def test_both_goalkeepers_required_before_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path,spec=self.candidate(root)
            spec['references'][-1]['approved']=False;write_json(path,spec)
            with self.assertRaisesRegex(ValueError,'A/goalkeeper'):
                freeze(path,root/'bad',require_complete=True)
            self.assertFalse((root/'bad').exists())

    def test_load_rechecks_complete_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path,_=self.candidate(root);freeze(path,root/'frozen')
            bundle=root/'frozen/frozen-references.json';spec=read_json(bundle)
            spec['references']=[r for r in spec['references'] if r['role']!='goalkeeper']
            write_json(bundle,spec)
            with self.assertRaisesRegex(ValueError,'goalkeeper'):load_frozen(bundle)

    def test_approval_requires_all_review_and_origin_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            _,spec=self.candidate(Path(tmp))
            for field,value in [('torso_verified',False),('halo_visible',True),('reviewer',''),
                                ('image_sha256','2'*64),('source_segment_id','other')]:
                bad=copy.deepcopy(spec);bad['references'][0]['visual_review'][field]=value
                self.assertFalse(completeness(bad)['ready'],field)
            bad=copy.deepcopy(spec);bad['references'][-1]['visual_review']['goalkeeper_evidence']=''
            self.assertFalse(completeness(bad)['ready'])
            bad=copy.deepcopy(spec);bad['references'][0]['source_sha256']=None
            self.assertFalse(completeness(bad)['ready'])

    def test_reject_review_never_freezes_and_preserves_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path,spec=self.candidate(root);before=sha256(path)
            decisions=root/'decisions.json'
            entries=[dict(r['visual_review'],label=r['label'],decision='reject',notes='Rejected synthetic fixture') for r in spec['references']]
            write_json(decisions,dict(candidate_sha256=before,reviewer='test',references=entries))
            report=review(path,root/'review',decisions,freeze_complete=True)
            self.assertFalse(report['references_frozen'])
            self.assertEqual(report['rejected_count'],7)
            self.assertFalse((root/'review/frozen').exists())
            self.assertEqual(sha256(path),before)
            checked=read_json(root/'review/reviewed-candidates.json')
            self.assertEqual(len(checked['references']),7)
            self.assertTrue(all(not r['approved'] for r in checked['references']))

    def test_changed_candidate_review_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path,spec=self.candidate(root)
            decisions=root/'decisions.json';write_json(decisions,dict(candidate_sha256='0'*64,reviewer='test'))
            with self.assertRaisesRegex(ValueError,'unchanged'):review(path,root/'bad',decisions)
            self.assertFalse((root/'bad').exists())

    def test_complete_review_freezes_only_after_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path,spec=self.candidate(root)
            decisions=root/'decisions.json'
            write_json(decisions,dict(candidate_sha256=sha256(path),reviewer='test',
                references=[dict(r['visual_review'],label=r['label'],goalkeeper_verified=r.get('goalkeeper_verified',False)) for r in spec['references']]))
            report=review(path,root/'review',decisions,freeze_complete=True)
            self.assertTrue(report['references_frozen'])
            self.assertEqual(report['frozen_count'],7)

    def test_visual_flags_do_not_override_bad_geometry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path,spec=self.candidate(root)
            spec['references'][0]['box']=[0,0,10,80];write_json(path,spec)
            decisions=root/'decisions.json'
            write_json(decisions,dict(candidate_sha256=sha256(path),reviewer='test',
                references=[dict(r['visual_review'],label=r['label']) for r in spec['references']]))
            report=review(path,root/'review',decisions,freeze_complete=True)
            self.assertEqual(report['status'],'invalid_reference_no_freeze')
            self.assertFalse(report['references_frozen'])
            self.assertFalse((root/'review/frozen').exists())


if __name__ == '__main__':unittest.main()
