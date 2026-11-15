"""Explicit v2 metadata for pure scorer unit tests, not real visual evidence."""
def person(label='p', x=0, role='player', classification='H', **changes):
    value = dict(label=label,box=[x,0,x+10,40],role=role,classification=classification,
        evaluable=role in ('player','goalkeeper','referee') and classification!='uncertain',
        partial=False,lighting='sun',occlusion='isolated')
    value.update(changes)
    return value


def truth_for(people, count=1):
    return dict(schema_version=2,stage='draft',reviewed=True,reviewer='unit-test',
        annotations_complete=True,evaluation_mode='full_frame',sample_kind='main',excluded_segments=[],
        frames=[dict(frame=i,timestamp_s=i,image=f'{i}.png',image_sha256=f'{i+1:064x}',
            pixel_sha256=f'{i+1:064x}',source_sha256='a'*64,width=200,height=100,
            source_segment_id=f'segment-{i}',group_id=f'group-{i}',temporal_origin_known=True,
            source_time_s=i*60,preparation='synthetic metadata for pure unit tests',
            evaluation_region=[[[0,0],[100,0],[100,100],[0,100]]],people=people) for i in range(count)])


def prediction(box=None, classification='H', role='player', **changes):
    value = dict(box=[0,0,10,40] if box is None else box,classification=classification,role=role)
    value.update(changes)
    return value
