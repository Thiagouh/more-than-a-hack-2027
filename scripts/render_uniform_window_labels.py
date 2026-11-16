"""Render only observed, identity-reviewed pixels; never read predicted classes."""
import argparse
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
from src.team_identification.manifests import read_json,write_json,load_image
from src.team_identification.pilot import sha256


def render(batch_path,out):
    batch=read_json(batch_path)
    out.mkdir(parents=True,exist_ok=False)
    font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial Bold.ttf',26)
    small=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',13)
    assets=[]
    for page in range((len(batch['sequences'])+15)//16):
        sheet=Image.new('RGB',(1600,1120),'#eef1f4');draw=ImageDraw.Draw(sheet)
        for i,case in enumerate(batch['sequences'][page*16:(page+1)*16]):
            x=i%4*400;y=i//4*280
            draw.text((x+8,y+3),str(case['id']),font=font,fill='#152b3a')
            for j,k in enumerate([0,25,40,50]):
                row=case['frames'][k];tx=x+j*100
                if row.get('crop'):
                    if row.get('association_verified') is not True:raise ValueError('Unverified pixels cannot appear as observations')
                    path=batch_path.parent/row['crop']
                    if sha256(path)!=row['crop_sha256']:raise ValueError('Pixels changed')
                    crop=load_image(path);scale=min(92/crop.width,205/crop.height)
                    crop=crop.resize((round(crop.width*scale),round(crop.height*scale)),Image.Resampling.NEAREST)
                    sheet.paste(crop,(tx+(100-crop.width)//2,y+37+(205-crop.height)//2))
                else:
                    draw.text((tx+6,y+110),'Sem imagem',font=small,fill='#53606c')
                    draw.text((tx+12,y+128),'verificada',font=small,fill='#53606c')
                draw.text((tx+26,y+256),f'{(k-50)/25:g}s',font=small,fill='#30495a')
        path=out/f'rotular-{page+1}.png';sheet.save(path)
        assets.append(dict(image=path.name,sha256=sha256(path)))
    write_json(out/'manifest.json',dict(batch_sha256=sha256(batch_path),contact_sheets=assets,
        rendering='Nearest-neighbor enlargement of real reviewed crops only; no generated pixels or class suggestions',
        instructions='One label per sequence ID; central target. Missing verified observations are blank. Incerto is permitted.'))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--batch',required=True,type=Path);p.add_argument('--out',required=True,type=Path);a=p.parse_args();render(a.batch,a.out)
