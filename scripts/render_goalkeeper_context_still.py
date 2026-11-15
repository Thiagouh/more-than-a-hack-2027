"""Render a real RGB still with an independently computed contextual hypothesis.

Not tactical tracking, not a before/after model benchmark. Manual context review
and automatic visual/possession execution are explicitly distinguished.
"""
import argparse
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
from src.team_identification.manifests import read_json,write_json
from src.team_identification.pilot import sha256


def render(image_path,result_path,out):
    spec=read_json(result_path)
    if spec.get('schema')!='goalkeeper-context-hypotheses-v1' or len(spec['results'])!=1:
        raise ValueError('One independently computed contextual result required')
    row=spec['results'][0]
    if row['image_sha256']!=sha256(image_path):raise ValueError('Still image mismatch')
    image=Image.open(image_path).convert('RGB')
    result=row['contextual_hypothesis'];evidence=row['context_evidence']
    team=result['team'];name={'H':'Barcelona (H)','A':'Athletic (A)',None:'Incerto'}[team]
    out.mkdir(parents=True,exist_ok=False)
    canvas=Image.new('RGB',(1600,1140),'#f2f5f7');d=ImageDraw.Draw(canvas)
    fontpath=Path('/System/Library/Fonts/Supplemental/Arial.ttf')
    def font(n):return ImageFont.truetype(str(fontpath),n) if fontpath.exists() else ImageFont.load_default(size=n)
    def text(x,y,value,size=24,color='#193044'):
        d.text((x,y),value,font=font(size),fill=color)
    def lines(x,y,values,size=24,color='#193044',step=35):
        for i,value in enumerate(values):text(x,y+i*step,value,size,color)
    text(28,25,'Goleiro com a bola | regra contextual',38)
    text(28,81,'Quadro real da gravação enviada · relógio visível 00:18 · primeiro tempo',24,'#536577')
    # Preserve the whole wide view, including scoreboard and defended goal.
    photo=image.resize((1120,630),Image.Resampling.LANCZOS);canvas.paste(photo,(28,150))
    box=row['visual_decision']['box'];scale=1120/image.width
    x,y,X,Y=[int(v*scale) for v in box]
    x+=28;X+=28;y+=150;Y+=150
    d.rounded_rectangle((x-5,y-5,X+5,Y+5),radius=5,outline='#21ef96',width=4)
    label='Goleiro · '+name
    d.rounded_rectangle((x-25,y-50,x+310,y-12),radius=6,fill='#123a34')
    text(x-15,y-45,label,22,'white')
    # Outcome of the actual contextual function, not an invented SAM prediction.
    d.rounded_rectangle((1172,150,1572,338),radius=16,fill='#184ca7' if team else '#746d54')
    lines(1192,173,['EQUIPE PELA REGRA',name],size=27,color='white',step=50)
    text(1192,282,'Hipótese contextual',21,'#d7e6ff')
    d.rounded_rectangle((1172,358,1572,646),radius=16,fill='white')
    text(1192,380,'Indícios revisados',27)
    lines(1192,428,['1º tempo confirmado','Gol defendido à esquerda','Uniforme verde distinto','Goleiro junto ao próprio gol'],size=22,step=44)
    d.rounded_rectangle((1172,666,1572,876),radius=16,fill='#e5eaf0')
    text(1192,688,'POSSE AUTOMÁTICA',25)
    text(1192,738,'Não reavaliada',30)
    lines(1192,792,['Este teste identifica a equipe.','Não mede erro de posse.'],size=21,step=32)
    # Show actual hand/ball evidence; no synthetic ball or player positions.
    zoom=image.crop((1160,460,1490,880));zoom.thumbnail((220,280))
    canvas.paste(zoom,(28,808))
    d.rectangle((27,807,28+zoom.width,808+zoom.height),outline='#b5c4ce',width=1)
    text(270,812,'Bola nas mãos: visível neste quadro',30)
    lines(270,866,['Verde + gol à esquerda + primeiro tempo → Barcelona (H).',
        'A regra foi executada sobre indícios revisados manualmente.',
        'Não houve detecção automática pelo SAM nem nova execução da posse.',
        'O resultado anterior “desconhecido” não foi medido neste quadro.'],size=23,step=39)
    lines(1172,914,['Amostra: 31 s da gravação','Relógio da partida: 00:18',
        'Sem sincronização tactical','Sem gabarito oficial neste still'],size=20,step=33)
    text(28,1105,'Imagem real anotada · contexto manual + regra executada · não é uma simulação do tracking',20,'#536577')
    path=out/'goleiro-barcelona-contexto.png';canvas.save(path)
    write_json(out/'provenance.json',dict(image=str(image_path.resolve()),image_sha256=sha256(image_path),
        contextual_result=str(result_path.resolve()),contextual_result_sha256=sha256(result_path),
        output_sha256=sha256(path),renderer_sha256=sha256(__file__),recording_sample_s=31,
        visible_match_clock='00:18',tactical_alignment_known=False,synthetic_pixels_in_scene=False,
        manual_context_review=True,visual_model_executed=False,possession_model_reexecuted=False,
        classification=result['classification'],contextual_reason=result['reason']))
    print(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image',type=Path,required=True);p.add_argument('--result',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();render(a.image,a.result,a.out)


if __name__=='__main__':main()
