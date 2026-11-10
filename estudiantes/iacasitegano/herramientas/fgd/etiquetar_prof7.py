"""NOTA DE ENTREGA (documentacion, no se usa al jugar): este es el script con el que se generaron
data/etiquetas_prof7.jsonl. Se corrio en nuestro laboratorio con la variante "FGEH" del buscador (mismo
expectiminimax, pero con tabla de jugadas-pista en vez del ordenamiento barato, y con
CAL_B0=0.003005, CAL_STM=0.08276) y los pesos de data/pesos_eval3_FGE_etiquetador.npz. El valor de
alfa-beta no depende del orden de jugadas, asi que las etiquetas son del mismo tipo que las que daria el
buscador entregado, pero ESTE archivo no corre aqui tal cual: necesita las partidas oficiales (carpetas
oficiales/ y g411/, no incluidas) y el codigo de FGEH, que no se entrega. Se conserva para que se vea el
procedimiento: 12 posiciones por partida (ply entre 8 y n-4), busqueda por profundidad iterativa hasta 7
sin tope de nodos, y se guarda el valor de la raiz.
Etiqueta posiciones de partidas oficiales con el valor de una busqueda PROFUNDA (sin tope de nodos).
Uso: python etiquetar.py <shard> <nshards> <profundidad> <salida.jsonl> [maxpos]
Reanudable: salta posiciones ya etiquetadas. Necesita oficiales/ (descomprimir oficiales.tgz aqui) y g411/."""
import sys,glob,json,time,random,numpy as np
from pathlib import Path
AQUI=Path(__file__).resolve().parent; sys.path.insert(0,str(AQUI/'sistema')); sys.path.insert(0,str(AQUI/'FGEH'))
from motor import Estado, Jugada
import busqueda_eval3 as b
with np.load(AQUI/'FGEH/data/pesos_eval3_ligero.npz') as d: b.cargar_pesos(d['pesos'])
b.calentar()
shard,nsh,PROF,out=int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]),sys.argv[4]; maxpos=int(sys.argv[5]) if len(sys.argv)>5 else 10**9
files=sorted(glob.glob(str(AQUI/'oficiales/*/*/partida_*.json'))+glob.glob(str(AQUI/'g411/g*.json')))
rng=random.Random(7); pos=[]   # (archivo, ply)
for f in files:
    n=len(json.load(open(f))['jugadas']); ks=sorted(rng.sample(range(8,max(n-4,9)),min(12,max(n-12,1))))
    for k in ks: pos.append((f,k))
rng.shuffle(pos)
hechos=set()
if Path(out).exists():
    for l in open(out):
        try: r=json.loads(l); hechos.add((r['f'],r['k']))
        except: pass
cache={}
def estado_en(f,k):
    d=cache.get(f)
    if d is None:
        cache.clear(); d=json.load(open(f)); cache[f]=d
    e=Estado.inicial()
    for j in d['jugadas'][:k]:
        c=j.get('combate'); jug=Jugada.desde_texto(j['jugada'])
        e=e.aplicar(jug,gana_ataque=bool(c['gano'])) if c else e.aplicar(jug)
    return e
nuevos=0; t0=time.time()
with open(out,'a') as fo:
    for i,(f,k) in enumerate(pos):
        if i%nsh!=shard or i//nsh>=maxpos: continue
        key=(Path(f).as_posix().split('/')[-3]+'/'+Path(f).as_posix().split('/')[-2]+'/'+Path(f).name if 'oficiales' in f else Path(f).name,k)
        if key in hechos: continue
        e=estado_en(f,k)
        if e.terminado(): continue
        t,p,ci=b._tablero_desde_estado(e); pesos,g=b._pesos_y_ganancia()
        cont=np.zeros(2,np.int64); tc=np.zeros(2*b.TT_TAM,np.uint64); tv=np.zeros(2*b.TT_TAM,np.float64); o=dd=-1
        for dep in range(1,PROF+1): o,dd,v,ab,comp=b._mejor_jugada_tt(t,p,ci,dep,pesos,g,o,dd,cont,0,tc,tv)
        fo.write(json.dumps(dict(f=key[0],k=k,tab=''.join(str(int(x)) for x in t),pts=[int(p[0]),int(p[1])],col=int(ci),v=float(v),mov=[int(o),int(dd)],nod=int(cont[0]),prof=PROF))+'\n'); fo.flush(); nuevos+=1
        if nuevos%50==0: print(nuevos,round(time.time()-t0),'s',flush=True)
print('listo',nuevos)
