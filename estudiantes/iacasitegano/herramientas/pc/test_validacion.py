"""Contratos que evitan fuga, olvido y reutilización del examen."""
import os
os.environ['NUMBA_DISABLE_JIT']='1'
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from validacion import preparar, seleccionar, examinar, cargar
from entrenar_replay import ajustar
from lanzar import commands
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[4]/'sistema'))
from motor import Estado
from fase5_rivales_robustos import montecarlo

class Puertas(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.base=np.zeros(275);self.base[267:]=[1,0,0,0,0,0,1,0]
        np.savez(self.root/'base.npz',pesos=self.base)
        w=self.base.copy();w[0]=1
        np.savez(self.root/'bueno.npz',pesos=w)
        self.dataset(self.root/'d.npz','nuevo',signal=True)
        self.dataset(self.root/'h.npz','historico',signal=False)
    def tearDown(self):self.temp.cleanup()
    def dataset(self,path,prefix,signal=True,counts=(32,16,16)):
        X=[];y=[];group=[];state=[];fold=[];turn=[];ply=[]
        for f,count in enumerate(counts):
            for g in range(count):
                for i in range(4):
                    x=np.zeros(267);x[0]=(-1 if i%2 else 1) if signal else 0
                    X.append(x);y.append(int(i%2==0));group.append(f'{prefix}_{f}_{g}')
                    state.append(f'{prefix}_{f}_{g}_{i}');fold.append(f);turn.append(i%2);ply.append([5,25,65,85][i])
        np.savez(path,X=X,y=y,grupo=group,estado=state,fold=fold,turno=turn,ply=ply)
    def freeze(self):
        return preparar(self.root/'d.npz',self.root/'h.npz',self.root/'base.npz',self.root/'gate','nuevo')
    def test_pipeline_y_examen_una_vez(self):
        self.freeze();r=seleccionar(self.root/'gate',self.root/'bueno.npz','uno')
        self.assertTrue(r['pasa']);self.assertFalse(r['test_evaluado'])
        e=examinar(self.root/'gate','uno');self.assertTrue(e['apto_para_puerta_tecnica']);self.assertFalse(e['promovido'])
        with self.assertRaises(FileExistsError):examinar(self.root/'gate','uno')
        with self.assertRaisesRegex(ValueError,'Examen ya abierto'):seleccionar(self.root/'gate',self.root/'bueno.npz','dos')
    def test_fuga_grupos_y_estados(self):
        for key in ['grupo','estado']:
            with self.subTest(key=key):
                d=cargar(self.root/'d.npz');d[key][np.flatnonzero(d['fold']==1)[0]]=d[key][0]
                np.savez(self.root/'fuga.npz',**d)
                with self.assertRaisesRegex(ValueError,'Fuga'):preparar(self.root/'fuga.npz',self.root/'h.npz',self.root/'base.npz',self.root/'gate','nuevo')
    def test_rechazo_por_retencion_aunque_mejore_nuevo(self):
        self.dataset(self.root/'h.npz','historico',signal=True)
        d=cargar(self.root/'h.npz');d['y']=1-d['y'];np.savez(self.root/'h.npz',**d)
        self.freeze();r=seleccionar(self.root/'gate',self.root/'bueno.npz','uno')
        self.assertTrue(r['checks']['mejora_validacion']);self.assertFalse(r['checks']['retencion']);self.assertFalse(r['pasa'])
        with self.assertRaisesRegex(ValueError,'Selección rechazada'):examinar(self.root/'gate','uno')
        self.assertFalse((self.root/'gate/EXAMEN_ABIERTO.json').exists())
    def test_dos_grupos_no_son_validacion_suficiente(self):
        self.dataset(self.root/'d.npz','nuevo',counts=(32,2,16));self.freeze()
        r=seleccionar(self.root/'gate',self.root/'bueno.npz','uno')
        self.assertFalse(r['checks']['cobertura_validacion']);self.assertFalse(r['pasa'])
    def test_datos_alterados_se_rechazan(self):
        self.freeze();p=self.root/'gate/validation.npz';p.write_bytes(p.read_bytes()+b'alterado')
        with self.assertRaisesRegex(ValueError,'modificado'):seleccionar(self.root/'gate',self.root/'bueno.npz','uno')
    def test_purga_cruces_con_replay_y_retencion(self):
        d=cargar(self.root/'d.npz');h=cargar(self.root/'h.npz')
        d['estado']=d['estado'].astype('U80')
        for f,hf in [(0,1),(1,0),(2,0)]:
            row=np.flatnonzero(d['fold']==f)[0]
            d['estado'][row]=h['estado'][np.flatnonzero(h['fold']==hf)[f]]
        np.savez(self.root/'d.npz',**d);c=self.freeze()
        self.assertEqual(c['purga_historico'],{'train':1,'validation':1,'test':1})
        for name in ['validation','test']:
            part=cargar(self.root/'gate'/(name+'.npz'))
            self.assertFalse(set(part['estado'])&set(h['estado'][np.isin(h['fold'],[0,1])]))
        part=cargar(self.root/'gate/train.npz')
        self.assertFalse(set(part['estado'])&set(h['estado'][h['fold']==1]))
    def test_mejor_candidato_y_azar_correcto(self):
        self.freeze();w=self.base.copy();w[0]=.5;np.savez(self.root/'menos.npz',pesos=w)
        seleccionar(self.root/'gate',self.root/'menos.npz','menos');seleccionar(self.root/'gate',self.root/'bueno.npz','mejor')
        with self.assertRaisesRegex(ValueError,'mejor candidato'):examinar(self.root/'gate','menos')
        self.assertTrue(examinar(self.root/'gate','mejor')['pasa'])
        e=Estado.desde_dict({'blancas':['a8'],'negras':['a9','b9'],'turno':'blancas','puntos':{'blancas':2,'negras':2}})
        j,info=montecarlo(e,7,1000,1)
        self.assertEqual(str(j),'a8-b9');self.assertTrue(.44<info['eval']<.56)
    def test_gradiente_y_colas_no_solapan(self):
        rng=np.random.default_rng(6);X=rng.normal(size=(60,3));y=rng.integers(0,2,60);ref=np.array([.1,.2,-.3])
        w=ajustar(X,y,np.ones(60),ref,1e10);np.testing.assert_allclose(w,ref,atol=1e-7)
        cmd=commands(self.root,self.root/'jobs',4,64,60,53000000)
        seeds=[int(c[c.index('--semilla')+1]) for c in cmd]
        self.assertEqual(len(set(seeds)),4);self.assertTrue(all(b-a>=32 for a,b in zip(seeds,seeds[1:])))

if __name__=='__main__':unittest.main()
