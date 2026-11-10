"""Regresiones de datos/modelos offline; no ejecutan partidas ni benchmarks."""
import os
for nombre in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[nombre]='1'
os.environ['NUMBA_DISABLE_JIT']='1'
import sys
from pathlib import Path
import unittest
import numpy as np

EQUIPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(EQUIPO),str(EQUIPO.parents[1]/'sistema')]
import eval3
from fase5_features import features,probabilidades,particion_hash
from fase5_modelos import ridge,logistica,escala,pesos_grupos,genetico,fitness_genoma
from fase5_rivales import uct,estilo
from motor import Estado
from fase5_adaptacion import ajustar


class Fase5(unittest.TestCase):
    def test_actualizacion_centrada(self):
        rng=np.random.default_rng(17);X=rng.normal(size=(40,3));y=rng.integers(0,2,40).astype(float)
        g=np.array([f'g{i//4}' for i in range(40)]);ref=np.array([.4,-.3,.1])
        w=ajustar(X,y,g,ref,1e10)
        np.testing.assert_allclose(w,ref,atol=1e-7)
        a=ajustar(X,y,g,np.zeros(3),100)
        b=logistica(X,y,100,pesos_grupos(g))
        np.testing.assert_allclose(a,b,atol=1e-7)

    def test_uct_azar_y_respuesta_rival(self):
        e=Estado.desde_dict({'blancas':['a8'],'negras':['a9','b9'],'turno':'blancas',
                            'puntos':{'blancas':2,'negras':2}})
        j,info=uct(e,7,1000,False)
        self.assertEqual(str(j),'a8-b9')
        self.assertTrue(.44<info['aristas'][0]['valor']<.56)
        e=Estado.desde_dict({'blancas':['a8','e4'],'negras':['i2','f5'],'turno':'blancas',
                            'puntos':{'blancas':2,'negras':2}})
        j,info=uct(e,7,512,False)
        self.assertIn(str(j),['a8-a9','a8-b9'])
        mala=next(x for x in info['aristas'] if x['jugada']=='e4-e5')
        self.assertLess(mala['valor'],.4)
        for nombre in ['agresivo','carrera']:
            j,_=estilo(e,nombre,7)
            self.assertIn(str(j),['a8-a9','a8-b9'])

    def test_features_y_perspectivas(self):
        tablero=np.zeros((1,81),np.int8)
        tablero[0,[0,10,38,71]]=1
        tablero[0,[9,39,40,80]]=2
        columnas=np.r_[np.arange(253),np.arange(263,266)]
        for color in (0,1):
            for a in range(3):
                for b in range(3):
                    pts=np.array([[a,b]])
                    calculado=features(tablero,pts,np.array([color]))[0]
                    esperado=eval3.features_anchas_py(tablero[0],pts[0],color)
                    np.testing.assert_allclose(calculado[columnas],esperado[columnas],atol=0)
                    opuesto=features(tablero,pts,np.array([1-color]))[0]
                    np.testing.assert_allclose(calculado,-opuesto,atol=0)

    def test_ridge_y_gradiente_logistico(self):
        rng=np.random.default_rng(7)
        X=rng.normal(size=(80,4))
        w=np.array([.7,-1,.3,.1])
        np.testing.assert_allclose(ridge(X,X@w,1e-8),w,atol=1e-8)
        y=(rng.random(80)<probabilidades(X@w)).astype(float)
        fitted=logistica(X,y,1)
        s=escala(X)
        grad=(X/s).T@(y-probabilidades(X@fitted)) - fitted*s
        self.assertLess(np.max(np.abs(grad)),1e-6)

    def test_grupos_y_hash_estable(self):
        g=np.array(['a']*2+['b']*5)
        peso=pesos_grupos(g)
        self.assertAlmostEqual(peso[g=='a'].sum(),peso[g=='b'].sum())
        a=particion_hash(['a','b','c'],sal='prueba')
        b=particion_hash(['x','c','a','b'],sal='prueba')
        np.testing.assert_array_equal(a,b[[2,3,1]])

    def test_ga_reproducible_y_referencia(self):
        rng=np.random.default_rng(7)
        Z=rng.normal(size=(50,4));y=rng.integers(0,2,50);peso=np.ones(50)
        a=genetico(Z,y,peso);b=genetico(Z,y,peso)
        self.assertEqual(a,b)
        self.assertEqual(a['evaluaciones'],72)
        self.assertLessEqual(a['fitness'],fitness_genoma(np.ones(3),Z,y,peso,'logistica'))
        self.assertTrue(all(.75<=x<=1.25 for x in a['mejor_genoma']))


if __name__=='__main__':unittest.main()
