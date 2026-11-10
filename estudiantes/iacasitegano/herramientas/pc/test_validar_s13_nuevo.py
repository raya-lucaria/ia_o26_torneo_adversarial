"""Regresiones diminutas para aislamiento de datos y arranque; sin búsqueda/JIT."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import validar_s13_nuevo as v
from fase8a_soporte import Budget


class ContratosS13(unittest.TestCase):
    def setUp(self):self.budget=Budget()
    def tearDown(self):self.budget.wait()
    def test_espejo_puntos_y_turno(self):
        b=np.zeros(81,np.int8);b[3]=1;b[49]=2
        boards=np.array([b,b.reshape(9,9)[:,::-1].ravel(),b,b])
        ids=v.estados(boards,np.array([[0,0],[0,0],[1,0],[0,0]],np.int8),[0,0,0,1])
        self.assertEqual(ids[0],ids[1]);self.assertEqual(len(set(ids)),3)

    def test_purga_cruzada(self):
        ids=np.array(['viejo','viejo','mixto','mixto','limpio'])
        families=np.array(['carrera','carrera','carrera','agresivo','agresivo'])
        keep,counts=v.purgar(ids,families,{'viejo'})
        self.assertEqual(keep.tolist(),[False,False,False,False,True])
        self.assertEqual(counts['union'],4)

    def test_no_arranca_sin_liberacion(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'gate.json';p.write_text('{"estado":"ventana_ocupada"}')
            with patch.object(v,'contrato',return_value={}),patch.object(v,'run') as run:
                with self.assertRaises(ValueError):v.continuar(Path(td),p)
                run.assert_not_called()
                self.assertFalse((Path(td)/'PIPELINE_S13_ACTIVO.json').exists())

    def test_uct_opaco_y_banderas_de_cobertura(self):
        root=v.MAIN/'resultados/medicion/fase5_mac_k8s_20261029/laboratorio_mac'
        hashes={
            'buscador_sha256':root/'material/profesor_utilidad/busqueda_eval3.py',
            'script_sha256':root/'estudiantes/iacasitegano/herramientas/fase5_generar_robusto.py',
            'pesos_sha256':root/'estudiantes/iacasitegano/data/pesos_eval3_ligero.npz',
            'eval3_sha256':root/'estudiantes/iacasitegano/eval3.py',
            'motor_sha256':root/'sistema/motor/nucleo.py',
            'rivales_sha256':root/'estudiantes/iacasitegano/herramientas/fase5_rivales.py',
            'rivales_robustos_sha256':root/'estudiantes/iacasitegano/herramientas/fase5_rivales_robustos.py'}
        manifest=dict(partidas=16,profundidad=7,tope_nodos=3000000,uct_iteraciones=4096,
                      mc_muestras=128,mc_rollouts=20,**{k:v.sha(p) for k,p in hashes.items()})
        with tempfile.TemporaryDirectory() as td:
            out=Path(td);(out/'CONTRATO.json').write_text('{}')
            c=dict(semillas_base=[98000000,99000000,100000000,101000000],root=str(root),exclusiones=[],limites=v.LIMITES)
            for worker,seed in enumerate(c['semillas_base']):
                folder=out/f'lotes/worker_{worker}';(folder/'partidas').mkdir(parents=True);(folder/'shards').mkdir()
                (folder/'progreso.json').write_text(json.dumps(dict(estado='completo',completadas=16)))
                (folder/'manifest.json').write_text(json.dumps(dict(manifest,semilla=seed)))
                for game in range(16):
                    family=v.FAMILIAS[(game//2)%4];color='blancas' if game%2==0 else 'negras'
                    stem=f'{game:04d}_{family}_{color}'
                    j=folder/'partidas'/(stem+'.json');s=folder/'shards'/(stem+'.npz')
                    if family=='uct':
                        j.write_text('UCT NO JSON: prohibido consumir etiqueta')
                        s.write_bytes(b'UCT NO NPZ: prohibido consumir features/etiqueta')
                        continue
                    board=np.zeros((1,81),np.int8);case=worker*16+game
                    board[0,(case*2)%81]=1;board[0,(case*2+1)%81]=2
                    np.savez(s,tablero=board,puntos=np.array([[0,0]],np.int8),turno=np.array([game%2],np.int8),
                             ply=np.array([game]),ganador_blancas=np.array([1]))
                    meta={'semilla_maestra':seed+game//2,'pareja':game//2,'juego':0,color:'profesor_offline'}
                    j.write_text(json.dumps({'meta':meta,'shard_sha256':v.sha(s)}))
            with patch.object(v,'contrato',return_value=c):
                result=v.preparar(out)
            self.assertEqual(len(result['reserva_uct']),16)
            self.assertFalse(result['reserva_uct_evaluada'])
            self.assertEqual(len(result['fuentes']),48)
            self.assertFalse(result['cobertura']['pasa']) # Sin posiciones tardías suficientes.
            with np.load(out/'validacion_nueva.npz',allow_pickle=False) as z:
                self.assertNotIn('uct',set(z['familia']))

    def test_exposicion_unica_e_idempotencia(self):
        real=v.MAIN/'resultados/medicion/fase5_subespacio13_20261029/ajuste/candidato.npz'
        base=v.MAIN/'resultados/medicion/fase5_lote256_mac_20261029/validacion/base.npz'
        with tempfile.TemporaryDirectory() as td:
            out=Path(td);(out/'CONTRATO.json').write_text('{}')
            X=np.zeros((12,267));X[:,0]=np.tile([-1,0,1],4)
            file=out/'validacion_nueva.npz'
            np.savez(file,X=X,y=np.tile([0,1,1],4),grupo=np.array([str(i//2) for i in range(12)]),
                     turno=np.tile([0,1],6),ply=np.arange(12)*10,familia=np.tile(v.FAMILIAS[:3],4))
            ready=dict(datos_sha256=v.sha(file),cobertura={'pasa':True},contrato_sha256=v.sha(out/'CONTRATO.json'))
            (out/'VALIDACION_LISTA.json').write_text(json.dumps(ready))
            (out/'VALIDACION_LISTA.sha256').write_text(v.sha(out/'VALIDACION_LISTA.json'))
            c=dict(base=str(base),candidato=str(real),limites=v.LIMITES)
            with patch.object(v,'contrato',return_value=c):
                first=v.evaluar(out);self.assertEqual(first,v.evaluar(out))
                self.assertTrue((out/'VALIDACION_ABIERTA.json').exists())
                (out/'RESULTADO_VALIDACION.json').unlink()
                with self.assertRaises(FileExistsError):v.evaluar(out)


if __name__=='__main__':unittest.main()
