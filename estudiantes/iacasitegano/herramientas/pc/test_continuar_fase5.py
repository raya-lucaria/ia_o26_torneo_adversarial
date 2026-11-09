"""Comprueba que datos incompletos o fuentes cambiantes nunca alcancen ajuste."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from continuar_fase5 import continuar

class Flujo(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)/'root';self.out=Path(self.temp.name)/'out'
        for name in ['material/profesor_utilidad/busqueda_eval3.py','estudiantes/iacasitegano/data/pesos_eval3_ligero.npz','estudiantes/iacasitegano/herramientas/fase5_generar_robusto.py','estudiantes/iacasitegano/herramientas/fase5_rivales.py','estudiantes/iacasitegano/herramientas/fase5_rivales_robustos.py','estudiantes/iacasitegano/eval3.py','sistema/motor/nucleo.py']:
            f=self.root/name;f.parent.mkdir(parents=True,exist_ok=True);f.write_text('fixture')
    def tearDown(self):self.temp.cleanup()
    def test_no_entrena_con_datos_incompletos(self):
        with patch('continuar_fase5.run',return_value=dict(interrupted=False,exit_codes=[0]*4,complete=False)),patch('continuar_fase5.entrenar') as fit:
            r=continuar(self.root,self.out,.01)
        self.assertEqual(r['fase'],'pausa_datos_incompletos');fit.assert_not_called()
        self.assertFalse((self.out/'PIPELINE_ACTIVO.json').exists())
    def test_fuente_cambiada_impide_reanudar(self):
        with patch('continuar_fase5.run',return_value=dict(interrupted=False,exit_codes=[0]*4,complete=False)):
            continuar(self.root,self.out,.01)
        (self.root/'material/profesor_utilidad/busqueda_eval3.py').write_text('otra fuente')
        with patch('continuar_fase5.run') as gen, self.assertRaisesRegex(ValueError,'Protocolo modificado'):
            continuar(self.root,self.out,.01)
        gen.assert_not_called()
    def test_cambio_durante_generacion_impide_entrenar(self):
        def generation(*args):
            (self.root/'material/profesor_utilidad/busqueda_eval3.py').write_text('cambió')
            return dict(interrupted=False,exit_codes=[0]*4,complete=True)
        with patch('continuar_fase5.run',side_effect=generation),patch('continuar_fase5.entrenar') as fit,self.assertRaisesRegex(ValueError,'Fuentes cambiaron'):
            continuar(self.root,self.out,.01)
        fit.assert_not_called()
    def test_cadena_real_y_reanudacion_sin_reabrir_test(self):
        import json,shutil
        from test_validacion import Puertas
        from validacion import sha,examinar
        fixture=Puertas();fixture.setUp();self.addCleanup(fixture.tearDown)
        main=Path(self.temp.name)/'main'
        base=main/'estudiantes/iacasitegano/data/pesos_eval3_ligero.npz'
        base.parent.mkdir(parents=True);shutil.copyfile(fixture.root/'base.npz',base)
        def combine(lotes,salida):
            shutil.copyfile(fixture.root/'d.npz',salida)
            salida.with_suffix('.json').write_text(json.dumps(dict(datos_sha256=sha(salida),groups={'0':32,'1':16,'2':16})))
        with patch('continuar_fase5.MAIN',main),patch('continuar_fase5.HIST',fixture.root/'h.npz'),patch('continuar_fase5.protocolo',return_value={'fixture':True}),patch('continuar_fase5.run',return_value=dict(interrupted=False,exit_codes=[0]*4,complete=True)),patch('continuar_fase5.combinar',side_effect=combine),patch('continuar_fase5.examinar',wraps=examinar) as exam:
            first=continuar(self.root,self.out,.01);second=continuar(self.root,self.out,.01)
        self.assertEqual(first['fase'],'pendiente_tecnica_y_partidas_oficiales')
        self.assertEqual(second['fase'],first['fase']);self.assertFalse(first['promovido'])
        self.assertEqual(exam.call_count,1)

if __name__=='__main__':unittest.main()
