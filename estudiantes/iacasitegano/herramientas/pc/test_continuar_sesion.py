"""La continuación respeta parada/rechazo y nunca crea un bucle de reintentos."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from continuar_sesion_fase5 import supervisar

class Continuacion(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        (self.root/'PROTOCOLO.json').write_text('{}')
    def tearDown(self):self.temp.cleanup()
    def test_no_rescata_interrupcion_ni_rechazo(self):
        for phase in ['interrumpido','rechazado_seleccion','rechazado_test','error_revisar','pendiente_tecnica_y_partidas_oficiales']:
            with self.subTest(phase=phase),patch('continuar_sesion_fase5.subprocess.Popen') as proc:
                (self.root/'ESTADO.json').write_text(json.dumps(dict(fase=phase)))
                r=supervisar(self.root,self.root,1,.01)
                self.assertEqual(r['fase'],'sin_reinicio');proc.assert_not_called()
    def test_pausa_reanuda_una_vez_subproceso_real(self):
        (self.root/'ESTADO.json').write_text(json.dumps(dict(fase='pausa_datos_incompletos')))
        worker=self.root/'worker.py'
        worker.write_text("from pathlib import Path\nimport json\np=Path(__file__).parent\n(p/'marcador').write_text('una ejecucion')\n(p/'ESTADO.json').write_text(json.dumps({'fase':'pendiente_tecnica_y_partidas_oficiales'}))\n")
        with patch('continuar_sesion_fase5.comando',return_value=[sys.executable,str(worker)]) as cmd:
            r=supervisar(self.root,self.root,1,.01);second=supervisar(self.root,self.root,1,.01)
        self.assertEqual(r['exit'],0);self.assertEqual(second['fase'],'sin_reinicio')
        self.assertEqual(cmd.call_count,1);self.assertFalse((self.root/'AUTO_ACTIVO.json').exists())

if __name__=='__main__':unittest.main()
