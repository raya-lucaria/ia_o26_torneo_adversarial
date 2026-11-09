import unittest
from fractions import Fraction as F
from fase5_oraculo_cotas import Estado,Jugada,Oraculo,analizar_caso,probabilidad

def atacar(color='blancas'):
    return Estado.desde_dict(dict(blancas=['a8'] if color=='blancas' else ['a1','b1'],
        negras=['a9','b9'] if color=='blancas' else ['a2'],turno=color,puntos={'blancas':2,'negras':2}))

class Cotas(unittest.TestCase):
    def test_terminal_y_horizonte_desconocido(self):
        e=atacar();o=Oraculo('blancas',0)
        self.assertEqual(o.valor(e,0),(F(-1),F(1)))
        self.assertEqual(o.accion(e,Jugada.desde_texto('a8-b9'),1),(F(0),F(1)))
    def test_ataque_ambos_colores_y_oraculo_exhaustivo(self):
        def exhaustive(e,objective):
            if e.terminado():return F(1 if e.ganador()==objective else -1)
            vals=[sum(probabilidad(p)*exhaustive(h,objective) for p,h in e.siguiente(j)) for j in e.jugadas()]
            return(max if e.turno==objective else min)(vals)
        for color,move in [('blancas','a8-b9'),('negras','a2-b1')]:
            e=atacar(color);j=Jugada.desde_texto(move)
            reference=sum(probabilidad(p)*exhaustive(h,color) for p,h in e.siguiente(j))
            self.assertEqual(reference,0)
            self.assertEqual(Oraculo(color,1000).accion(e,j,2),(reference,reference))
    def test_dominancia_solo_cuando_separan_cotas(self):
        state=dict(blancas=['a8','e4'],negras=['i2','f5'],turno='blancas',puntos={'blancas':2,'negras':2})
        r=analizar_caso(dict(id='fixture',estado=state,color_propio='blancas',jugada_observada='e4-e5'),2,2000,1)
        self.assertTrue(r['dominancia_demostrada']);self.assertIn('a8-a9',r['alternativas_demostrablemente_mejores'])
        self.assertEqual(r['cotas']['e4-e5'],['-1','-1'])
    def test_presupuesto_no_inventa_valor_y_probabilidad(self):
        e=atacar();self.assertEqual(Oraculo('blancas',0).valor(e,20),(F(-1),F(1)))
        self.assertEqual(probabilidad(.65),F(13,20))
        with self.assertRaises(ValueError):probabilidad(.12345)
    def test_acciones_no_visitadas_conservan_ambas_cotas(self):
        e=Estado.desde_dict(dict(blancas=['e4','i8'],negras=['a2','f5'],turno='blancas',puntos={'blancas':2,'negras':2}))
        # La primera acción pierde si el rival juega a2-a1. Una acción posterior
        # gana inmediatamente. Agotar presupuesto no permite olvidar esa opción.
        for objective in ['blancas','negras']:
            self.assertEqual(Oraculo(objective,2).valor(e,2),(F(-1),F(1)))

if __name__=='__main__':unittest.main()
