"""Evaluación lineal acotada desde quien tiene el turno; no es una red."""
import json
import math
from pathlib import Path

NOMBRES = ('constante', 'puntos_diferencia', 'material_diferencia',
           'avance_max_propio', 'avance_max_rival', 'avance_total_diferencia',
           'a_un_paso_propio', 'a_un_paso_rival', 'apoyo_diferencia')

NOMBRES_V2 = ('constante','puntos','material','tres_avanzados','movilidad',
              'apoyo','anotaciones_disponibles','exposicion_rival_menos_propia',
              'amenaza_tercer_punto')


def rasgos_v2(estado):
    t = estado._tablero
    puntos = estado._puntos
    avances = [[], []]
    movilidad = [0, 0]
    apoyos = [0, 0]
    anotar = [0., 0.]
    riesgo = [{}, {}]

    def apoyo(pos, c):
        f, col = divmod(pos, 9)
        f -= 1 if c == 0 else -1
        if not 0 <= f < 9:
            return 0
        return sum(int(t[f*9+col+d]) == c+1 for d in (-1,1) if 0 <= col+d < 9)

    for pos in range(81):
        pieza = int(t[pos])
        if not pieza:
            continue
        c = pieza-1
        f,col = divmod(pos,9)
        avances[c].append(f if c == 0 else 8-f)
        ap = apoyo(pos,c)
        apoyos[c] += ap
        destino_fila = f + (1 if c == 0 else -1)
        mejor_anotacion = 0.
        for d in (-1,0,1):
            if not 0 <= col+d < 9 or not 0 <= destino_fila < 9:
                continue
            destino = destino_fila*9+col+d
            v = int(t[destino])
            if v == pieza or (d == 0 and v != 0):
                continue
            movilidad[c] += 1
            p = 1. if v == 0 else min(.9,max(.2,.5+.15*(ap-apoyo(destino,1-c))))
            if v:
                riesgo[1-c][destino] = max(riesgo[1-c].get(destino,0.),p)
            if destino_fila == (8 if c == 0 else 0):
                # Cada peón puede anotar como máximo una vez en el próximo turno.
                mejor_anotacion = max(mejor_anotacion,p)
        anotar[c] += mejor_anotacion
    c = estado._color; r = 1-c
    return (1.,float(puntos[c]-puntos[r])/3.,(len(avances[c])-len(avances[r]))/18.,
            (sum(sorted(avances[c],reverse=True)[:3])-sum(sorted(avances[r],reverse=True)[:3]))/24.,
            (movilidad[c]-movilidad[r])/54.,(apoyos[c]-apoyos[r])/36.,
            (anotar[c]-anotar[r])/3.,(sum(riesgo[r].values())-sum(riesgo[c].values()))/18.,
            (anotar[c]*int(puntos[c]==2)-anotar[r]*int(puntos[r]==2))/3.)


def rasgos(estado):
    t = estado._tablero
    color = estado._color
    datos = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    for pos in range(81):
        pieza = int(t[pos])
        if pieza == 0:
            continue
        c = pieza - 1
        fila, col = divmod(pos, 9)
        avance = fila if c == 0 else 8 - fila
        d = datos[c]
        d[0] += 1
        d[1] = max(d[1], avance)
        d[2] += avance
        d[3] += avance == 7
        atras = fila - (1 if c == 0 else -1)
        if 0 <= atras < 9:
            for delta in (-1, 1):
                if 0 <= col + delta < 9 and int(t[atras*9+col+delta]) == pieza:
                    d[4] += 1
    a, b = datos[color], datos[1-color]
    return (1., (int(estado._puntos[color])-int(estado._puntos[1-color]))/3.,
            (a[0]-b[0])/18., a[1]/8., b[1]/8., (a[2]-b[2])/144.,
            a[3]/18., b[3]/18., (a[4]-b[4])/36.)


class Valor:
    def __init__(self, ruta, usar_finales=True):
        datos = json.loads(Path(ruta).read_text())
        if datos['formato']==4:
            from geometria import rasgos_v4 as nuevos,NOMBRES_V4 as nombres
            self.funcion_rasgos=nuevos
        elif datos['formato']==3:
            from geometria import rasgos as nuevos,NOMBRES as nombres
            self.funcion_rasgos=nuevos
        else:
            self.funcion_rasgos = rasgos if datos['formato']==1 else rasgos_v2
            nombres = NOMBRES if datos['formato']==1 else NOMBRES_V2
        if datos['formato'] not in (1,2,3,4) or datos['rasgos'] != list(nombres):
            raise ValueError('Modelo incompatible')
        self.pesos = tuple(float(x) for x in datos['pesos'])
        if len(self.pesos) != len(nombres) or any(not math.isfinite(x) for x in self.pesos):
            raise ValueError('Pesos inválidos')
        self.version = datos['version']
        self.formato = datos['formato']
        if self.formato>=3:
            import numpy as np
            self.pesos_array=np.asarray(self.pesos,dtype=np.float64)
        self.finales = {}
        archivo=Path(ruta).parent/'finales.npz'
        if self.formato>=3 and usar_finales and archivo.exists():
            import numpy as np
            with np.load(archivo,allow_pickle=False) as f:
                claves=f['estados'];valores=f['valores'];resueltos=f['resueltos']
                if claves.shape!=(len(valores),22) or resueltos.shape!=valores.shape or len(valores)>100000 or not np.all(resueltos):
                    raise ValueError('Tabla de finales incompatible')
                if not np.isfinite(valores).all() or np.any((valores<0)|(valores>1)):
                    raise ValueError('Valores de finales invalidos')
                self.finales={k.tobytes():float(v) for k,v in zip(claves,valores)}

    def evaluar(self, estado):
        if self.finales:
            import numpy as np
            if np.count_nonzero(estado._tablero)<=3:
                from compacto import identidad
                exacto=self.finales.get(identidad(estado))
                if exacto is not None:return exacto
        return self.evaluar_bruto(estado)

    def evaluar_bruto(self,estado):
        if self.formato>=3:
            import numpy as np
            z=float(np.dot(self.funcion_rasgos(estado),self.pesos_array))
        else:
            z = sum(x*w for x,w in zip(self.funcion_rasgos(estado), self.pesos))
        return 1. / (1. + math.exp(-max(-20., min(20., z))))
