"""Búsqueda adversarial con azar exacto, reloj y transposiciones.

El valor siempre se mide desde el jugador que mueve en ``estado``. Por eso,
después de una jugada, el valor del hijo se convierte con ``1 - V(hijo)``.
Un combate no se muestrea: se recorren sus dos resultados y se ponderan con la
probabilidad oficial del motor.
"""
import time
import struct
from compacto import identidad


class TiempoAgotado(Exception):
    """Señal interna: no se guarda una búsqueda incompleta en la caché."""


def _firma(estado):
    """Identidad completa que acompaña al hash Zobrist del motor."""
    return identidad(estado)


class TablaTransposicion:
    """Valores exactos a la profundidad buscada dentro de una decisión."""

    registro = struct.Struct('<Q22sHdB')

    def __init__(self, capacidad=50000):
        self.capacidad = capacidad
        self._datos = bytearray(capacidad * self.registro.size)
        self._entradas = 0
        self.aciertos = 0

    def leer(self, estado, profundidad):
        if not self.capacidad:
            return None
        posicion = ((estado.clave ^ (profundidad * 104729)) % self.capacidad) * self.registro.size
        clave, firma, prof, valor, ocupada = self.registro.unpack_from(self._datos, posicion)
        if ocupada and clave == estado.clave and prof == profundidad and firma == _firma(estado):
            self.aciertos += 1
            return valor
        return None

    def guardar(self, estado, profundidad, valor):
        if not self.capacidad:
            return
        posicion = ((estado.clave ^ (profundidad * 104729)) % self.capacidad) * self.registro.size
        if not self._datos[posicion + self.registro.size - 1]:
            self._entradas += 1
        self.registro.pack_into(self._datos, posicion, estado.clave, _firma(estado), profundidad, valor, 1)

    @property
    def entradas(self):
        return self._entradas


def terminal(estado):
    ganador = estado.ganador()
    return None if ganador is None else float(ganador == estado.turno)


def _ordenar(estado, jugadas, evaluar, plazo=float('inf'), reloj=time.perf_counter, ordenar=True):
    """Primero victorias ciertas y después hojas prometedoras.

    El orden no cambia el valor de expectiminimax; sí aprovecha mejor el reloj
    en la siguiente profundidad.
    """
    candidatos = []
    for posicion, jugada in enumerate(jugadas):
        if reloj() >= plazo:
            raise TiempoAgotado
        resultados = estado.siguiente(jugada)
        victoria_cierta = all(h.ganador() == estado.turno for p, h in resultados if p > 0.0)
        estimado = 0.
        if ordenar:
            for p,h in resultados:
                if reloj() >= plazo:
                    raise TiempoAgotado
                t = terminal(h)
                estimado += p * (1. - (evaluar(h) if t is None else t))
        candidatos.append((victoria_cierta, estimado, -posicion, jugada, resultados))
    if ordenar:
        candidatos.sort(key=lambda x:x[:3], reverse=True)
    return [(fila[3],fila[4],fila[0]) for fila in candidatos]


def valor(estado, profundidad, evaluar, plazo, tabla, reloj=time.perf_counter, ordenar=True,
          alfa=0.,beta=1.,podar=True):
    if reloj() >= plazo:
        raise TiempoAgotado
    resultado = terminal(estado)
    if resultado is not None:
        return resultado
    if profundidad == 0:
        return evaluar(estado)
    previo = tabla.leer(estado, profundidad)
    if previo is not None:
        return previo
    # Generación perezosa cuando no hay ordenamiento: no crea hermanos de más.
    if ordenar:
        opciones = _ordenar(estado, estado.jugadas(), evaluar, plazo, reloj)
    else:
        opciones = ((j,None,False) for j in estado.jugadas())
    alfa_inicial=alfa;mejor=0.
    for j,resultados,_ in opciones:
        v=valor_accion(estado,j,profundidad,evaluar,plazo,tabla,reloj,resultados,ordenar,alfa,beta,podar)
        mejor=max(mejor,v);alfa=max(alfa,mejor)
        if podar and mejor>=beta:
            return mejor  # cota inferior: nunca se guarda como valor exacto
    if not podar or (mejor>alfa_inicial and mejor<beta):
        tabla.guardar(estado, profundidad, mejor)
    return mejor


def valor_accion(estado, jugada, profundidad, evaluar, plazo,
                 tabla=None, reloj=time.perf_counter, resultados=None, ordenar=True,
                 alfa=0.,beta=1.,podar=True):
    """Valor de una acción, incluyendo todos los resultados de un combate."""
    if tabla is None:
        tabla = TablaTransposicion()
    rs=estado.siguiente(jugada) if resultados is None else resultados
    if len(rs)==1:
        return 1.-valor(rs[0][1],profundidad-1,evaluar,plazo,tabla,reloj,ordenar,
                        1.-beta,1.-alfa,podar)
    # En azar cada hijo se busca con ventana completa. Solo se corta la acción
    # si su máxima contribución restante en [0,1] ya no puede superar alfa.
    total=0.;restante=1.
    for p,h in rs:
        if p<=0:continue
        total+=p*(1.-valor(h,profundidad-1,evaluar,plazo,tabla,reloj,ordenar,0.,1.,podar))
        restante=max(0.,restante-p)
        if podar and total+restante<alfa-1e-12:
            return total+restante  # cota superior, dominada por una acción ya vista
    return total


def decidir(estado, evaluar, presupuesto, reloj=time.perf_counter, capacidad=50000, ordenar=False, profundidad_max=None,podar=True):
    """Devuelve siempre una jugada legal de la última profundidad completa."""
    inicio = reloj()
    plazo = inicio + max(0.0, presupuesto)
    jugadas = estado.jugadas()
    if not jugadas:
        raise ValueError('Estado terminal')
    mejor = jugadas[0]  # respaldo legal incluso con un reloj ya agotado
    valor_mejor = None
    prof_completa = 0
    tabla = TablaTransposicion(capacidad)
    orden = [(j,None,False) for j in jugadas]
    try:
        orden = _ordenar(estado, jugadas, evaluar, plazo, reloj, ordenar)
        for j,rs,gana in orden:
            if gana:
                return j,{'prof':1,'eval':1.,'busqueda_s':reloj()-inicio,'tt_entradas':0,'tt_aciertos':0}
    except TiempoAgotado:
        pass
    profundidad = 1
    while reloj() < plazo:
        try:
            valores=[];alfa=0.
            for j,rs,_ in orden:
                v=valor_accion(estado,j,profundidad,evaluar,plazo,tabla,reloj,rs,ordenar,alfa,1.,podar)
                valores.append((v,j));alfa=max(alfa,v)
        except TiempoAgotado:
            break
        valor_mejor, mejor = max(valores, key=lambda item: item[0])
        orden.sort(key=lambda fila: fila[0] != mejor)
        prof_completa = profundidad
        if profundidad_max is not None and profundidad >= profundidad_max:
            break
        profundidad += 1
    return mejor, {
        'prof': prof_completa,
        'eval': valor_mejor,
        'busqueda_s': reloj() - inicio,
        'tt_entradas': tabla.entradas,
        'tt_aciertos': tabla.aciertos,
    }
