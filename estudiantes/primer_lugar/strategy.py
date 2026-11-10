"""Búsqueda adversarial con un evaluador lineal versionado."""
from motor import Estado, reportar
from valor import Valor
from busqueda import decidir


def preparar(config):
    global modelo,turnos
    turnos=0
    modelo=Valor(config.carpeta_datos/'modelo.json')
    inicial=Estado.inicial()
    inicial.siguiente(inicial.jugadas()[0])
    _=inicial.clave
    modelo.evaluar(inicial)


def jugar(estado, tiempo_restante):
    global turnos
    turnos+=1
    reserva=2.5
    disponible=max(0., tiempo_restante-reserva)
    presupuesto=min(.65, disponible/45., disponible*.06)
    if modelo.formato>=3:
        disponible=max(0.,tiempo_restante-3.)
        critico=max(estado.puntos.values())==2
        restantes=max(12,48-turnos)
        presupuesto=min(1.2,disponible/restantes*(1.5 if critico else 1.),disponible*.10)
    accion,info=decidir(estado,modelo.evaluar,presupuesto)
    reportar(**info, modelo=modelo.version)
    return accion
