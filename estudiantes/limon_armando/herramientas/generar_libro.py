"""Genera el libro de aperturas data/libro_aperturas.json (offline, antes del torneo).

Uso (desde la raíz del repo, con .venv activo):

    python estudiantes/limon_armando/herramientas/generar_libro.py <segundos_por_posicion> [procesos] [uct|plano]

Para cada posición de apertura, corre la búsqueda de un bot (por omisión el UCT, `strategy.decidir`;
con `plano`, `plano.jugar`) con mucho más tiempo que en una partida y guarda la jugada elegida:
{clave del estado: "e2-e3"}. Solo lo lee el bot UCT con LIBRO = True; el bot plano no usa libro.
Posiciones cubiertas:
  - Blancas, 1.ª jugada: la posición inicial;
  - Negras, 1.ª jugada: después de cada una de las 27 primeras jugadas de Blancas;
  - Blancas, 2.ª jugada: después de nuestra 1.ª jugada y cada respuesta de Negras;
  - Negras, 2.ª jugada: después de cada 1.ª de Blancas, nuestra respuesta y cada 2.ª de Blancas.
En esas jugadas todavía no hay peones en contacto, así que no interviene ningún dado: la
posición siguiente depende solo de las jugadas. El archivo solo tiene claves y jugadas (datos,
no código); `strategy.py` lo usa solo si la jugada guardada es legal.
"""

import json
import os
import sys
import types
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from motor import Estado  # noqa: E402

CARPETA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SALIDA = os.path.join(CARPETA, "data", "libro_aperturas.json")


def _pensar(args):
    """Jugada del bot para `estado` pensando `segundos` (en un proceso aparte)."""
    estado, segundos, semilla, algoritmo = args
    if algoritmo == "plano":
        import plano
        plano.FACTOR_CRITICA = plano.FACTOR_QUIETA = 1.0  # mismo tiempo en toda posición
        plano.preparar(types.SimpleNamespace(color=estado.turno, semilla=semilla, reloj_s=30.0,
                                             carpeta_datos=os.path.join(CARPETA, "data")))
        # plano.jugar busca (tiempo_restante - RESERVA_S) / 25 segundos.
        return estado.clave, str(plano.jugar(estado, plano.RESERVA_S + 25 * segundos))
    import strategy  # cada proceso carga su propio bot
    strategy.preparar(types.SimpleNamespace(color=estado.turno, semilla=semilla, reloj_s=30.0,
                                            carpeta_datos=os.path.join(CARPETA, "data")))
    strategy.RELOJ = "fijo"
    strategy.LIBRO = False
    # Con la regla «fijo» se piensa (tiempo_restante - reserva) / 25.
    jugada = strategy.decidir(estado, strategy.reserva + 25 * segundos)
    return estado.clave, str(jugada)


def _siguiente(estado, jugada):
    (_, hijo), = estado.siguiente(jugada)  # sin contacto: un solo resultado
    return hijo


def main(segundos, procesos, algoritmo="uct"):
    libro = {}
    with Pool(procesos) as pool:
        def resolver(estados):
            res = pool.map(_pensar, [(e, segundos, k, algoritmo) for k, e in enumerate(estados)])
            libro.update({str(c): j for c, j in res})
            return {c: j for c, j in res}

        s0 = Estado.inicial()
        jug = lambda e: next(j for j in e.jugadas() if str(j) == libro[str(e.clave)])  # noqa: E731
        resolver([s0])                                                   # Blancas 1
        tras_b1 = [_siguiente(s0, j) for j in s0.jugadas()]
        resolver(tras_b1)                                                # Negras 1
        s1 = _siguiente(s0, jug(s0))
        resolver([_siguiente(s1, j) for j in s1.jugadas()])              # Blancas 2
        tras_n1 = [_siguiente(e, jug(e)) for e in tras_b1]
        resolver([_siguiente(e, j) for e in tras_n1 for j in e.jugadas()])  # Negras 2
    with open(SALIDA, "w") as f:
        json.dump(libro, f, indent=0, sort_keys=True)
    print(f"{len(libro)} posiciones en {os.path.normpath(SALIDA)}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(float(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else os.cpu_count(),
         sys.argv[3] if len(sys.argv) > 3 else "uct")
