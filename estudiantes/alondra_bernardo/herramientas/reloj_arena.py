"""Arena con reloj real, sin Docker: dos carpetas de bots (su strategy.py) en un mismo proceso.

Cada bot carga sus propios módulos y recibe su reloj como en el torneo (sin el costo fijo de
Docker). Sirve para comparar versiones contra rivales que juegan por tiempo.

python estudiantes/alondra_bernardo/herramientas/reloj_arena.py estudiantes/A estudiantes/B \
    --partidas 360 --reloj 30 --procesos 8 [--semilla 0] [--reloj-a 60]
"""
import argparse, importlib.util, random, sys, time
from multiprocessing import Pool
from pathlib import Path
from types import SimpleNamespace

from motor import Estado

BOTS = {}
AUX = ("rapido", "buscador", "strategy")


def cargar(carpeta, alias):
    """Importa strategy.py de `carpeta` con sus módulos propios, sin mezclarlos con los del otro bot."""
    for m in AUX:
        sys.modules.pop(m, None)
    sys.path.insert(0, str(carpeta))
    spec = importlib.util.spec_from_file_location(alias, Path(carpeta) / "strategy.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    # Compilar aquí, mientras sys.modules todavía apunta a los módulos de este bot.
    mod.preparar(SimpleNamespace(color="blancas", reloj_s=30.0, preparar_s=10.0, semilla=0,
                                 carpeta_datos=Path(carpeta) / "data"))
    sys.path.pop(0)
    return mod


def iniciar(a, b, reloj, reloj_a=None):
    BOTS.update(a=cargar(a, "bot_a"), b=cargar(b, "bot_b"), ruta_a=Path(a), ruta_b=Path(b), reloj=reloj,
                reloj_a=reloj_a or reloj)


def partida(k):
    rng = random.Random(k)
    a_blancas = k % 2 == 0
    colores = {"a": "blancas" if a_blancas else "negras", "b": "negras" if a_blancas else "blancas"}
    for q in "ab":
        cfg = SimpleNamespace(color=colores[q], reloj_s=BOTS["reloj_a"] if q == "a" else BOTS["reloj"], preparar_s=10.0, semilla=k,
                              carpeta_datos=BOTS["ruta_" + q] / "data")
        BOTS[q].preparar(cfg)
    resto = {"a": BOTS["reloj_a"], "b": BOTS["reloj"]}
    e = Estado.inicial()
    while e.ganador() is None:
        q = "a" if e.turno == colores["a"] else "b"
        t = time.perf_counter()
        r = BOTS[q].jugar(e, resto[q])
        resto[q] -= time.perf_counter() - t
        if resto[q] <= 0:  # pierde por reloj
            return (q == "b", True)
        j = next(x for x in e.jugadas() if str(x) == str(r))
        probs, hijos = zip(*e.siguiente(j))
        e = rng.choices(hijos, weights=probs)[0]
    return (e.ganador() == colores["a"], False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("a"); p.add_argument("b")
    p.add_argument("--partidas", type=int, default=120)
    p.add_argument("--reloj", type=float, default=6.0)
    p.add_argument("--reloj-a", type=float, default=None, help="reloj distinto para A")
    p.add_argument("--procesos", type=int, default=8)
    p.add_argument("--semilla", type=int, default=0)
    x = p.parse_args()
    with Pool(x.procesos, initializer=iniciar, initargs=(x.a, x.b, x.reloj, x.reloj_a)) as pool:
        r = pool.map(partida, range(x.semilla, x.semilla + x.partidas), chunksize=1)
    g = sum(w for w, _ in r)
    print(f"{Path(x.a).name} vs {Path(x.b).name}: {g}-{len(r) - g} (reloj {x.reloj_a or x.reloj}/{x.reloj} s, {sum(t for _, t in r)} por tiempo)")


if __name__ == "__main__":
    main()
