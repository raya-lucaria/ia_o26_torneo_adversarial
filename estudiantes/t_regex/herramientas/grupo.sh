#!/bin/sh
# Enfrenta una versión (carpeta, relativa a estudiantes/t_regex) contra el grupo de referencia:
# v1, v3 y v5 (100 000 nodos, 400 partidas), montecarlo (300 000 nodos, 200) y random (100).
# Uso, dentro de la imagen de bots desde estudiantes/t_regex: sh herramientas/grupo.sh ../t_regex_v8
A="$1"
for v in v1 v3 v5; do
    python herramientas/arena.py --a "$A" --b ../t_regex_$v --pares 200 --nodos 100000 | grep -E "A gana" | sed "s/^/contra $v: /"
done
python herramientas/arena.py --a "$A" --b montecarlo:1700 --pares 100 --nodos 300000 | grep -E "A gana" | sed "s/^/contra montecarlo: /"
python herramientas/arena.py --a "$A" --b random --pares 50 --nodos 100000 | grep -E "A gana" | sed "s/^/contra random: /"
