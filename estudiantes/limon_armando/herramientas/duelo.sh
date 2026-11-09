#!/bin/bash
# Uso: duelo.sh <bot> <rival> <partidas> <semilla> <log>
# Un flujo de partidas bot vs rival. Reintenta si `jugar` falla al arrancar (carrera con
# otros flujos al limpiar contenedores); si falla ya jugando, no reintenta (no duplicar).
for intento in 1 2 3 4 5; do
  inicio=$SECONDS
  ./torneo jugar "$1" "$2" --partidas "$3" --semilla "$4" > "$5" 2>&1 && exit 0
  (( SECONDS - inicio > 90 )) && exit 1
  sleep $((RANDOM % 5 + 1))
done
exit 1
