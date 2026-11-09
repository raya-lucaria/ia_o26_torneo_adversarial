#!/bin/bash
# Uso (desde la raíz del repo): hacer_variante.sh <nombre> 'CONST = valor' ...
# Copia el bot a notas_privadas/experimentos/<nombre> y cambia constantes de strategy.py o plano.py.
set -e
nombre=$1; shift
dest=notas_privadas/experimentos/$nombre
rm -rf -- "notas_privadas/experimentos/$nombre"
cp -r estudiantes/limon_armando "$dest"
rm -rf -- "$dest/__pycache__" "$dest/herramientas"
sed -i "s/^nombre = .*/nombre = \"$nombre\"/" "$dest/equipo.toml"
for asign in "$@"; do
  c=${asign%% =*}
  archivo=$(grep -l "^$c = " "$dest/plano.py" "$dest/strategy.py" | head -1)  # el bot activo es plano.py
  [ -n "$archivo" ] || { echo "no existe $c"; exit 1; }
  sed -i "s|^$c = [^#]*|$asign  |" "$archivo"
  grep -n "^$c = " "$archivo"
done
