"""Rasgos espaciales compilados. Los corredores son aproximaciones estáticas.

Se conserva cada bando por separado y un histograma de las tres casillas de
frente de cada peón. El modelo no confunde contacto frontal con ataque diagonal.
"""
import numpy as np
from numba import njit

BASE = ('puntos','material','avance_1','avance_2','avance_3','movilidad',
        'apoyos','anotaciones','tercer_punto','ataques','mejor_ataque',
        'riesgo','amenazados','bloqueados','contacto_frontal',
        'corredores_estaticos','avance_corredor','avance_con_dos_puntos')
POR_LADO = len(BASE)+27
NOMBRES = ('constante',)+tuple(f'{lado}_{nombre}' for lado in ('propio','rival')
           for nombre in BASE+tuple(f'frente_{i}' for i in range(27)))


@njit(cache=False)
def apoyo(t,pos,c):
    fila=pos//9-(1 if c==0 else -1);col=pos%9;n=0
    if fila<0 or fila>=9:return 0
    if col>0 and t[fila*9+col-1]==c+1:n+=1
    if col<8 and t[fila*9+col+1]==c+1:n+=1
    return n


@njit(cache=False)
def calcular(t,color,puntos):
    datos=np.zeros((2,POR_LADO),np.float64)
    riesgos=np.zeros((2,81),np.float64)
    for c in range(2):
        datos[c,0]=puntos[c]/3.
        paso=1 if c==0 else -1
        for pos in range(81):
            if t[pos]!=c+1:continue
            fila=pos//9;col=pos%9;avance=fila if c==0 else 8-fila
            datos[c,1]+=1./18
            a=avance/8.
            for k in range(2,5):
                if a>datos[c,k]:a,datos[c,k]=datos[c,k],a
            ap=apoyo(t,pos,c);datos[c,6]+=ap/36.
            nf=fila+paso
            if nf<0 or nf>=9:continue
            patron=0;factor=1;movs=0;anotar=0.
            for delta in range(-1,2):
                nc=col+delta
                # Fuera del tablero se codifica como bloqueado por pieza propia.
                v=c+1 if nc<0 or nc>=9 else t[nf*9+nc]
                categoria=0 if v==0 else (1 if v==c+1 else 2)
                patron+=categoria*factor;factor*=3
                if nc<0 or nc>=9 or v==c+1 or (delta==0 and v!=0):continue
                movs+=1;p=1.
                if v:
                    p=min(.9,max(.2,.5+.15*(ap-apoyo(t,nf*9+nc,1-c))))
                    datos[c,9]+=1./36
                    datos[c,10]=max(datos[c,10],p)
                    riesgos[1-c,nf*9+nc]=max(riesgos[1-c,nf*9+nc],p)
                if nf==(8 if c==0 else 0):anotar=max(anotar,p)
            reflejo=(patron%3)*9+((patron//3)%3)*3+patron//9
            datos[c,18+min(patron,reflejo)]+=1./18
            datos[c,5]+=movs/54.
            datos[c,7]+=anotar/3.
            datos[c,8]+=anotar/3. if puntos[c]==2 else 0.
            datos[c,13]+=1./18 if movs==0 else 0.
            datos[c,14]+=1./18 if t[nf*9+col]==2-c else 0.
            # Camino a meta sin piezas ni amenazas diagonales rivales actuales.
            # No presupone que el oponente se quede quieto durante una partida.
            frente=np.zeros(9,np.uint8);frente[col]=1
            rr=fila+paso;libre=True
            while rr>=0 and rr<9:
                siguiente=np.zeros(9,np.uint8)
                for cc in range(9):
                    if not frente[cc]:continue
                    for delta in range(-1,2):
                        nc=cc+delta
                        if nc<0 or nc>=9 or t[rr*9+nc]!=0:continue
                        atfila=rr+paso;atacado=False
                        if atfila>=0 and atfila<9:
                            if nc>0 and t[atfila*9+nc-1]==2-c:atacado=True
                            if nc<8 and t[atfila*9+nc+1]==2-c:atacado=True
                        if not atacado:siguiente[nc]=1
                frente=siguiente
                if frente.sum()==0:libre=False;break
                rr+=paso
            if libre:
                datos[c,15]+=1./18
                datos[c,16]=max(datos[c,16],avance/8.)
        datos[c,17]=datos[c,2] if puntos[c]==2 else 0.
    for c in range(2):
        for pos in range(81):
            datos[c,11]+=riesgos[c,pos]/18.
            datos[c,12]+=1./18 if riesgos[c,pos]>0 else 0.
    salida=np.empty(1+2*POR_LADO,np.float64);salida[0]=1.
    salida[1:1+POR_LADO]=datos[color]
    salida[1+POR_LADO:]=datos[1-color]
    return salida


def rasgos(estado):
    return calcular(estado._tablero,estado._color,estado._puntos)


def pesos_iniciales():
    # Hipótesis inicial explícita, sin elección mediante validación.
    base=np.array([5.,1.,.7,.6,.7,.5,.5,3.,5.,.2,.1,-1.,-.3,-.3,-.2,.5,.5,1.])
    pesos=np.zeros(len(NOMBRES));pesos[1:1+len(BASE)]=base
    pesos[1+POR_LADO:1+POR_LADO+len(BASE)]=-base
    return pesos


# Interacciones explícitas: una captura con el rival adelantado no significa lo
# mismo que una captura cuando ya ganamos la carrera. Son rasgos, no reglas que
# declaren por su cuenta una victoria.
EXTRAS=tuple(f'carrera_{contexto}_{d}' for contexto in ('general','propio_dos','rival_dos') for d in range(-8,9))+(
    'ataque_en_carrera_atrasada','riesgo_en_carrera_atrasada','ataque_a_ultimo_rival',
    'ultimo_propio_sin_tercer_punto','ultimo_rival_sin_tercer_punto',
    'amenaza_propia_con_rival_adelantado','amenaza_rival_con_propio_adelantado','carrera_ambos_dos')
NOMBRES_V4=NOMBRES+EXTRAS


@njit(cache=False)
def calcular_v4(t,color,puntos):
    base=calcular(t,color,puntos);salida=np.zeros(len(NOMBRES)+59,np.float64)
    salida[:len(NOMBRES)]=base
    a=1;b=1+POR_LADO;o=len(NOMBRES)
    diferencia=int(round(8*(base[a+2]-base[b+2])))
    indice=max(0,min(16,diferencia+8))
    salida[o+indice]=1.
    salida[o+17+indice]=1. if puntos[color]==2 else 0.
    salida[o+34+indice]=1. if puntos[1-color]==2 else 0.
    atraso=1. if diferencia<0 else 0.
    salida[o+51]=base[a+10]*atraso
    salida[o+52]=base[a+11]*atraso
    salida[o+53]=base[a+10] if base[b+1]<=1.01/18 else 0.
    salida[o+54]=1. if base[a+1]<=1.01/18 and puntos[color]<2 else 0.
    salida[o+55]=1. if base[b+1]<=1.01/18 and puntos[1-color]<2 else 0.
    salida[o+56]=base[a+7]*atraso
    salida[o+57]=base[b+7]*(1.-atraso)
    salida[o+58]=float(diferencia)/8. if puntos[color]==2 and puntos[1-color]==2 else 0.
    return salida


def rasgos_v4(estado):
    return calcular_v4(estado._tablero,estado._color,estado._puntos)


def pesos_iniciales_v4():
    return np.concatenate((pesos_iniciales(),np.zeros(len(EXTRAS))))
