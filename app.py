import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pydicom as dc
import math
import plotly.express as px

#from IPython.display import HTML, Latex, Markdown, clear_output, display, FileLink
#from pydicom import dcmread
#from pydicom.filebase import DicomBytesIO
from scipy import fftpack, stats
from scipy.optimize import curve_fit
from skimage import color
from skimage.feature import canny
from skimage.transform import hough_circle, hough_circle_peaks

#from docx.shared import Cm, Emu, Inches, Mm
#from docxtpl import DocxTemplate, InlineImage


import os
import io
import uuid
import math

sns.set_theme(color_codes=True)
np.seterr(divide='ignore', invalid='ignore')

# Constantes y tolerancias


TC005_tolerancia = '\u00B1 2 mm'

TC007_TipoKernel = "H20s"
TC007_Medio_lado = 38.0
TC007_HoughParameters = [3, 0, 50, [99, 100, 101], 1]

TC012_tolerancia = ['\u00B1 5 UH', '\u00B1 20 UH']
TC014_tolerancia = ['\u00B1 4 UH']
TC015_tolerancia = ['\u00B1 5 UH']
TC017_tolerancia = ['3.5 mm 3%']
TC017_deseable = ['6 mm 0.8%']

# TC016_centros = np.array([[-2.39, -55.71], [-31.1, -47.51], [-59.81,   3.08],
#                         [-30.42,  52.98], [27.69,  52.29], [56.4,   2.39], [27., -47.51]])

TC014_centros = np.array([[0, 0], [-50, 0], [50,   0],
                         [0,  -50], [0,  50]])

TC014_posiciones = ['Centro', 'Izquierda', 'Derecha',
                    'Arriba', 'Abajo']

TC016_centros = np.array([[-0.34, -57.08], [-29.73, -49.56], [-58.45,   0.34],
                         [-29.73,  50.25], [28.37,  50.93], [57.08,   0.34], [28.37, -49.56]])

TC016_materiales = ['Air', 'PMP', 'LDPE',
                    'Polystirene', 'Acrylic', 'Delrin', 'Teflon']
TC016_edensity = [0.004, 2.851, 3.160, 3.335, 3.833, 4.557, 6.243]


TC018_HoughParameters = [3, 0, 50, [99, 100], 1]
TC018_Limits = [87.5, 112.5]
TC018_OverResolution = 0.1
TC018_MaxAngle = 6
TC018_StepAngle = 0.5
TC018_Angles = [0, 30, 60, 90, 120, 150, 180]
Catphan_ExtRadius = 100  # radio externo del catphan


#def initvariables():
#    if 'template' not in st.session_state:
#        st.session_state.template = DocxTemplate("templates/CT_template.docx")
#    if 'context' not in st.session_state:
#        st.session_state.context = {}
#    if 'dirname' not in st.session_state and 'template' in st.session_state and 'context' in st.session_state:
#        dirname = str(uuid.uuid4())
#        os.mkdir(dirname)
#        os.mkdir(dirname+'/img')
#        st.session_state.dirname = dirname
#    if 'TC007_table_axial' not in st.session_state:
#        st.session_state.TC007_table_axial = pd.DataFrame()


#if 'dirname' not in st.session_state or 'template' not in st.session_state or 'context' not in st.session_state:
    #initvariables()


# if (st.session_state.images['TC005_ini']) is not None:
#    print(st.session_state.images['TC005_ini'])
#    image = InlineImage(st.session_state.template,
#                    st.session_state.images['TC005_ini'], Cm(8))


def AddImageFile(File, name, width=350):
    if os.path.exists('tmp/'+name+'.png'):
        os.remove('tmp/'+name+'.png')
    ImageFile = plt.imread(File)
    plt.imsave('tmp/'+name+'.png', ImageFile, cmap='gray')
    st.session_state.images[name] = 'tmp/'+name+'.png'
    st.image(ImageFile, width=width)





@st.cache()
def addtable(table, name):
    X = []
    for index, row in table.iterrows():
        X.append(row.to_dict())
    st.session_state.context[name] = X


def CambioEnLoader(nombre, fichero):
    if 'context' in st.session_state:
        if nombre in st.session_state.context:
            if type(st.session_state[fichero]) is list:
                if (len(st.session_state[fichero]) == 0):
                    del st.session_state.context[nombre]
            else:
                del st.session_state.context[nombre]


def createCircularMask(shape, center=None, radius=None):

    if center is None:  # use the middle of the image
        center = [int(shape[1] / 2), int(shape[0] / 2)]
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], shape[1] -
                     center[0], shape[0] - center[1])

    Y, X = np.ogrid[:shape[0], :shape[1]]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

    mask = dist_from_center <= radius
    return mask


def NormalizeMTF(Matrix, Agua, Aire):
    return (Matrix - Aire) / (Agua - Aire)


def round_up_to_odd(f):
    return (np.ceil(f * 10) // 2 * 2 - 2) / 10


def GetMTF(X):
    return np.absolute(fftpack.fft(np.gradient(X)))


def CalcularMTF(X, data):
    return np.array(
        [
            np.interp(x, np.sort(data.mean(axis=0)[
                      0:50]), np.arange(10, 0, -0.2))
            for x in X
        ]
    )


def ObtenerMTFDeImagenes(images, HoughParameters=TC018_HoughParameters, Limits=TC018_Limits, OverResolution=TC018_OverResolution, ValoresSemaforos=np.array([0.5, 0.1, 0.02])):

    Resolution = images[0].PixelSpacing[0]
    accums, cx, cy, radii = BusquedaCirculos(
        images[0].pixel_array, HoughParameters, Resolution
    )

    DistanceMap = np.arange(
        Limits[0] / Resolution,
        Limits[1] / Resolution,
        OverResolution,
    )
    WaterMask = createCircularMask(
        images[0].pixel_array.shape,
        center=(cx[0], cy[0]),
        radius=int(Catphan_ExtRadius * 0.9 / Resolution)
        * np.logical_not(
            createCircularMask(
                images[0].pixel_array.shape,
                center=(cx[0], cy[0]),
                radius=int(Catphan_ExtRadius * 0.8 / Resolution),
            )
        ),
    )
    AirMask = createCircularMask(
        images[0].pixel_array.shape,
        center=(cx[0], cy[0]),
        radius=int(Catphan_ExtRadius * 1.3 / Resolution)
        * np.logical_not(
            createCircularMask(
                images[0].pixel_array.shape,
                center=(cx[0], cy[0]),
                radius=int(Catphan_ExtRadius * 1.1 / Resolution),
            )
        ),
    )

    HUAgua = WaterMask * images[0].pixel_array

    HUAire = AirMask * images[0].pixel_array

    Angulo = math.sqrt(2 * Resolution /
                       Catphan_ExtRadius)  # En radianes
    FrecuencyResolution = (
        0.01 / Resolution
    )  # Resolución en frecuencia tomada según IEC62220-1
    FrecuenciaNyquist = 1 / 2.0 / Resolution
    # factor para el "overfitting"
    Oversampling = int(1 / math.tan(Angulo))
    NumeroDePuntos = int(100 * Oversampling)
    comp = (
        math.pi
        / 2
        / FrecuenciaNyquist
        * np.arange(0, FrecuenciaNyquist, FrecuencyResolution)
    ) / np.sin(
        math.pi
        / 2
        / FrecuenciaNyquist
        * np.arange(0, FrecuenciaNyquist, FrecuencyResolution)
    )

    image_PA = np.array([X.pixel_array for X in images])
    Limites = {
        ("Y", "Sup", "Min"): np.max(
            [0, cy[0] - int(Catphan_ExtRadius / Resolution) - 50]
        ),
        ("Y", "Inf", "Max"): np.min(
            [
                image_PA.shape[0] - 1,
                cy[0] + int(Catphan_ExtRadius / Resolution) + 50,
            ]
        ),
        ("X", "Izq", "Min"): np.max(
            [0, cx[0] - int(Catphan_ExtRadius / Resolution) - 50]
        ),
        ("X", "Dch", "Max"): np.min(
            [
                image_PA.shape[1] - 1,
                cx[0] + int(Catphan_ExtRadius / Resolution) + 50,
            ]
        ),
    }
    Data_Y = np.array(
        [
            [
                X[
                    Limites["X", "Izq",
                            "Min"]: Limites["X", "Izq", "Min"]
                    + 100,
                    cy[0]: cy[0] + Oversampling,
                ],
                X[
                    Limites["X", "Izq",
                            "Min"]: Limites["X", "Izq", "Min"]
                    + 100,
                    cy[0] - Oversampling: cy[0],
                ],
                X[
                    Limites["X", "Dch", "Max"]
                    - 100: Limites["X", "Dch", "Max"],
                    cy[0]: cy[0] + Oversampling,
                ],
                X[
                    Limites["X", "Dch", "Max"]
                    - 100: Limites["X", "Dch", "Max"],
                    cy[0] - Oversampling: cy[0],
                ],
            ]
            for X in image_PA
        ]
    )
    Data_X = np.array(
        [
            [
                X[
                    Limites["X", "Izq",
                            "Min"]: Limites["X", "Izq", "Min"]
                    + 100,
                    cy[0]: cy[0] + Oversampling,
                ],
                X[
                    Limites["X", "Izq",
                            "Min"]: Limites["X", "Izq", "Min"]
                    + 100,
                    cy[0] - Oversampling: cy[0],
                ],
                X[
                    Limites["X", "Dch", "Max"]
                    - 100: Limites["X", "Dch", "Max"],
                    cy[0]: cy[0] + Oversampling,
                ],
                X[
                    Limites["X", "Dch", "Max"]
                    - 100: Limites["X", "Dch", "Max"],
                    cy[0] - Oversampling: cy[0],
                ],
            ]
            for X in image_PA
        ]
    )
    for X in Data_X:
        X[0] = np.flip(X[0], axis=0)
        X[3] = np.flip(X[3], axis=0)
    Data_Y1 = np.reshape(
        Data_Y,
        (
            Data_Y.shape[0] * Data_Y.shape[1],
            Data_Y.shape[2] * Data_Y.shape[3],
        ),
    )
    Data_X1 = np.reshape(
        Data_X,
        (
            Data_X.shape[0] * Data_X.shape[1],
            Data_X.shape[2] * Data_X.shape[3],
        ),
    )
    Data_X1 = np.array(
        [np.reshape(X, (X.shape[0] * X.shape[1]))
         for Y in Data_X for X in Y]
    )
    Data_Y1 = [
        NormalizeMTF(X, HUAgua[HUAgua > 0.01].mean(),
                     HUAire[HUAire > 0.01].mean())
        for X in Data_Y1
    ]
    Data_X1 = [
        NormalizeMTF(X, HUAgua[HUAgua > 0.01].mean(),
                     HUAire[HUAire > 0.01].mean())
        for X in Data_X1
    ]
    MTFY = np.array([GetMTF(X) for X in Data_Y1])
    MTFX = np.array([GetMTF(X) for X in Data_X1])
    MTFX_Data = np.around(
        np.vstack([np.arange(0, 10, 0.2), MTFX.mean(
            axis=0)[0:50]]).transpose(), 4
    )
    MTFY_Data = np.around(
        np.vstack([np.arange(0, 10, 0.2), MTFY.mean(
            axis=0)[0:50]]).transpose(), 4
    )
    SemaforosX = np.around(
        CalcularMTF(np.array(ValoresSemaforos), MTFX), 2
    )
    SemaforosY = np.around(
        CalcularMTF(np.array(ValoresSemaforos), MTFY), 2
    )
    MTF = pd.DataFrame(MTFX_Data, columns=['frecuencia', 'MTFX'])
    MTF['MTFY'] = MTFY_Data[:, 1]
    MTF['MTF'] = MTF[['MTFX', 'MTFY']].mean(axis=1)

    fig_2, ax_2 = plt.subplots(1)
    sns.regplot(x='frecuencia', y='MTFX',
                data=MTF, order=4, scatter=False)
    sns.regplot(x='frecuencia', y='MTFY',
                data=MTF, order=4, scatter=False)
    sns.regplot(x='frecuencia', y='MTF',
                data=MTF, order=4, scatter=False)
    ax_2.set(xlabel='Frecuencia ($cm^{-1}$)')
    ax_2.set(ylabel='MTF')
    ax_2.legend(['MTF X', 'MTF Y', 'MTF'])
    MTF = MTF.set_index('frecuencia')
    Semaforos = pd.DataFrame([SemaforosX, SemaforosY],
                             columns=ValoresSemaforos.astype(str))
    Semaforos = Semaforos.append(
        Semaforos.mean(axis=0), ignore_index=True)
    Semaforos.index = [
        "MTFX (cm\u207B\u00B9)", 'MTFY (cm\u207B\u00B9)', 'MTF (cm\u207B\u00B9)']
    fig = px.scatter(MTF, trendline="rolling", trendline_options=dict(window=3),
                     title="MTF")
    fig.data = [t for t in fig.data if t.mode == "lines"]
    # trendlines have showlegend=False by default
    fig.update_layout({'legend_title_text': ''})
    fig.update_traces(showlegend=True)
    fig.update_traces(
        hovertemplate='%{y}')
    fig.update_layout(
        title='MTF', xaxis_title='frecuencia (cm\u207B\u00B9)', yaxis_title='MTF', hovermode='x')
    return fig_2, Semaforos, fig


def sigmoid(x, lower, upper, xmid, wsig):
    return lower + (upper - lower) / (1 + np.exp((xmid - x) / wsig))


def DSigmoid(x, center, lower1, lower2, upper1, upper2, xmid1, xmid2, wsig1, wsig2):
    xout1 = x[x < center]
    xout2 = x[x >= center]
    out1 = sigmoid(xout1, lower1, upper1, xmid1, wsig1)
    out2 = sigmoid(xout2, lower2, upper2, xmid2, wsig2)
    return np.concatenate((out1, out2))


def DefinirXY(image, centro, radiopixel):
    out = dict()
    L = (
        np.linspace(
            centro.x - radiopixel,
            centro.x + radiopixel,
            num=2 * radiopixel + 1,
        ).astype(np.int16),
        np.linspace(
            centro.y - radiopixel,
            centro.y + radiopixel,
            num=2 * radiopixel + 1,
        ).astype(np.int16),
    )
    out["x"] = [
        L[0] - L[0].min(),
        L[0] - L[0].min(),
        L[1] - L[1].min(),
        L[1] - L[1].min(),
    ]
    out["y"] = (
        np.array(
            [
                [
                    image.pixel_array[centro.y - radiopixel - 1, L[0]],
                    image.pixel_array[centro.y + radiopixel - 1, L[0]],
                    image.pixel_array[L[1], centro.x - radiopixel - 1],
                    image.pixel_array[L[1], centro.x + radiopixel - 1],
                ],
                [
                    image.pixel_array[centro.y - radiopixel, L[0]],
                    image.pixel_array[centro.y + radiopixel, L[0]],
                    image.pixel_array[L[1], centro.x - radiopixel],
                    image.pixel_array[L[1], centro.x + radiopixel],
                ],
                [
                    image.pixel_array[centro.y - radiopixel + 1, L[0]],
                    image.pixel_array[centro.y + radiopixel + 1, L[0]],
                    image.pixel_array[L[1], centro.x - radiopixel + 1],
                    image.pixel_array[L[1], centro.x + radiopixel + 1],
                ],
            ]
        )
        + float(image.RescaleIntercept)
    )
    # out["y"] = out["y"] - out["y"].min(axis=0)
    return out


def CondicionesIniciales(x, y):
    hwhmExtrem = np.where(y >= y.max() / 2)[0]
    out = dict()
    out["center"] = int((x[hwhmExtrem[-1]] + x[hwhmExtrem[0]]) / 2)
    out["lower1"] = out["lower2"] = y[0]
    out["upper1"] = out["upper2"] = y.max()
    out["xmid1"] = x[hwhmExtrem[0]]
    out["xmid2"] = x[hwhmExtrem[-1]]
    out["wsig1"] = 0.5
    out["wsig2"] = -0.5
    return out


def EspesorDeCorte(image, centro, radiopixel):
    espesor = np.array([])
    cov = np.array([])
    espesor_pre = np.array([])
    Perfiles = DefinirXY(image, centro, radiopixel)
    for X, Y in zip(Perfiles["x"], BestRow(image, Perfiles)):
        X0 = CondicionesIniciales(
            X.astype(np.float32) * image.PixelSpacing[0], Y)
        # espesor_pre=np.append(espesor_pre,(X0['xmid2'] - X0['xmid1']) * 0.42 * image.PixelSpacing[0])
        popt, pcov = curve_fit(
            DSigmoid,
            X.astype(np.float32) * image.PixelSpacing[0],
            Y,
            p0=[
                X0["center"],
                X0["lower1"],
                X0["lower2"],
                X0["upper1"],
                X0["upper2"],
                X0["xmid1"],
                X0["xmid2"],
                X0["wsig1"],
                X0["wsig2"],
            ],
            method="dogbox",
        )
        # espesor=np.append(espesor,(popt[6] - popt[5]) * 0.42 * image.PixelSpacing[0])
        # espesor=np.append(espesor,popt)
        espesor = np.append(espesor, (popt[6] - popt[5]) * 0.42)
        # cov=np.append(cov,np.sqrt(np.diag(pcov)))
    return espesor  # .reshape(4,9)#,cov#,espesor_pre


def BestRow(image, Perfiles):
    out = []
    for X, Y in zip(Perfiles["x"], np.transpose(Perfiles["y"], (1, 0, 2))):
        yout = np.array([])
        # print(X)
        for J in Y:
            X0 = CondicionesIniciales(X.astype(np.float32), J)
            yout = np.append(
                yout, (X0["xmid2"] - X0["xmid1"]) *
                0.42 * image.PixelSpacing[0]
            )
        try:
            out.append(
                Y[
                    np.where(
                        np.abs(
                            yout - image.SliceThickness
                            == np.abs(yout.min() - image.SliceThickness)
                        )
                    )[0][0]
                ]
            )
        except:
            pass
            # print('Fallo')
    return np.array(out)


class Point:
    def __init__(self, x_init, y_init):
        self.x = x_init
        self.y = y_init

    def shiftX(self, d):
        return Point(self.x + d, self.y)

    def shiftY(self, d):
        return Point(self.x, self.y + d)

    def __repr__(self):
        return "".join(["Point(", str(self.x), ",", str(self.y), ")"])

    def __add__(a, b):
        return Point(a.x + b.x, a.y + b.y)

    def __getitem__(self, i):
        return Point(self.x[i], self.y[i])


def BusquedaCirculos(image, HoughParameters, PixelSize):
    edges = canny(
        image,
        sigma=HoughParameters[0],
        low_threshold=HoughParameters[1],
        high_threshold=HoughParameters[2],
    )
    hough_radii = np.array(HoughParameters[3])
    hough_radii = (hough_radii / PixelSize).astype(np.int32)
    hough_res = hough_circle(edges, hough_radii)
    return hough_circle_peaks(
        hough_res, hough_radii, total_num_peaks=HoughParameters[4]
    )


def TC016_recta(imagen, centros=TC016_centros, densidad=TC016_edensity, radio=5, materiales=TC016_materiales, HoughParameters=TC018_HoughParameters):

    MatrizOriginal = imagen.pixel_array+imagen.RescaleIntercept
    tamanoPixel = imagen.PixelSpacing[0]
    accums, cx, cy, radii = BusquedaCirculos(
        imagen.pixel_array, HoughParameters, tamanoPixel
    )
    MitadPixeles = int(MatrizOriginal.shape[0]/2)
    pixelcentros = (centros/tamanoPixel).astype(np.int32) + \
        np.array([cx[0], cy[0]])
    # pixelcentros = (centros/tamanoPixel).astype(np.int32)+np.array([255,248])#MitadPixeles
    m = np.array([])
    x = np.array([])
    y = np.array([])
    for material, densidad, centro in zip(materiales, densidad, pixelcentros):
        out = MatrizOriginal[createCircularMask(MatrizOriginal.shape, center=centro, radius=int(
            radio/tamanoPixel))].astype(np.float32)
        m = np.concatenate((m, np.repeat(material, out.shape)))
        x = np.concatenate((x, np.repeat(densidad, out.shape)))
        y = np.concatenate((y, out))
    FF = pd.DataFrame(columns=['m', 'x', 'y'])
    FF['m'] = m
    FF['x'] = x
    FF['y'] = y
    return FF


def TC014_valores(imagen, centros=TC014_centros, radio=5, posiciones=TC014_posiciones, HoughParameters=TC018_HoughParameters):

    MatrizOriginal = imagen.pixel_array+imagen.RescaleIntercept
    tamanoPixel = imagen.PixelSpacing[0]
    accums, cx, cy, radii = BusquedaCirculos(
        imagen.pixel_array, HoughParameters, tamanoPixel
    )
    MitadPixeles = int(MatrizOriginal.shape[0]/2)
    pixelcentros = (centros/tamanoPixel).astype(np.int32) + np.array([cx[0], cy[0]])
    # pixelcentros = (centros/tamanoPixel).astype(np.int32)+np.array([255,248])#MitadPixeles
    media = np.array([])
    desvest = np.array([])
    for centro in pixelcentros:
        out = MatrizOriginal[createCircularMask(MatrizOriginal.shape, center=centro, radius=int(
            radio/tamanoPixel))]
        print(MatrizOriginal[createCircularMask(MatrizOriginal.shape, center=centro, radius=int(radio/tamanoPixel))].mean())
        media=np.append(media,np.array([out.mean()]))
        desvest=np.append(desvest,np.array([out.std()]))
    FF = pd.DataFrame()
    FF['posicion'] = posiciones
    FF['media'] = media
    FF['std'] = desvest
    return FF

def TC016_llenar_tabla(tabla, materiales=TC016_materiales):
    Media = np.array([])
    Std = np.array([])
    for m in materiales:
        subconjunto_table = tabla[tabla['m'] == m]
        np.append(Media, subconjunto_table['y'].mean())
        np.append(Std, subconjunto_table['y'].std())
    FF = pd.DataFrame(columns=['Material', 'Media', 'Std'])
    FF['Material'] = materiales
    FF['Media'] = Media
    FF['Std'] = Std
    FF = FF.set_index('Material')
    return FF

def DimensionArray(lista):
    return np.zeros(lista[0].pixel_array.shape+tuple([len(lista)]))

# Constantes 2
#TC007_tolerancia_axial = np.repeat('\u00B1 1 mm', espesores_de_corte_axial)
#TC007_tolerancia_helix = np.repeat('\u00B1 1 mm', espesores_de_corte_helix)

# st.session_state.context

# Pruebas

st.title('Herramientas de cálculo CC CT')


# Espesor de corte 

st.markdown('## Espesor de Corte [TC007](https://drive.google.com/open?id=14adFusPK1sqlF9hLEqKl86HMZ03XSFzf&disco=AAAADH081e4)')

TC007_file1_raw = st.file_uploader(
    "", ['dcm'], True, key='TC007_image1_raw', on_change=CambioEnLoader('TC007', 'TC007_axial'), help="Arrastre los ficheros adquiridos del módulo de Catphan CTP732")
#TC007_table_axial = np.zeros((1, 3))

if TC007_file1_raw is not None:
    TC007_files = [dc.read_file(x) for x in TC007_file1_raw]
if len(TC007_files) > 0:
    TC007_table_axial = np.zeros((len(TC007_files), 3))
    TC007_HS_Pixel = int(np.round(TC007_Medio_lado / TC007_files[0].PixelSpacing[0]))
    TC007_accums, TC007_cx, TC007_cy, TC007_radii = BusquedaCirculos(
            TC007_files[0].pixel_array,
            TC007_HoughParameters,
            TC007_files[0].PixelSpacing[0],
        )
    TC007_centro = Point(TC007_cx[0], TC007_cy[0])
    for i, TC007_file in enumerate(TC007_files):

        #TC007_file = dc.read_file(TC007_file1_raw)
        TC007_Perfiles = DefinirXY(TC007_file, TC007_centro, TC007_HS_Pixel)
        TC007_table_axial[i, 0] = TC007_file.SliceThickness
        TC007_table_axial[i, 1] = EspesorDeCorte(
        TC007_file,TC007_centro, TC007_HS_Pixel).mean()
        TC007_table_axial[i, 2] = EspesorDeCorte(
            TC007_file,TC007_centro, TC007_HS_Pixel).std()
    TC007_table_axial=pd.DataFrame(TC007_table_axial,columns=['Nominal','Media','Std'])
    TC007_table_axial['Nominal']=TC007_table_axial['Nominal'].astype(int)
    #TC007_table_axial = pd.DataFrame({'Nominal': TC007_table_axial[:, 0].astype(str), 'Media': TC007_table_axial[:, 1].astype(
    #    str), 'Std': TC007_table_axial[:, 2].astype(str)})  # , 'Tolerancia': TC007_tolerancia_axial})
    #TC007_table_axial['Tolerancia'] = np.repeat(
    #    TC007_tolerancia_axial, len(TC007_files))
    #TC007_table_axial.groupby("Nominal")["Media"].mean()
    # st.dataframe(st.session_state.TC007_table_axial)
    #st.write( {'Nominal': TC007_file.SliceThickness, 'Media': EspesorDeCorte(TC007_file, TC007_Perfiles)[0].mean(), 'Std': EspesorDeCorte(TC007_file, TC007_Perfiles)[0].std(), 'Tolerancia': TC007_tolerancia_axial}, ignore_index=True)
    TC007_table_axial=TC007_table_axial.groupby("Nominal").agg({'Media':'mean', 'Std':lambda x: np.sqrt(np.sum(x**2))})
    TC007_table_axial=TC007_table_axial.round(2)
    TC007_table_axial.index.name='Espesor'
    TC007_table_axial['Media']=TC007_table_axial['Media'].astype(str)
    TC007_table_axial['Std']=TC007_table_axial['Std'].astype(str)
    TC007_table_axial['Tolerancia']='\u00B1 1 mm'
    csv = TC007_table_axial.to_csv().encode('iso-8859-8')
    st.download_button(
     label="Bajar datos como CSV",
     data=csv,
     file_name='EspesorCorte.csv',
     mime='text/csv',
        )
    TC007_table_axial.reset_index(inplace=True)
    TC007_table_axial

#-------------------------------------------------------------------------------------------------------------------------------------------------------












# 2.5 Valores de los números CT en distintos materiales. Linealidad y escala de contraste [TC016]

st.markdown(
    '## Uniformidad espacial del número CT [TC016](https://drive.google.com/open?id=14adFusPK1sqlF9hLEqKl86HMZ03XSFzf&disco=AAAADH081k4)')

TC016_file1_raw = st.file_uploader(
    "", ['dcm'], True, key='TC016_image1_raw',help="Utilizar de nuevo las imágenes del módulo CTP732")
if TC016_file1_raw is not None:
    TC016_files = [dc.read_file(x) for x in TC016_file1_raw]
if len(TC016_files) > 0:
    TC016_file=TC016_files[0]
    TC016_file_Matrix=DimensionArray(TC016_files)
    for i,row in enumerate(TC016_files):
        TC016_file_Matrix[:,:,i]=row.pixel_array
    TC016_file.PixelData=TC016_file_Matrix.mean(axis=2).astype(np.uint16).tobytes()
    TC016_fig_1, TC016_ax_1 = plt.subplots(figsize=(1, 1))
    #TC016_file = dc.read_file(TC016_file1)
    TC016_im = TC016_ax_1.imshow(TC016_file.pixel_array, cmap='gray')
    TC016_ax_1.grid(False)
    TC016_ax_1.set_axis_off()
    TC016_col1, TC016_col2 = st.columns(2)
    with TC016_col1:
        st.pyplot(TC016_fig_1)
    with TC016_col2:
        TC016_fig_2, TC016_ax_2 = plt.subplots(figsize=(6, 5))
        TC016_table = TC016_recta(TC016_file)
        TC016_ax_2 = sns.regplot(x="x", y="y", data=TC016_table,
                                 x_estimator=np.mean)
        TC016_ax_2.set(xlabel='EDensity $10^{23}\: e\cdot cm^{-3}$')
        TC016_ax_2.set(ylabel='HU')
        #TC016_ax_2.get_figure().savefig(
        #    st.session_state.dirname+"/img/TC016_regresion.png")
        #st.session_state.context['TC016_regresion'] = InlineImage(
        #    st.session_state.template, st.session_state.dirname+"/img/TC016_regresion.png", Cm(12))
        st.pyplot(TC016_fig_2)
    # get coeffs of linear fit
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        TC016_table['x'], TC016_table['y'])
    #st.write(slope, intercept, r_value, p_value, std_err)
    # st.write(TC016_table[TC016_table['m']=='Air']['y'].mean())
    csv2 = TC016_llenar_tabla(TC016_table).to_csv().encode('iso-8859-8')
    st.download_button(
     label="Bajar datos como CSV",
     data=csv2,
     file_name='UniformidadCT.csv',
     mime='text/csv',
        )
    st.dataframe(TC016_llenar_tabla(TC016_table))



# 2.7 Resolución espacial [TC018]

st.markdown(
    '## Homogeneidad y Resolución espacial') # [TC018](https://drive.google.com/open?id=14adFusPK1sqlF9hLEqKl86HMZ03XSFzf&disco=AAAADH081mY)')


TC018_file1_abdomen = st.file_uploader(
    "MTF abdomen", ['dcm'], True, key='TC018_images_abdomen_raw',help="Volcaremos las imágenes del módulo CTP729 (homogenidad)", on_change=CambioEnLoader('TC018_img_abdomen', 'TC018_images_abdomen_raw'))
if TC018_file1_abdomen is not None:
    TC018_image_abdomen = [dc.read_file(x) for x in TC018_file1_abdomen]
if len(TC018_image_abdomen) > 0:
    TC018_MTF_imagen_abdomen, TC018_MTF_valores_abdomen, TC018_MTF_plotly_abdomen = ObtenerMTFDeImagenes(
        TC018_image_abdomen,)
    # Quitar el comentario para gráfica en seaborn
    # st.pyplot(TC018_MTF_imagen_abdomen)
    #TC018_MTF_plotly_abdomen.write_image(
    #    st.session_state.dirname+"/img/TC018_img_abdomen.png")
    #st.session_state.context['TC018_img_abdomen'] = InlineImage(
    #    st.session_state.template, st.session_state.dirname+"/img/TC018_img_abdomen.png", Cm(12))
    st.plotly_chart(TC018_MTF_plotly_abdomen)
    csv3 = TC018_MTF_valores_abdomen.to_csv().encode('utf-8')
    st.download_button(
     label="Bajar datos como CSV",
     data=csv3,
     file_name='MTFvalores.csv',
     mime='text/csv',
        )
    st.table(TC018_MTF_valores_abdomen)
    TC014_homogeneidad=TC014_valores(TC018_image_abdomen[0])
    TC014_homogeneidad=TC014_homogeneidad.set_index('posicion')
    csv4 = TC018_MTF_valores_abdomen.to_csv().encode('utf-8')
    st.download_button(
     label="Bajar datos como CSV",
     data=csv4,
     file_name='ValoresHomogeneidad.csv',
     mime='text/csv',
        ) 
    st.table(TC014_homogeneidad)
