from osgeo import gdal, ogr

from skimage import segmentation
from skimage.exposure import histogram, adjust_gamma, equalize_hist
from skimage.feature import local_binary_pattern, hog, greycomatrix, greycoprops
from skimage.filters import median, gaussian, sobel, laplace
from skimage import img_as_uint, img_as_ubyte, img_as_float64

from scipy.stats import entropy, kurtosis
from scipy.ndimage import variance
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift

import pandas as pd

import tensorflow as tf
from sklearn import svm
from sklearn import metrics

from typing import Union
import numpy as np
import os
from tempfile import gettempdir

"""
"""
# tf.compat.v1.enable_eager_execution()

# - Constantes - #
TIFF = 'GTiff'
RNA = 0
SVM = 1
FT_GAMA = 0
FT_EQLHIST = 1
FT_MEDIANA = 2
FT_GAUSS = 3
FT_SOBEL = 4
FT_LAPLACE = 5
FT_FPB_IDEAL = 6
FT_FPA_IDEAL = 7
CI_RGB = {1: gdal.GCI_RedBand, 2: gdal.GCI_GreenBand, 3: gdal.GCI_BlueBand}
DTYPE_CVT = {gdal.GDT_Byte: np.uint8, gdal.GDT_UInt16: np.uint16}


# - Funções de Manipulação de Raster - #

def get_bandas(raster):
    """
    Verifica se o arquivo é um gdal.Dataset e retorna apenas a banda ou lista de bandas

    Args:
        raster (Union[gdal.Dataset, gdal.Band]): gdal.Dataset ou gdal.Band que será checado

    Returns:
        Union[list of gdal.Band, gdal.Band]: gdal.Band ou uma lista de gdal.Band com cada banda encontrada no gdal.Dataset

    """

    if type(raster) is gdal.Dataset:
        # contagem de bandas no raster
        raster_qtd = raster.RasterCount
        if raster_qtd == 1:
            return raster.GetRasterBand(1)
        else:
            bands = list()
            for b in range(1, raster_qtd + 1):
                bands.append(raster.GetRasterBand(b))
            return bands
    elif type(raster) is gdal.Band:
        return raster
    else:
        print('Erro: Não é Band nem Dataset')
        return None


def check_resol_radio(rasters):
    """
    Verifica se os rasters estão na mesma resolução radiométrica para manipulações deste

    Args:
        rasters (Union[list of gdal.Dataset, list of gdal.Band]):

    Returns:
        bool: True se os rasters/bandas estão na mesma resolução
    """

    if type(rasters) is not list:
        rasters = list(rasters)

    rradios = set()
    for raster in rasters:
        tp = type(raster)
        if tp is gdal.Dataset:
            for b in range(1, raster.RasterCount + 1):
                rradios.add(raster.GetRasterBand(b).DataType)
        elif tp is gdal.Band:
            rradios.add(raster.DataType)
        else:
            print('Erro no tipo de arquivo')
            return None

        print(rradios)
        if len(rradios) > 1:
            return False

    return True


def check_projection(rasters):
    """

    Args:
        rasters (list of gdal.Dataset):

    Returns:
        bool: True se rasters possuem mesma projeção e False se não
    """

    if type(rasters) is not list:
        print("Erro de tipo")
        return None

    projs = set()
    for raster in rasters:
        if type(raster) is not gdal.Dataset:
            print("Erro de tipo")
            return None
        else:
            projs.add(raster.GetProjection())

            if len(projs) > 1:
                return False

    return True


def check_num_bandas(rasters):
    """

    Args:
        rasters:

    Returns:
        int: total de bandas ou 0 se numero de bandas são diferentes ou vazia

    """
    tp = type(rasters)
    if tp is gdal.Dataset:
        tot = rasters.RasterCount

    elif tp is gdal.Band:
        tot = 1

    elif tp is list:
        tot = rasters[0].RasterCount

        for raster in rasters[1:]:
            if type(raster) is not gdal.Dataset:
                print("Erro de tipo")
                return None
            else:
                if raster.RasterCount != tot:
                    return 0

    else:
        print("Erro de tipo")
        return None

    return tot


def cria_destino(path, nome, desc, ext='tif', extra=None):
    """ Cria caminho de onde será salvo algum Dataset.

    Args:
        path (str): diretório do arquivo
        nome (str): nome do arquivo
        ext (str): extensão que será salvo o arquivo
        desc (str): caminho de um Dataset que será manipulado (usado caso não seja passado path ou nome
        extra (str): informação acrescentada depois do nome do arquivo

    Returns:
        str: caminho do arquivo
    """

    if not nome:
        nome = f'{os.path.splitext(os.path.basename(desc))[0]}'
        if extra:
            nome += f'_{extra}'

    if not path:
        path = os.path.dirname(desc)

    return f'{path}/{nome}.{ext}'


def csv_str_list2list(str_list, tipo):
    """
    Função para converter lista salva em csv para o tipo correto dos itens

    Args:
        str_list (str): lista em string, geralmente como fica salvo lista em csv pelo Pandas
        tipo (function): função de conversão do tipo. Ex: float, int

    Returns (list): Retorna uma lista com os itens do tipo do parâmetro

    """
    # ToDo: checar se é lista e conv é função
    return [tipo(x) for x in str_list.strip('[]').split(', ')]


def compor_rgb(r, g, b, ext=TIFF, path=None, nome=None):
    """
    Une 3 rasters de única banda em um raster de composição colorida RGB

    Args:
        r (Union[gdal.Dataset, gdal.Band]): gdal.Dataset referente a banda Red
        g (Union[gdal.Dataset, gdal.Band]): gdal.Dataset referente a banda Green
        b (Union[gdal.Dataset, gdal.Band]): gdal.Dataset referente a banda Blue
        ext (str): gdal.Driver name. Default = 'GTiff'
        path (str): caminho do diretório de saída do arquivo raster
        nome (str): nome para o raster. Default = gerará automaticamente

    Returns:
        gdal.Dataset: gdal.Dataset com 3 rasters referente a composição colorida RGB

    """

    # - checagem de possiveis erros

    if path:
        if not os.path.exists(path):
            print('Diretório não existe')
            return None

    gd_r = get_bandas(r)
    if not gd_r or type(gd_r) is list:
        print('Erro no arquivo raster')
        return None  # ou padronizar erros - Erro no arquivo raster

    gd_g = get_bandas(g)
    if not gd_g or type(gd_g) is list:
        print('Erro no arquivo raster')
        return None  # ou padronizar erros - Erro no arquivo raster

    gd_b = get_bandas(b)
    if not gd_b or type(gd_b) is list:
        print('Erro no arquivo raster')
        return None  # ou padronizar erros - Erro no arquivo raster

    if not check_resol_radio([gd_r, gd_g, gd_b]):
        print('Rasters com resoluções radiométrica diferentes')
        return None  # padronizar erros - Rasters com resoluções radiométrica diferentes

    # salvando o tipo de dado do raster, referente a resolução radiométrica
    dtype = gd_r.DataType

    # salvando informações geográficas do raster
    col = r.RasterXSize
    row = r.RasterYSize
    geo_transf = r.GetGeoTransform()
    proj = r.GetProjection()

    if col != g.RasterXSize or col != b.RasterXSize or\
            row != g.RasterYSize or row != b.RasterYSize:
        print('Rasters em posições diferentes')
        return None  # erro - Rasters em posições diferentes

    if geo_transf != g.GetGeoTransform() or geo_transf != b.GetGeoTransform():  # pode dar sempre diferente
        print('Rasters com geotransformações diferentes')
        return None  # erro - Rasters com geotransformações diferentes

    if not check_projection([r, g, b]):
        print('Rasters com projeções diferentes')
        return None  # erro - Rasters com projeções diferentes

    # criando novo arquivo raster da composição
    driver = gdal.GetDriverByName(ext)
    dest = cria_destino(path, nome, r.GetDescription(), extra='comp_RGB')
    comp = driver.Create(dest, col, row, 3, dtype, ['PHOTOMETRIC=RGB'])

    # adicionando as informações geográficas
    comp.SetGeoTransform(geo_transf)
    comp.SetProjection(proj)

    # escrevendo os dados das bandas no raster
    bands = [gd_r, gd_g, gd_b]
    for b in range(3):
        rb = b + 1
        comp.GetRasterBand(rb).WriteArray(bands[b].ReadAsArray(0, 0, col, row))

    # atualizando as alterações no raster
    comp.FlushCache()

    return comp


def alterar_ref_espacial(raster, ref_nova, path=None, nome=None):
    """
    Altera a referência espacial do Dataset, utilizando o padrão EPSG

    Args:
        raster (gdal.Dataset): gdal.Dataset que se deseja alterar referência
        ref_nova (Union[int, str]): tipo de referência pelo padrão EPSG. Exemplo: 4628
        path (str): caminho do diretório de saída do arquivo raster. Default:
        nome (str): nome para o raster. Default: {nome atual}_EPSG:{ref_nova}

    Returns:
        gdal.Dataset: gdal.Dataset atualizado com nova referência espacial

    """

    # checando possíveis erros
    if type(raster) is not gdal.Dataset:
        print('Não é um gdal.Dataset')
        return None
    if type(ref_nova) is not int:
        try:
            ref_nova = int(ref_nova)
        except TypeError:
            print('Projeção precisa ser só o valor inteiro do padrão EPSG')

        return None

    if path:
        if not os.path.exists(path):
            print('Diretório não existe')
            return None

    # criando caminho do arquivo que será gerado
    dest = cria_destino(path, nome, raster.GetDescription(), extra=f'EPSG-{ref_nova}')

    # executando a função de utilidade Warp para atualizar referencia espacial
    raster_ref = gdal.Warp(dest, raster, dstSRS=f'EPSG:{ref_nova}', outputType=raster.GetRasterBand(1).DataType)
    if not raster_ref:
        print('Erro na projeção')

    return raster_ref


def f16t8bits(raster, noData=0, path=None, nome=None):
    """
    Converte a resolução radiométrica de 16 bits para 8 bits de forma escalável

    Args:
        raster (gdal.Dataset): raster de entrada para conversão
        path (str):
        nome (str):

    Returns:
        gdal.Dataset: raster convertido para 8 bits

    """

    # checando possíveis erros
    if path:
        if not os.path.exists(path):
            print('Diretório não existe')
            return None

    if type(raster) is not gdal.Dataset:
        print("Erro de tipo")
        return None

    if raster.GetRasterBand(1).DataType == gdal.GDT_Byte:
        print("Já está em 8 bits")
        return raster

    if raster.GetRasterBand(1).DataType != gdal.GDT_UInt16:
        print("Não é 16 bits")
        return None

    # criando caminho do arquivo que será gerado
    dest = cria_destino(path, nome, raster.GetDescription(), extra='8bits')

    # executando a função de utilidade Warp para atualizar resolução radiométrica
    nraster = gdal.Translate(dest, raster, scaleParams=[0, 65535, 0, 255], outputType=gdal.GDT_Byte, noData=noData)

    return nraster


def mosaicar(rasters, nodata_value=0, path=None, nome=None):
    """
    Constrói mosaico de rasters utilizando gdal.Warp.
    Necessário que os rasters estejam com mesma resolução radiométrica, sistema de coordenadas e numero de bandas

    Args:
        rasters (list of gdal.Dataset): lista dos raster que serão mosaicados
        nodata_value (int): valor que será considerado nulo
        path (str): caminho do diretório de saída do arquivo raster
        nome (str): nome para o raster

    Returns:
        gdal.Dataset: gdal.Dataset com o raster final mosaicado

    """

    # checagem de possíveis erros
    if path:
        if not os.path.exists(path):
            print('Diretório não existe')
            return None

    if type(rasters) is not list or len(rasters) < 2:
        print('Não é uma lista de rasters para mosaicar')
        return None

    if not check_resol_radio(rasters):
        print('Rasters com resoluções radiométrica diferentes')
        return None  # padronizar erros - Rasters com resoluções radiométrica diferentes

    if not check_projection(rasters):  # talvez tenha que usar a classe de projeção pra comparar de fato já que tem posições diferentes (por ora parece que nao)
        print('Rasters com projeções diferentes')
        return None  # erro - Rasters com projeções diferentes

    if not check_num_bandas(rasters):
        print('Rasters com quantidade diferente de bandas')
        return None

    # prepara diretório que salvará o raster mosaicado
    dest = cria_destino(path, nome, rasters[0].GetDescription(), extra='mosaico')

    # utiliza função de utilidade Warp
    msc = gdal.Warp(dest, rasters, srcNodata=nodata_value, dstNodata=nodata_value, multithread=True)

    # define a forma de interpretação de cor do raster
    rc = msc.RasterCount
    if msc:  # Substituir pra try e expect
        if rc == 1:
            msc.GetRasterBand(1).SetColorInterpretation(gdal.GCI_GrayIndex)
        else:
            msc.GetRasterBand(1).SetColorInterpretation(gdal.GCI_RedBand)
            msc.GetRasterBand(2).SetColorInterpretation(gdal.GCI_GreenBand)
            msc.GetRasterBand(3).SetColorInterpretation(gdal.GCI_BlueBand)

        msc.FlushCache()

    else:
        print('Erro no mosaico')

    return msc


def recortar(raster, shape_path, where=None, nodata_value=0, path=None, nome=None):
    """Recorta um raster a partir de um shape

    Args:
        raster (gdal.Dataset): raster que sofrerá o recorte
        shape_path (str): diretório do shape + nome
        where (str): cláusula where no shape para recorte
        nodata_value (int): valor considerado sem dado do raster
        path (str): diretório que será salvo
        nome (str): nome do arquivo que será salvo

    Returns:
        gdal.Dataset: raster recortado pelo shape

    """

    # checagem de possíveis erros
    if path:
        if not os.path.exists(path):
            print('Diretório não existe')
            return None

    if not os.path.isfile(shape_path):
        print('Erro no arquivo/diretório')

    # prepara diretório que salvará o raster mosaicado
    dest = cria_destino(path, nome, raster.GetDescription(), extra='croped')
    # utiliza função de utilidade Warp
    rec = gdal.Warp(dest, raster, cutlineDSName=shape_path, srcNodata=nodata_value, dstNodata=nodata_value,
                    cutlineWhere=where, cropToCutline=True, outputType=raster.GetRasterBand(1).DataType,
                    multithread=True)

    if not rec:
        print('Erro no recorte')
        return None

    rec.FlushCache()

    return rec


def shp2raster(shape, pixel_wh=30, field=None, path=None, nome=None):
    """Converte um arquivo .shp (ESRI shapefile) em um arquivo raster de acordo com as classes do field escolhido

    Args:
        shape (Union[ogr.DataSource, str]): caminho do arquivo raster ou raster já carregado por drive GDAL
        pixel_wh (int): precisão do pixel, referente a resolução espacial. em m (30m = bandas multiespectrais landsat)
        field (str): nome do campo referência para conversão
        path (str): caminho do diretório onde será salvo o raster
        nome (str): nome do arquivo raster

    Returns:
        (gdal.Dataset, dict): raster codificado resultante da conversão do shape e dicionário dos códigos das classes

    """

    # Checando tipos e possíveis erros
    tp_shp = type(shape)
    if tp_shp is str:
        shp = ogr.Open(shape, 0)
    elif tp_shp is ogr.DataSource:
        shp = shape
    else:
        print("Erro de tipo")
        return None

    dest = cria_destino(path, nome, shp.GetDescription(), extra='shp2ras')
    if not dest:
        print('Erro: Arquivo destino não foi criado')
        return None

    lyr_shp = shp.GetLayer()
    ldef_shp = lyr_shp.GetLayerDefn()
    fi_shp = ldef_shp.GetFieldIndex(field)  # numero de indíce do field referência

    if fi_shp < 0:
        print('Erro: field não existe')
        return None

    # criando shape temporário (a API GDAL não permite salvar alterações no arquivo principal
    # e prevenir erros no arquivo fonte)
    driver = ogr.GetDriverByName('ESRI Shapefile')
    shp_aux = driver.CreateDataSource(os.path.join(gettempdir(), 'tmp.shp'))

    # criando layer do shape auxiliar com mesmas características do shape fonte
    lyr_shp_aux = shp_aux.CreateLayer(lyr_shp.GetName(), lyr_shp.GetSpatialRef(), lyr_shp.GetGeomType())
    ldef_shp_aux = lyr_shp_aux.GetLayerDefn()

    # copiando configuração do campo geométrico do shape para o shape auxiliar
    geom_old = ldef_shp.GetGeomFieldDefn(0)
    geom_new = ogr.GeomFieldDefn(geom_old.GetName())
    geom_new.SetType(geom_old.GetType())
    geom_new.SetSpatialRef(geom_old.GetSpatialRef())

    # criando o field que será salvo os códigos no shape auxiliar
    field_new = ogr.FieldDefn('code')
    field_new.SetType(0)
    ldef_shp_aux.AddFieldDefn(field_new)

    # adicionando os fields no layer do shape auxiliar
    ldef_shp_aux.AddGeomFieldDefn(geom_new)
    fi_shp_aux = ldef_shp_aux.GetFieldIndex('code')

    # checando o tipo de dado do field
    field_type = ldef_shp.GetFieldDefn(fi_shp).type

    # auxiliar para criar codificação caso já não seja codificado o field referência
    id_aux = 1

    # dicionário dos códigos das classes
    dic_codes = dict()
    for i in range(lyr_shp.GetFeatureCount()):
        # salvando a classe da feature atual
        ftre_old = lyr_shp.GetFeature(i)
        classe = ftre_old.GetField(fi_shp)

        # salvando no dicionário caso a classe ainda não tinha sido carregada
        if classe not in dic_codes:
            if field_type == 0:  # caso já seja um campo codificado
                dic_codes[classe] = classe
            else:
                dic_codes[classe] = id_aux
                id_aux += 1

        # criando a feature a ser inserida no shape auxiliar
        ftre = ogr.Feature(ldef_shp_aux)

        # adicionando o codigo e geometria da feature
        ftre.SetField(fi_shp_aux, dic_codes[classe])
        ftre.SetGeometry(ftre_old.GetGeometryRef())

        # adicionando a feature no shape auxiliar
        lyr_shp_aux.CreateFeature(ftre)

    # extent do shape auxiliar que será utilizado como referência pra criar o raster
    x_mn, x_mx, y_mn, y_mx = lyr_shp.GetExtent()
    print(lyr_shp.GetExtent())
    cols = int((x_mx - x_mn) / pixel_wh)
    rows = int((y_mx - y_mn) / pixel_wh)
    if not cols or not rows:
        print('Coluna ou Largura está zerada')
        return None

    if len(dic_codes) < 256:
        dtype = gdal.GDT_Byte
    else:
        dtype = gdal.GDT_UInt16

    # criando e preparando as informações geográficas do Raster
    raster = gdal.GetDriverByName('GTiff').Create(dest, cols, rows, 1, dtype)
    raster.SetGeoTransform((x_mn, pixel_wh, 0, y_mx, 0, -pixel_wh))
    raster.SetProjection(lyr_shp.GetSpatialRef().ExportToWkt())
    raster.GetRasterBand(1).SetNoDataValue(0)

    # transformando o shape auxiliar em raster
    gdal.RasterizeLayer(raster, [1], lyr_shp_aux, options=['ATTRIBUTE=code'])

    # atualizando as alterações no raster
    raster.FlushCache()

    # excluindo o shape auxiliar
    path_t = shp_aux.GetDescription()
    shp_aux = None
    os.remove(path_t)

    return raster, dic_codes


def gdal2nparray(raster):
    """

    Args:
        raster (gdal.Dataset):

    Returns: np.ndarray:

    """
    raster_t = type(raster)

    if raster_t is gdal.Dataset:
        bandas = get_bandas(raster)

        if len(bandas) > 3:
            print('Utilizando somente as 3 primeiras bandas')
            bandas = bandas[:3]

        for i in range(len(bandas)):
            bandas[i] = bandas[i].ReadAsArray()

        np_raster = np.dstack(tuple(bandas))

    elif raster_t is np.ndarray:
        np_raster = raster

    else:
        print('Erro no tipo')
        return None

    return np_raster


def segmentar(raster, scale, min_size, path=None, nome=None):
    """Segmenta um raster em regiões por meio do algoritmo felzenszwalb

    Args:
        raster (gdal.Dataset): Raster a ser segmentado
        scale (int): Scale do algoritmo felzenszwalb
        min_size (int): min_size do algoritmo felzenszwalb
        path (str): diretório do arquivo
        nome (str): nome do arquivo

    Returns:
        (gdal.Dataset): Raster segmentado, onde cada região terá um valor de identificação único em seus pixels.


    """
    if path:
        if not os.path.exists(path):
            print('Diretório não existe')
            return None

    np_raster = gdal2nparray(raster)

    if np_raster:

        np_seg = segmentation.felzenszwalb(np_raster, scale=scale, min_size=min_size)
        np_seg = np_seg + 1

        dest = cria_destino(path, nome, raster.GetDescription(), extra='segmentation')
        seg = gdal.GetDriverByName('GTiff').Create(dest, raster.RasterXSize, raster.RasterYSize, 1, gdal.GDT_UInt16)
        seg.SetGeoTransform(raster.GetGeoTransform())
        seg.SetProjection(raster.GetProjection())
        seg.GetRasterBand(1).WriteArray(np_seg)
        seg.FlushCache()

        if not seg:
            print('Erro ao criar raster')
            return None

        return seg

    return None


def calc_total_pixels(banda):
    larg, alt = banda.shape
    return larg * alt


def calc_assimetria(banda, mean, std, npix):
    soma = banda - mean
    soma = soma ** 3
    ndvp = std ** 3 * npix

    return (1/ndvp) * np.sum(soma)


def extrair_carac_regiao(regiao, caracteristicas, params=None):
    """
        Extração das caracterísitcas:
            - hog: Histogram of Oriented Gradients
            - media: media dos valores do pixel de cada banda
            - dsv_p: desvio padrão dos valores do pixel de cada banda
            - ast: assimetria
            - var: variancia
            - ent: entropia
            - crt: curtose
            - glcm: Grey level Co-occurrence Matrix (contraste, dissimilaridade, homogeneidade, ASM, energia, correlação)
            - lbp: Local Binary Pattern
    Args:
        regiao (Union[np.ndarray, np.ma.core.MaskedArray]): região de imagem que será extraída as características
        caracteristicas (list of str): lista de características a serem extraídas
        params (dict): parametros para o algoritmo de extração

    Returns:
        (list of float): lista de características extraídas
    """

    features = pd.DataFrame()
    b1 = regiao[..., 0]
    b2 = regiao[..., 1]
    b3 = regiao[..., 2]

    # total de pixels
    b1_npix = calc_total_pixels(b1)
    b2_npix = calc_total_pixels(b2)
    b3_npix = calc_total_pixels(b3)

    # média
    b1_mean = b2_mean = b3_mean = mean = None
    if 'media' in caracteristicas:
        b1_mean = np.mean(b1)
        b2_mean = np.mean(b2)
        b3_mean = np.mean(b3)

        features = features.assign(media=[[b1_mean, b2_mean, b3_mean, mean]])

    b1_std = b2_std = b3_std = std = None
    # desvio padrão
    if 'dsv_p' in caracteristicas:
        b1_std = np.std(b1)
        b2_std = np.std(b2)
        b3_std = np.std(b3)

        features = features.assign(dsv_p=[[b1_std, b2_std, b3_std, std]])

    # assimetria
    if 'ast' in caracteristicas:
        if not b1_mean:
            b1_mean = np.mean(b1)
            b2_mean = np.mean(b2)
            b3_mean = np.mean(b3)
        if not b1_std:
            b1_std = np.std(b1)
            b2_std = np.std(b2)
            b3_std = np.std(b3)

        b1_ast = calc_assimetria(b1, b1_mean, b1_std, b1_npix)
        b2_ast = calc_assimetria(b2, b2_mean, b2_std, b2_npix)
        b3_ast = calc_assimetria(b3, b3_mean, b3_std, b3_npix)

        features = features.assign(ast=[[b1_ast, b2_ast, b3_ast]])

    # variancia
    if 'var' in caracteristicas:
        b1_var = variance(b1)
        b2_var = variance(b1)
        b3_var = variance(b1)

        features = features.assign(var=[[b1_var, b2_var, b3_var]])

    # histograma
    if 'ent' or 'crt' in caracteristicas:
        b1_hst, _ = histogram(b1, nbins=b1.max())  # alterar nbins de acordo com dtype
        b2_hst, _ = histogram(b2, nbins=b2.max())  # alterar nbins de acordo com dtype
        b3_hst, _ = histogram(b3, nbins=b3.max())  # alterar nbins de acordo com dtype

        # entropia  - hst
        if 'ent' in caracteristicas:
            b1_ent = entropy(b1_hst)
            b2_ent = entropy(b2_hst)
            b3_ent = entropy(b3_hst)

            features = features.assign(ent=[[b1_ent, b2_ent, b3_ent]])

        # curtose - hst
        if 'crt' in caracteristicas:
            b1_crt = kurtosis(b1_hst)
            b2_crt = kurtosis(b2_hst)
            b3_crt = kurtosis(b3_hst)

            features = features.assign(crt=[[b1_crt, b2_crt, b3_crt]])

    if not params:
        params = dict()
    if type(params) is not dict:
        print('Params precisa ser do tipo dict')
        return None

    # lbp
    if 'lbp' in caracteristicas:
        if 'P' in params:
            p = params['P']
            if type(p) is not int:
                print('P precisa ser um valor inteiro')
                return None
        else:
            p = 8

        if 'R' in params:
            r = params['R']
            if type(r) is not int and type(r) is not float:
                print('R precisa ser um valor float')
                return None
        else:
            r = 1

        b1_lbp = local_binary_pattern(b1, p, r, method='ror')
        b2_lbp = local_binary_pattern(b2, p, r, method='ror')
        b3_lbp = local_binary_pattern(b3, p, r, method='ror')

        b1_lbp_h, _ = histogram(b1_lbp.ravel())
        b2_lbp_h, _ = histogram(b2_lbp.ravel())
        b3_lbp_h, _ = histogram(b3_lbp.ravel())

        b1_min = b1_lbp_h.min()
        b2_min = b2_lbp_h.min()
        b3_min = b3_lbp_h.min()

        # ToDo: melhorar normalização
        #   -> checar se só serão utilizados o primeiro e o último do histograma
        #       (possivelmente mudar o nbins do histogram resolve)
        b1_lbp_h = (b1_lbp_h - b1_min) / (b1_lbp_h.max() - b1_min)
        b2_lbp_h = (b2_lbp_h - b2_min) / (b2_lbp_h.max() - b2_min)
        b3_lbp_h = (b3_lbp_h - b3_min) / (b3_lbp_h.max() - b3_min)

        features = features.assign(lbp_b1=[list(b1_lbp_h)], lbp_b2=[list(b2_lbp_h)], lbp_b3=[list(b3_lbp_h)])

    # hog
    if 'hog' in caracteristicas:
        if 'pixels_per_cell' in params:
            pixels_per_cell = params['pixels_per_cell']
            if type(pixels_per_cell) is not tuple and len(pixels_per_cell) != 2:
                print('Erro no parametro pixels_per_cell. Ex: (8, 8)')
                return None
        else:
            pixels_per_cell = (8, 8)

        if 'pixels_per_cell' in params:
            cells_per_block = params['pixels_per_cell']
            if type(cells_per_block) is not tuple and len(cells_per_block) != 2:
                print('Erro no parametro cells_per_block. Ex: (3, 3)')
                return None
        else:
            cells_per_block = (8, 8)

        h = hog(regiao, block_norm='L2-Hys', visualize=False, feature_vector=True, multichannel=True,
                pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block)
        features = features.assign(hog=[list(h)])

    # glcm
    if 'glcm' in caracteristicas:
        if 'distances' in params:
            distances = params['distances']
            if type(distances) is list:
                for dist in distances:
                    if type(dist) is not int:
                        print('Valores de distancia devem ser inteiros')
                        return None
            else:
                print('Distancia devem estar em lista')
                return None
        else:
            distances = [1]

        if 'angles' in params:
            angles = params['angles']
            if type(angles) is list:
                for ang in angles:
                    if type(ang) is not float and type(ang) is not int:
                        print('Valores de angulos devem ser float')
                        return None
            else:
                print('Angulos devem estar em lista')
                return None
        else:
            angles = [np.pi / 2]

        b1_glcm = greycomatrix(b1, distances, angles, levels=b1.max() + 1)
        b2_glcm = greycomatrix(b2, distances, angles, levels=b2.max() + 1)
        b3_glcm = greycomatrix(b3, distances, angles, levels=b3.max() + 1)

        glcm_res = list()
        # contrast
        glcm_res.append(greycoprops(b1_glcm, 'contrast')[0][0])
        glcm_res.append(greycoprops(b2_glcm, 'contrast')[0][0])
        glcm_res.append(greycoprops(b3_glcm, 'contrast')[0][0])

        # dissimilarity
        glcm_res.append(greycoprops(b1_glcm, 'dissimilarity')[0][0])
        glcm_res.append(greycoprops(b2_glcm, 'dissimilarity')[0][0])
        glcm_res.append(greycoprops(b3_glcm, 'dissimilarity')[0][0])

        # homogeneity
        glcm_res.append(greycoprops(b1_glcm, 'homogeneity')[0][0])
        glcm_res.append(greycoprops(b2_glcm, 'homogeneity')[0][0])
        glcm_res.append(greycoprops(b3_glcm, 'homogeneity')[0][0])

        # energy
        glcm_res.append(greycoprops(b1_glcm, 'energy')[0][0])
        glcm_res.append(greycoprops(b2_glcm, 'energy')[0][0])
        glcm_res.append(greycoprops(b3_glcm, 'energy')[0][0])

        # correlation
        glcm_res.append(greycoprops(b1_glcm, 'correlation')[0][0])
        glcm_res.append(greycoprops(b2_glcm, 'correlation')[0][0])
        glcm_res.append(greycoprops(b3_glcm, 'correlation')[0][0])

        # ASM
        glcm_res.append(greycoprops(b1_glcm, 'ASM')[0][0])
        glcm_res.append(greycoprops(b2_glcm, 'ASM')[0][0])
        glcm_res.append(greycoprops(b3_glcm, 'ASM')[0][0])

        features = features.assign(glcm=[glcm_res])

    return features


def extrair_caracteristicas(raster, mask, caracteristicas, params=None):
    """
        Separa a imagem nas regiões definidas da máscara e executa a função de extração para cada região.
        Extração das caracterísitcas:
            - hog: Histogram of Oriented Gradients
            - media: media dos valores do pixel de cada banda
            - dsv_p: desvio padrão dos valores do pixel de cada banda
            - var: variancia
            - entr: entropia
            - crt: curtose
            - glcm: Grey level Co-occurrence Matrix (contraste, dissimilaridade, homogeneidade, ASM, energia, correlação)
            - lbp: Local Binary Pattern
    Args:
        raster (Union[gdal.Dataset, np.ndarray]): raster a sofrer a extração de características
        mask (Union[gdal.Dataset, np.ndarray]): raster máscara que define as divisões das regiões
        caracteristicas (list of str): lista de características a serem extraídas
        params (dict): parametros para o algoritmo de extração

    Returns:
        (pd.DataFrame): DataFrame com todas as características extraídas de todas regiões
    """

    # checando o preparando raster
    raster_t = type(raster)

    if raster_t is gdal.Dataset:
        bandas = get_bandas(raster)
        # readasarray converte todas 3 bandas

        for i in range(len(bandas)):
            bandas[i] = bandas[i].ReadAsArray()

        np_raster = np.dstack(tuple(bandas))
        # np_raster = gdal2nparray(bandas)

    elif raster_t is np.ndarray:
        np_raster = raster

    else:
        print('!Erro no tipo')
        return None

    # checando o preparando mask
    mask_t = type(mask)

    if mask_t is gdal.Dataset:
        banda = get_bandas(mask)
        np_mask = banda.ReadAsArray()
    elif mask_t is np.ndarray:
        np_mask = mask
    else:
        print('Erro no tipo')
        return None

    for c in caracteristicas:
        if c not in ['hog', 'media', 'dsv_p', 'var', 'entr', 'crt', 'glcm', 'lbp']:
            print('Erro: característica inválida')
            return None

    features = pd.DataFrame()
    regioes = np.unique(np_mask)
    regioes = np.delete(regioes, 0)
    for reg in regioes:
        rows, cols = np.where(np_mask == reg)

        np_reg = np_raster[min(rows): max(rows) + 1, min(cols): max(cols) + 1, :].copy()
        np_mask_reg = np_mask[min(rows): max(rows) + 1, min(cols): max(cols) + 1].copy()

        r, c = np.where(np_mask_reg == reg)
        np_mask_reg[r, c] = 0
        np_reg = np.ma.masked_array(
            np_reg,
            np.dstack((np_mask_reg, np_mask_reg, np_mask_reg)),
            fill_value=0
        )

        ftr_regiao = extrair_carac_regiao(np_reg, caracteristicas, params)
        if ftr_regiao is None:
            return None

        ftr_regiao = ftr_regiao.assign(reg=int(reg))

        features = features.append(ftr_regiao, ignore_index=True, sort=False)

    return features


def create_npmask(np_raster, no_data):
    """Função auxiliar que cria uma mascara para ignorar a região sem informação do raster original

    Args:
        np_raster (np.ndarray):
        no_data (int): valor considerado como nenhum dado no raster

    Returns:
        (np.ndarray): mascara onde 0 será o valor da região ignorada e 1 o valor da região válida
    """
    np_mask = np.ones(np_raster.shape, np.uint8)
    coords = np.where(np_raster == no_data)
    np_mask[coords] = 0
    return np_mask


def create_fmask(shape, r, filtro):
    """ Função auxiliar que cria a mascara para os filtros de passa alta e baixa ideal

    Args:
        shape (tuple): shape da imagem no domínio de fourier
        r (int): raio de distância do centro da imagem onde será definido o filtro
        filtro(int): tipo de filtro FT_FPB_IDEAL ou FT_FPA_IDEAL

    Returns:
        (np.ndarray): mascará de acordo com o filtro ideal a ser processado
    """

    rows, cols = shape
    cy, cx = rows // 2, cols // 2
    y, x = np.ogrid[:rows, :cols]
    reg = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)

    if not r:
        r = min(cx, cy, cols - cx, rows - cy)
    if filtro == FT_FPB_IDEAL:
        reg_mask = reg <= r
    else:
        reg_mask = reg > r

    return reg_mask


def realcar(np_raster, np_filtrada):
    """ Função auxiliar que incrementa um imagem filtrada de passa alta a imagem original

    Args:
        np_raster (np.ndarray): imagem original
        np_filtrada (np.ndarray): imagem com filtro de passa alta

    Returns:
        (np.ndarray): imagem realçada
    """
    return np_raster + np_filtrada


def alto_reforco(np_raster, np_filtrada, k=1):
    """ Função auxiliar que executa o processo de alto_reforço

    Args:
        np_raster (np.ndarray): imagem original
        np_filtrada (np.ndarray): imagem filtrada
        k (float): constante que multiplica mascara de nitidez

    Returns:
        (np.ndarray): imagem resultante do alto reforço
    """

    mask_nitidez = np_raster - np_filtrada
    return np_raster + k * mask_nitidez


def float2uint(np_raster, dtype):
    """ Função auxiliar para converter imagens em float, resultantes de filtros, para uint8 ou uint16

    Args:
        np_raster (np.ndarray): imagem em float
        dtype (np.dtype): dtype a ser convertido

    Returns:
        (np.ndarray): imagem em uint8 ou uint16
    """

    if dtype == np.uint16:
        return img_as_uint(np_raster)
    elif dtype == np.uint8:
        return img_as_ubyte(np_raster)

    print('Datatype não reconhecido')
    return None


def filtrar(raster, filtro, params, alto_ref=False, realce=False, path=None, nome=None):
    """ Executa um filtro em um raster georreferenciado. O raster é convertido em uma imagem não georreferenciada,
    onde é executado o filtro e depois é criado novamente em raster georreferenciado.
    filtros disponíveis: correção gama, equalização do histograma, mediana, gaussiano, sobel, laplace,
    passa baixa ideal (domínio da frequencia) e passa alta ideal (domínio da frequencia).

    Possível utilizar alto reforço (high boost) nos filtros de mediana, gaussiano e passa baixa ideal (alto_ref=True).

    Possível utilizar realce nos filtros de passa baixa (realce=True).


    parametros disponíveis por filtro:
        - FT_GAMA: 'gamma', 'gain'
        - FT_MEDIANA: 'kernel'
        - FT_GAUSS: 'sigma'
        - FT_LAPLACE: 'lpc_oper'
        - FT_FPB_IDEAL e FT_FPA_IDEAL: 'ft_raio'

    Args:
        raster(gdal.Dataset): Raster gdal a ser filtrado
        filtro(int): Código do filtro disponível. Utilizar as constantes FT_...
        params(dict): Dicionário com nome e valores dos parametros do filtro a ser processado
        alto_ref(bool): Executar alto reforço com filtros de passa baixa
        realce(bool): Executar realce com filtros de passa alta
        path (str): diretório do arquivo raster resultante
        nome (str): nome do arquivo raster resultante

    Returns:
        (gdal.Dataset): Raster gdal filtrado
    """

    if path:
        if not os.path.exists(path):
            print('Diretório não existe')
            return None

    raster_count = raster.RasterCount
    np_raster = gdal2nparray(raster)
    if np_raster is None:
        return None

    if not params:
        params = dict()

    np_filtrada = None
    if filtro == FT_GAMA:
        extra = 'ft_gama'
        # ToDo: confirmar se funciona em imagem composta diretamente
        gamma = params['gamma'] if 'gamma' in params else 1
        if gamma < 0:
            print('Valor de gamma não pode ser negativo')
            return None
        gain = params['gain'] if 'gain' in params else 1
        np_filtrada = adjust_gamma(np_raster, gamma, gain)

    elif filtro == FT_EQLHIST:
        extra = 'ft_equal_hist'
        np_mask = create_npmask(np_raster, raster.GetRasterBand(1).GetNoDataValue())
        if raster_count > 1:
            aux = list()
            for i in range(raster_count):
                np_band = np_raster[..., i]
                aux.append(equalize_hist(np_band, np_band.max(), np_mask[..., i]))
            np_filtrada = np.dstack(aux)
        else:
            np_filtrada = equalize_hist(np_raster, np_raster.max(), np_mask)
        np_filtrada = float2uint(np_filtrada, np_raster.dtype)

    elif filtro == FT_MEDIANA:
        extra = 'ft_mediana'
        np_mask = create_npmask(np_raster, raster.GetRasterBand(1).GetNoDataValue())
        shape = params['kernel'] if 'kernel' in params else (3, 3)
        if type(shape) is not tuple:
            print('tipo de dado do kernel inválido')
            return None
        else:
            if len(shape) != 2:
                print('Shape do kernel inválido. exemplo correto: (x, y)')
                return None
        selem = np.ones(shape)
        aux = list()
        if raster_count > 1:
            for i in range(raster_count):
                aux.append(median(np_raster[..., i], selem, mask=np_mask[..., i]))
            np_filtrada = np.dstack(tuple(aux))
        else:
            np_filtrada = median(np_raster, selem, mask=np_mask)

        if alto_ref:
            extra += '_alto_ref'
            np_filtrada = alto_reforco(np_raster, np_filtrada)

    elif filtro == FT_GAUSS:
        extra = 'ft_gauss'
        if raster.RasterCount > 1:
            multichannel = True
        else:
            multichannel = False
        sigma = params['sigma'] if 'sigma' in params else 1
        np_filtrada = gaussian(np_raster, sigma, multichannel=multichannel, preserve_range=True)

        if alto_ref:
            extra += '_alto_ref'
            np_filtrada = alto_reforco(np_raster, np_filtrada)

    elif filtro == FT_SOBEL:
        extra = 'ft_sobel'
        np_mask = create_npmask(np_raster, raster.GetRasterBand(1).GetNoDataValue())
        if raster_count > 1:
            aux = list()
            for i in range(raster_count):
                aux.append(sobel(np_raster[..., i], np_mask[..., i]))
            np_filtrada = np.dstack(tuple(aux))

        else:
            np_filtrada = sobel(np_raster, np_mask)

        np_filtrada = float2uint(np_filtrada, np_raster.dtype)

        if realce:
            extra += '_realce'
            np_filtrada = realcar(np_raster, np_filtrada)

    elif filtro == FT_LAPLACE:
        extra = 'ft_laplace'
        np_mask = create_npmask(np_raster, raster.GetRasterBand(1).GetNoDataValue())
        ksize = params['lpc_oper'] if 'lpc_oper' in params else 3
        if raster_count > 1:
            aux = list()
            for i in range(raster_count):
                aux.append(laplace(np_raster[..., i], ksize, np_mask[..., i]))
            np_filtrada = np.dstack(tuple(aux))
        else:
            np_filtrada = laplace(np_raster, ksize, np_mask)

        lpc_min = np_filtrada.min()
        if lpc_min < 0:
            np_filtrada = np_filtrada - lpc_min

        if realce:
            extra += '_realce'
            np_filtrada = realcar(np_raster, np_filtrada)

    elif filtro == FT_FPB_IDEAL or FT_FPA_IDEAL:
        extra = 'ft_fourier'
        if filtro == FT_FPB_IDEAL:
            extra += '_pb_ideal'
        else:
            extra += '_pa_ideal'

        r = params['ft_raio'] if 'ft_raio' in params else None
        mask_reg = create_fmask(np_raster[..., 0].shape, r, filtro)

        rst_ft = np.fft.fft2(np_raster)
        rst_ftst = fftshift(rst_ft)
        rst_ftst[mask_reg] = 0
        rst_r_ft = np.fft.ifftshift(rst_ftst)
        np_filtrada = np.abs(np.fft.ifft2(rst_r_ft))
        r, c, b = np.where(np_raster == 0)
        np_filtrada[r, c, b] = 0
        if realce:
            extra += '_realce'
            np_filtrada = realcar(np_raster, np_filtrada)

        if alto_ref:
            extra += '_alto_ref'
            np_filtrada = alto_reforco(np_raster, np_filtrada)

    else:
        print('Erro: filtro não reconhecido')
        return None

    if np_filtrada is not None:
        # converter em raster
        col = raster.RasterXSize
        row = raster.RasterYSize
        geo_transf = raster.GetGeoTransform()
        proj = raster.GetProjection()
        dtype = raster.GetRasterBand(1).DataType

        dest = cria_destino(path, nome, raster.GetDescription(), extra=extra)
        driver = gdal.GetDriverByName(TIFF)
        raster_filtrada = driver.Create(dest, col, row, raster_count, dtype, ['PHOTOMETRIC=RGB'])

        # adicionando as informações geográficas
        raster_filtrada.SetGeoTransform(geo_transf)
        raster_filtrada.SetProjection(proj)

        # escrevendo os dados das bandas no raster
        if raster_count == 1:
            raster_filtrada.GetRasterBand(1).WriteArray(np_filtrada)
        else:
            for i in range(raster_count):
                raster_filtrada.GetRasterBand(i + 1).WriteArray(np_filtrada[..., i])

        # atualizando as alterações no raster
        raster_filtrada.FlushCache()

        return raster_filtrada

    print('Erro no filtro')
    return None


def series2array(df, cols_ignore):
    """

    Args:
        df(pd.DataFrame):
        cols_ignore(Union[str, list]):

    Returns:

    """

    if type(cols_ignore) is str:
        cols_ignore = [cols_ignore]

    for col in df.columns:
        if col not in cols_ignore:
            df[col] = df[col].map(lambda x: np.array(x))
    return df


def prepara_features(df, class_col='classe'):
    """
    
    Args:
        df (pd.DataFrame): 
        class_col (str): 

    Returns:
        (list, pd.Series)

    """
    target = df.pop(class_col)
    data_flatted = [np.concatenate(x).ravel().tolist() for x in df.values]
    return data_flatted, target.values


def treinar_modelo(df_train, tipo_class, df_eval=None, class_col='classe', id_col=None, val_split=0.75,
                   classificador=None, params=None):
    """
    Ajusta um modelo de classificação a partir de uma base de conhecimento, podendo ser uma Rede Neural Artificial (RNA)
    -Multilayer Perceptron- ou Máquina de Vetores de Suporte (SVM).
    parametros possíveis de cada classificador (utilizando o dicicionário params):
        - RNA:
            - hidden_units: lista com a quantidade de nós em cada layer. exemplo: [50, 50, 50], default: [10]
            - learning rate: valor de aprendizado do algoritmo otimizador. default: 0.01
            - steps: repetições do treinamento (backpropagation). Padrão 10000
        - SVM:
            - kernel: tipo de kernel trick (linear, poly, rbg, sigmoid). default: linear
            - degree: graus do kernel trick polinomial (poly). default: 3
            - gamma: coeficiente do kernel (auto, scale)
            - coef0: termo idependente dos kernel tricks sigmoid e poly
            - tol: tolerancia para o critério de parada. default: 1e-3


    Args:
        df_train (pd.DataFrame): base de dados para treinamento
        tipo_class (int): tipo de classificador (rh.RNA, rh.SVM)
        df_eval (pd.DataFrame): base de dados para teste (Opcional)
        class_col (str): nome da coluna com as classes
        id_col (str): nome da coluna de identificação única de cada tupla de dado, se houver
        val_split(float): valor utilizado para dividir a base de treinamento, caso não houver uma base para testes
        classificador (Union[tf.estimator.DNNClassifier, svm.SVC]): classificador para treinar em uma nova base
        params (dict): parametros para criação dos classificadores

    Returns:

    """

    # checando erros
    if type(df_train) is not pd.DataFrame:
        print('Erro: erro do tipo da DataFrame')
        return None

    accepted_params = ['learning_rate', 'hidden_units', 'steps', 'kernel', 'degree', 'gamma', 'coef0', 'tol']
    if not params:
        params = dict()
    else:
        for key in params:
            if key not in accepted_params:
                print('Parametro não reconhecido')
                return None

    # copiando dataframe de treinamento
    df_train_c = df_train.copy()
    # retirando coluna de id único caso tenha no dataframe
    if id_col:
        df_train_c.pop(id_col)
    df_train_c = series2array(df_train_c, class_col)

    if df_eval is not None:
        df_eval_c = df_eval.copy()
        if id_col:
            df_eval_c.pop(id_col)
        df_eval_c = series2array(df_eval_c, class_col)
    else:
        if 0 < val_split < 1:
            all_data = df_train_c.sample(len(df_train_c)).reset_index(drop=True)
            split = int(len(df_train_c) * val_split)
            df_train_c = all_data.iloc[:split].reset_index(drop=True)
            df_eval_c = all_data.iloc[split:].reset_index(drop=True)
        else:
            print('Sem dados para avaliar')
            return None

    data_train_flatted, target = prepara_features(df_train_c, class_col)
    data_eval_flatted, target_ev = prepara_features(df_eval_c, class_col)

    if tipo_class == RNA:

        # salvando labels e quantidade
        lbs = np.sort(np.unique(target))
        n_classes = len(lbs)

        # readequando nivel de labels para 0 -> n_classes - 1
        lbs_map = list(enumerate(lbs))
        lbs_dic = dict()
        for lb in lbs_map:
            lbs_dic[lb[1]] = lb[0]
        lbs_dic_ = dict(zip(lbs_dic.values(), lbs_dic.keys()))
        for i in range(len(target)):
            target[i] = lbs_dic[target[i]]
        for i in range(len(target_ev)):
            target_ev[i] = lbs_dic[target_ev[i]]

        # transformando em Tensor
        ds_train = tf.data.Dataset.from_tensor_slices((data_train_flatted, target))
        ds_train = ds_train.shuffle(buffer_size=len(df_train_c))
        ds_eval = tf.data.Dataset.from_tensor_slices((data_eval_flatted, target_ev))
        ds_eval = ds_eval.batch(1)

        # salvando o shape das features e criando entrada da RNA
        shape_ftr = tf.compat.v1.data.get_output_shapes(ds_train)
        shape_ftr = (shape_ftr[0].num_elements(), shape_ftr[1].num_elements())

        aux = len(df_train_c) // 10
        batch_size = aux if aux > 0 else 1
        ds_train = ds_train.batch(batch_size)

        feature_col = [tf.feature_column.numeric_column(key='x', shape=shape_ftr)]

        def train_input_fn():
            def gen1(a, b):
                return {'x': a}, b

            ds = ds_train.map(gen1)
            itr = tf.compat.v1.data.make_one_shot_iterator(ds)
            data, labels = itr.get_next()
            return data, labels

        def predict_input_fn():
            def gen1(a, b):
                return {'x': a}, b

            ds = ds_eval.map(gen1)
            itr = tf.compat.v1.data.make_one_shot_iterator(ds)
            data, _ = itr.get_next()
            return data, None

        if classificador is None:
            learning_rate = params['learning_rate'] if 'learning_rate' in params else 0.01
            classificador = tf.estimator.DNNClassifier(
                hidden_units=params['hidden_units'] if 'hidden_units' in params else [10],
                n_classes=n_classes,
                feature_columns=feature_col,
                optimizer=tf.compat.v1.train.GradientDescentOptimizer(learning_rate)
            )

        if type(classificador) is tf.estimator.DNNClassifier:
            steps = params['steps'] if 'steps' in params else 10000
            classificador.train(train_input_fn, steps=steps)
            target_pr = np.array([pr['class_ids'][0] for pr in classificador.predict(predict_input_fn)])
            # retornando o valor original dos labels
            for i in range(len(target_pr)):
                target_pr[i] = lbs_dic_[target_pr[i]]
            for i in range(len(target_ev)):
                target_ev[i] = lbs_dic_[target_ev[i]]
        else:
            print('Erro: tipo de classificador não reconhecido')
            return None

    elif tipo_class == SVM:
        if classificador is None:
            classificador = svm.SVC(
                kernel=params['kernel'] if 'kernel' in params else 'linear',
                degree=params['degree'] if 'degree' in params else 3,
                gamma=params['gamma'] if 'gamma' in params else 'auto',
                coef0=params['coef0'] if 'coef0' in params else 0.0,
                tol=params['tol'] if 'tol' in params else 1e-3
            )
        if type(classificador) is svm.SVC:
            classificador.fit(data_train_flatted, target)
            target_pr = classificador.predict(data_eval_flatted)
        else:
            print('Erro: tipo de classificador não reconhecido')
            return None

    else:
        print('Erro: algoritmo de aprendizado não reconhecido')
        return None

    avaliacao = {
        'accuracy': metrics.accuracy_score(target_ev, target_pr),
        'balanced_accuracy': metrics.balanced_accuracy_score(target_ev, target_pr),
        'precision_micro': metrics.precision_score(target_ev, target_pr, average='micro'),
        'recall_micro': metrics.recall_score(target_ev, target_pr, average='micro'),
        'f1_micro': metrics.f1_score(target_ev, target_pr, average='micro'),
        'precision': metrics.precision_score(target_ev, target_pr, average='macro'),
        'recall': metrics.recall_score(target_ev, target_pr, average='macro'),
        'f1': metrics.f1_score(target_ev, target_pr, average='macro'),
        'brier_score_loss': metrics.brier_score_loss(target_ev, target_pr),
        'confusion_matrix': metrics.confusion_matrix(target_ev, target_pr)
    }

    if tipo_class == SVM:
        return classificador, avaliacao
    if tipo_class == RNA:
        return classificador, avaliacao, lbs_dic_


def classificar(raster, mask, caracteristicas, classificador, path=None, nome=None, lbs_dict=None):
    """
        Executa o processo de classificação do raster, em que cada pixel da região assumirá o valor da classe calculada
        Necessário passar o classificador construído na função treinar_modelo

    Args:
        raster (gdal.Dataset): raster a ser extraída as características e classificada
        mask (gdal.Dataset): raster máscara que define as regiões segmentadas
        caracteristicas (list): lista com nome das características a serem extraídas
        classificador (Union[tf.estimator.DNNClassifier, svm.SVC]):
        path (str): diretório que será salvo o raster classificado
        nome (str): nome do arquivo que será salvo o raster classificado
        lbs_dict (dict): dicionário de conversão dos labels para uso do classificador RNA

    Returns:

    """

    if path:
        if not os.path.exists(path):
            print('Diretório não existe')
            return None
    dest = cria_destino(path, nome, raster.GetDescription(), extra='class')

    if type(classificador) is not tf.estimator.DNNClassifier and type(classificador) is not svm.SVC:
        print('Erro do classificador')
        return None

    if not caracteristicas:
        print('Necessário lista de características')
        return None

    # checando e preparando raster
    raster_t = type(raster)
    if raster_t is gdal.Dataset:
        bandas = get_bandas(raster)
        # readasarray converte todas 3 bandas
        for i in range(len(bandas)):
            bandas[i] = bandas[i].ReadAsArray()
        np_raster = np.dstack(tuple(bandas))
        # np_raster = gdal2nparray(bandas)
    else:
        print('Erro no tipo')
        return None

    # checando o preparando mask
    mask_t = type(mask)
    if mask_t is gdal.Dataset:
        mask_b = get_bandas(mask)

        np_mask = mask_b.ReadAsArray()
    else:
        print('Erro no tipo')
        return None

    features_ext = extrair_caracteristicas(np_raster, np_mask, caracteristicas)
    if features_ext is None:
        print('Erro na extração')
        return None
    features, reg = prepara_features(features_ext, 'reg')

    if type(classificador) is tf.estimator.DNNClassifier:
        if not lbs_dict:
            print('Necessário dicionário de labels')
            return None
        elif not len(lbs_dict):
            print('Dicionário de labels vazio')
            return None

        ds_eval = tf.data.Dataset.from_tensor_slices(features)
        ds_eval = ds_eval.batch(1)

        def predict_input_fn():
            def gen1(a):
                return {'x': a}

            ds = ds_eval.map(gen1)
            itr = tf.compat.v1.data.make_one_shot_iterator(ds)
            data = itr.get_next()
            return data, None

        target_pr = np.array([pr['class_ids'][0] for pr in classificador.predict(predict_input_fn)])

        # corrigindo ao label correto
        for i in range(len(target_pr)):
            target_pr[i] = lbs_dict[target_pr[i]]

    elif type(classificador) is svm.SVC:
        target_pr = classificador.predict(features)

    col = mask.RasterXSize
    row = mask.RasterYSize
    geo_transf = mask.GetGeoTransform()
    proj = mask.GetProjection()

    driver = gdal.GetDriverByName(TIFF)
    raster_class = driver.Create(dest, col, row, 1, gdal.GDT_Byte)

    # adicionando as informações geográficas
    raster_class.SetGeoTransform(geo_transf)
    raster_class.SetProjection(proj)

    np_class = np.copy(np_mask)
    for i in range(len(reg)):
        rows, cols = np.where(np_class == reg[i])
        np_class[rows, cols] = target_pr[i]

    # escrevendo os dados das bandas no raster
    raster_class.GetRasterBand(1).WriteArray(np_class)

    # atualizando as alterações no raster
    raster_class.FlushCache()

    return raster_class
