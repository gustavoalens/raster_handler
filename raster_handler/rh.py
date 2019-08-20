from osgeo import gdal, ogr

from skimage import segmentation
from skimage.exposure import histogram
from skimage.color import rgb2grey
from skimage.feature import local_binary_pattern, hog, greycomatrix, greycoprops
from skimage import img_as_uint

from scipy.stats import entropy, kurtosis
from scipy.ndimage import variance

import pandas as pd

import tensorflow as tf

from typing import Union
import numpy as np
import os
from tempfile import gettempdir

"""
"""
# tf.compat.v1.enable_eager_execution()

# - Constantes - #
TIFF = 'GTiff'
RN = 0
SVM = 1
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
    dest = cria_destino(path, nome, raster.GetDescription(), extra=f'EPSG:{ref_nova}')

    # executando a função de utilidade Warp para atualizar referencia espacial
    raster_ref = gdal.Warp(dest, raster, dstSRS=f'EPSG-{ref_nova}', outputType=raster.GetRasterBand(1).DataType)
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
        raster:

    Returns:

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


def calc_energia(hst, npix):
    prob = np.array(hst, dtype=np.float)
    prob /= npix
    return np.sum(prob ** 2)


def extrair_carac_regiao(regiao, caracteristicas):
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

    Returns:
        (list of float): lista de características extraídas
    """

    features = pd.DataFrame()
    b1 = regiao[..., 0]
    b2 = regiao[..., 1]
    b3 = regiao[..., 2]
    reg_g = img_as_uint(rgb2grey(regiao))

    # total de pixels
    b1_npix = calc_total_pixels(b1)
    b2_npix = calc_total_pixels(b2)
    b3_npix = calc_total_pixels(b3)
    npix = calc_total_pixels(reg_g)

    # média
    b1_mean = b2_mean = b3_mean = mean = None
    if 'media' in caracteristicas:
        b1_mean = np.mean(b1)
        b2_mean = np.mean(b2)
        b3_mean = np.mean(b3)
        mean = np.mean(reg_g)

        features = features.assign(media=[[b1_mean, b2_mean, b3_mean, mean]])

    b1_std = b2_std = b3_std = std = None
    # desvio padrão
    if 'dsv_p' in caracteristicas:
        b1_std = np.std(b1)
        b2_std = np.std(b2)
        b3_std = np.std(b3)
        std = np.std(reg_g)

        features = features.assign(dsv_p=[[b1_std, b2_std, b3_std, std]])

    # assimetria
    if 'ast' in caracteristicas:
        if not b1_mean:
            b1_mean = np.mean(b1)
            b2_mean = np.mean(b2)
            b3_mean = np.mean(b3)
            mean = np.mean(reg_g)
        if not b1_std:
            b1_std = np.std(b1)
            b2_std = np.std(b2)
            b3_std = np.std(b3)
            std = np.std(reg_g)

        b1_ast = calc_assimetria(b1, b1_mean, b1_std, b1_npix)
        b2_ast = calc_assimetria(b2, b2_mean, b2_std, b2_npix)
        b3_ast = calc_assimetria(b3, b3_mean, b3_std, b3_npix)
        ast = calc_assimetria(reg_g, mean, std, npix)

        features = features.assign(ast=[[b1_ast, b2_ast, b3_ast, ast]])

    # variancia
    if 'var' in caracteristicas:
        b1_var = variance(b1)
        b2_var = variance(b1)
        b3_var = variance(b1)
        var = variance(reg_g)

        features = features.assign(var=[[b1_var, b2_var, b3_var, var]])

    # histograma
    if 'ent' or 'crt' in caracteristicas:
        b1_hst, _ = histogram(b1, nbins=b1.max())  # alterar nbins de acordo com dtype
        b2_hst, _ = histogram(b2, nbins=b2.max())  # alterar nbins de acordo com dtype
        b3_hst, _ = histogram(b3, nbins=b3.max())  # alterar nbins de acordo com dtype
        hst, _ = histogram(reg_g, nbins=reg_g.max())

        # entropia  - hst
        if 'ent' in caracteristicas:
            b1_ent = entropy(b1_hst)
            b2_ent = entropy(b2_hst)
            b3_ent = entropy(b3_hst)
            ent = entropy(hst)

            features = features.assign(ent=[[b1_ent, b2_ent, b3_ent, ent]])

        # curtose - hst
        if 'crt' in caracteristicas:
            b1_crt = kurtosis(b1_hst)
            b2_crt = kurtosis(b2_hst)
            b3_crt = kurtosis(b3_hst)
            crt = kurtosis(hst)

            features = features.assign(crt=[[b1_crt, b2_crt, b3_crt, crt]])

    # lbp
    if 'lbp' in caracteristicas:
        b1_lbp = local_binary_pattern(b1, 1, np.pi / 2)
        b2_lbp = local_binary_pattern(b2, 1, np.pi / 2)
        b3_lbp = local_binary_pattern(b3, 1, np.pi / 2)
        lbp = local_binary_pattern(reg_g, 1, np.pi / 2)

        b1_lbp_h, _ = histogram(b1_lbp.ravel())
        b2_lbp_h, _ = histogram(b2_lbp.ravel())
        b3_lbp_h, _ = histogram(b3_lbp.ravel())
        lbp_h, _ = histogram(lbp)

        b1_min = b1_lbp_h.min()
        b2_min = b2_lbp_h.min()
        b3_min = b3_lbp_h.min()
        min = lbp_h.min()

        # ToDo: melhorar normalização
        #   -> checar se só serão utilizados o primeiro e o último do histograma
        #       (possivelmente mudar o nbins do histogram resolve)
        b1_lbp_h = (b1_lbp_h - b1_min) / (b1_lbp_h.max() - b1_min)
        b2_lbp_h = (b2_lbp_h - b2_min) / (b2_lbp_h.max() - b2_min)
        b3_lbp_h = (b3_lbp_h - b3_min) / (b3_lbp_h.max() - b3_min)

        b1_lbp_h = (b1_lbp_h - b1_min) / (b1_lbp_h.max() - b1_min)
        b2_lbp_h = (b2_lbp_h - b2_min) / (b2_lbp_h.max() - b2_min)
        b3_lbp_h = (b3_lbp_h - b3_min) / (b3_lbp_h.max() - b3_min)
        lbp_h = (lbp_h - min) / (lbp_h.max() - min)
        features = features.assign(lbp_b1=[list(b1_lbp_h)], lbp_b2=[list(b2_lbp_h)], lbp_b3=[list(b3_lbp_h)], lbp=[list(lbp_h)])

    # hog
    if 'hog' in caracteristicas:
        h = hog(regiao, block_norm='L2-Hys', visualize=False, feature_vector=True, multichannel=True)
        features = features.assign(hog=[list(h)])

    # glcm
    if 'glcm' in caracteristicas:
        b1_glcm = greycomatrix(b1, [10], [np.pi/2], levels=b1.max() + 1)
        b2_glcm = greycomatrix(b2, [10], [np.pi/2], levels=b2.max() + 1)
        b3_glcm = greycomatrix(b3, [10], [np.pi/2], levels=b3.max() + 1)
        glcm = greycomatrix(reg_g, [10], [np.pi/2], levels=reg_g.max() + 1)

        glcm_res = list()
        # contrast
        glcm_res.append(greycoprops(b1_glcm, 'contrast')[0][0])
        glcm_res.append(greycoprops(b2_glcm, 'contrast')[0][0])
        glcm_res.append(greycoprops(b3_glcm, 'contrast')[0][0])
        glcm_res.append(greycoprops(glcm, 'contrast')[0][0])

        # dissimilarity
        glcm_res.append(greycoprops(b1_glcm, 'dissimilarity')[0][0])
        glcm_res.append(greycoprops(b2_glcm, 'dissimilarity')[0][0])
        glcm_res.append(greycoprops(b3_glcm, 'dissimilarity')[0][0])
        glcm_res.append(greycoprops(glcm, 'dissimilarity')[0][0])

        # homogeneity
        glcm_res.append(greycoprops(b1_glcm, 'homogeneity')[0][0])
        glcm_res.append(greycoprops(b2_glcm, 'homogeneity')[0][0])
        glcm_res.append(greycoprops(b3_glcm, 'homogeneity')[0][0])
        glcm_res.append(greycoprops(glcm, 'homogeneity')[0][0])

        # energy
        glcm_res.append(greycoprops(b1_glcm, 'energy')[0][0])
        glcm_res.append(greycoprops(b2_glcm, 'energy')[0][0])
        glcm_res.append(greycoprops(b3_glcm, 'energy')[0][0])
        glcm_res.append(greycoprops(glcm, 'energy')[0][0])

        # correlation
        glcm_res.append(greycoprops(b1_glcm, 'correlation')[0][0])
        glcm_res.append(greycoprops(b2_glcm, 'correlation')[0][0])
        glcm_res.append(greycoprops(b3_glcm, 'correlation')[0][0])
        glcm_res.append(greycoprops(glcm, 'correlation')[0][0])

        # ASM
        glcm_res.append(greycoprops(b1_glcm, 'ASM')[0][0])
        glcm_res.append(greycoprops(b2_glcm, 'ASM')[0][0])
        glcm_res.append(greycoprops(b3_glcm, 'ASM')[0][0])
        glcm_res.append(greycoprops(glcm, 'ASM')[0][0])

        features = features.assign(glcm=[glcm_res])

    return features


def extrair_caracteristicas(raster, mask, caracteristicas):
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

    features = pd.DataFrame(columns=caracteristicas)
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

        ftr_regiao = extrair_carac_regiao(np_reg, caracteristicas)
        ftr_regiao = ftr_regiao.assign(reg=int(reg))

        features = features.append(ftr_regiao, ignore_index=True, sort=False)

    return features


def filtrar(raster, path=None, nome=None):
    """

    Args:
        raster:
        path:
        nome:

    Returns:

    """
    if path:
        if not os.path.exists(path):
            print('Diretório não existe')
            return None

    dest = cria_destino(path, nome, raster.GetDescription())


def treinar_modelo(df_train, tipo_class, class_col='classe', id_col=None, df_eval=None):
    """

    Args:
        df_train (pd.DataFrame): dados de treinamento
        tipo_class (str): tipo de classificador (rna, svm)
        class_col (str):
        id_col (str):
        df_eval (pd.DataFrame: dados para testar

    Returns:

    """
    # ToDo: verificar item por item do dataframe se são object(ou é str ou será lista)
    #   -> se for str, transformar em category
    #   -> se for lista, transformar em ndarray
    #   -> se for float ou int, colocar em lista e converter para ndarray
    df_train_c = df_train.copy()

    if id_col:
        df_train_c.pop(id_col)

    target = df_train_c.pop(class_col)

    data_train_flatted = [np.concatenate(x).ravel().tolist() for x in df_train_c.values]
    nodos = len(data_train_flatted[0])

    ds_train = tf.data.Dataset.from_tensor_slices((data_train_flatted, target.values))
    ds_train = ds_train.shuffle(buffer_size=len(df_train_c))
    aux = int(nodos / 10)
    batch_size = aux if aux > 0 else 1
    ds_train = ds_train.batch(batch_size)
    # ToDo: estudar sobre batch e definir valor

    if tipo_class.lower() == 'rna':
        keras = tf.keras
        modelo = keras.Sequential([
            # keras.layers.DenseFeatures(features_columns),
            keras.layers.Input(nodos, name='input'),
            keras.layers.Dense(30, tf.nn.relu, name='inter_1'),
            keras.layers.Dense(30, tf.nn.relu, name='inter_2'),
            keras.layers.Dense(1, tf.nn.sigmoid, name='final')
        ])  # ToDo: definir ultima camada com número de quantidade de classes

        # ToDo: verificar melhor forma de calcular as métricas
        modelo.compile(
            optimizer=tf.compat.v1.train.GradientDescentOptimizer(0.01),
            loss=keras.losses.mean_squared_error,
            # optimizer='adam',
            # loss='binary_crossentropy',
            metrics=[
                # keras.metrics.mean_squared_error,
                # keras.metrics.mean_absolute_error,
                # keras.metrics.sparse_categorical_accuracy,
                # keras.metrics.sparse_categorical_crossentropy,
                # keras.metrics.sparse_top_k_categorical_accuracy,
                # tf.metrics.false_negatives
                'accuracy'
                # tf.metrics.mean_absolute_error
                # tf.metrics.root_mean_squared_error,
                # tf.metrics.accuracy,
                # tf.metrics.false_negatives
            ]  # ,
            # run_eagerly=True
        )

        modelo.fit(ds_train, epochs=3)
        modelo.predict_classes(ds_train, None)
        predict = modelo.predict(ds_train)
        acc = tf.summary.scalar('accuracy', tf.metrics.accuracy(target, predict))

        print(list(tf.metrics.precision(target, predict)))
        return modelo, ds_train


def classificar(raster, classificador, path=None, nome=None):
    """

    Args:
        raster:
        classificador:
        path:
        nome:

    Returns:

    """
    if path:
        if not os.path.exists(path):
            print('Diretório não existe')
            return None

    dest = cria_destino(path, nome, raster.GetDescription())
