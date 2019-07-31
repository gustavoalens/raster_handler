from osgeo import gdal, ogr
import numpy as np
from skimage import segmentation
from skimage.exposure import histogram
from skimage.color import rgb2grey
from skimage.feature import local_binary_pattern
from skimage import img_as_uint
from scipy.stats import entropy, kurtosis
from scipy.ndimage import variance
import pandas as pd

from typing import Union
import os
from tempfile import gettempdir

"""
"""

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


# ToDo: tornar só desc como obrigatório, pegar demais informações dele
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


# def cria_destino(path, nome, ext, extra=None):
#
#     if not ext:
#         print('Erro: Falta ext')
#         return None
#     if not path:
#         print('Erro: Falta path')
#         return None
#     if not nome:
#         print('Erro: Falta nome')
#         return None
#
#     if extra:
#         nome += f'_{extra}'
#
#     return f'{path}/{nome}.{ext}'


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
        ref_nova (int): tipo de referência pelo padrão EPSG. Exemplo: 4628
        path (str): caminho do diretório de saída do arquivo raster. Default:
        nome (str): nome para o raster. Default: {nome atual}_EPSG:{ref_nova}

    Returns:
        gdal.Dataset: gdal.Dataset atualizado com nova referência espacial

    """

    # checando possíveis erros
    if type(raster) is not gdal.Dataset:
        print('Não é um gdal.Dataset')
        return None
    if type(ref_nova) is not int:  # ToDo: tentar converter para int antes
        print('Projeção precisa ser padrão EPSG')
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


def segmentar(raster, scale, min_size, path=None, nome=None):
    """

    Args:
        raster (gdal.Dataset):
        path (str):
        nome (str):

    Returns:


    """
    if path:
        if not os.path.exists(path):
            print('Diretório não existe')
            return None

    # ToDo: pegar bandar separadas; transformar em arrays, juntar num np.array (imagem) e iniciar segmentação
    # checando o preparando raster
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

    # seg2 = segmentation.quickshift(img, kernel_size=6, max_dist=15, sigma=0.5)
    np_seg = segmentation.felzenszwalb(np_raster, scale=scale, min_size=min_size)
    np_seg = np_seg + 1

    # if not np_seg:
    #     print('Erro na segmentação')
    #     return None

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
    # print(seg)
    # fig = plt.figure(figsize=(8, 8))
    # fig.add_subplot(2, 1, 1)
    # plt.imshow(segmentation.mark_boundaries(img, seg2))
    # fig.add_subplot(2, 1, 2)
    # plt.imshow(segmentation.mark_boundaries(img, seg4))
    # plt.show()


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


def extrair_carac_regiao(regiao):
    """
        Extração das caracterísitcas:
    Args:
        regiao (Union[np.ndarray, np.ma.core.MaskedArray]): região de imagem que será extraída as características

    Returns:
        (list of float): lista de características extraídas
    """
    features = list()
    b1 = regiao[..., 0]
    b2 = regiao[..., 1]
    b3 = regiao[..., 2]
    reg_g = img_as_uint(rgb2grey(regiao))

    # total de pixels
    b1_npix = calc_total_pixels(b1)
    b2_npix = calc_total_pixels(b2)
    b3_npix = calc_total_pixels(b3)
    npix = calc_total_pixels(reg_g)

    # b1_npix = len(b1)
    # b2_npix = len(b2)
    # b3_npix = len(b3)
    # npix = len(reg_g)

    # média  # ToDO: fazer essa só dos histogramas?
    b1_mean = np.mean(b1)
    b2_mean = np.mean(b2)
    b3_mean = np.mean(b3)
    mean = np.mean(reg_g)
    features.append(b1_mean)
    features.append(b2_mean)
    features.append(b3_mean)
    features.append(mean)

    # desvio padrão
    b1_std = np.std(b1)
    b2_std = np.std(b2)
    b3_std = np.std(b3)
    std = np.std(reg_g)
    features.append(b1_std)
    features.append(b2_std)
    features.append(b3_std)
    features.append(std)

    # assimetria
    features.append(calc_assimetria(b1, b1_mean, b1_std, b1_npix))
    features.append(calc_assimetria(b2, b2_mean, b2_std, b2_npix))
    features.append(calc_assimetria(b3, b3_mean, b3_std, b3_npix))
    features.append(calc_assimetria(reg_g, mean, std, npix))

    # variancia
    features.append(variance(b1))
    features.append(variance(b2))
    features.append(variance(b3))
    features.append(variance(reg_g))

    # b1_levels = b1.max() + 1
    # b2_levels = b2.max() + 1
    # b3_levels = b3.max() + 1
    # levels = reg_g.max() + 1

    # histograma
    b1_hst = histogram(b1)[0]  # alterar nbins de acordo com dtype
    b2_hst = histogram(b2)[0]  # alterar nbins de acordo com dtype
    b3_hst = histogram(b3)[0]  # alterar nbins de acordo com dtype
    hst = histogram(reg_g)[0]

    # energia - hst  # ToDO: Checar esse calculo (por ora não achei se isso está correto)
    features.append(calc_energia(b1_hst, b1_npix))
    features.append(calc_energia(b2_hst, b2_npix))
    features.append(calc_energia(b3_hst, b3_npix))  # ToDO: npix deve ser referente ao histograma, idem abaixo
    features.append(calc_energia(hst, npix))

    # entropia  - hst
    features.append(entropy(b1_hst))
    features.append(entropy(b2_hst))
    features.append(entropy(b3_hst))
    features.append(entropy(hst))

    # curtose - hst
    features.append(kurtosis(b1_hst))
    features.append(kurtosis(b2_hst))
    features.append(kurtosis(b3_hst))
    features.append(kurtosis(hst))

    # lbp_hst
    b1_lbp = local_binary_pattern(b1, 10, np.pi / 2)
    b2_lbp = local_binary_pattern(b2, 10, np.pi / 2)
    b3_lbp = local_binary_pattern(b3, 10, np.pi / 2)
    lbp = local_binary_pattern(reg_g, 10, np.pi / 2)

    b1_lbp_hst = histogram(b1_lbp)[0]
    b2_lbp_hst = histogram(b2_lbp)[0]
    b3_lbp_hst = histogram(b3_lbp)[0]
    lbp_hst = histogram(lbp)[0]

    b1_npix = calc_total_pixels(b1_lbp)
    b2_npix = calc_total_pixels(b2_lbp)
    b3_npix = calc_total_pixels(b3_lbp)
    npix = calc_total_pixels(lbp)

    # # energia - lbp_hst # ToDO: idem energia - hst
    features.append(calc_energia(b1_lbp_hst, b1_npix))
    features.append(calc_energia(b2_lbp_hst, b2_npix))
    features.append(calc_energia(b3_lbp_hst, b3_npix))
    features.append(calc_energia(lbp_hst, npix))

    # entropia  - lbp_hst
    features.append(entropy(b1_lbp_hst))
    features.append(entropy(b2_lbp_hst))
    features.append(entropy(b3_lbp_hst))
    features.append(entropy(lbp_hst))

    # curtose - lbp_hst
    features.append(kurtosis(b1_lbp_hst))
    features.append(kurtosis(b2_lbp_hst))
    features.append(kurtosis(b3_lbp_hst))
    features.append(kurtosis(lbp_hst))

    # # glcm
    # b1_glcm = greycomatrix(b1, [1], [1], levels=b1_levels)
    # b2_glcm = greycomatrix(b2, [1], [1], levels=b2_levels)
    # b3_glcm = greycomatrix(b3, [1], [1], levels=b3_levels)
    # glcm = greycomatrix(reg_g, [1], [1], levels=levels)
    #
    # # contrast
    # features.append(greycoprops(b1_glcm, 'contrast')[0][0])
    # features.append(greycoprops(b2_glcm, 'contrast')[0][0])
    # features.append(greycoprops(b3_glcm, 'contrast')[0][0])
    # features.append(greycoprops(glcm, 'contrast')[0][0])
    #
    # # dissimilarity
    # features.append(greycoprops(b1_glcm, 'dissimilarity')[0][0])
    # features.append(greycoprops(b2_glcm, 'dissimilarity')[0][0])
    # features.append(greycoprops(b3_glcm, 'dissimilarity')[0][0])
    # features.append(greycoprops(glcm, 'dissimilarity')[0][0])
    #
    # # homogeneity
    # features.append(greycoprops(b1_glcm, 'homogeneity')[0][0])
    # features.append(greycoprops(b2_glcm, 'homogeneity')[0][0])
    # features.append(greycoprops(b3_glcm, 'homogeneity')[0][0])
    # features.append(greycoprops(glcm, 'homogeneity')[0][0])
    #
    # # energy
    # features.append(greycoprops(b1_glcm, 'energy')[0][0])
    # features.append(greycoprops(b2_glcm, 'energy')[0][0])
    # features.append(greycoprops(b3_glcm, 'energy')[0][0])
    # features.append(greycoprops(glcm, 'energy')[0][0])
    #
    # # correlation
    # features.append(greycoprops(b1_glcm, 'correlation')[0][0])
    # features.append(greycoprops(b2_glcm, 'correlation')[0][0])
    # features.append(greycoprops(b3_glcm, 'correlation')[0][0])
    # features.append(greycoprops(glcm, 'correlation')[0][0])
    #
    # # ASM
    # features.append(greycoprops(b1_glcm, 'ASM')[0][0])
    # features.append(greycoprops(b2_glcm, 'ASM')[0][0])
    # features.append(greycoprops(b3_glcm, 'ASM')[0][0])
    # features.append(greycoprops(glcm, 'ASM')[0][0])

    return features


def extrair_caracteristicas(raster, mask):
    """
        Separa a imagem nas regiões definidas da máscara e executa a função de extração para cada região
    Args:
        raster (Union[gdal.Dataset, np.ndarray]): raster a sofrer a extração de características
        mask (Union[gdal.Dataset, np.ndarray]): raster máscara que define as divisões das regiões

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

    elif raster_t is np.ndarray:
        np_raster = raster

    else:
        print('Erro no tipo')
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

    features_list = list()
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

        ftr_regiao = [reg]

        ftr_regiao += extrair_carac_regiao(np_reg)
        features_list.append(ftr_regiao)

    columns = ['fid',
               'media_b1', 'media_b2', 'media_b3', 'media',
               'dsv_p_b1', 'dsv_p_b2', 'dsv_p_b3', 'dsv_p',
               'ass_b1', 'ass_b2', 'ass_b3', 'ass',
               'varc_b1', 'varc_b2', 'varc_b3', 'varc',
               'energ_b1', 'energ_b2', 'energ_b3', 'energ',
               'etrp_b1', 'etrp_b2', 'etrp_b3', 'etrp',
               'crts_b1', 'crts_b2', 'crts_b3', 'crts',
               'lbp_energ_b1', 'lbp_energ_b2', 'lbp_energ_b3', 'lbp_energ',
               'lbp_etrp_b1', 'lbp_etrp_b2', 'lbp_etrp_b3', 'lbp_etrp',
               'lbp_crts_b1', 'lbp_crts_b2', 'lbp_crts_b3', 'lbp_crts',
               ]

    return pd.DataFrame(features_list, columns=columns).set_index('fid')


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


def treinar_class(raster, tclass, mask=None, features=None):
    pass


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
