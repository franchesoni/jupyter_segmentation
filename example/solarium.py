from pathlib import Path
import glob
import os
import zipfile
from os.path import join, isfile

import numpy as np
import iio
import skimage.transform


def load_series(zip_path, outdir):
    """
    Load the spectral filename series selected by the user.
    Parameters
    ----------
    : Namespace
        Contains the parameter of the command line.
    Return
    ------
    series: dict
        Contains the spectral filenames.
    gt_series: list
        Contains the ground truth masks, if any.
    rgb_series: dict
        Contains the spectral filenames.
    """

    # décompactage de l'archive. Lors de ce décompactage, les arborescences
    # sont conservées dans le répertoire de sortie.
    f = zipfile.ZipFile(zip_path)
    # print("FICHIER :", zip_path)
    f.extractall(path=outdir)
    # print("RÉPERTOIRE:", outdir)
    # print("FICHIERS:", f.namelist())
    files = [
        join(outdir, fic) for fic in f.namelist() if isfile(join(outdir, fic))
    ]
    myfiles = sorted(files)
    f.close()
    # print("[INFO Python] Fichiers myfiles", myfiles)
    # print("[INFO Python] Fichier myfiles[0]", os.path.abspath(myfiles[0]))

    # convert the Namespace object into a dictionary

    bandes = [
        "B01",
        "B02",
        "B03",
        "B04",
        "B05",
        "B06",
        "B07",
        "B08",
        "B8A",
        "B09",
        "B10",
        "B11",
        "B12",
    ]

    series = {}
    # search for spectral files
    for b in bandes:
        series[b] = []
        for x in os.walk(outdir):
            for y in sorted(glob.glob(join(x[0], "*" + b + ".tif"))):
                series[b] += [y]

    # search for the ground truth masks
    gt_series = None
    for x in os.walk(outdir):
        if "manual_segmentation_masks" in x[0]:
            gt_series = sorted(glob.glob(join(x[0], "*")))

    # search for the RGB series
    rgb_series = {}
    for b in ["B02", "B03", "B04"]:
        rgb_series[b] = []
        for x in os.walk(outdir):
            for y in sorted(glob.glob(join(x[0], "*" + b + ".tif"))):
                rgb_series[b] += [y]

    # vérification de la présence des bandes
    for s in series:
        liste = series[s]
        if len(liste) == 0:
            print("[FATAL] Missing band %s" % (s))
            exit()
    # print("[INFO] Spectral file loading done.")

    for s in rgb_series:
        liste = rgb_series[s]
        if len(liste) == 0:
            print("[FATAL] Missing band %s" % (s))
            exit()
    # print("[INFO] RGB file loading done.")

    return series, gt_series, rgb_series


def read_panel_image(site_zip_Path, date_index, normalize=False):
    series, gt_series, rgb_series = load_series(str(site_zip_Path), outdir=f"figs/{site_zip_Path.stem}")
    date_index = np.clip(0, len(series["B02"]) - 1, date_index)
    rgb_image = np.stack(
        [
            iio.read(rgb_series["B04"][date_index])[..., 0],
            iio.read(rgb_series["B03"][date_index])[..., 0],
            iio.read(rgb_series["B02"][date_index])[..., 0],
        ],
        axis=2,
    )

    factors = {
        "B01": 6,
        "B02": 1,
        "B03": 1,
        "B04": 1,
        "B05": 2,
        "B06": 2,
        "B07": 2,
        "B08": 1,
        "B8A": 2,
        "B09": 6,
        "B10": 6,
        "B11": 2,
        "B12": 2,
    }
    multispectral_image = np.stack(
        [
            skimage.transform.rescale(
                iio.read(series[b][date_index])[..., 0],
                scale=factors[b],
                order=5,
            )
            for b in series
        ],
        axis=2,
    )
    if normalize:
        return (norm_fn(multispectral_image)*255).astype(np.uint8), (norm_fn(rgb_image)*255).astype(np.uint8)
    return multispectral_image, rgb_image


def norm_fn(x):
    return (x - x.min()) / (x.max() - x.min())


def norm_fn_per_channel(x):
    return (x - x.min((0, 1))[None, None]) / (x.max((0, 1)) - x.min((0, 1)))[
        None, None
    ]

def demo():
    date_index = 5
    site_zip_Path = Path("/home/franchesoni/mine/creations/phd/material/data/kayrros/solarium/concho_valley.zip")
    site_name = site_zip_Path.stem
    (Path('figs') / site_name).mkdir(parents=True, exist_ok=True)

    ms_img, rgb_img = read_panel_image(site_zip_Path, date_index, normalize=True)

    iio.write(f"figs/{site_name}.png", rgb_img)

if __name__ == '__main__':
    pass
    # demo()