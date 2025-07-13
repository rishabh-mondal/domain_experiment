import geopandas as gpd
import math, os, requests
from shapely.geometry import box
from PIL import Image
from tqdm import tqdm
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

# === 1. Load and get extent ===
geojson_path="/home/summer_interns/ruddhi_intern/arcgis_small_historical/delhi_airshed_small.geojson"
download_tiles_dir = "/home/summer_interns/ruddhi_intern/arcgis_small_historical/tiles_downloaded"


gdf = gpd.read_file(geojson_path)
minx, miny, maxx, maxy = gdf.total_bounds

# === 2. Setup ===
zoom = 17
time_ids = {
    # 2016: 3515,
    # 2017: 577,
    # 2018: 13161,
    # 2019: 6036,
    2020: 23001,
    2021: 1049,
    # 2022: 42663,
    # 2023: 11475,
    # 2024: 41468,
    # 2025: 36557,
}

def deg2num(lat, lon, zoom):
    n = 2**zoom
    xt = (lon + 180) / 360 * n
    yt = (1 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * n
    return int(xt), int(yt)

def tile_bounds(x, y, z):
    n = 2**z
    lon1 = x / n * 360.0 - 180.0
    lat1 = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    lon2 = (x + 1) / n * 360.0 - 180.0
    lat2 = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))
    return box(lon1, lat2, lon2, lat1)

# === 3. Compute tile range ===
x_min, y_max = deg2num(miny, minx, zoom)
x_max, y_min = deg2num(maxy, maxx, zoom)

roi_geom = gdf.geometry.unary_union
tile_w, tile_h = 256, 256

# === 4. Loop through each timeId ===
for year, timeId in time_ids.items():
    print(f" Processing year {year} (timeId={timeId})...")

    url_tpl = (
        "https://wayback.maptiles.arcgis.com/arcgis/rest/services/"
        "world_imagery/wmts/1.0.0/default028mm/mapserver/tile/"
        f"{timeId}/{{z}}/{{y}}/{{x}}"
    )
    
    year_dir = f"{download_tiles_dir}/{year}"
    os.makedirs(year_dir, exist_ok=True)

    downloaded_tiles = []
    # === Create session with retries ===
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=1,  # Waits 1s, 2s, 4s, etc.
        status_forcelist=[500, 502, 503, 504],
        raise_on_status=False
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    # === Replace download section ===
    for x in tqdm(range(x_min, x_max + 1), desc=f"X Tiles ({year})"):
        for y in range(y_min, y_max + 1):
            tile_poly = tile_bounds(x, y, zoom)
            if roi_geom.intersects(tile_poly):
                url = url_tpl.format(z=zoom, x=x, y=y)
                path = f"{year_dir}/{zoom}_{x}_{y}.png"
                if not os.path.exists(path):
                    try:
                        r = session.get(url, timeout=10)
                        if r.status_code == 200:
                            with open(path, "wb") as f:
                                f.write(r.content)
                    except Exception as e:
                        print(f"⚠️ Failed {x},{y}: {e}")

    # === 5. Stitch tiles for this year ===
    tile_w, tile_h = 256, 256
    year_dir = f"tiles_downloaded/{year}"

    # === Extract tile coordinates from filenames ===
    downloaded_tiles = []
    for fname in os.listdir(year_dir):
        if fname.endswith(".png") and fname.startswith(f"{zoom}_"):
            _, x, y = fname.replace(".png", "").split("_")
            downloaded_tiles.append((int(x), int(y)))

    # === Stitch tiles ===
    xs, ys = zip(*downloaded_tiles)
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    width = (max_x - min_x + 1) * tile_w
    height = (max_y - min_y + 1) * tile_h

    stitched = Image.new("RGB", (width, height))
    for x, y in downloaded_tiles:
        path = f"{year_dir}/{zoom}_{x}_{y}.png"
        img = Image.open(path)
        px = (x - min_x) * tile_w
        py = (y - min_y) * tile_h
        stitched.paste(img, (px, py))

    out_path = f"/home/summer_interns/ruddhi_intern/arcgis_small_historical/test/delhi_airshed_small_{year}.png"
    stitched.save(out_path)
    print(f"Image for {year} saved as '{out_path}'")
