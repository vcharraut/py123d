# Captures top-down semantic + depth tiles of a CARLA town (static layers only). Python 3.12 + carla wheel, no numpy.
# Needs a running server (one map switch per server is reliable; Town01_Opt crashes 0.9.16, use Town01):
#   PYTHONPATH=<whl_dir> python3.12 scripts/opendrive_parity/capture_carla.py Town03 <capture_dir> [Town03_Opt]
import json, math, sys, queue
from pathlib import Path
import carla

town, out = sys.argv[1], Path(sys.argv[2]) / sys.argv[1]
level = sys.argv[3] if len(sys.argv) > 3 else f"{town}_Opt"
out.mkdir(parents=True, exist_ok=True)
H, FOV, W, STEP = 300.0, 20.0, 1024, 80.0  # altitude above ground, fov, pixels, tile step
client = carla.Client("localhost", 2000); client.set_timeout(120)
world = client.get_world()
if not world.get_map().name.endswith(level):
    world = client.load_world(level, reset_settings=True)
for layer in () if not level.endswith("_Opt") else ("Buildings", "Foliage", "Props", "StreetLights", "Walls", "ParkedVehicles", "Particles"):
    world.unload_map_layer(getattr(carla.MapLayer, layer))
hide = [o.id for lab in (carla.CityObjectLabel.Vegetation, carla.CityObjectLabel.Poles, carla.CityObjectLabel.TrafficLight,
        carla.CityObjectLabel.TrafficSigns, carla.CityObjectLabel.Buildings, carla.CityObjectLabel.Fences, carla.CityObjectLabel.Static,
        carla.CityObjectLabel.Dynamic, carla.CityObjectLabel.Walls, carla.CityObjectLabel.Other) for o in world.get_environment_objects(lab)]
world.enable_environment_objects(set(hide), False)
settings = world.get_settings(); settings.synchronous_mode = True; settings.fixed_delta_seconds = 0.05; world.apply_settings(settings)

wps = world.get_map().generate_waypoints(20.0)
xs, ys = [w.transform.location.x for w in wps], [w.transform.location.y for w in wps]
x0, x1, y0, y1 = min(xs) - 60, max(xs) + 60, min(ys) - 60, max(ys) + 60
zmax = max(w.transform.location.z for w in wps)
lib = world.get_blueprint_library()
cams = {}
for kind in ("semantic_segmentation", "depth"):
    bp = lib.find(f"sensor.camera.{kind}")
    bp.set_attribute("image_size_x", str(W)); bp.set_attribute("image_size_y", str(W)); bp.set_attribute("fov", str(FOV))
    cams[kind] = world.spawn_actor(bp, carla.Transform(carla.Location(0, 0, zmax + H), carla.Rotation(pitch=-90)))
qs = {k: queue.Queue() for k in cams}
for k, c in cams.items():
    c.listen(qs[k].put)

tiles = []
nx, ny = math.ceil((x1 - x0) / STEP), math.ceil((y1 - y0) / STEP)
for i in range(nx):
    for j in range(ny):
        tf = carla.Transform(carla.Location(x0 + (i + 0.5) * STEP, y0 + (j + 0.5) * STEP, zmax + H), carla.Rotation(pitch=-90))
        for c in cams.values():
            c.set_transform(tf)
        for _ in range(3):  # let streaming settle
            frame = world.tick()
        imgs = {}
        for k, q in qs.items():
            img = q.get(timeout=60)
            while img.frame < frame:
                img = q.get(timeout=60)
            imgs[k] = img
        name = f"t{i:03d}_{j:03d}"
        for k, img in imgs.items():
            (out / f"{name}_{k}.bgra").write_bytes(bytes(img.raw_data))
        tiles.append({"name": name, "x": tf.location.x, "y": tf.location.y, "z": tf.location.z})
    print(f"{town} column {i + 1}/{nx}", flush=True)
(out / "meta.json").write_text(json.dumps({"W": W, "FOV": FOV, "tiles": tiles,
    "labels": {n: int(getattr(carla.CityObjectLabel, n)) for n in ("Roads", "RoadLines", "Sidewalks", "Terrain", "Ground", "Water", "Bridge", "GuardRail", "RailTrack")}}))
for c in cams.values():
    c.destroy()
settings.synchronous_mode = False; world.apply_settings(settings)
