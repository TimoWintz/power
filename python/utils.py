import numpy as onp
import requests
from stravalib import Client
import googlemaps
import json
import os
import ruptures as rpt
from copy import deepcopy
from scipy.optimize import bisect, least_squares, minimize, Bounds, NonlinearConstraint
from pyowm import OWM

from jax import numpy as np
from jax import grad, jit
# again, this only works on startup!
from jax.config import config
config.update("jax_enable_x64", True)

class IGNAlti:
    max_slice = 20
    def __init__(self, secret):
        self.login = secret["login"]
        self.password = secret["password"]
        self.key = secret["key"]

    def format_line(self, pts):
        lon_str = "|".join([str(l["lng"]) for l in pts])
        lat_str = "|".join([str(l["lat"]) for l in pts])
        return lon_str, lat_str

    def correct_elevation(self, pts):
        for i in range(len(pts) // self.max_slice + 1):
            s = slice(i * self.max_slice, (i+1)* self.max_slice)
            lon_str, lat_str = self.format_line(pts[s])
            url = "https://{0}:{1}@wxs.ign.fr/{2}/alti/rest/elevationLine.json?lon={3}&lat={4}&zonly=true".format(self.login, self.password, self.key, lon_str, lat_str)
            r = requests.get(url)
            body = r.content.decode()
            x = json.loads(body)
            if 'elevations' in x:
                for j in range(len(x["elevations"])):
                    idx = i*self.max_slice + j
                    pts[idx]["alt"] = x["elevations"][j]["z"]

class SnapToRoad:
    max_slice = 50
    def __init__(self, key):
        self.key = key

    def snap_to_roads(self, d):
        client = googlemaps.Client(self.key)
        for i in range(len(d) // self.max_slice + 1):
            s = slice(i * self.max_slice, (i+1)* self.max_slice)
            x = client.snap_to_roads(d[s])
            for a in x:
                idx = i*self.max_slice + a["originalIndex"]
                d[idx]["lat"] = a["location"]["latitude"]
                d[idx]["lng"] = a["location"]["longitude"]

class Segment:
    def __init__(self):
        self.points = []
        self.has_gps = False
        self.has_power = False

    def load_gpx(gpx_file):
        pass

    def from_strava_segment(access_token, segment_id, resolution="medium"):
        client = Client(access_token=access_token)
        segment_stream = client.get_segment_streams(segment_id, types=["latlng", "distance", "altitude"],
            resolution=resolution)
        res = Segment()
        n_pts = len(segment_stream["latlng"].data)
        res.points = []
        for i in range(n_pts):
            res.points.append({
                "lat" : segment_stream["latlng"].data[i][0],
                "lng" : segment_stream["latlng"].data[i][1],
                "alt" : segment_stream["altitude"].data[i],
                "distance" : segment_stream["distance"].data[i]
            })
        res.has_gps = True
        res.has_power = False
        return res

    def from_strava_effort(access_token, segment_effort_id, resolution="medium"):
        client = Client(access_token=access_token)
        segment_stream = client.get_effort_streams(segment_effort_id, types=["latlng", "distance", "altitude", "watts", "time"],
            resolution=resolution)
        return Segment._from_stream(segment_stream)

    def _from_stream(stream):
        res = Segment()
        res.has_power = False
        n_pts = len(stream["latlng"].data)
        res.points = []
        res.test = stream
        for i in range(n_pts):
            res.points.append({
                "lat" : stream["latlng"].data[i][0],
                "lng" : stream["latlng"].data[i][1],
                "alt" : stream["altitude"].data[i],
                "distance" : stream["distance"].data[i],
                "time" : stream["time"].data[i]
            })
            if "watts" in stream.keys():
                res.points[-1]["power"] = stream["watts"].data[i]
                res.has_power = True
        res.has_gps = True
        return res

    def from_activity(access_token, activity_id, resolution="medium"):
        client = Client(access_token=access_token)
        activity_stream = client.get_activity_streams(activity_id, types=["latlng", "distance", "altitude", "watts", "time"],
            resolution=resolution)
        return Segment._from_stream(activity_stream)

    def snap_to_roads(self, gmaps_api_key):
        s = SnapToRoad(gmaps_api_key)
        s.snap_to_roads(self.points)

    def correct_elevation(self, secret, service="ign"):
        ign = IGNAlti(secret)
        ign.correct_elevation(self.points)

min_speed = 0.1
def _total_time(lengths, speed):
    return np.sum(lengths / speed)

total_time_fn = jit(_total_time)
g_total_time_fn = jit(grad(_total_time, 1))

def _nominal_power(lengths, grades, power_model_params, speed):
    power = []
    for i in range(len(speed)):
        power.append(PowerModel.power_from_speed(power_model_params, grades[i], speed[i]))
    power = np.array(power)
    t = lengths / speed
    return (1 / np.sum(t) * np.sum( t * power**4 ) )**(1/4)

nominal_power_fn = jit(_nominal_power)
g_nominal_power_fn = jit(grad(_nominal_power, 3))


def optimal_power(lengths, grades, power_model_params, target_nominal_power, eps=0.01):
    # t = np.array([p["duration"] for p in self.intervals])
    lam = 4.0
    nominal_power = 0.0
    f = lambda v : total_time_fn(lengths, v)
    gf = lambda v : g_total_time_fn(lengths, v)
    h = lambda v : nominal_power_fn(lengths, grades, power_model_params, v)
    gh = lambda v : g_nominal_power_fn(lengths, grades, power_model_params, v)
    v0 = PowerModel.vmin * np.ones(len(lengths))
    lb = v0
    ub = PowerModel.vmax * np.ones(len(lengths))
    bounds = Bounds(lb, ub)
    constraint = NonlinearConstraint(h, jac=gh, lb=np.array([0]), ub=np.array([target_nominal_power]))
    # constraint = NonlinearConstraint(h, lb=np.array([0]), ub=np.array([target_nominal_power]))
    r = minimize(fun=f, jac=gf, x0=v0, constraints=constraint, bounds=bounds).x
    # r = minimize(fun=f, x0=v0).x
    return r

class Profile:
    def __init__(self, segment):
        self.segment = segment
        self.points = []
        self.intervals = []
        self.has_power = False

    def _pt_idx(self, idx):
        return deepcopy(self.segment.points[idx])


    def _interval(self, idx):
        length = self.points[idx+1]["distance"] - self.points[idx]["distance"]
        elevation = self.points[idx+1]["alt"] - self.points[idx]["alt"]

        distance = length if idx == 0 else self._interval(idx - 1)["distance"] + length
        cum_elevation = elevation if idx == 0 else self._interval(idx - 1)["cum_elevation"] + elevation
        res = {
            "length" : length,
            "elevation" : elevation,
            "grade" : elevation / length * 100,
            "distance" : distance,
            "cum_elevation" : cum_elevation
        }
        if self.segment.has_power:
            p = 0
            T = self.segment.points[self.indices[idx+1]]["time"] - self.segment.points[self.indices[idx]]["time"]
            dist = self.segment.points[self.indices[idx+1]]["distance"] - self.segment.points[self.indices[idx]]["distance"]
            for i in range(self.indices[idx], self.indices[idx+1]):
                dt = self.segment.points[i+1]["time"] - self.segment.points[i]["time"]
                p += self.segment.points[i]["power"] * dt
            p /= T
            res["power"] = p
            res["speed"] = dist / T
            res["duration"] = T
            self.has_power = True
            return res
        else:
            return res

    def optimal_power(self, power_model_params, target_nominal_power):
        assert(len(power_model_params) == 4)
        print("params = ", power_model_params)
        lengths = np.array([x["length"] for x in self.intervals])
        grades = np.array([x["grade"] for x in self.intervals])
        power = optimal_power(lengths, grades, power_model_params, target_nominal_power)
        return self.intervals_from_power(power, power_model_params)

    def cluster_grades(self, pen=10):
        p = self.segment.points
        diff_alt = [p[i+1]["alt"] - p[i]["alt"] for i in range(len(self.segment.points) - 1)]
        diff_dist = [p[i+1]["distance"] - p[i]["distance"] for i in range(len(self.segment.points) - 1)]
        grade = np.array(diff_alt) / np.array(diff_dist)
        algo= rpt.Pelt(model="rbf").fit(grade)
        self.indices = algo.predict(pen=pen)
        self.indices.insert(0,0)
        self.points= [self._pt_idx(i) for i in self.indices]
        self.intervals = [self._interval(i) for i in range(len(self.points) - 1)]

    def cluster_power(self, pen=10):
        power = np.array([p["power"] for p in self.segment.points])
        algo= rpt.Pelt(model="rbf").fit(power)
        self.indices = algo.predict(pen=pen)
        self.indices.insert(0,0)
        self.points= [self._pt_idx(i) for i in self.indices]
        self.intervals = [self._interval(i) for i in range(len(self.points) - 1)]



    def intervals_from_power(self, speed, power_model):
        result = deepcopy(self.intervals)
        for i in range(len(self.intervals)):
            result[i]["duration"] = self.intervals[i]["length"] / speed[i]
            result[i]["speed"] = speed[i]
            result[i]["speed_kmh"] = speed[i]*3.6
            result[i]["power"] = float(PowerModel.power_from_speed(power_model, self.intervals[i]["grade"], speed[i]))
            if i > 0:
                prev_t = result[i-1]["time"]
            else:
                prev_t = 0
            result[i]["time"] = prev_t + result[i]["duration"]
            result[i]["average_kmh"] = result[i]["distance"] / result[i]["time"] * 3.6
        return result

class PowerModel:
    vmax = 100
    vmin = 0.1
    gravity = 9.81

    def power_from_speed(params, grade, v, wind_speed=0.0):
        mass, drivetrain_efficiency, Cda, Cr = params
        return 1/drivetrain_efficiency * v * (Cda * (v - wind_speed) * (v - wind_speed) + Cr + mass * PowerModel.gravity * np.sin(np.arctan(grade / 100)))

    def speed_from_power(params, grade, power, wind_speed=0.0):
        return bisect(lambda x: power - PowerModel.power_from_speed(params, grade, x), 0, PowerModel.vmax)

    def estimate_parameters(mass, profile, drivetrain_efficiency=None, Cr=None):
        if not profile.has_power:
            raise Exception("Profile must have power data")

        speed = np.array([p["speed"] for p in profile.intervals])
        power = np.array([p["power"] for p in profile.intervals])
        grade = np.array([p["grade"] for p in profile.intervals])

        lb = np.array([0.8, 0.1, 0.001])
        ub = np.array([1.0, 0.5, 0.01])

        eps = 1e-5

        if drivetrain_efficiency is not None:
            lb[0] = drivetrain_efficiency
            ub[0] = drivetrain_efficiency+eps

        if Cr is not None:
            lb[2] = drivetrain_efficiency
            ub[2] = drivetrain_efficiency+eps

        bounds = Bounds(lb, ub)
        l = np.array([i["duration"] for i in profile.intervals])

        map = lambda x : l * (PowerModel.power_from_speed([mass, x[0], x[1], x[2]], grade, speed) - power)
        res = least_squares(map, lb, bounds=(lb, ub)).x
        return np.array([mass, res[0], res[1], res[2]])

if __name__ == "__main__":
    print("testing IGN API...")
    secrets = {}
    d = os.path.dirname(__file__)
    with open(os.path.join(d, "ign_secrets.txt")) as f:
        secrets["key"] = f.readline().strip()
        secrets["login"] = f.readline().strip()
        secrets["password"] = f.readline().strip()
    print(secrets)
    ign = IGNAlti(secrets)
    lat = 45.188529
    lon = 5.724524
    pts = [{"lat" : lat, "lng" : lon}]
    ign.correct_elevation(pts)
    print("Grenoble altitude = ", pts[0]["alt"])
    print("OK")

    print("testing strava segment stream...")
    access_token = "629fac9d17abaa701b50513709c1bf48b14a68b5"
    segment_id = 8061351
    segment = Segment.from_strava_segment(access_token, segment_id)
    print("OK")

    print("testing strava segment effort...")
    segment_effort_id = 2713736074022118840
    segment_2 = Segment.from_strava_effort(access_token, segment_effort_id)
    print("OK")

    print("testing snap to roads...")
    with open(os.path.join(d, "gmaps_secrets.txt")) as f:
        secret = f.readline().strip()
    segment.snap_to_roads(secret)
    print("OK")

    print("testing fix altitude...")
    segment_2.correct_elevation(secrets)
    print("OK")

    print("Approximating profile...")
    profile = Profile(segment_2)
    profile.cluster_grades(3)
    print("OK")

    print("Testing power model")
    mass = 73
    drivetrain_efficiency = 0.98
    Cda = 0.25
    Cr = 0.003
    P = 290
    power_model = [mass, drivetrain_efficiency, Cda, Cr]
    speed = 5.0
    grade = 10
    power = PowerModel.power_from_speed(power_model, speed, grade)
    r = PowerModel.speed_from_power(power_model, power, grade)
    print("speed = ", speed, " / ", r)
    print("OK")
    print("Testing est cda")
    # m = PowerModel.estimate_parameters(mass, profile, drivetrain_efficiency=drivetrain_efficiency, Cr=Cr)
    # print("estimated cda = ", m.Cda)
    print("OK")
    print("Testing optimal power")
    target_nominal_power = 266
    r = profile.optimal_power(power_model, target_nominal_power)
    print("OK")

    print("Testing openweathermap")
    with open(os.path.join(d, "owm_secrets.txt")) as f:
        owm_key = f.readline().strip()
    ow = OWM(owm_key)
    print("OK")

