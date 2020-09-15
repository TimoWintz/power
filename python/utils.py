import numpy as np
import requests
from stravalib import Client
import googlemaps
import json
import os
import ruptures as rpt
from copy import deepcopy
from scipy.optimize import bisect, least_squares, minimize, Bounds, NonlinearConstraint
from pyowm import OWM

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

# g_nominal_power_fn = jit(grad(nominal_power_fn, 0))

class Profile:
    def __init__(self, segment):
        self.segment = segment
        self.points = []
        self.intervals = []
        self.has_power = False

    def _pt_idx(self, idx):
        return deepcopy(self.segment.points[idx])

    def nominal_power(self, p, power_model):
        t = self.estimate_durations(p, power_model)
        return (1 / np.sum(t) * np.sum( t * p**4 ) )**(1/4)

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

    def estimate_durations(self, power, power_model):
        result = np.zeros(len(self.intervals))
        for i in range(len(self.intervals)):
            spd = power_model.speed_from_power(power[i], self.intervals[i]["grade"])
            result[i] = self.intervals[i]["length"] / spd
        return result

    def optimal_power(self, power_model, target_nominal_power, eps=0.01):
        # t = np.array([p["duration"] for p in self.intervals])
        lam = 4.0
        nominal_power = 0.0
        f = lambda p : np.sum(self.estimate_durations(p, power_model))
        p0 = target_nominal_power * np.ones(len(self.intervals))
        constraint = NonlinearConstraint(lambda p: np.array([self.nominal_power(p, power_model)]), lb=np.array([0]), ub=np.array([target_nominal_power]))
        r = minimize(fun=f, x0=p0, constraints=constraint).x
        return self.intervals_from_power(r, power_model)

    def intervals_from_power(self, power, power_model):
        result = deepcopy(self.intervals)
        for i in range(len(self.intervals)):
            spd = power_model.speed_from_power(power[i], self.intervals[i]["grade"])
            result[i]["duration"] = self.intervals[i]["length"] / spd
            result[i]["speed"] = spd
            result[i]["speed_kmh"] = spd*3.6
            result[i]["power"] = power[i]
            if i > 0:
                prev_t = result[i-1]["time"]
            else:
                prev_t = 0
            result[i]["time"] = prev_t + result[i]["duration"]
            result[i]["average_kmh"] = result[i]["distance"] / result[i]["time"] * 3.6
        return result

class PowerModel:
    vmax = 100
    gravity = 9.81

    def __init__(self, mass, drivetrain_efficiency, Cda, Cr):
        self.mass = mass
        self.drivetrain_efficiency = drivetrain_efficiency
        self.Cda = Cda
        self.Cr = Cr

    def power_from_speed(self, v, grade, wind_speed=0.0): # wind is + for tailwind, - for headwind
        return PowerModel._power_from_speed(self.mass, self.drivetrain_efficiency, self.Cda, self.Cr, v, grade, wind_speed)

    def speed_from_power(self, power, grade, wind_speed=0.0):
        return PowerModel._speed_from_power(self.mass, self.drivetrain_efficiency, self.Cda, self.Cr, power, grade, wind_speed)

    def _power_from_speed(mass, drivetrain_efficiency, Cda, Cr, v, grade, wind_speed=0.0):
        return 1/drivetrain_efficiency * v * (Cda * (v - wind_speed) * (v - wind_speed) + Cr + mass * PowerModel.gravity * np.sin(np.arctan(grade / 100)))

    def _speed_from_power(mass, drivetrain_efficiency, Cda, Cr, power, grade, wind_speed=0.0):
        return bisect(lambda x: power - PowerModel._power_from_speed(mass, drivetrain_efficiency, Cda, Cr, x, grade), 0, PowerModel.vmax)

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

        map = lambda x : l * ( PowerModel._power_from_speed(mass, x[0], x[1], x[2], speed, grade) - power)
        res = least_squares(map, lb, bounds=(lb, ub)).x
        return PowerModel(mass, res[0], res[1], res[2])

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
    power_model = PowerModel(mass, drivetrain_efficiency, Cda, Cr)
    speed = 5.0
    grade = 10
    power = power_model.power_from_speed(speed, grade)
    r = power_model.speed_from_power(power, grade)
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

