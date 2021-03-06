import numpy as onp
import requests
from stravalib import Client
import googlemaps
import json
import os
import ruptures as rpt
from copy import deepcopy
from scipy.optimize import bisect, least_squares, minimize, Bounds, NonlinearConstraint, shgo
from pyowm import OWM

from jax import numpy as np
from jax import grad, jit, jacfwd
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
        self.has_temperature = False

    def load_gpx(gpx_file):
        pass

    def from_strava_segment(access_token, segment_id, resolution="low"):
        client = Client(access_token=access_token)
        segment_stream = client.get_segment_streams(segment_id, types=["latlng", "distance", "altitude"],
            resolution=resolution)
        res = Segment()
        segment = client.get_segment(segment_id)
        res.name = segment.name
        n_pts = len(segment_stream["latlng"].data)
        res.points = []
        for i in range(n_pts):
            res.points.append({
                "lat" : segment_stream["latlng"].data[i][0],
                "lng" : segment_stream["latlng"].data[i][1],
                "alt" : segment_stream["altitude"].data[i],
                "distance" : segment_stream["distance"].data[i],
                "temp": 20
            })
        res.has_gps = True
        res.has_power = False
        res.has_temperature = False
        return res

    def from_strava_efforts(access_token, segment_id, resolution="low"):
        res = []
        client = Client(access_token=access_token)
        efforts = client.get_segment_efforts(segment_id)
        for effort in efforts:
            res.append(Segment.from_strava_effort(access_token, effort.id))
        return res

    def from_strava_effort(access_token, segment_effort_id, resolution="low"):
        client = Client(access_token=access_token)
        segment_stream = client.get_effort_streams(segment_effort_id, types=["latlng", "distance", "altitude", "watts", "time", "temp"],
            resolution=resolution)
        segment = client.get_segment_effort(segment_effort_id)
        res = Segment._from_stream(segment_stream)
        res.name = segment.name
        return res

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
            if "temp" in stream.keys():
                res.points[-1]["temp"] = stream["temp"].data[i]
                self.has_temperature = True
            else:
                res.points[-1]["temp"] = 20
        res.has_gps = True
        return res

    def from_activity(access_token, activity_id, resolution="low"):
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

def _normalized_power(power_model_params, lengths, grades, headwind, altitude, temp, speed):
    power = []
    for i in range(len(speed)):
        power.append(PowerModel.power_from_speed(power_model_params, grades[i], speed[i], altitude[i], temp[i], headwind[i]))
    power = np.array(power)
    t = lengths / speed
    return (1 / np.sum(t) * np.sum( t * power**4 ) )**(1/4)

normalized_power_fn = jit(_normalized_power)
g_normalized_power_fn = jit(grad(_normalized_power, 6))

def _average_power(power_model_params, lengths, grades, headwind, altitude, temp, speed):
    power = []
    for i in range(len(speed)):
        power.append(PowerModel.power_from_speed(power_model_params, grades[i], speed[i], altitude[i], temp[i], headwind[i]))
    power = np.array(power)
    t = lengths / speed
    return (1 / np.sum(t) * np.sum( t * power ) )

average_power_fn = jit(_average_power)
g_average_power_fn = jit(grad(_average_power, 6))

def optimal_power(power_model_params, lengths, grades, headwind, altitudes, temp, target_normalized_power,
    constant_power=False, constant_velocity=False, use_normalized=True):

    min_power = 0.5 * target_normalized_power 
    max_power = 4.0 * target_normalized_power

    if headwind is None:
        headwind = np.zeros(len(lengths))
    if temp is None:
        temp = np.zeros(len(lengths))
    lam = 4.0
    normalized_power = 0.0
    f = lambda v : total_time_fn(lengths, v)
    gf = lambda v : g_total_time_fn(lengths, v)

    print("length = ", lengths)
    print("grade = ", grades)
    print("headwind= ", headwind)
    print("altitude= ", altitudes)
    print("temp= ", temp)

    constraints = []
    if use_normalized:
        h = lambda v : normalized_power_fn(power_model_params, lengths, grades, headwind, altitudes, temp, v)
        gh = lambda v : g_normalized_power_fn(power_model_params, lengths, grades, headwind, altitudes, temp, v)
    else:
        h = lambda v : average_power_fn(power_model_params, lengths, grades, headwind, altitudes, temp, v)
        gh = lambda v : g_average_power_fn(power_model_params, lengths, grades, headwind, altitudes, temp, v)
    constraints.append(NonlinearConstraint(h, jac=gh, lb=np.array([0]), ub=np.array([target_normalized_power])))

    if constant_velocity:
        v = 5*np.ones(len(grades))
        eps = 0.001
        while h(v) < target_normalized_power:
            v *= (1+eps)
        return onp.array(v)


    v0 = np.array([PowerModel.speed_from_power(power_model_params, grades[i], target_normalized_power, altitudes[i], temp[i], headwind[i]) for i in range(len(grades))])
    print("v0 = ", v0)
    if constant_power:
        print("NP = ", h(v0))
        return onp.array(v0)
    lb = np.zeros(len(lengths))
    ub = np.array([PowerModel.speed_from_power(power_model_params, grades[i], max_power, altitudes[i], temp[i], headwind[i]) for i in range(len(grades))])
    bounds = Bounds(lb, ub)

    # constraint = NonlinearConstraint(h, lb=np.array([0]), ub=np.array([target_normalized_power]))
    print("fun = ", f(v0))
    r = minimize(fun=f, jac=gf, x0=v0, constraints=constraints, bounds=bounds)
    if not r.success:
        r = minimize(fun=f, jac=gf, x0=v0, constraints=constraints, bounds=bounds, method="trust-constr")
    print("fun = ", f(r.x))
    print("NP = ", h(r.x))
    print("OPTIM RESULT")
    print(r)
    return r.x

class Wind:
    def __init__(self, speed, direction_deg):
        self.speed = speed / 3.6
        self.direction_rad = np.pi / 180 * direction_deg

    def average_tangential_component(self, profile):
        headwind = onp.zeros(len(profile.intervals))
        for i in range(len(profile.intervals)):
            v = 0
            pt1 = profile.points[i]
            pt2 = profile.points[i+1]
            angle = self.direction_rad - angle_from_coordinates(pt1["lat"], pt1["lng"], pt2["lat"], pt2["lng"])
            headwind[i] = np.cos(angle) * self.speed
        return headwind

class Profile:
    def __init__(self, segment=None):
        if segment is not None:
            self.segment = segment
            self.has_power = self.segment.has_power
            self.has_temperature = self.segment.has_temperature
            self.points = deepcopy(self.segment.points)
            self.intervals = [self._interval_indices(i, i+1) for i in range(len(self.points)-1)]
            self.update()
        else:
            self.points = []
            self.intervals = []
            self.has_power = False
            self.has_temperature = False
        self.has_wind = False

    def update(self):
        for i, interval in enumerate(self.intervals):
            if i == 0:
                interval["total_elevation"] = max(interval["elevation"], 0)
            else:
                interval["total_elevation"] = self.intervals[i-1]["total_elevation"] + max(interval["elevation"], 0)

    def _average_power(self, i1, i2):
        p = 0
        T = self.points[i2]["time"] - self.points[i1]["time"]
        dist = self.points[i2]["distance"] - self.points[i1]["distance"]
        for i in range(i1, i2):
            dt = (self.points[i+1]["time"] - self.points[i]["time"])
            p += self.points[i]["power"] * dt
        p /= T
        return p

    def _interval_indices(self, i1, i2):
        length = self.points[i2]["distance"] - self.points[i1]["distance"]
        elevation = self.points[i2]["alt"] - self.points[i1]["alt"]
        distance = self.points[i2]["distance"]
        res = {
            "length" : length,
            "elevation" : elevation,
            "grade" : elevation / length * 100,
            "distance" : distance,
            "altitude": self.points[i1]["alt"],
            "temp": self.points[i1]["temp"]
        }
        if "time" in self.points[i2]:
            res["duration"] = self.points[i2]["time"] - self.points[i1]["time"]
            res["time"] = self.points[i2]["time"]
            res["speed"] = res["length"] / res["duration"]
        if self.has_power:
            power = self._average_power(i1, i2)
            res["power"] = power
        return res

    def set_temp(self, temp):
        for i, interval in enumerate(self.intervals):
            interval["temp"] = temp

    def add_wind(self, wind_speed, wind_direction):
        wind = Wind(wind_speed, wind_direction)
        headwind = wind.average_tangential_component(self)
        for i, interval in enumerate(self.intervals):
            interval["headwind"] = headwind[i]
        self.has_wind = True

    def optimal_power(self, power_model_params, target_normalized_power,
                constant_power=False,
                constant_velocity=False,
                use_normalized=True):
        print("Constant = ", constant_power)
        assert(len(power_model_params) == 4)
        print("params = ", power_model_params)
        if self.has_wind:
            headwind = np.array([x["headwind"] for x in self.intervals])
        else:
            headwind = onp.zeros(len(self.intervals))
        lengths = np.array([x["length"] for x in self.intervals])
        temps = np.array([x["temp"] for x in self.intervals])
        altitudes = np.array([x["altitude"] for x in self.intervals])
        grades = np.array([x["grade"] for x in self.intervals])
        speed = optimal_power(power_model_params, lengths, grades, headwind, altitudes, temps, target_normalized_power,
            constant_power=constant_power, constant_velocity=constant_velocity, use_normalized=use_normalized)
        return {"normalized_power" : float(normalized_power_fn(power_model_params, lengths, grades, headwind, altitudes, temps, speed)),
                "intervals" : self.intervals_from_power(speed, power_model_params)}

    def regularize(self, pen=10, key="grade"):
        values = np.array([x[key] for x in self.intervals])
        algo = rpt.Pelt(model="rbf").fit(values)
        self.indices = algo.predict(pen=pen)
        self.indices.insert(0,0)
        self.intervals = [self._interval_indices(self.indices[i], self.indices[i+1]) for i in range(len(self.indices) - 1)]
        self.points= [self.points[i] for i in self.indices]
        self.update()

    def intervals_from_power(self, speed, power_model):
        result = deepcopy(self.intervals)
        for i in range(len(self.intervals)):
            if self.has_wind:
                wind = self.intervals[i]["headwind"]
            else:
                wind = 0.0
            result[i]["headwind"] = wind
            result[i]["duration"] = self.intervals[i]["length"] / speed[i]
            result[i]["speed"] = speed[i]
            result[i]["speed_kmh"] = speed[i]*3.6
            result[i]["power"] = float(PowerModel.power_from_speed(power_model, self.intervals[i]["grade"], speed[i], self.intervals[i]["altitude"], self.intervals[i]["temp"], wind))
            if i > 0:
                prev_t = result[i-1]["time"]
            else:
                prev_t = 0
            result[i]["time"] = prev_t + result[i]["duration"]
            result[i]["average_kmh"] = result[i]["distance"] / result[i]["time"] * 3.6
        return result

def angle_from_coordinates(lat1, lng1, lat2, lng2):
    dlng = lng2 - lng1
    y = np.sin(dlng) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlng)
    return np.arctan2(y, x)


class PowerModel:
    vmax = 100
    vmin = 0.1
    gravity = 9.81

    def pressure_from_altitude(alt):
        return 1013.25 * (1-0.0065*alt/288.15)**(5.255)

    def air_density_pressure(temperature_C, pressure_hpa):
        R_air = 287
        temperature_K = temperature_C + 273
        pressure_pa = 100*pressure_hpa
        air_density = pressure_pa / temperature_K / R_air
        return air_density

    def air_density(alt, temperature_C):
        return PowerModel.air_density_pressure(temperature_C, PowerModel.pressure_from_altitude(alt))

    def power_from_speed(params, grade, v, altitude=0, temperature_C=20, wind_speed=0.0):
        mass, drivetrain_efficiency, Cda, Cr = params
        rho = PowerModel.air_density(altitude, temperature_C)
        return  1/drivetrain_efficiency * v * (0.5 * rho * Cda * np.sign(v+wind_speed) *(v + wind_speed) * (v + wind_speed) + Cr * mass * PowerModel.gravity + mass * PowerModel.gravity * np.sin(np.arctan(grade / 100)))

    def speed_from_power(params, grade, power, alt=0, temperature_C=20, wind_speed=0.0):
        return bisect(lambda x: power - PowerModel.power_from_speed(params, grade, x, alt, temperature_C, wind_speed), 0, PowerModel.vmax)

    def estimate_parameters(profile, Cda=None, drivetrain_efficiency=None, Cr=None, mass=None):
        if not profile.has_power:
            raise Exception("Profile must have power data")

        speed = np.array([p["speed"] for p in profile.intervals])
        power = np.array([p["power"] for p in profile.intervals])
        grade = np.array([p["grade"] for p in profile.intervals])
        altitude = np.array([p["altitude"] for p in profile.intervals])
        if profile.has_wind:
            headwind = np.array([p["headwind"] for p in profile.intervals])
        else:
            headwind = np.zeros(len(profile.intervals))
        temp = np.array([p["temp"] for p in profile.intervals])

        def static_or_not(x, i):
            if i == 0:
                if mass is not None:
                    return mass 
                else:
                    return x[i]
            if i == 1:
                if drivetrain_efficiency is not None:
                    return drivetrain_efficiency
                else:
                    return x[i]
            elif i == 2:
                if Cda is not None:
                    return Cda
                else:
                    return x[i]
            elif i == 3:
                if Cr is not None:
                    return Cr
                else:
                    return x[i]

        lb = np.array([0.0, 0.8, 0.01, 0.0001])
        ub = np.array([200.0, 1.0, 0.6, 0.5])
        l = np.array([i["duration"] for i in profile.intervals])

        map = lambda x : l * (PowerModel.power_from_speed([static_or_not(x, 0), static_or_not(x, 1), static_or_not(x, 2), static_or_not(x, 3)],
            grade, speed, altitude, temp, headwind) - power)
        res = least_squares(map, lb, bounds=(lb, ub)).x
        return np.array([static_or_not(res, 0), static_or_not(res, 1), static_or_not(res, 2), static_or_not(res, 3)])

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
    target_normalized_power = 266
    r = profile.optimal_power(power_model, target_normalized_power)
    print("OK")

    print("Testing openweathermap")
    with open(os.path.join(d, "owm_secrets.txt")) as f:
        owm_key = f.readline().strip()
    ow = OWM(owm_key)
    print("OK")

