from stravalib import Client
import requests
import numpy as np
import pwlf
import json
from copy import deepcopy
from scipy import optimize
import sys
import time
"""
sys.path.append("./_skbuild/linux-x86_64-3.10/cmake-build/")
try:
    import _core as opt_power
except:
    print("hello")
    # import opt_power
"""
import opt_power


class IGNAlti:
    max_slice = 20

    def __init__(self):
        return

    def format_line(self, pts):
        lon_str = "|".join([str(l["lng"]) for l in pts])
        lat_str = "|".join([str(l["lat"]) for l in pts])
        return lon_str, lat_str

    def correct_elevation(self, pts):
        for i in range(len(pts) // self.max_slice + 1):
            s = slice(i * self.max_slice, (i+1) * self.max_slice)
            lon_str, lat_str = self.format_line(pts[s])
            url = "https://wxs.ign.fr/essentiels/alti/rest/elevation.json?lon={}&lat={}&zonly=true".format(
                lon_str, lat_str)
            r = requests.get(url)
            body = r.content.decode()
            x = json.loads(body)
            if 'elevations' in x:
                for j in range(len(x["elevations"])):
                    idx = i*self.max_slice + j
                    pts[idx]["alt"] = x["elevations"][j]


class Segment:
    def __init__(self):
        self.points = []
        self.has_gps = False
        self.has_watts = False
        self.has_temperature = False

    @staticmethod
    def load_gpx(gpx_file):
        pass

    @staticmethod
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
                "lat": segment_stream["latlng"].data[i][0],
                "lng": segment_stream["latlng"].data[i][1],
                "alt": segment_stream["altitude"].data[i],
                "distance": segment_stream["distance"].data[i],
                "temp": 20
            })
        res.has_gps = True
        res.has_power = False
        res.has_temperature = False
        return res

    @staticmethod
    def from_strava_efforts(access_token, segment_id, resolution="low"):
        res = []
        client = Client(access_token=access_token)
        efforts = client.get_segment_efforts(segment_id)
        for effort in efforts:
            res.append(Segment.from_strava_effort(access_token, effort.id))
        return res

    @staticmethod
    def from_strava_effort(access_token, segment_effort_id, resolution="low"):
        client = Client(access_token=access_token)
        segment_stream = client.get_effort_streams(segment_effort_id, types=["latlng", "distance", "altitude", "watts", "time", "temp"],
                                                   resolution=resolution)
        segment = client.get_segment_effort(segment_effort_id)
        res = Segment._from_stream(segment_stream)
        res.name = segment.name
        return res

    @staticmethod
    def from_strava_activity(access_token, activity_id, resolution="low"):
        client = Client(access_token=access_token)
        activity = client.get_activity(activity_id)
        streams = client.get_activity_streams(
            activity_id, types=["latlng", "distance", "altitude"], resolution='medium')
        res = Segment._from_stream(streams)
        res.name = activity.name
        return res

    @staticmethod
    def from_strava_route(access_token, route_id, resolution="low"):
        client = Client(access_token=access_token)
        route = client.get_route(route_id)
        streams = client.get_route_streams(
            route_id)
        res = Segment._from_stream(streams)
        res.name = route.name
        return res

    @staticmethod
    def _from_stream(stream):
        res = Segment()
        res.has_power = False
        n_pts = len(stream["latlng"].data)
        res.points = []
        res.test = stream
        origin_distance = stream["distance"].data[0]
        for i in range(n_pts):
            if i > 0 and stream["distance"].data[i] == res.points[-1]["distance"]:
                continue
            res.points.append({
                "lat": stream["latlng"].data[i][0],
                "lng": stream["latlng"].data[i][1],
                "alt": stream["altitude"].data[i],
                "distance": stream["distance"].data[i] - origin_distance,
            })
            if "time" in stream.keys():
                res.points[-1]["time"] = stream["time"].data[i]
            if "watts" in stream.keys():
                res.points[-1]["watts"] = stream["watts"].data[i]
                res.has_watts = True
            if "temp" in stream.keys():
                res.points[-1]["temp"] = stream["temp"].data[i]
                res.has_temperature = True
            else:
                res.points[-1]["temp"] = 20
        res.has_gps = True
        res.track_points = deepcopy(res.points)
        return res

    @staticmethod
    def from_activity(access_token, activity_id, resolution="low"):
        client = Client(access_token=access_token)
        activity_stream = client.get_activity_streams(activity_id, types=["latlng", "distance", "altitude", "watts", "time"],
                                                      resolution=resolution)
        return Segment._from_stream(activity_stream)

    def correct_elevation(self):
        ign = IGNAlti()
        ign.correct_elevation(self.points)


class Wind:
    def __init__(self, speed, direction_deg):
        self.speed = speed / 3.6
        self.direction_rad = np.pi / 180 * direction_deg

    def average_tangential_component(self, profile):
        headwind = onp.zeros(len(profile.intervals))
        for i in range(len(profile.intervals)):
            pt1 = profile.points[i]
            pt2 = profile.points[i+1]
            angle = self.direction_rad - \
                angle_from_coordinates(
                    pt1["lat"], pt1["lng"], pt2["lat"], pt2["lng"])
            headwind[i] = np.cos(angle) * self.speed
        return headwind


class Profile:
    def __init__(self, segment=None):
        if segment is not None:
            self.segment = segment
            self.has_watts = self.segment.has_watts
            self.has_temperature = self.segment.has_temperature
            self.points = deepcopy(self.segment.points)
            self.intervals = [self._interval_indices(
                i, i+1) for i in range(len(self.points)-1)]
            self.update()
        else:
            self.points = []
            self.intervals = []
            self.has_watts = False
            self.has_temperature = False
        self.has_wind = False
        self.has_est_power = False

    def update(self):
        for i, interval in enumerate(self.intervals):
            if i == 0:
                interval["total_elevation"] = max(interval["elevation"], 0)
            else:
                interval["total_elevation"] = self.intervals[i -
                                                             1]["total_elevation"] + max(interval["elevation"], 0)

    def _average_power(self, i1, i2):
        p = 0
        T = self.points[i2]["time"] - self.points[i1]["time"]
        dist = self.points[i2]["distance"] - self.points[i1]["distance"]
        for i in range(i1, i2):
            dt = (self.points[i+1]["time"] - self.points[i]["time"])
            p += self.points[i]["watts"] * dt
        p /= T
        return p

    def _interval_indices(self, i1, i2):
        length = self.points[i2]["distance"] - self.points[i1]["distance"]
        elevation = self.points[i2]["alt"] - self.points[i1]["alt"]
        distance = self.points[i2]["distance"]
        res = {
            "length": length,
            "elevation": elevation,
            "grade": elevation / length * 100,
            "distance": distance,
            "altitude": self.points[i1]["alt"],
            "temp": self.points[i1]["temp"]
        }
        if "time" in self.points[i2]:
            res["duration"] = self.points[i2]["time"] - self.points[i1]["time"]
            res["time"] = self.points[i2]["time"]
            res["speed"] = res["length"] / res["duration"]
        if self.has_watts:
            power = self._average_power(i1, i2)
            res["watts"] = power
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

    def get_array(self, key):
        if key is None or (key == "headwind" and not self.has_wind):
            return np.zeros(len(self.intervals))
        return np.array([x[key] for x in self.intervals])

    def optimal_power(self, mass, drivetrain_efficiency, Cda, Cr, threshold_power, wp0):
        cp_model_params = opt_power.CPModelParams(wp0, threshold_power)
        physics_params = opt_power.PhysicsParams(
            mass, drivetrain_efficiency, Cda, Cr)
        length_array = self.get_array("length")
        grade_array = self.get_array("grade")
        altitude_array = self.get_array("altitude")
        temp_array = self.get_array("temp")
        headwind_array = self.get_array("headwind")
        segment_params = opt_power.SegmentParams(physics_params, length_array,
                                                 grade_array,
                                                 altitude_array,
                                                 temp_array,
                                                 headwind_array)
        initial_speed = opt_power.speed_from_power(
            threshold_power * np.ones_like(length_array), segment_params)
        average_speed = opt_power.average_speed(
            initial_speed, segment_params)
        min_speed = opt_power.speed_from_power(
            0.75 * threshold_power * np.ones_like(length_array), segment_params)
        max_speed = opt_power.speed_from_power(
            (threshold_power + wp0 / 300) * np.ones_like(length_array), segment_params)

        def jac_wp_bal(x):
            j = np.zeros((len(x), len(x)))
            opt_power.wp_bal(x, segment_params, cp_model_params, jac=j)
            return j
        constraints = optimize.NonlinearConstraint(
            lambda x: opt_power.wp_bal(x, segment_params, cp_model_params),
            jac=jac_wp_bal,
            lb=np.zeros(len(length_array)), ub=np.inf)

        print(f"initial speed = {initial_speed}")
        print(
            f"initial time = {opt_power.total_time(initial_speed, segment_params)}")

        bounds = optimize.Bounds(min_speed, max_speed)
        print(grade_array)
        print(initial_speed)

        start = time.time()
        x = optimize.minimize(lambda speed: opt_power.total_time(speed, segment_params), x0=initial_speed,
                              jac=lambda speed: -length_array / speed**2,
                              bounds=bounds, constraints=[constraints])
        end = time.time()
        print("Time consumed in opt: ", end - start)

        wp_bal = opt_power.wp_bal(x.x, segment_params, cp_model_params)
        target_speed = x.x
        print(f"wp_bal = {wp_bal}")
        print(f"speed = {target_speed}")
        print(f"grade = {grade_array}")
        print(f"speed = {target_speed}")
        average_speed = opt_power.average_speed(
            target_speed, segment_params)
        print(f"average speed = {average_speed}")
        power = opt_power.power_from_speed(target_speed, segment_params)
        print(f"power = {power}")
        print(
            f"final time = {opt_power.total_time(target_speed, segment_params)}")
        return {"intervals": self.intervals_from_speed(
            target_speed, average_speed, power, wp_bal)}

    def regularize(self, pen=10):
        if pen < 1:
            pen = 1
        pen = int(pen)
        dist = np.array([x["distance"] for x in self.intervals])
        alti = np.array([x["altitude"] for x in self.intervals])

        idx = list(range(len(dist)))
        while len(idx) > pen + 1:
            lam = [(dist[idx[i]] - dist[idx[i-1]]) / (dist[idx[i+1]] -
                                                      dist[idx[i-1]]) for i in range(1, len(idx)-1)]
            middle_alti = np.array([lam[i-1] * alti[idx[i+1]] +
                                    (1-lam[i-1]) * alti[idx[i-1]] for i in range(1, len(idx) - 1)])
            true_alti = np.array([alti[i] for i in idx[1:len(idx)-1]])
            best_idx = np.argmin(np.abs(middle_alti - true_alti))
            idx.remove(idx[best_idx + 1])
        self.indices = idx
        self.indices[-1] = len(dist)
        self.intervals = [self._interval_indices(
            self.indices[i], self.indices[i+1]) for i in range(len(self.indices) - 1)]
        self.points = [self.points[i] for i in self.indices]
        self.update()

    def regularize_(self, pen=10):
        pen = int(pen)
        dist = np.array([x["distance"] for x in self.intervals])
        print(dist)
        alti = np.array([x["altitude"] for x in self.intervals])
        my_pwlf = pwlf.PiecewiseLinFit(dist, alti)
        fit = my_pwlf.fitfast(pen, pop=10)
        i0 = 0
        self.indices = [i0]
        for b in fit[1:]:
            while b > dist[i0]:
                i0 += 1
            if i0 > self.indices[-1]:
                self.indices.append(i0)
        self.indices[-1] = len(dist)

        self.intervals = [self._interval_indices(
            self.indices[i], self.indices[i+1]) for i in range(len(self.indices) - 1)]
        self.points = [self.points[i] for i in self.indices]
        self.update()

    def intervals_from_speed(self, target_speed, average_speed, power, wp_bal):
        result = deepcopy(self.intervals)
        for i in range(len(self.intervals)):
            if self.has_wind:
                wind = self.intervals[i]["headwind"]
            else:
                wind = 0.0
            result[i]["headwind"] = float(wind)
            result[i]["duration"] = float(
                self.intervals[i]["length"] / average_speed[i])
            result[i]["speed"] = float(target_speed[i])
            result[i]["est_power"] = float(power[i])
            result[i]["wp_bal"] = max(float(wp_bal[i]), 0)

            if i > 0:
                prev_t = result[i-1]["time"]
            else:
                prev_t = 0
            result[i]["time"] = prev_t + result[i]["duration"]
            result[i]["average_speed"] = result[i]["distance"] / \
                result[i]["time"]
        return result


def angle_from_coordinates(lat1, lng1, lat2, lng2):
    dlng = lng2 - lng1
    y = np.sin(dlng) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * \
        np.cos(lat2) * np.cos(dlng)
    return np.arctan2(y, x)


if __name__ == "__main__":
    print("testing strava segment stream...")
    access_token = "eb0fb1b6f9f5b641ceae0156fbe5393a3f797218"
    segment_id = 8061351
    segment = Segment.from_strava_segment(access_token, segment_id)
    print("OK")

    print("Approximating profile...")
    profile = Profile(segment)
    profile.regularize(5)
    print("OK")

    print("Testing optimal power")
    mass = 73
    drivetrain_efficiency = 0.98
    Cda = 0.25
    Cr = 0.003
    P = 290
    wp0 = 20000
    speed = 5.0
    grade = 10
    target_normalized_power = 266
    cp_model_params = opt_power.CPModelParams(wp0, P)
    physics_params = opt_power.PhysicsParams(
        mass, drivetrain_efficiency, Cda, Cr)
    print(type(physics_params))
    segment_params = opt_power.SegmentParams(physics_params, profile.get_array("length"), profile.get_array("grade"), profile.get_array("altitude"),
                                             profile.get_array("temp"), profile.get_array("headwind"))
    speed = 5 * np.ones(len(profile.intervals))
    speed[2] = 22
    # print(profile.get_array("grade"))
    # print(profile.get_array("length"))
    wp_bal, jac = opt_power.wp_bal(speed, segment_params, cp_model_params)
    r = profile.optimal_power(
        mass, drivetrain_efficiency, Cda, Cr, target_normalized_power, 20000)
    print("OK")
