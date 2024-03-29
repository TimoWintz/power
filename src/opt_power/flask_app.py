#!/usr/bin/env python3
from flask import render_template, request, redirect, make_response, session, send_from_directory
from flask import Flask
from stravalib import Client
from opt_power import auth, utils
import os
import time
from flask_restful import Resource, Api
from flask_restful import reqparse
from flask_cors import CORS
import logging
import shelve


app = Flask(__name__)
app.secret_key = 'blablabla'
api = Api(app)
cors = CORS(app)
# mycache = SharedMemoryDict(name='segments', size=1000000)
# print("cache = ", list(mycache.keys()))
logger = logging.getLogger(__name__)

#
TMP_FILE = "/tmp/puissance"


def setup_cache():
    mycache = shelve.open(TMP_FILE, 'c')
    mycache.close()


def read_cache(key):
    mycache = shelve.open(TMP_FILE, 'r')
    val = mycache.get(key, None)
    mycache.close()
    return val


def write_cache(key, val):
    mycache = shelve.open(TMP_FILE, 'w')
    mycache[key] = val
    mycache.close()


use_ign = False
ign_secrets = {}
if use_ign:
    d = os.path.dirname(__file__)
    with open(os.path.join(d, "ign_secrets.txt")) as f:
        ign_secrets["key"] = f.readline().strip()
        ign_secrets["login"] = f.readline().strip()
        ign_secrets["password"] = f.readline().strip()

try:
    with open(os.path.join(d, "gmaps_secrets.txt")) as f:
        gmaps_secrets = f.readline().strip()
except:
    gmaps_secrets = None


@app.route('/auth')
def main():
    try:
        client = Client(access_token=session['access_token'])
        athlete = client.get_athlete()
        print(athlete)
        return redirect('/index.html')

    except:
        with open('strava_secrets.txt') as f:
            client_id = f.readline().strip()
            client_secret = f.readline().strip()
        access_token = auth.get_token(client_id, client_secret)
        if access_token == None:
            return auth.redirect_auth(client_id)
        return redirect('/index.html')


class Auth(Resource):
    def get(self):
        try:
            client = Client(access_token=session['access_token'])
            athlete = client.get_athlete()
            return {"auth": True, "firstname": athlete.firstname, "lastname": athlete.lastname}
        except:
            return {"auth": False}


api.add_resource(Auth, "/api/auth")


@app.route('/')
def index():
    return redirect("/index.html")


segment_parser = reqparse.RequestParser()
segment_parser.add_argument(
    'type', type=str, help='Type (segment or effort)', default="segment", location='args')
segment_parser.add_argument('id', type=int, help='Segment id', location='args')
segment_parser.add_argument('use_ign', type=int, default=0, location='args')
segment_parser.add_argument('use_gmaps', type=int, default=0, location='args')

profile_parser = reqparse.RequestParser()
profile_parser.add_argument(
    'pen', type=float, help='Penalty, higher means less divisions in the profile', default=3.0, location='args')
profile_parser.add_argument(
    'key', type=str, help='What key to use for profile segmentation', default="grade", location='args')

pm_parser = reqparse.RequestParser()
pm_parser.add_argument('mass', type=float, help='Total mass', location='args')
pm_parser.add_argument('drivetrain_efficiency', type=float, location='args',
                       help='Drivetrain efficiency (0.0 - 1.0)', default=0.98)
pm_parser.add_argument(
    'Cda', type=float, help='Air friction coefficient Cda', default=0.35, location='args')
pm_parser.add_argument(
    'Cr', type=float, help='Rolling resistance coefficicent Cr', default=0.003, location='args')
pm_parser.add_argument('wind_speed', type=float, location='args',
                       help='Wind speed', default=0.00)
pm_parser.add_argument('wind_direction', type=float, location='args',
                       help='Wind direction', default=0.00)
pm_parser.add_argument(
    'temp', type=float, help='Temperature (C)', default=20.00, location='args')
pm_parser.add_argument('optim_type', type=str, location='args',
                       help='Type of optimization', default='np')
pm_parser.add_argument('wp0', type=float, help='Wp',
                       default=20.0, location='args')
pm_parser.add_argument('power', type=float, help='threshold power',
                       default=270.0, location='args')


def get_segment(args):
    start = time.time()
    id = args["id"]
    desc = str(frozenset(args.items()))
    print("segment = {}".format("segment"))
    segment = read_cache(desc)
    if segment is None:
        print("type = ", args["type"])
        if args["type"] == "segment":
            segment = utils.Segment.from_strava_segment(
                session['access_token'], id)
        elif args["type"] == "effort":
            segment = utils.Segment.from_strava_effort(
                session['access_token'], id)
        elif args["type"] == "activity":
            segment = utils.Segment.from_strava_activity(
                session['access_token'], id)
        elif args["type"] == "route":
            segment = utils.Segment.from_strava_route(
                session['access_token'], id)

        if args["use_ign"]:
            segment.correct_elevation()
        write_cache(desc, segment)
    elapsed = (time.time()-start)
    logger.info("Got segment in {}".format(elapsed))
    return segment


def get_profile(segment_args, profile_args):
    print(segment_args)
    print(profile_args)
    start = time.time()
    segment_desc = str(frozenset(segment_args.items()))
    profile_desc = str(profile_args['pen']) + "_" + segment_desc
    profile = read_cache(profile_desc)
    if profile is None:
        segment = get_segment(segment_args)
        profile = utils.Profile(segment)
        profile.regularize(profile_args['pen'])
        write_cache(profile_desc, profile)
    elapsed = (time.time()-start)
    logger.info("Got profile in {}".format(elapsed))
    return profile


class OptimalPower(Resource):
    def get(self):
        segment_args = segment_parser.parse_args()
        profile_args = profile_parser.parse_args()
        profile = get_profile(segment_args, profile_args)
        pm_args = pm_parser.parse_args()
        start = time.time()
        if pm_args["wind_speed"] > 0:
            profile.add_wind(pm_args["wind_speed"], pm_args["wind_direction"])
        profile.set_temp(pm_args["temp"])
        r = profile.optimal_power(
            pm_args["mass"], pm_args["drivetrain_efficiency"], pm_args["Cda"],  pm_args["Cr"], pm_args["power"], 1000 * pm_args["wp0"])
        elapsed = (time.time()-start)
        logger.info("Got optimal power in {}".format(elapsed))
        return r


api.add_resource(OptimalPower, "/api/optimal_power")


class Profile(Resource):
    def get(self):
        segment_args = segment_parser.parse_args()
        profile_args = profile_parser.parse_args()
        profile = get_profile(segment_args, profile_args)
        pm_args = pm_parser.parse_args()
        power_model = [
            pm_args["mass"], pm_args["drivetrain_efficiency"], pm_args["Cda"], pm_args["Cr"]]
        if pm_args["wind_speed"] > 0:
            profile.add_wind(pm_args["wind_speed"], pm_args["wind_direction"])
        profile.set_temp(pm_args["temp"])
        if segment_args["type"] == "effort":
            intervals = profile.intervals_from_speed(power_model)
        else:
            intervals = profile.intervals

        res = {
            "segment_name": profile.segment.name,
            "segment_points": profile.segment.points,
            "profile_points": profile.points,
            "profile_intervals": intervals
        }
        return res


api.add_resource(Profile, "/api/profile")


class EstimateParameters(Resource):
    def get(self):
        segment_args = segment_parser.parse_args()
        profile_args = profile_parser.parse_args()
        pm_args = pm_parser.parse_args()
        profile = get_profile(segment_args, profile_args)
        if pm_args["wind_speed"] > 0:
            profile.add_wind(pm_args["wind_speed"], pm_args["wind_direction"])
        if not profile.has_temperature:
            print("Using provided temperature")
            profile.set_temp(pm_args["temp"])
        mass = None if pm_args["mass"] == 0. else pm_args["mass"]
        Cda = None if pm_args["Cda"] == 0. else pm_args["Cda"]
        Cr = None if pm_args["Cr"] == 0. else pm_args["Cr"]
        drivetrain_efficiency = None if pm_args["drivetrain_efficiency"] == 0. else pm_args["drivetrain_efficiency"]
        params = utils.PowerModel.estimate_parameters(
            profile, Cda=Cda, drivetrain_efficiency=drivetrain_efficiency, Cr=Cr, mass=mass)
        return {
            "mass": float(params[0]),
            "drivetrain_efficiency": float(params[1]),
            "Cda": float(params[2]),
            "Cr": float(params[3])
        }


api.add_resource(EstimateParameters, "/api/parameters")


@ app.route('/<path:path>')
def send_js(path):
    return send_from_directory('static', path)


setup_cache()

if __name__ == "__main__":
    app.run(debug=True)
