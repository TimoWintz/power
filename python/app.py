#!/usr/bin/env python3
from flask import render_template, request, redirect, make_response, session
from flask import Flask
from stravalib import Client
import auth
import os
from flask_restful import Resource, Api
from flask_restful import reqparse
from flask_cors import CORS
import shelve

import utils

app = Flask(__name__)
app.secret_key = 'blablabla'
api = Api(app)
cors = CORS(app)
mycache = shelve.open('puissance', flag='c')
print("cache = ", list(mycache.keys()))

use_ign = True
ign_secrets = {}
if use_ign:
    d = os.path.dirname(__file__)
    with open(os.path.join(d, "ign_secrets.txt")) as f:
        ign_secrets["key"] = f.readline().strip()
        ign_secrets["login"] = f.readline().strip()
        ign_secrets["password"] = f.readline().strip()

@app.route('/static/<path:path>')
def send_js(path):
    return send_from_directory('js', path)


@app.route('/')
def main():
    try:
        client = Client(access_token=session['access_token'])
        athlete = client.get_athlete()
        return redirect('/static/index.html')
    except:
        with open('strava_secrets.txt') as f:
            client_id = f.readline().strip()
            client_secret= f.readline().strip()
        access_token = auth.get_token(client_id, client_secret)
        if access_token == None:
            return auth.redirect_auth(client_id)
        return redirect('/')

class Auth(Resource):
    def get(self):
        try:
            client = Client(access_token=session['access_token'])
            athlete = client.get_athlete()
            return {"auth" : True, "firstname" : athlete.firstname, "lastname" : athlete.lastname}
        except:
            return {"auth" : False}
api.add_resource(Auth, "/api/auth")


segment_parser = reqparse.RequestParser()
segment_parser.add_argument('type', type=str, help='Type (segment or effort)', default="segment")
segment_parser.add_argument('id', type=int, help='Segment id')
segment_parser.add_argument('use_ign', type=bool, default=True)

profile_parser = reqparse.RequestParser()
profile_parser.add_argument('pen', type=float, help='Penalty, higher means less divisions in the profile', default=3.0)

pm_parser = reqparse.RequestParser()
pm_parser.add_argument('mass', type=float, help='Total mass')
pm_parser.add_argument('drivetrain_efficiency', type=float, help='Drivetrain efficiency (0.0 - 1.0)', default=0.98)
pm_parser.add_argument('Cda', type=float, help='Air friction coefficient Cda', default=0.3)
pm_parser.add_argument('Cr', type=float, help='Rolling resistance coefficicent Cr', default=0.003)


def get_segment(args):
    id = args["id"]
    desc = str(frozenset(args.items()))
    if desc in mycache:
        print("using cache")
        segment = mycache[desc]
    else:
        if args["type"] == "segment":
            segment = utils.Segment.from_strava_segment(session['access_token'], id)
        elif args["type"] == "effort":
            segment = utils.Segment.from_strava_effort(session['access_token'], id)
        if args["use_ign"]:
            segment.correct_elevation(ign_secrets)
        mycache[desc] = segment
        mycache.sync()
    return segment

def get_profile(segment_args, profile_args):
    segment_desc = str(frozenset(segment_args.items()))
    profile_desc = str(profile_args['pen']) + "_" + segment_desc
    if profile_desc in mycache:
        print("using profile cache")
        profile = mycache[profile_desc]
    else:
        segment = get_segment(segment_args)
        profile = utils.Profile(segment)
        profile.cluster_grades(profile_args['pen'])
        mycache[profile_desc] = profile
        mycache.sync()
    return profile

opt_parser = reqparse.RequestParser()
opt_parser.add_argument('power', type=float, help='Nominal power')
opt_parser.add_argument('format', type=str, default="json", help='Nominal power')

class OptimalPower(Resource):
    def get(self):
        segment_args = segment_parser.parse_args()
        profile_args = profile_parser.parse_args()
        profile = get_profile(segment_args, profile_args)
        pm_args = pm_parser.parse_args()
        power_model = utils.PowerModel(**pm_args)
        opt_args = opt_parser.parse_args()
        r = profile.optimal_power(power_model, opt_args["power"])
        return r

api.add_resource(OptimalPower, "/api/optimal_power")


if __name__ == "__main__":
    try:
        app.run()
    except:
        shelve.close()
