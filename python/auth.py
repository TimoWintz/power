from flask import session, request, redirect
from stravalib import Client

def redirect_auth(client_id):
    client = Client()
    url = client.authorization_url(client_id=client_id,
                                   redirect_uri=request.url)
    return redirect(url)

def get_token(client_id, client_secret):
    access_token = None#session.get('access_token')
    if access_token != None:
        return access_token
    code = request.args.get('code')
    if code == None:
        return None
    client = Client()
    access_token = client.exchange_code_for_token(client_id=client_id, client_secret=client_secret, code=code)
    session['access_token'] = access_token["access_token"]
    return access_token
