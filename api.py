from flask import Flask, request
from flask_restplus import Resource, Api, reqparse

from scouter import Scouter

app = Flask(__name__)
api = Api(app, version='1.0', title="Location intelligence API",
    description='REST API that provides location intelligence data',)
api.namespaces.clear()
ns = api.namespace('peers-insight', 
                   description='Get insight into peers by your location')
scouter = Scouter()

peers_parser = reqparse.RequestParser()
peers_parser.add_argument('lat', type=float, required=True, help='Latitude of your location')
peers_parser.add_argument('lon', type=float, required=True, help='Longitude of your location')

@ns.route('/peers')
@ns.expect(peers_parser) 
class Peers(Resource):
    def get(self):
        """
        Returns ranked list of restaurants in similar location
        """
        args = request.args
        latitude = float(args['lat'])
        longitude = float(args['lon'])
        ret = scouter.similarly_located_restaurants(longitude, latitude)
        return ret

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)