import cv2
from flask import Flask, request, Response
import jsonpickle
import numpy as np

class Server:

    def __init__(self, tfnet=None):
        # Initialize the Flask application
        self.app = Flask(__name__)
        self.port = 5000

        # route http posts to this method
        @self.app.route('/api/test', methods=['POST'])
        def test():
            r = request
            # convert string of image data to uint8
            nparr = np.fromstring(r.data, np.uint8)
            # decode image
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            cv2.imwrite('server_dir/image_out.png', img)

            # Call the tensorflow prediction
            if tfnet:
                print('Running object prediction ...')
                tfnet.predict(inp_path='server_dir/')

            # The output image is encoded in b64 inside json and returned
            # to the client
            with open('server_dir/out/image_out.png', 'rb') as ret_img:
                response_pickled = jsonpickle.encode(ret_img.read())
                return Response(response=response_pickled, status=200, mimetype="image/png")
    
    def run(self):
        self.app.run(host="0.0.0.0", port=self.port)