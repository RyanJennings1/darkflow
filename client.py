#!/usr/bin/env python3
"""
  Name: client.py
  Author: Ryan Jennings
  Date: 2018-09-21
"""
import argparse
import base64
import socket

import requests

def main(address=None, filename=None):
    """Main function"""
    PORT = 5000
    if address: addr = 'http://' + address + ':' + str(PORT)
    else: addr = 'http://' + socket.gethostbyname(socket.gethostname()) + ':' + str(PORT)

    url = addr + '/api/test'

    print("Using url: ", url)

    if not filename: filename = 'test_dir/test.png'

    try:
        data = post_image(url, filename)
        imgdata = base64.b64decode(data.json()['py/b64'])
        with open('outputimg.png', 'wb') as f:
            f.write(imgdata)
    except Exception as exc:
        print('Error attempting to post image: ', exc)

def post_image(url, img_name):
    """Post image and return response"""

    content_type = 'image/png'
    headers = {'content-type': content_type}

    print('Reading image "' + img_name + '" ... ')

    with open(img_name, 'rb') as f:
        img = f.read()
        print("Sending POST ... ")
        response = requests.post(url, data=img, headers=headers)
    return response

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--address', '-a', type=str, help='IP address to post to')
    parser.add_argument('--file', '-f', type=str, help='Send specified file')
    args = parser.parse_args()
    if args.address: print('\nUsing address: ', args.address)
    if args.file: print('\nUsing file: ', args.file)
    main(address=args.address, filename=args.file)
