from .defaults import argHandler #Import the default arguments
import os
from .net.build import TFNet
from .server import Server

def cliHandler(args):
    FLAGS = argHandler()
    FLAGS.setDefaults()
    FLAGS.parseArgs(args)

    # make sure all necessary dirs exist
    def _get_dir(dirs):
        for d in dirs:
            this = os.path.abspath(os.path.join(os.path.curdir, d))
            if not os.path.exists(this): os.makedirs(this)
    
    requiredDirectories = [FLAGS.imgdir, FLAGS.binary, FLAGS.backup, os.path.join(FLAGS.imgdir,'out')]
    if FLAGS.summary:
        requiredDirectories.append(FLAGS.summary)

    _get_dir(requiredDirectories)

    # fix FLAGS.load to appropriate type
    try: FLAGS.load = int(FLAGS.load)
    except: pass

    tfnet = TFNet(FLAGS)
    
    if FLAGS.demo:
        tfnet.camera()
        exit('Demo stopped, exit.')

    if FLAGS.train:
        print('Enter training ...'); tfnet.train()
        if not FLAGS.savepb: 
            exit('Training finished, exit.')

    if FLAGS.savepb:
        print('Rebuild a constant version ...')
        tfnet.savepb(); exit('Done')

    # tfnet.predict()
    print('Should be predicting here')

    # Set up server
    server = Server(tfnet=tfnet)
    server.run()

    """
    Ask if they want another prediction.
    Will be replaced with an interface
    for serving img post requests
    val = 'y'
    while (val == 'y'):
        val = input('Predict new dir? [y/N] ')
        if val == 'y':
            # new_dir = input('Enter dir name: ')
            # tfnet.predict(inp_path=new_dir)
    """
