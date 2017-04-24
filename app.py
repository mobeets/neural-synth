import json
import os.path
import cherrypy
import numpy as np
import conf
from model import Generator

class Root(object):
    def __init__(self):
        self.gen_vrnn = Generator('static/models/vrnn.h5', model_type='vrnn')
        self.gen_vae = Generator('static/models/vae.h5', model_type='vae')
        self.gen_source = 'vae'

    @cherrypy.expose
    def index(self):
        with open('index.html') as f:
            return f.read()

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @cherrypy.tools.json_in()
    def decode(self):
        result = {"operation": "request", "result": "success"}
        gen_mdl = self.gen_vrnn if self.gen_source == 'vrnn' else self.gen_vae
        result["output"] = gen_mdl.generate_as_notes(cherrypy.request.json)
        return result

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @cherrypy.tools.json_in()
    def encode(self):
        result = {"operation": "request", "result": "success"}
        result["output"] = self.gen_vae.encode_as_notes(cherrypy.request.json)
        return result

def main():
    cherrypy.config.update(conf.settings)

    root_app = cherrypy.tree.mount(Root(), '/', conf.root_settings)
    root_app.merge(conf.settings)

    if hasattr(cherrypy.engine, "signal_handler"):
        cherrypy.engine.signal_handler.subscribe()
    if hasattr(cherrypy.engine, "console_control_handler"):
        cherrypy.engine.console_control_handler.subscribe()
    cherrypy.engine.start()
    cherrypy.engine.block()

if __name__ == '__main__':
    main()
