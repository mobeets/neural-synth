import json
import os.path
import cherrypy
import numpy as np
import conf
# from model import Generator, detect_chord

Generator = lambda x: None
detect_chord = lambda x: None

class Root(object):
    def __init__(self):
        # self.gen_mdl = Generator('static/models/vrnn.h5', model_type='vrnn')
        # self.gen_mdl = Generator('static/models/vae.h5', model_type='vae')
        self.gen_mdl = Generator('static/models/clvae2.h5', model_type='clvae')

    @cherrypy.expose
    def index(self):
        with open('index.html') as f:
            return f.read()

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @cherrypy.tools.json_in()
    def changekey(self):
        result = {"operation": "request", "result": "success"}
        result["output"] = self.gen_mdl.change_key(cherrypy.request.json)
        return result

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @cherrypy.tools.json_in()
    def decode(self):
        result = {"operation": "request", "result": "success"}
        result["output"] = self.gen_mdl.generate_as_notes(cherrypy.request.json)
        return result

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @cherrypy.tools.json_in()
    def detect(self):
        result = {"operation": "request", "result": "success"}
        result["output"] = detect_chord(cherrypy.request.json)
        return result

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @cherrypy.tools.json_in()
    def encode(self):
        result = {"operation": "request", "result": "success"}
        result["output"] = self.gen_mdl.encode_as_notes(cherrypy.request.json)
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
