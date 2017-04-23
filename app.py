import json
import os.path
import cherrypy
import numpy as np
import conf
from model import Generator

class Root(object):
    def __init__(self):
        self.model_file = 'static/model/lcvrnn15.h5'
        self.generator = Generator(self.model_file)

    @cherrypy.expose
    def index(self):
        with open('index.html') as f:
            return f.read()

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @cherrypy.tools.json_in()
    def decode(self):
        result = {"operation": "request", "result": "success"}
        content = cherrypy.request.json
        x = self.generator.generate_as_notes(content)
        result["output"] = x
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
