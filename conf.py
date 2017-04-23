import os
CURDIR = os.path.dirname(os.path.abspath(__file__))
ROOTDIR = CURDIR
PORT_NUM = 8080

settings = {
    'global': {
        'server.socket_host': '0.0.0.0',
        'server.socket_port': int(os.environ.get('PORT', str(PORT_NUM))),
        'server.environment': 'development',
    },
}

root_settings = {
    '/': {
        'tools.staticdir.root': ROOTDIR,
    },
    '/static': {
        'tools.staticdir.on': True,
        'tools.staticdir.dir': 'static',
        'tools.expires.on'    : True,
        'tools.expires.secs'  : 1
    },
}
