import signal
import sys
import logging

from baistro._boot import boot
from baistro.api.api import ai_api
from baistro.config.config import AppConfig

from baistro.helper import ts
from flask import Flask, render_template, url_for
from flask_cors import CORS

# todo: https://stackoverflow.com/a/16993115/2073149
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
pil_logger = logging.getLogger('PIL')
pil_logger.setLevel(logging.INFO)

s = boot()

app = Flask(__name__)
CORS(app)


def on_signal(signal_number, frame):
    logging.info(f'shutting down {signal_number}')
    s.shutdown()


# signal.signal(signal.SIGKILL, on_signal)
signal.signal(signal.SIGTERM, on_signal)
signal.signal(signal.SIGINT, on_signal)


@app.route('/')
def route_home():
    links = []
    for rule in app.url_map.iter_rules():
        if rule.endpoint == 'static':
            continue
        url = url_for(rule.endpoint, **(rule.defaults or {}))
        methods = rule.methods.copy()
        if 'HEAD' in methods:
            methods.remove('HEAD')
        if 'OPTIONS' in methods:
            methods.remove('OPTIONS')
        links.append((url, list(methods), rule.endpoint, rule.arguments, rule.defaults))
    return render_template('index.html', version=AppConfig.APP_ENV, links=links)


@app.route('/ping')
def route_api():
    return {
        "now": ts.now_iso(micros=False),
    }


ai_api(app, s)
