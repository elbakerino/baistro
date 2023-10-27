from datetime import datetime
from typing import Union


def now_iso(micros: bool = True):
    return datetime.utcnow().strftime(f'%Y-%m-%d %H:%M:%S{".%f" if micros else ""}+0000')


def format_iso(ts: Union[int, float], micros: bool = True):
    return datetime.utcfromtimestamp(ts).strftime(f'%Y-%m-%d %H:%M:%S{".%f" if micros else ""}+0000')
