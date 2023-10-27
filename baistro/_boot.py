from baistro.config.config import AppConfig


class Services(object):

    def __init__(
        self,
    ):
        def shutdown():
            pass

        self.shutdown = shutdown
        # just to ensure always loaded when services are loaded
        self.config = AppConfig


def boot(
):
    return Services()
