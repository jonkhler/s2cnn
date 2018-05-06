# pylint: disable=R,C,E1101
import threading
import time
from functools import wraps


class WaitPrint(threading.Thread):
    def __init__(self, t, message):
        super().__init__()
        self.t = t
        self.message = message
        self.running = True

    def stop(self):
        self.running = False

    def run(self):
        for _ in range(int(self.t // 0.1)):
            time.sleep(0.1)
            if not self.running:
                return
        print(self.message, end="")


def show_running(func):
    @wraps(func)
    def g(*args, **kargs):
        x = WaitPrint(
            2,
            "{}({})... ".format(
                func.__name__,
                ", ".join(
                    [repr(x) for x in args] +
                    ["{}={}".format(key, repr(value)) for key, value in kargs.items()]
                )
            )
        )
        x.start()
        t = time.perf_counter()
        r = func(*args, **kargs)
        if x.is_alive():
            x.stop()
        else:
            print("done in {:.0f} seconds".format(time.perf_counter() - t))
        return r
    return g
