from multiprocessing import Process, Queue

class Multiprocessor():

    def __init__(self, timeout, default_ret):
        self.timeout = timeout
        self.default_ret = default_ret
        self.process = None
        self.queue = Queue()

    @staticmethod
    def _wrapper(func, queue, args, kwargs):
        ret = func(*args, **kwargs)
        queue.put(ret)

    def run(self, func, *args, **kwargs):
        args2 = [func, self.queue, args, kwargs]
        p = Process(target=self._wrapper, args=args2)
        self.process = p
        p.start()

    def wait(self):
        ret = self.default_ret
        p = self.process
        p.join(self.timeout)
        if p.is_alive():
            print("did not finish in time")
            p.terminate()
            p.join()
        else:
            ret = self.queue.get()
        return ret