import traceback


def init_queue(q):
    global queue
    queue = q

def error_handler(e):
    print(dir(e), "\n")
    print("-->{}<--".format(e.__cause__))
    traceback.print_exception(type(e), e, e.__traceback__)
