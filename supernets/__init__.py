import importlib


def find_supernet_using_name(supernet_name):
    supernet_filename = "supernets." + supernet_name + '_supernet'
    supernetlib = importlib.import_module(supernet_filename)
    supernet = None
    target_supernet_name = supernet_name.replace('_', '') + 'supernet'
    for name, cls in supernetlib.__dict__.items():
        if name.lower() == target_supernet_name.lower():
            supernet = cls

    if supernet is None:
        print("In %s.py, there should be a class of supernet with class name that matches %s in lowercase." %
              (supernet_filename, target_supernet_name))
        exit(0)

    return supernet


def get_option_setter(supernet_name):
    supernet_class = find_supernet_using_name(supernet_name)
    return supernet_class.modify_commandline_options


def create_supernet(opt, verbose=True):
    supernet = find_supernet_using_name(opt.supernet)
    instance = supernet(opt)
    if verbose:
        print("supernet [%s] was created" % type(instance).__name__)
    return instance
