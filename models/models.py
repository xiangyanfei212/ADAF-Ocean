import copy
models = {}

def register(name):
    '''
    Model classes can be registered and managed dynamically using the @register decorator.
    '''
    def decorator(cls):
        models[name] = cls
        return cls
    return decorator


def make(model_spec, args=None, load_sd=False, load_sd_from_senseiver=False):
    '''
    The make function encapsulates model initialization logic, including argument customization and state dictionary loading.

    '''
    if args is not None:
        model_args = copy.deepcopy(model_spec['args'])
        model_args.update(args)
    else:
        model_args = model_spec['args']
    # print(f'models: {models}')
    model = models[model_spec['name']](**model_args)

    if load_sd:
        # new_state_dict = {}
        # for k, v in model_spec['state_dict'].items():
        #     new_key = k.replace("module.", "")
        #     new_state_dict[new_key] = v
        new_state_dict = {}
        for k, v in model_spec['state_dict'].items():
            if "module." in k:
                new_key = k.replace("module.", "")
            else:
                new_key = k
            new_state_dict[new_key] = v
            # print(f'k={k}, new_key:{new_key}')
        model.load_state_dict(new_state_dict)

    if load_sd_from_senseiver:
        new_state_dict = {}
        for k, v in model_spec['state_dict'].items():
            new_key = k.replace("module.", "", 1) 
            new_state_dict[new_key] = v
            # print(f'k={k}, new_key:{new_key}')
        model.load_state_dict(new_state_dict)

    return model
