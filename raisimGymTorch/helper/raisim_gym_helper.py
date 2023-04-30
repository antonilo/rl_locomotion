from ruamel.yaml import dump, RoundTripDumper
from shutil import copyfile
import datetime
import os
import ntpath

#def generate_string_from_config(config):
#    float_vals = []
#    for k in config['environment'].keys(): 
#        if k.endswith('Coeff'):
#            float_vals.append(config['environment'][k])
#    float_vals = [str(v) for v in float_vals]
#    return '_'.join(float_vals)

class ConfigurationSaver:
    def __init__(self, log_dir, save_items, config = None, overwrite = False):
        if config is not None:
            self._data_dir = log_dir + '/' #+ generate_string_from_config(config)
        else:
            self._data_dir = log_dir + '/' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        os.makedirs(self._data_dir, exist_ok = overwrite)

        if config is not None:
            with open(self._data_dir + '/cfg.yaml', 'w') as fptr:
                dump(config, stream = fptr, Dumper = RoundTripDumper)

        if save_items is not None:
            for save_item in save_items:
                base_file_name = ntpath.basename(save_item)
                copyfile(save_item, self._data_dir + '/' + base_file_name)

    @property
    def data_dir(self):
        return self._data_dir
        

def TensorboardLauncher(directory_path):
    from tensorboard import program
    import webbrowser
    # learning visualizer
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', directory_path])
    url = tb.launch()
    print("[RAISIM_GYM] Tensorboard session created: "+url)
    webbrowser.open_new(url)
