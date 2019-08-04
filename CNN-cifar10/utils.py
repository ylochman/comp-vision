import os
import json
import pandas as pd
pd.set_option('display.max_colwidth', -1)

class AverageMeter(object):
    """Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def log_schd(scheduler):
    s = scheduler.__class__.__name__ + '('
    for param in ['gamma', 'step_size']:
        s = s + param + ': ' + str(eval('scheduler.'+param)) + ', ' 
    s = s[:-2] + ')'
    return s

def log_opt(optimizer):
    s = optimizer.__class__.__name__ + '('
    for param in optimizer.defaults:
        s = s + param + ': ' + str(optimizer.defaults[param]) + ', ' 
    s = s[:-2] + ')'
    return s

def log_in_json(net, criterion, batch_size, optimizer, scheduler,
                n_epochs, device, current_time, test_accuracy=None):
    log_config_names = ['model', 'criterion', 'batch_size', 'optimizer',
                        'scheduler', 'n_epochs', 'device']
    log_configs = ['net', 'criterion', 'batch_size', 'log_opt(optimizer)',
                   'log_schd(scheduler)', 'n_epochs', 'device']
    log_dict = {}
    for name, config in zip(log_config_names, log_configs):
        log_dict[name] = str(eval(config))
    if test_accuracy is not None:
        log_dict['test_accuracy'] = "{:.2f}%".format(test_accuracy)
    json_filepath = 'logs/tensorboard/' + current_time + '/configs.json'
    with open(json_filepath, 'w') as f:
        json.dump(log_dict, f, indent=4)


def get_summary_df(logpath='./logs/tensorboard'):
    for i, timestamp in enumerate(sorted(os.listdir(logpath))):
        jsonfile = '{}/{}/configs.json'.format(logpath, timestamp)
        try:
            with open(jsonfile, 'r') as f:
                data = json.load(f)
            data['timestamp'] = timestamp
            series = pd.DataFrame(pd.Series(data))
            if i == 0:
                df = series
            else:
                df = pd.concat([df, series], axis=1, sort=False)
        except FileNotFoundError:
            print('No {} found'.format(jsonfile))
    df = df.T.set_index('timestamp')
    df.model = df.model.apply(lambda x: ' '.join(x.split('\n')))
    return df.sort_values('test_accuracy', ascending=False)