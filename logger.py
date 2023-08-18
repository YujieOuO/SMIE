import numpy as np
import logging
from config import *

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
 
    return logger

class Log:
    @ex.capture
    def __init__(self, log_path) -> None:
        self.batch_data = dict()
        self.epoch_data = dict()
        self.max_data = {'best_epoch':-1, 'test_acc':-1}
        self.logger = get_logger(log_path)
        self.logger.info('Start')

    def update_batch(self, name, value):
        if name not in self.batch_data:
            self.batch_data[name] =  list()
        self.batch_data[name].append(value)

    @ex.capture
    def update_epoch(self, epoch, epoch_num, train_mode):
        self.logger.info('Epoch:[{}/{}]'.format(epoch + 1 , epoch_num))
        for name in self.batch_data.keys():
            if name not in self.epoch_data:
                self.epoch_data[name] = list()
            epoch_value = np.mean(self.batch_data[name])
            self.epoch_data[name].append(epoch_value)
            self.batch_data[name] = list()
            if 'test/cls_acc' in name and epoch_value > self.max_data['test_acc']:
                self.max_data['test_acc'] = epoch_value
                self.max_data['best_epoch'] = epoch
            self.logger.info("{}: {}".format(name, self.epoch_data[name][-1]))
        if "loadweight" in train_mode:
            self.logger.info("Epoch:[{}] get the best test acc: {}"
                        .format(self.max_data['best_epoch'], self.max_data['test_acc']))
