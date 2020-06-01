# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import abc
import os
import time
import sys
import yaml

from paddle import fluid

from paddlerec.core.utils import envs


class EngineMode:
    """
    There are various engine designed for different runing environment.
    """
    SINGLE = 1
    CLUSTER = 2
    LOCAL_CLUSTER = 3


class FleetMode:
    """
    Paddle Distributed train support: ParameterServer/Collective/PSlib
    """
    PS = 1
    COLLECTIVE = 2
    PSLIB = 3


class Device:
    """
    PaddleRec Support CPU/GPU, XPU will comming soon
    """
    CPU = 1
    GPU = 2
    # XPU =3


class Trainer(object):
    """R
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, config=None):
        self._status_processor = {}
        self.model = None
        self.inference_models = []
        self.increment_models = []
        self._exector_context = {}
        self._context = {'status': 'uninit', 'is_exit': False}
        self._config_yaml = config
        self._context["config_yaml"] = self._config_yaml

        with open(config, 'r') as rb:
            self._config = yaml.load(rb.read(), Loader=yaml.FullLoader)

        self._context["env"] = self._config
        self._model = {}
        self._dataset = {}
        envs.set_global_envs(self._config)
        envs.update_workspace()
        self._runner_name = envs.get_global_env("mode")
        self._context["runner_name"] = self._runner_name

        print("PaddleRec: Runner {} Begin".format(self._runner_name))
        self.which_device()
        self.which_engine()
        self.which_fleet_mode()
        self.which_executor_mode()
        self.legality_check()

    def which_device(self):
        device = envs.get_global_env("runner." + self._runner_name + ".device",
                                     default_value="CPU")
        if device.upper() == 'GPU':
            self.device = Device.GPU
            self._place = fluid.CUDAPlace(0)
            self._exe = fluid.Executor(self._place)
        elif device.upper() == "CPU":
            self.device = Device.CPU
            self._place = fluid.CPUPlace()
            self._exe = fluid.Executor(self._place)
        else:
            raise ValueError("Not Support device {}".format(device))
        self._context["exe"] = self._exe
        self._context["place"] = self._place

    def which_engine(self):
        engine = envs.get_global_env("runner." + self._runner_name + ".engine",
                                     default_value="SINGLE")
        if engine.upper() == "SINGLE":
            self.engine = EngineMode.SINGLE
            self.is_fleet = False
        elif engine.upper() == "LOCAL_CLUSTER":
            self.engine = EngineMode.LOCAL_CLUSTER
            self.is_fleet = True
        elif engine.upper() == "CLUSTER":
            self.engine = EngineMode.CLUSTER
            self.is_fleet = True
        else:
            raise ValueError("Not Support Engine {}".format(engine))
        self._context["is_fleet"] = self.is_fleet

    def which_fleet_mode(self):
        fleet_mode = envs.get_global_env("runner." + self._runner_name + ".fleet_mode",
                                         default_value="PS")
        if fleet_mode.upper() == "PS":
            self.fleet_mode = FleetMode.PS
        elif fleet_mode.upper() == "Collective":
            self.fleet_mode = FleetMode.COLLECTIVE
        elif fleet_mode.upper() == "PSLIB":
            self.fleet_mode = FleetMode.PSLIB
        else:
            raise ValueError("Not Support Fleet Mode {}".format(fleet_mode))

    def which_executor_mode(self):
        executor_mode = envs.get_global_env("runner." + self._runner_name + ".executor_mode",
                                            default_value="train")
        if executor_mode.upper() not in ["TRAIN", "INFER"]:
            raise ValueError(
                "Not Support Executor Mode {}".format(executor_mode))
        if executor_mode.upper() == "TRAIN":
            self.is_infer = False
        else:
            self.is_infer = True
        self._context["is_infer"] = self.is_infer

    def legality_check(self):
        if self.device == Device.CPU:
            assert self.fleet_mode != FleetMode.COLLECTIVE, "Not Support CPU with Collective Mode"

        if self.is_infer:
            assert self.engine == EngineMode.SINGLE, "Not Support Distributed Infer "

    @abc.abstractmethod
    def processor_register(self):
        pass

    def regist_context_processor(self, status_name, processor):
        """
        regist a processor for specify status
        """
        self._status_processor[status_name] = processor

    def context_process(self, context):
        """
        select a processor to deal specify context
        Args:
            context : context with status
        Return:
            None : run a processor for this status
        """
        if context['status'] in self._status_processor:
            self._status_processor[context['status']](context)
        else:
            self.other_status_processor(context)

    def other_status_processor(self, context):
        """
        if no processor match context.status, use defalut processor
        Return:
            None, just sleep in base
        """
        print('unknow context_status:%s, do nothing' % context['status'])
        time.sleep(60)

    def reload_train_context(self):
        """
        context maybe update timely, reload for update
        """
        pass

    def run(self):
        """
        keep running by statu context.
        """
        while True:
            self.reload_train_context()
            self.context_process(self._context)
            if self._context['is_exit']:
                break


def user_define_engine(engine_yaml):
    with open(engine_yaml, 'r') as rb:
        _config = yaml.load(rb.read(), Loader=yaml.FullLoader)
    assert _config is not None

    envs.set_runtime_environs(_config)

    train_location = envs.get_global_env("engine.file")
    train_dirname = os.path.dirname(train_location)
    base_name = os.path.splitext(os.path.basename(train_location))[0]
    sys.path.append(train_dirname)
    trainer_class = envs.lazy_instance_by_fliename(base_name,
                                                   "UserDefineTraining")
    return trainer_class
