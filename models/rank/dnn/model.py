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

import math

import paddle.fluid as fluid

from paddlerec.core.utils import envs
from paddlerec.core.model import ModelBase


class Model(ModelBase):
    def __init__(self, config):
        ModelBase.__init__(self, config)

    def _init_hyper_parameters(self):
        self.is_distributed = True if envs.get_fleet_mode().upper(
        ) == "PSLIB" else False
        self.sparse_feature_number = envs.get_global_env(
            "hyper_parameters.sparse_feature_number")
        self.sparse_feature_dim = envs.get_global_env(
            "hyper_parameters.sparse_feature_dim")
        self.learning_rate = envs.get_global_env(
            "hyper_parameters.optimizer.learning_rate")
        self.dense_feature_dim = envs.get_global_env(
            "hyper_parameters.dense_feature_dim")

    def input_data(self):
        dense_input = fluid.data(name="dense_input",
                                 shape=[-1, self.dense_feature_dim],
                                 dtype="float32")

        sparse_input_ids = [
            fluid.data(name="C" + str(i),
                       shape=[-1, 1],
                       lod_level=1,
                       dtype="int64") for i in range(1, 27)
        ]

        label = fluid.data(name="label", shape=[-1, 1], dtype="float32")

        inputs = [dense_input] + sparse_input_ids + [label]
        return inputs

    def net(self, inputs, is_infer=False):
        def embedding_layer(input):
            return fluid.layers.embedding(
                input=input,
                is_sparse=True,
                size=[self.sparse_feature_number, self.sparse_feature_dim],
                param_attr=fluid.ParamAttr(
                    name="SparseFeatFactors",
                    initializer=fluid.initializer.Uniform()),
            )
        
        sparse_embed_seq = list(map(embedding_layer, inputs[1:-1]))

        concated = fluid.layers.concat(sparse_embed_seq + inputs[0:1], axis=1)

        # with fluid.device_guard("gpu"):
        fc1 = fluid.layers.fc(
            input=concated,
            size=400,
            act="relu",
            param_attr=fluid.ParamAttr(initializer=fluid.initializer.Normal(
                scale=1 / math.sqrt(concated.shape[1]))), name="fc1"
        )

        fc2 = fluid.layers.fc(
            input=fc1,
            size=400,
            act="relu",
            param_attr=fluid.ParamAttr(initializer=fluid.initializer.Normal(
                scale=1 / math.sqrt(fc1.shape[1]))), name="fc2"
        )

        fc3 = fluid.layers.fc(
            input=fc2,
            size=400,
            act="relu",
            param_attr=fluid.ParamAttr(initializer=fluid.initializer.Normal(
                scale=1 / math.sqrt(fc2.shape[1]))), name="fc3"
        )

        label = fluid.layers.cast(inputs[-1], dtype="int64")

        predict = fluid.layers.fc(
            input=fc3,
            size=2,
            act="softmax",
            param_attr=fluid.ParamAttr(initializer=fluid.initializer.Normal(
                scale=1 / math.sqrt(fc3.shape[1]))),
        )

        self.predict = predict

        auc, batch_auc, _ = fluid.layers.auc(input=predict,
                                            label=label,
                                            num_thresholds=2**12,
                                            slide_steps=20)
        fluid.layers.Print(auc, message="training auc_var")

        if is_infer:
            self._infer_results["AUC"] = auc
            self._infer_results["BATCH_AUC"] = batch_auc
            return

        self._metrics["AUC"] = auc
        self._metrics["BATCH_AUC"] = batch_auc

        cost = fluid.layers.cross_entropy(input=predict, label=label)
        avg_cost = fluid.layers.reduce_sum(cost)
        self._cost = avg_cost

    def optimizer(self):
        optimizer = fluid.optimizer.Adam(self.learning_rate, lazy_mode=True)
        return optimizer

    def infer_net(self):
        pass
