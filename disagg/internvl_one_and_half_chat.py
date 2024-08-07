import os
import numpy as np
import triton_python_backend_utils as pb_utils
import json

os.system(f"TMPDIR=/workdir pip3 install --cache-dir=/workdir --target=/workdir \
  https://s3plus.sankuai.com/automl-pkgs/torch/torch-2.2.1+cu118-cp39-cp39-linux_x86_64.whl \
  https://s3plus.sankuai.com/automl-pkgs/torch/torchvision-0.17.1+cu118-cp39-cp39-linux_x86_64.whl \
  https://s3plus.sankuai.com/automl-pkgs/flash-attn/flash_attn-2.5.9.post1-cp39-cp39-linux_x86_64.whl \
  numpy==1.24.4 einops lmdeploy==0.4.2 tokenizers==0.19.1 transformers==4.41.2 autoawq transformers_stream_generator urllib3==1.26.6 protobuf==3.20.0 timm  \
  -i http://pypi.sankuai.com/simple --trusted-host pypi.sankuai.com --retries=0")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
os.system(f"pip3 list")

from lmdeploy import pipeline, ChatTemplateConfig
from lmdeploy.messages import TurbomindEngineConfig
from lmdeploy.vl import load_image


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to intialize any state associated with this model.

        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """
        # You must parse model_config. JSON string is not parsed here
        pwd = os.path.abspath(os.path.dirname(__file__))
        self.max_batch_size = json.loads(args['model_config'])['max_batch_size']
        self.ckpt_path = f'{pwd}/checkpoint/models--OpenGVLab--InternVL-Chat-V1-5-Quant/'

        self.pipeline = pipeline(self.ckpt_path, 
                                 model_name='OpenGVLab/InternVL-Chat-V1-5', 
                                 backend_config=TurbomindEngineConfig(cache_max_entry_count=0.2,
                                                                      quant_policy=0,
                                                                      session_len=16384,
                                                                      tp=2,
                                                                      max_batch_size=1),
                                 chat_template_config=ChatTemplateConfig(model_name='internvl-internlm2'), 
                                 log_level='INFO')

    def create_batch(self, requests):
        # 从请求列表中创建一个批次
        batch_query = []
        batch_image = []
        batch_requests = []
        # while len(batch_query) < self.max_batch_size and len(requests) > 0:
        for request in requests:
            # request = requests.pop(0)
            batch_requests.append(request)

            query = pb_utils.get_input_tensor_by_name(request, "query")
            query = query.as_numpy()

            image = pb_utils.get_input_tensor_by_name(request, "image")
            image = image.as_numpy()

            batch_query.append(query)
            batch_image.append(image)

        batch_query = np.concatenate(batch_query)
        batch_image = np.concatenate(batch_image)

        return (batch_requests, batch_query, batch_image)


    def evaluate(self, query, image):
        # print("Processing queries:", len(query))
        query = [q[0].decode('utf-8') for q in query]
        # 
        images = [load_image(img[0].decode('utf-8')) for img in image]

        prompts = list(zip(query, images))

        # print("len", len(prompts))
        # import pdb;pdb.set_trace()
        responses = self.pipeline(prompts) #, do_preprocess=False)

        res = []
        for r in responses:
          res.append(r.text.encode('utf-8'))
        
        return res

    def execute(self, requests):
        """`execute` must be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference request is made
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse

        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest

        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """
        responses = []
        
        batch_requests, batch_query, batch_image = self.create_batch(requests)

        batch_outputs = self.evaluate(batch_query, batch_image)

        # print("length:", len(batch_requests), len(batch_outputs))

        # 后处理模型输出，并为每个请求创建响应
        for output in batch_outputs:
            output = np.array(output,  dtype=np.string_)

            out_tensor_0 = pb_utils.Tensor("response", output.astype(np.string_))
            response = pb_utils.InferenceResponse(output_tensors=[out_tensor_0])
            responses.append(response)


        # print("len req:", len(requests))
        # print("len resp:", len(responses))

        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')