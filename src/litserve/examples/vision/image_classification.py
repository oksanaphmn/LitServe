# Copyright The Lightning AI team.
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
import base64
import io

import torch
import torchvision
from PIL import Image
from torchvision import transforms
import litserve as ls


class ImageClassifierAPI(ls.LitAPI):
    def setup(self, device):
        self.image_processing = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.model = torchvision.models.resnet50(pretrained=True).to(device)

    def decode_request(self, request):
        image = request["image_data"]
        image = base64.b64decode(image)
        image = Image.open(io.BytesIO(image))
        image = self.image_processing(image)
        return image[None, :].to(self.device)

    def batch(self, inputs):
        return torch.cat(inputs, dim=0).to(self.device)

    @torch.inference_mode
    def predict(self, x):
        outputs = self.model(x)
        _, preds = torch.max(outputs, 1)
        return preds.tolist()

    def unbatch(self, outputs):
        return outputs

    def encode_response(self, output):
        if isinstance(output, list):
            output = output[0]
        return {"label": output}


if __name__ == "__main__":
    api = ImageClassifierAPI()
    server = ls.LitServer(api, max_batch_size=4, batch_timeout=0.1)
    server.run(port=8000)
