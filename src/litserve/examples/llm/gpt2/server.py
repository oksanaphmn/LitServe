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
from transformers import pipeline
import litserve as ls


class GPT2LitAPI(ls.LitAPI):
    def setup(self, device):
        self.generator = pipeline("text-generation", model="openai-community/gpt2", device=device)

    def predict(self, prompt, context):
        temperature = context["temperature"]
        max_tokens = context["max_tokens"]
        out = self.generator(prompt, temperature=temperature, max_length=max_tokens)
        yield out[0]["generated_text"]


if __name__ == "__main__":
    api = GPT2LitAPI()
    server = ls.LitServer(api, spec=ls.OpenAISpec())
    server.run(port=8000)
