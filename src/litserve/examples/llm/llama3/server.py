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
import litserve as ls
from litgpt import LLM


class Llama3API(ls.LitAPI):
    def setup(self, device):
        self.llm = LLM.load(
            "checkpoints/meta-llama/Meta-Llama-3-8B-Instruct",
            device_type="auto",
        )

    def predict(self, prompt, context):
        prompt = prompt[-1]["content"]
        response = self.llm.generate(
            prompt=prompt,
            temperature=context["temperature"],
            max_new_tokens=context["max_tokens"],
        )
        yield response


if __name__ == "__main__":
    api = Llama3API()
    server = ls.LitServer(api, spec=ls.OpenAISpec())
    server.run(port=8000)
