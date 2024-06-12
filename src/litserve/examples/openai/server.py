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
import os
from openai import OpenAI
import openai
import cohere
import logging

import litserve as ls

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class OpenAILitAPI(ls.LitAPI):
    def setup(self, device):
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.cohere_client = cohere.Client()

    def predict(self, prompt):
        try:
            output = self.client.chat.completions.create(messages=prompt, model="gpt-3.5-turbo")
            yield output.choices[0].message.json()
        except (openai.APIError, openai.AuthenticationError, openai.RateLimitError, openai.InternalServerError):
            logger.error("unable to connect with OpenAI API falling back to Cohere")

            message = prompt[-1]["content"]
            chat = self.cohere_client.chat(
                message=message,
                model="command-r",
            )
            yield chat.text


if __name__ == "__main__":
    api = OpenAILitAPI()
    server = ls.LitServer(api, spec=ls.OpenAISpec())
    server.run(port=8000)
